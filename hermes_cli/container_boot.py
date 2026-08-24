"""Container-boot reconciliation of per-profile gateway s6 services.

Wired into the image as /etc/cont-init.d/02-reconcile-profiles. Runs as root after
01-hermes-setup (the stage2 hook) has chowned the volume and seeded $HERMES_HOME, but
before s6-rc starts user services.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

log = logging.getLogger(__name__)

# Only this desired state auto-restarts; everything else (startup_failed, starting, stopped,
# missing) registers the slot down and waits for the user — no crash-loop of a broken gateway
# across `docker restart`. Older installs only have gateway_state; newer lifecycle commands
# persist desired_state separately so a transient runtime state can't erase operator intent.
_AUTOSTART_STATES = frozenset({"running"})
# Transient sub-states of a RUNNING gateway (not an operator stop, not a failed boot). A gateway
# hard-killed in one of them with no `desired_state` would otherwise stay DOWN on every later boot
# (observed: staging stranded at `draining`); map them to `running`, mirroring gateway/run.py.
# `starting` / `startup_failed` are excluded: auto-restarting a mid-boot death is the crash-loop.
# A gateway only ever reaches these while it is up and serving, so they are NOT an operator stop and NOT a
# failed boot: - `draining`  — written by the drain watcher / scale-to-zero go-dormant path when an
# in-flight quiesce begins (gateway/run.py). - `degraded`  — written when the gateway comes up with some
# platforms queued for retry, then "falls through to the normal running state" (gateway/run.py #5196): the
# process is up, serving cron + whatever platforms connected, and the reconnect watcher takes the rest from
# there. When a gateway is hard-killed *while in one of these states* (a container/VM recreate SIGTERMs it
# before `_stop_impl` reaches its terminal-state persist), the last value left in gateway_state.json is the
# transient sub-state. With no explicit `desired_state` to fall back to, treating that literal value as the
# autostart intent would leave the gateway DOWN on every subsequent boot — the gateway never comes back, the
# dashboard is up but messaging stays dark (observed on a relay-opted-in staging instance stranded at
# `draining`, 2026-06; `degraded` is the same wedge class). Map these transient sub-states to `running` so a
# stranded marker reads as the run-intent it actually represents. This mirrors gateway/run.py's #42675
# handling, which persists `running` (not the mid-shutdown `draining`) when an unexpected signal tears the
# gateway down — extended here to the case where the gateway died before it could persist anything at all.
_TRANSIENT_RUNNING_STATES = frozenset({"draining", "degraded"})
# Container-namespaced state is garbage post-restart (an equal PID is a different process).
_STALE_RUNTIME_FILES = ("gateway.pid", "processes.json")

ReconcileActionLabel = Literal["started", "registered", "skipped"]


@dataclass(frozen=True)
class ReconcileAction:
    """One profile's outcome from a single reconciliation pass."""
    profile: str
    prior_state: str | None
    action: ReconcileActionLabel
    # "clean" / "unclean" (sentinel still says running — SIGKILL/OOM/VM death) / "unknown". Boot
    # is the one place that can stamp a violent death into the volume-persisted log.
    prior_exit: str = "unknown"


def _slot_action(
    profile: str, profile_dir: Path, prior_state: str | None, start: bool) -> ReconcileAction:
    return ReconcileAction(profile=profile, prior_state=prior_state,
                           action="started" if start else "registered",
                           prior_exit=_read_prior_exit_label(profile_dir))


def reconcile_profile_gateways(
    *, hermes_home: Path, scandir: Path, dry_run: bool = False,
    container_argv: Sequence[str] | None = None) -> list[ReconcileAction]:
    """Recreate s6 service registrations for every persistent profile.

    Always registers a ``gateway-default`` slot for the root profile (the implicit profile at
    the top of ``$HERMES_HOME``): ``hermes_cli.gateway`` maps an empty profile suffix to it,
    so it is what ``hermes gateway start`` (no ``-p``) targets.

    Without it, bare ``hermes gateway start`` inside the container would land on ``s6-svc -u
    /run/service/gateway-default`` → uncaught ``CalledProcessError`` → traceback to the user (PR #30136
    review).
    """
    actions: list[ReconcileAction] = []
    # Under a multiplexing root gateway named slots are still registered but must not boot from
    # their persisted run intent, or they would become additional multiplex owners.
    from gateway.config import load_gateway_config
    from utils import is_truthy_value
    try:
        multiplex_profiles = load_gateway_config().multiplex_profiles
    except Exception:
        log.warning("Unable to load gateway configuration during container boot; using the "
                    "GATEWAY_MULTIPLEX_PROFILES override if set.", exc_info=True)
        multiplex_profiles = is_truthy_value(os.environ.get("GATEWAY_MULTIPLEX_PROFILES"))

    # A legacy `gateway run` container with no state yet seeds `running` (pre-s6 behavior).
    legacy_default_state = _maybe_migrate_legacy_gateway_run_state(
        hermes_home, container_argv=container_argv, dry_run=dry_run)
    default_prior_state = legacy_default_state or _read_desired_state(hermes_home)
    default_should_start = default_prior_state in _AUTOSTART_STATES
    if not dry_run:
        _cleanup_stale_runtime_files(hermes_home)
        _register_service(scandir, "default", start=default_should_start)
    actions.append(_slot_action("default", hermes_home, default_prior_state, default_should_start))

    profiles_root = hermes_home / "profiles"
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            # SOUL.md (seeded by `hermes profile create`) is the "real profile" marker.
            if not entry.is_dir() or not (entry / "SOUL.md").exists():
                continue
            # "default" is reserved for the root profile slot above.
            if entry.name == "default":
                log.warning("profiles/default/ exists — skipping to avoid colliding with the "
                            "reserved root-profile s6 slot")
                continue

            prior_state = _read_desired_state(entry)
            should_start = not multiplex_profiles and prior_state in _AUTOSTART_STATES
            if not dry_run:
                _cleanup_stale_runtime_files(entry)
                _register_service(scandir, entry.name, start=should_start)
            actions.append(_slot_action(entry.name, entry, prior_state, should_start))
    if not dry_run:
        _write_reconcile_log(hermes_home, actions)
    return actions


def _maybe_migrate_legacy_gateway_run_state(
    hermes_home: Path, *, container_argv: Sequence[str] | None, dry_run: bool) -> str | None:
    """Seed root gateway_state for pre-s6 `gateway run` containers (the tini image let users run
    the gateway as the container command; post-s6 it would register down and never start)."""
    state_file = hermes_home / "gateway_state.json"
    if state_file.exists():
        return None
    if os.environ.get("HERMES_GATEWAY_NO_SUPERVISE", "").lower() in ("1", "true", "yes"):
        return None
    argv = tuple(container_argv) if container_argv is not None else _read_container_argv()
    if not _is_legacy_gateway_run_request(argv):
        return None
    if not dry_run:
        import time
        state_file.write_text(json.dumps({
            "gateway_state": "running",
            "desired_state": "running",
            "timestamp": int(time.time()),
            "migrated_from": "legacy-container-cmd",
        }) + "\n", encoding="utf-8")
    return "running"


def _cmdline_argv(cmdline: Path) -> tuple[str, ...]:
    raw = cmdline.read_bytes()
    return tuple(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)


def _read_container_argv() -> tuple[str, ...]:
    """Best-effort argv of the container's main program (the one holding ``main-wrapper.sh``):
    PID 1 first (s6 v2 ``/init``), then every ``/proc/*/cmdline`` (s6 v3 ``s6-svscan``)."""
    def _cmdlines():
        yield Path("/proc/1/cmdline")
        try:
            for entry in Path("/proc").iterdir():
                if entry.name.isdigit():
                    yield entry / "cmdline"
        except OSError:
            return

    for cmdline in _cmdlines():
        try:
            argv = _cmdline_argv(cmdline)
        except OSError:
            continue
        if any("main-wrapper.sh" in part for part in argv):
            return argv
    return ()


def _strip_container_argv_prefix(argv: Sequence[str]) -> list[str]:
    """Strip the s6/wrapper prefix off the container argv, leaving the hermes args.

    Drops everything through the ``main-wrapper.sh`` token — the stable boundary the image
    owns — rather than peeling tokens positionally (which broke on the s6 v2→v3 bump).
    """
    args = list(argv)
    wrapper_idx = next((i for i, a in enumerate(args) if a.endswith("main-wrapper.sh")), None)
    if wrapper_idx is not None:
        args = args[wrapper_idx + 1 :]
    elif args and Path(args[0]).name == "init":  # defensive: `init` with no wrapper token
        args = args[1:]
    if args and args[0].endswith("entrypoint-dispatch.sh"):  # non-PID-1 dispatch shim
        args = args[1:]
    if args and Path(args[0]).name == "hermes":  # the wrapper re-execs `hermes <subcommand>`
        args = args[1:]
    return args


def _is_legacy_gateway_run_request(argv: Sequence[str]) -> bool:
    """True for Docker commands equivalent to `gateway run`."""
    args = _strip_container_argv_prefix(argv)
    if "--no-supervise" in args:
        return False
    return len(args) >= 2 and args[0] == "gateway" and args[1] == "run"


def _is_dashboard_container(argv: Sequence[str]) -> bool:
    """True when the container's command is the dashboard (which never supervises gateways)."""
    args = _strip_container_argv_prefix(argv)
    return bool(args) and args[0] == "dashboard"


def _read_desired_state(profile_dir: Path) -> str | None:
    """Persisted ``desired_state`` (operator intent), else legacy ``gateway_state``; missing or
    unparseable files count as "no state" so a corrupt file can't bork boot."""
    state_file = profile_dir / "gateway_state.json"
    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("could not read %s; treating as no prior state", state_file)
        return None
    desired_state = data.get("desired_state")
    if desired_state is not None:
        return desired_state
    gateway_state = data.get("gateway_state")
    return "running" if gateway_state in _TRANSIENT_RUNNING_STATES else gateway_state


def _cleanup_stale_runtime_files(profile_dir: Path) -> None:
    """Remove PID-namespace-bound runtime files that would confuse the new gateway's checks."""
    for name in _STALE_RUNTIME_FILES:
        (profile_dir / name).unlink(missing_ok=True)


def _read_prior_exit_label(profile_dir: Path) -> str:
    """Exception-free ``lifecycle_ledger.read_prior_exit_label`` — forensics never block boot."""
    try:
        from gateway.lifecycle_ledger import read_prior_exit_label
        return read_prior_exit_label(profile_dir)
    except Exception:
        return "unknown"


def _write_exec(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _register_service(scandir: Path, profile: str, *, start: bool) -> None:
    """Recreate the s6 service slot for one profile.

    Mirrors ``S6ServiceManager.register_profile_gateway`` but sets start state via the ``down``
    marker (cont-init.d runs before s6-svscan has a control socket). Built in a sibling temp
    dir and ``Path.replace``d into place so an interrupted write never leaves a half-built dir.

    This matches :meth:`S6ServiceManager.register_profile_gateway` (PR #30136 review item O4) — even though
    cont-init.d runs before s6-svscan starts scanning, an atomic publication keeps the contract uniform
    between the two registration paths and protects against a half-populated dir if the script is
    interrupted mid-write.
    """
    import shutil

    from hermes_cli.service_manager import (
        S6ServiceManager, _seed_supervise_skeleton, validate_profile_name)

    validate_profile_name(profile)
    service_dir = scandir / f"gateway-{profile}"
    # Dot-prefixed so s6-svscan skips the staging dir: a non-dotted name gets supervised AS ROOT
    # by a concurrent rescan, creating a root-owned ``supervise/`` → EACCES in the seed below.
    tmp_dir = service_dir.with_name("." + service_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    try:
        (tmp_dir / "type").write_text("longrun\n", encoding="utf-8")
        # Manager's own rendering keeps both registration paths consistent; per-profile env
        # comes from the profile's config.yaml, so extra_env is empty.
        _write_exec(tmp_dir / "run", S6ServiceManager._render_run_script(profile, extra_env={}))
        _write_exec(tmp_dir / "finish", S6ServiceManager._render_finish_script())
        (tmp_dir / "log").mkdir()
        _write_exec(tmp_dir / "log" / "run", S6ServiceManager._render_log_run(profile))
        if not start:  # `hermes -p <profile> gateway start` brings it up later (s6-svc -u)
            (tmp_dir / "down").touch()
        # Pre-create supervise/ with hermes ownership BEFORE publishing so s6-supervise inherits
        # it and runtime s6-svc calls as the hermes user won't EACCES.
        _seed_supervise_skeleton(tmp_dir)
        if service_dir.exists():
            shutil.rmtree(service_dir)
        tmp_dir.replace(service_dir)  # atomic publish
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


# ~3000 lines ≈ a year of daily reboots on a 5-profile container; rotated to .1 when crossed.
_LOG_ROTATE_BYTES = 256 * 1024


def _write_reconcile_log(hermes_home: Path, actions: list[ReconcileAction]) -> None:
    """Append one line per profile to $HERMES_HOME/logs/container-boot.log (rotated to ``.1``) —
    a separate greppable file for "why didn't my profile come back up".

    Size-bounded: when the file exceeds ``_LOG_ROTATE_BYTES`` (defaults to 256 KiB ≈ 3000 reconcile lines),
    the current file is renamed to ``container-boot.log.1`` (replacing any previous rotation) before the new
    entries are appended. This gives long- lived containers a soft cap of ~512 KiB across the two files
    without pulling in logrotate or s6-log machinery just for this one append-only file (PR #30136 review
    item O3).
    """
    import time
    log_dir = hermes_home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "container-boot.log"
    try:
        if log_path.exists() and log_path.stat().st_size >= _LOG_ROTATE_BYTES:
            log_path.replace(log_dir / "container-boot.log.1")
    except OSError as exc:  # non-fatal — keep appending rather than lose the entry
        log.warning("could not rotate %s: %s", log_path, exc)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with log_path.open("a", encoding="utf-8") as f:
        for a in actions:
            f.write(f"{ts} profile={a.profile} prior_state={a.prior_state} "
                    f"action={a.action} prior_exit={a.prior_exit}\n")


def main() -> int:
    """Entry point invoked from /etc/cont-init.d/02-reconcile-profiles."""
    # A dashboard-only container must not reconcile: with a shared bind-mounted HERMES_HOME both
    # containers race to flock() the same s6-log files → "Resource busy" restart storm. Detected
    # from PID 1 argv, not an operator flag (a flag can be forgotten in a hand-written manifest).
    if _is_dashboard_container(_read_container_argv()):
        print("reconcile: skipping (dashboard container — does not need per-profile gateways)")
        return 0

    hermes_home = Path(os.environ.get("HERMES_HOME", "/opt/data"))
    scandir = Path(os.environ.get("S6_PROFILE_GATEWAY_SCANDIR", "/run/service"))
    actions = reconcile_profile_gateways(hermes_home=hermes_home, scandir=scandir)
    for a in actions:
        print(f"reconcile: profile={a.profile} prior_state={a.prior_state} action={a.action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
