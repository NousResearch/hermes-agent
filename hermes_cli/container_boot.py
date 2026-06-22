"""Container-boot reconciliation of per-profile gateway s6 services.

Service directories under /run/service/ live on **tmpfs** and are wiped
on every container restart. Profile directories under
``$HERMES_HOME/profiles/<name>/`` live on the persistent VOLUME, and
each one records its gateway's last state in ``gateway_state.json``.
This module bridges the two: on every container boot, walk the
persistent profiles, recreate the s6 service slots, and auto-start
only those whose last recorded state was ``running``.

Wired into the image as /etc/cont-init.d/02-reconcile-profiles by the
Dockerfile (Phase 4 Task 4.0). Runs as root after 01-hermes-setup
(the stage2 hook) has chowned the volume and seeded $HERMES_HOME, but
before s6-rc starts user services.

Without this module, every ``docker restart`` would silently wipe
every per-profile gateway, even though the user's profiles still
exist on disk.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

# Only this prior state triggers automatic restart. Everything else
# (startup_failed, starting, stopped, missing) registers the slot in
# the down state and waits for explicit user action — this avoids the
# crash-loop where a broken gateway keeps being restarted across
# `docker restart` cycles.
_AUTOSTART_STATES = frozenset({"running"})

# Stale runtime files we sweep before recreating service slots. These
# all hold container-namespaced state (PIDs, process tables) that's
# garbage post-restart — a numerically-equal PID in the new container
# is a different process. See the Risk Register in the plan.
_STALE_RUNTIME_FILES = ("gateway.pid", "processes.json")

ReconcileActionLabel = Literal["started", "registered", "skipped"]


@dataclass(frozen=True)
class ReconcileAction:
    """One profile's outcome from a single reconciliation pass."""
    profile: str
    prior_state: str | None
    action: ReconcileActionLabel


def reconcile_profile_gateways(
    *,
    hermes_home: Path,
    scandir: Path,
    dry_run: bool = False,
) -> list[ReconcileAction]:
    """Recreate s6 service registrations for every persistent profile.

    Always registers a ``gateway-default`` slot for the root profile
    (the implicit profile that lives at the top of ``$HERMES_HOME``,
    not under ``profiles/``). The dispatcher in ``hermes_cli.gateway``
    maps an empty profile suffix to ``gateway-default``, so this slot
    is what ``hermes gateway start`` (no ``-p``) targets. Without it,
    bare ``hermes gateway start`` inside the container would land on
    ``s6-svc -u /run/service/gateway-default`` → uncaught
    ``CalledProcessError`` → traceback to the user (PR #30136 review).

    The default slot's prior state is read from
    ``$HERMES_HOME/gateway_state.json`` (sibling to the profile root,
    not under ``profiles/``); stale runtime files there are swept the
    same way as for named profiles.

    Args:
        hermes_home: The container's HERMES_HOME (typically /opt/data).
            Profiles live under ``<hermes_home>/profiles/<name>/``;
            the default profile lives at ``<hermes_home>`` itself.
        scandir: The s6 dynamic scandir (typically /run/service). Service
            directories are created at ``<scandir>/gateway-<profile>/``.
        dry_run: When True, walk and return the action list without
            touching the filesystem. For tests and `--dry-run` debug.

    Returns:
        One :class:`ReconcileAction` per profile, in this order:
        ``default`` first, then named profiles in directory order.
    """
    actions: list[ReconcileAction] = []

    # Default profile — always register, even if nothing has ever
    # populated the root profile dir. The slot exists so
    # ``hermes gateway start`` (no ``-p``) has somewhere to land;
    # auto-up only when the prior state was "running" (same rule as
    # named profiles).
    default_prior_state = _read_prior_state(hermes_home)
    default_should_start = default_prior_state in _AUTOSTART_STATES
    if not dry_run:
        _cleanup_stale_runtime_files(hermes_home)
        _register_service(scandir, "default", start=default_should_start)
    actions.append(ReconcileAction(
        profile="default",
        prior_state=default_prior_state,
        action="started" if default_should_start else "registered",
    ))

    profiles_root = hermes_home / "profiles"
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if not entry.is_dir():
                continue
            # SOUL.md is always seeded by `hermes profile create` (config.yaml
            # is not — that comes later via `hermes setup`). Use it as the
            # "real profile" marker so stray dirs (backups, manual mkdir)
            # aren't picked up.
            if not (entry / "SOUL.md").exists():
                continue
            # The "default" service name is reserved for the root
            # profile (above) — if a user has somehow created a
            # ``profiles/default/`` directory, skip it to avoid the
            # slot collision. Their gateway would still be reachable
            # via ``hermes -p default-named gateway start`` if they
            # rename the directory; we don't try to disambiguate here.
            if entry.name == "default":
                log.warning(
                    "profiles/default/ exists — skipping to avoid colliding "
                    "with the reserved root-profile s6 slot",
                )
                continue

            # Skip scaffold/template dirs (leading underscore, e.g.
            # '_template'). They carry a SOUL.md but are not runnable profiles,
            # and their names fail validate_profile_name — which would otherwise
            # raise inside _register_service and abort the ENTIRE reconcile,
            # leaving every later profile's gateway slot unregistered. Genuinely
            # invalid names (e.g. uppercase) are NOT caught here and still
            # surface as a hard error below.
            if entry.name.startswith("_"):
                log.info(
                    "skipping scaffold dir profiles/%s (not a runnable profile)",
                    entry.name,
                )
                continue

            prior_state = _read_prior_state(entry)
            should_start = prior_state in _AUTOSTART_STATES

            if not dry_run:
                _cleanup_stale_runtime_files(entry)
                _register_service(scandir, entry.name, start=should_start)

            actions.append(ReconcileAction(
                profile=entry.name,
                prior_state=prior_state,
                action="started" if should_start else "registered",
            ))

    if not dry_run:
        _write_reconcile_log(hermes_home, actions)
    return actions


def _read_prior_state(profile_dir: Path) -> str | None:
    """Read gateway_state.json's ``gateway_state`` field, or None if
    missing or unparseable. Unparseable counts as "no prior state" so
    we don't bork the whole reconciliation on a corrupt file."""
    state_file = profile_dir / "gateway_state.json"
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text()).get("gateway_state")
    except (OSError, json.JSONDecodeError):
        log.warning(
            "could not read %s; treating as no prior state", state_file,
        )
        return None


def _cleanup_stale_runtime_files(profile_dir: Path) -> None:
    """Remove gateway.pid and processes.json — they reference PIDs in
    the dead container's process namespace and would otherwise confuse
    the newly-started gateway's process-mismatch checks."""
    for name in _STALE_RUNTIME_FILES:
        (profile_dir / name).unlink(missing_ok=True)


def _register_service(scandir: Path, profile: str, *, start: bool) -> None:
    """Recreate the s6 service slot for one profile.

    Mirrors the rendering in :func:`S6ServiceManager.register_profile_gateway`,
    but here we control the start state directly via the ``down`` marker
    file (s6-svscan honors it on rescan). Cannot use the manager
    directly because the cont-init.d phase runs as root before
    s6-svscan starts scanning the dynamic scandir — the manager's
    ``s6-svscanctl -a`` call would fail with no control socket.

    Atomicity: build the new layout in a sibling temp directory and
    rename it into place via :meth:`Path.replace`. This matches
    :meth:`S6ServiceManager.register_profile_gateway` (PR #30136
    review item O4) — even though cont-init.d runs before s6-svscan
    starts scanning, an atomic publication keeps the contract uniform
    between the two registration paths and protects against a
    half-populated dir if the script is interrupted mid-write.
    """
    import shutil

    from hermes_cli.service_manager import (
        S6ServiceManager,
        _seed_supervise_skeleton,
        validate_profile_name,
    )

    validate_profile_name(profile)
    service_dir = scandir / f"gateway-{profile}"
    tmp_dir = service_dir.with_name(service_dir.name + ".tmp")

    # Wipe any leftover tmp from a previous interrupted run.
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    try:
        (tmp_dir / "type").write_text("longrun\n")

        # Reuse the manager's run-script rendering — single source of
        # truth so register_profile_gateway and reconcile_profile_gateways
        # stay consistent. extra_env is empty here; users who need
        # per-profile env can set it via the profile's config.yaml
        # (which the gateway itself loads).
        run = tmp_dir / "run"
        run.write_text(S6ServiceManager._render_run_script(profile, extra_env={}))
        run.chmod(0o755)

        # Persistent log rotation (OQ8-C).
        log_subdir = tmp_dir / "log"
        log_subdir.mkdir()
        log_run = log_subdir / "run"
        log_run.write_text(S6ServiceManager._render_log_run(profile))
        log_run.chmod(0o755)

        # The presence of a `down` file tells s6-supervise to NOT
        # start the service when s6-svscan picks it up. User brings
        # it up explicitly with `hermes -p <profile> gateway start`
        # (which routes through the Phase 4
        # _dispatch_via_service_manager_if_s6 helper to `s6-svc -u`).
        if not start:
            (tmp_dir / "down").touch()

        # Pre-create the supervise/ skeleton with hermes ownership
        # BEFORE we publish the slot. Mirrors the same pre-creation
        # step in S6ServiceManager.register_profile_gateway — when
        # s6-svscan picks the published slot up, the s6-supervise it
        # spawns will EEXIST our dirs/FIFOs and inherit hermes
        # ownership, so runtime s6-svc / s6-svstat / s6-svwait calls
        # (all dispatched as the hermes user) won't hit EACCES. See
        # ``_seed_supervise_skeleton`` in service_manager.py for the
        # full rationale.
        _seed_supervise_skeleton(tmp_dir)

        # Publish atomically. Path.replace handles the existing-target
        # case the same way os.rename does on POSIX: the target is
        # silently replaced, so a previous reconcile pass's slot is
        # cleanly overwritten in one operation.
        if service_dir.exists():
            shutil.rmtree(service_dir)
        tmp_dir.replace(service_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _write_reconcile_log(
    hermes_home: Path, actions: list[ReconcileAction],
) -> None:
    """Append one line per profile to $HERMES_HOME/logs/container-boot.log.

    Operators inspect this to debug "why didn't my profile come back
    up". Keeping a separate log file (vs. mixing into agent.log) lets
    troubleshooters grep for "profile=foo" without wading through
    unrelated activity.

    Size-bounded: when the file exceeds ``_LOG_ROTATE_BYTES``
    (defaults to 256 KiB ≈ 3000 reconcile lines), the current file
    is renamed to ``container-boot.log.1`` (replacing any previous
    rotation) before the new entries are appended. This gives long-
    lived containers a soft cap of ~512 KiB across the two files
    without pulling in logrotate or s6-log machinery just for this
    one append-only file (PR #30136 review item O3).
    """
    import time
    log_dir = hermes_home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "container-boot.log"

    # Rotate before opening to append, so the new entries always land
    # in a fresh file when we crossed the threshold last time.
    try:
        if log_path.exists() and log_path.stat().st_size >= _LOG_ROTATE_BYTES:
            log_path.replace(log_dir / "container-boot.log.1")
    except OSError as exc:
        # Rotation failure is non-fatal — keep appending to the
        # existing file rather than losing the entry entirely.
        log.warning("could not rotate %s: %s", log_path, exc)

    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with log_path.open("a", encoding="utf-8") as f:
        for a in actions:
            f.write(
                f"{ts} profile={a.profile} prior_state={a.prior_state} "
                f"action={a.action}\n"
            )


# 256 KiB soft cap on container-boot.log; rotated to .1 when crossed.
# At ~80 B per reconcile-action line this is ~3000 lines, or about a
# year of daily reboots on a 5-profile container. Two files = ~512 KiB
# worst case. Tuned for visibility (small enough to grep / cat without
# scrolling forever) more than space (the persistent volume has GB).
_LOG_ROTATE_BYTES = 256 * 1024


def main() -> int:
    """Entry point invoked from /etc/cont-init.d/02-reconcile-profiles."""
    hermes_home = Path(os.environ.get("HERMES_HOME", "/opt/data"))
    scandir = Path(os.environ.get("S6_PROFILE_GATEWAY_SCANDIR", "/run/service"))
    actions = reconcile_profile_gateways(
        hermes_home=hermes_home, scandir=scandir,
    )
    for a in actions:
        print(
            f"reconcile: profile={a.profile} "
            f"prior_state={a.prior_state} action={a.action}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
