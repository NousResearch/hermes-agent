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

    Args:
        hermes_home: The container's HERMES_HOME (typically /opt/data).
            Profiles live under ``<hermes_home>/profiles/<name>/``.
        scandir: The s6 dynamic scandir (typically /run/service). Service
            directories are created at ``<scandir>/gateway-<profile>/``.
        dry_run: When True, walk and return the action list without
            touching the filesystem. For tests and `--dry-run` debug.

    Returns:
        One :class:`ReconcileAction` per profile, in directory order.
    """
    actions: list[ReconcileAction] = []
    profiles_root = hermes_home / "profiles"
    if not profiles_root.is_dir():
        return actions

    for entry in sorted(profiles_root.iterdir()):
        if not entry.is_dir():
            continue
        # SOUL.md is always seeded by `hermes profile create` (config.yaml
        # is not — that comes later via `hermes setup`). Use it as the
        # "real profile" marker so stray dirs (backups, manual mkdir)
        # aren't picked up.
        if not (entry / "SOUL.md").exists():
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
    """
    from hermes_cli.service_manager import (
        S6ServiceManager,
        validate_profile_name,
    )

    validate_profile_name(profile)
    service_dir = scandir / f"gateway-{profile}"
    service_dir.mkdir(parents=True, exist_ok=True)

    (service_dir / "type").write_text("longrun\n")

    # Reuse the manager's run-script rendering — single source of truth
    # so register_profile_gateway and reconcile_profile_gateways stay
    # consistent. extra_env is empty here; users who need per-profile
    # env can set it via the profile's config.yaml (which the gateway
    # itself loads).
    run = service_dir / "run"
    run.write_text(S6ServiceManager._render_run_script(profile, port=0, extra_env={}))
    run.chmod(0o755)

    # Persistent log rotation (OQ8-C).
    log_subdir = service_dir / "log"
    log_subdir.mkdir(exist_ok=True)
    log_run = log_subdir / "run"
    log_run.write_text(S6ServiceManager._render_log_run(profile))
    log_run.chmod(0o755)

    # The presence of a `down` file tells s6-supervise to NOT start
    # the service when s6-svscan picks it up. User brings it up
    # explicitly with `hermes -p <profile> gateway start` (which
    # routes through the Phase 4 _dispatch_via_service_manager_if_s6
    # helper to `s6-svc -u`).
    down_marker = service_dir / "down"
    if start:
        down_marker.unlink(missing_ok=True)
    else:
        down_marker.touch()


def _write_reconcile_log(
    hermes_home: Path, actions: list[ReconcileAction],
) -> None:
    """Append one line per profile to $HERMES_HOME/logs/container-boot.log.

    Operators inspect this to debug "why didn't my profile come back
    up". Keeping a separate log file (vs. mixing into agent.log) lets
    troubleshooters grep for "profile=foo" without wading through
    unrelated activity.
    """
    import time
    log_dir = hermes_home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with (log_dir / "container-boot.log").open("a", encoding="utf-8") as f:
        for a in actions:
            f.write(
                f"{ts} profile={a.profile} prior_state={a.prior_state} "
                f"action={a.action}\n"
            )


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
