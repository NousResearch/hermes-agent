"""The paragraph ``hermes update`` prints when this home still runs an old ``hermes dashboard`` /
``hermes serve`` backend beside (or instead of) the gateway.

Only an update whose own pre-update inventory (or restart phase) already saw such a backend, other than
a sibling profile's, looks further, so an update with none starts no process and prints exactly what it
printed before. The detection is then the report ``hermes gateway migrate --backends`` prints (``gateway_migrate_backends``):
it only reads (``ps``, ``/proc``, ``systemctl show``), never applies, stops, restarts or prompts. A failed
detection is logged, never raised: the update's outcome does not depend on it.
"""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path

logger = logging.getLogger(__name__)

#: After the profile-qualified ``hermes`` (``hermes -p NAME``), the way the migration prints its own steps.
PLAN_ARGS = "gateway migrate --backends --apply --dry-run"
#: Backends Hermes Desktop owns are its own business, never offered for migration.
_DESKTOP_OWNED = frozenset({"desktop-ssh", "desktop-local"})
#: Overall wall-clock budget for waiting on the post-update old-backend scan; on timeout the notice is skipped.
_SCAN_TIMEOUT_SECONDS = 5.0


def _serves_this_home(item, home: Path, host) -> bool:
    if not item.running and item.unit and item.scope and item.supervisor in ("systemd", "nix-systemd"):
        from hermes_cli.gateway_migrate_backends_plan import _for_this_home
        return _for_this_home(item, host)
    if item.this_home is not None:
        return bool(item.this_home)
    # A user unit that is not running and names no HERMES_HOME would run this user's own default home; a
    # system unit's user is not known here.
    if item.running or item.scope != "user":
        return False
    from hermes_cli.dashboard_procs import _normalized_home_for_compare
    from hermes_constants import _get_platform_default_hermes_home
    return (_normalized_home_for_compare(str(home))
            == _normalized_home_for_compare(str(_get_platform_default_hermes_home())))


def _describe(item) -> str:
    from hermes_cli.main_dashboard import _parse_dashboard_runtime
    runtime = _parse_dashboard_runtime(shlex.join(item.argv)) if item.argv else None
    command = f"hermes {runtime[0]}" if runtime else "hermes dashboard / hermes serve"
    where = []
    if item.unit:
        where.append(f"{item.supervisor} {item.scope + ' ' if item.scope else ''}unit {item.unit}")
    else:
        where.append({"manual": "started by hand", "detached": "running detached"}.get(item.supervisor, item.supervisor))
    where.append(f"PID {item.pid}" if item.pid else "installed, not running")
    if runtime:
        where.append(f"port {runtime[2]}")
    return f"`{command}` ({', '.join(where)})"


def plan_command(home: Path) -> str:
    """The dry-run command for *home*, typed the way the migration prints its steps (``hermes_command``:
    ``hermes -p NAME``, with the root pinned when it is not the standard one; PowerShell on Windows)."""
    from hermes_cli.gateway_migrate_backends_plan import hermes_command
    return f"{hermes_command(str(home), powershell=os.name == 'nt')} {PLAN_ARGS}"


def old_backend_notice(home: Path | None = None) -> str | None:
    """The paragraph, or None when no old backend serves *home* (the active home by default)."""
    from hermes_cli.backend_migration_marker import backend_retired_by_migration
    from hermes_cli.gateway_migrate_backends import collect_backends
    from hermes_cli.gateway_migrate_backends_host import SystemHost
    from hermes_constants import get_hermes_home
    home = Path(home) if home is not None else get_hermes_home()
    host = SystemHost(home)
    # A unit a migration stopped but could not make `disabled` (a `static` one) is not a backend left over.
    retired = backend_retired_by_migration(home)
    found = [item for item in collect_backends()
             if item.supervisor not in _DESKTOP_OWNED and (item.running or not retired)
             and _serves_this_home(item, home, host)]
    if not found:
        return None
    what = "; ".join(_describe(item) for item in found)
    noun = "an old Hermes backend" if len(found) == 1 else f"{len(found)} old Hermes backends"
    return (f"Found {noun} for this profile: {what}. Phones and browsers connected to it use a second copy "
            "of Hermes that runs its own chats, apart from the gateway. To see what moving it to the gateway "
            f"would do (this changes nothing): {plan_command(home)}")


def _sibling_profiles(plan) -> frozenset[str]:
    """The profiles the update's inventory names that are not this one (empty when this home is not a
    plain profile, so nothing is ruled out on a guess)."""
    from hermes_cli.profiles import get_active_profile_name
    mine = get_active_profile_name()
    if mine == "custom":
        return frozenset()
    return frozenset(str(name) for name in getattr(plan, "profiles", None) or ()) - {mine}


def update_saw_old_backend(plan, restart) -> bool:
    """Did this update already meet a serve/dashboard backend no Desktop owns that is not a sibling
    profile's? Reads only what the update collected (its pre-update plan and its restart bookkeeping),
    and never starts a process or opens a connection."""
    from hermes_cli.update_cmd_fleet import _is_serve_or_dashboard_unit
    from hermes_cli.update_inventory import (
        _SERVE_KINDS,
        CLIENT_OWNED_SERVE_SUPERVISORS,
        _serve_unit_matches_profile,
    )
    runtimes = [runtime for runtime in getattr(plan, "runtimes", None) or ()
                if runtime.kind in _SERVE_KINDS and runtime.supervisor not in CLIENT_OWNED_SERVE_SUPERVISORS]
    # A scope-qualified name (``user/hermes-dashboard``) is the same unit.
    units = [name for name in (str(unit).rsplit("/", 1)[-1].removesuffix(".service") for unit in (
        *getattr(restart, "restarted_services", ()), *getattr(restart, "failed_or_stale_units", ())))
        if _is_serve_or_dashboard_unit(f"{name}.service")]
    if not runtimes and not units:  # the common update: nothing else is resolved
        return False
    siblings = _sibling_profiles(plan)
    return (any(runtime.profile not in siblings for runtime in runtimes)
            or any(not any(_serve_unit_matches_profile(name, unit) for name in siblings) for unit in units))


def _run_with_deadline(fn, timeout: float):
    """Run *fn* on a daemon thread and return its result, or None on timeout or error."""
    from agent.memory_provider import spawn_context_thread
    from hermes_cli.update_cmd_common import _best_effort
    box: list = []

    def _target() -> None:
        with _best_effort("Old backend notice after update failed: %s"):
            box.append(fn())

    worker = spawn_context_thread(_target, name="update-old-backend-notice", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.debug("Old backend notice after update timed out after %.1fs", timeout)
        return None
    return box[0] if box else None


def print_old_backend_notice(plan, restart, *, timeout: float = _SCAN_TIMEOUT_SECONDS) -> None:
    """Print :func:`old_backend_notice` after an update that saw an old backend (``plan`` is the update's
    pre-update inventory, ``restart`` its restart outcome). Never raises, prints nothing on timeout or failure."""
    from hermes_cli.update_cmd_common import _best_effort
    with _best_effort("Old backend notice after update failed: %s"):  # never fails a committed update
        if not update_saw_old_backend(plan, restart):
            return
        notice = _run_with_deadline(old_backend_notice, timeout)
        if notice:
            print(f"\n{notice}")
