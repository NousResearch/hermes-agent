"""Home scoping for ``hermes update``'s fleet restart (#93349).

The restart phase enumerates ``hermes-gateway*``/``hermes-serve*`` units, ``ai.hermes.gateway*``
LaunchAgents and every ``gateway run`` process on the host. Those are HOST-wide namespaces: a
second Hermes install (another ``HERMES_HOME`` root under the same account, its own checkout and
venv) shares them, and the update used to restart that install's gateway too — including the
account's real ``hermes-gateway.service`` when a scratch home ran ``hermes update``.

The fleet an update owns is the set of homes its plan inventories: the updating root plus every
``<root>/profiles/<name>``. Ownership is judged from what a runtime actually runs on — the live
process environment (``HERMES_HOME``/``HOME``, via ``_hermes_home_for_pid``), the unit's declared
``Environment=`` or the plist's pinned ``HERMES_HOME`` — never from a unit label or an argv
substring. Unknown ownership is left alone: a restart we cannot prove is ours is somebody else's
outage.
"""

from __future__ import annotations

import shlex
from contextlib import suppress
from pathlib import Path


def _resolved(path) -> Path | None:
    try:
        return Path(str(path)).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def update_scope_homes() -> set[Path]:
    """Resolved homes the running update owns: the invoking home, the install root and its profiles."""
    homes: set[Path] = set()
    with suppress(Exception):
        from hermes_constants import get_hermes_home
        if (home := _resolved(get_hermes_home())) is not None:
            homes.add(home)
    with suppress(Exception):
        from hermes_cli.update_receipt import _profile_homes
        for _profile, home in _profile_homes():
            if (resolved := _resolved(home)) is not None:
                homes.add(resolved)
    return homes


def home_in_update_scope(home, scope: set[Path] | None = None) -> bool:
    """True when *home* (a path or string) is one of the homes this update owns."""
    if not home:
        return False
    resolved = _resolved(home)
    if resolved is None:
        return False
    return resolved in (update_scope_homes() if scope is None else scope)


def gateway_pid_in_update_scope(pid: int, scope: set[Path] | None = None) -> bool | None:
    """Does gateway *pid* run on a home this update owns? ``None`` when its home cannot be read."""
    from hermes_cli.dashboard_procs import _hermes_home_for_pid
    try:
        home = _hermes_home_for_pid(pid)
    except Exception:
        return None
    if home is None:
        return None
    return home_in_update_scope(home, scope)


def partition_gateway_pids_by_scope(pids, scope: set[Path] | None = None) -> tuple[list[int], list[tuple[int, str | None]]]:
    """``(owned, foreign)`` split of *pids*; ``foreign`` pairs each PID with its home (None = unreadable)."""
    scope = update_scope_homes() if scope is None else scope
    owned: list[int] = []
    foreign: list[tuple[int, str | None]] = []
    for pid in pids:
        verdict = gateway_pid_in_update_scope(pid, scope)
        if verdict:
            owned.append(pid)
        else:
            home = None
            if verdict is False:
                with suppress(Exception):
                    from hermes_cli.dashboard_procs import _hermes_home_for_pid
                    home = _hermes_home_for_pid(pid)
            foreign.append((pid, home))
    return owned, foreign


def systemd_unit_hermes_home(scope_cmd: list, svc_name: str) -> str | None:
    """Home the systemd unit *svc_name* runs on: its live MainPID's environment first, then the
    unit's declared ``Environment=HERMES_HOME``; for a user-scope unit that declares none, the
    user's own default home. ``None`` when nothing readable names a home."""
    from hermes_cli.update_cmd_fleet import _systemctl, _unit_main_pid

    pid = _unit_main_pid(scope_cmd, svc_name)
    if pid > 0:
        with suppress(Exception):
            from hermes_cli.dashboard_procs import _hermes_home_for_pid
            if (home := _hermes_home_for_pid(pid)) is not None:
                return home
    try:
        shown = _systemctl(list(scope_cmd) + ["show", svc_name, "--property=Environment", "--value"], timeout=10)
    except Exception:
        return None
    if getattr(shown, "returncode", 1) != 0:
        return None
    env_line = (getattr(shown, "stdout", "") or "").strip()
    try:
        tokens = shlex.split(env_line)
    except ValueError:
        tokens = env_line.split()
    for token in tokens:
        key, sep, value = token.partition("=")
        if sep and key == "HERMES_HOME" and value.strip():
            return value.strip()
    if "--user" in scope_cmd:
        return str(Path.home() / ".hermes")
    return None


def systemd_unit_in_update_scope(scope_cmd: list, svc_name: str, scope: set[Path] | None = None) -> bool | None:
    """Does unit *svc_name* belong to this update? ``None`` = ownership unreadable (leave it alone)."""
    home = systemd_unit_hermes_home(scope_cmd, svc_name)
    if home is None:
        return None
    return home_in_update_scope(home, scope)


def launchd_label_foreign_home(label: str, scope: set[Path] | None = None) -> str | None:
    """The HERMES_HOME a derived launchd label's installed plist pins when that home is NOT one of
    this update's — labels are account-global, so root B's default profile derives the same bare
    ``ai.hermes.gateway`` root A installed. ``None`` = ours, or no/unreadable plist (the locate step
    decides whether a job exists; only a proven foreign home is refused)."""
    import plistlib
    with suppress(Exception):
        from gateway.launchd_service import get_launchd_plist_path
        plist_path = get_launchd_plist_path().with_name(f"{label}.plist")
        if not plist_path.exists():
            return None
        data = plistlib.loads(plist_path.read_bytes())
        pinned = str(data["EnvironmentVariables"]["HERMES_HOME"])
        return None if home_in_update_scope(pinned, scope) else pinned
    return None


def describe_skipped_runtime(kind: str, name: str, home: str | None) -> str:
    """One notice line for a runtime the update leaves alone (foreign home or unreadable ownership)."""
    if home is None:
        return f"  ↷ {name}: {kind} whose Hermes home could not be read — left alone (not restarted)"
    return f"  ↷ {name}: {kind} of another Hermes home ({home}) — left alone (not restarted)"
