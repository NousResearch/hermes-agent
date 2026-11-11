"""Environment for the detached gateway-restart watcher (``hermes_cli.gateway._spawn_gateway_restart_watcher``).

The watcher hands its own ``os.environ`` to the gateway it respawns, so its env IS that gateway's env.
The updater that spawns it has already loaded its OWN profile's ``.env`` into ``os.environ``: a
sibling profile respawned with it starts with the updater's bot tokens, and a profile that declares
none of its own (an API-only profile) polls the root bot and steals its token lock (#135792).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _host_gateway_watcher_env() -> dict[str, str]:
    """Scrubbed default-profile env for a detached host-gateway respawn watcher."""
    from tools.environments.local import host_gateway_child_env
    env = host_gateway_child_env()
    env.pop("_HERMES_GATEWAY", None)
    return env


def _argv_profile_home(run_argv: list[str]) -> Optional[str]:
    """The home a ``--profile <name>`` / ``-p <name>`` argv runs on, else None."""
    for i, part in enumerate(run_argv):
        name = None
        if part in ("--profile", "-p") and i + 1 < len(run_argv):
            name = run_argv[i + 1]
        elif part.startswith("--profile="):
            name = part.split("=", 1)[1]
        if name:
            from hermes_cli.profiles import get_profile_dir
            return str(get_profile_dir(name))
    return None


def _profile_gateway_watcher_env(home: Optional[str]) -> Optional[dict[str, str]]:
    """The respawned profile gateway's env, by the rule ``update_cmd_posix_pause._replay_env`` uses.

    A gateway respawned on the launcher's OWN named profile keeps the launcher's environment (an
    exported bot token may be its only credential): None. Any other home gets that home's own
    secrets and none of the launcher's (``served_profile_child_env``).
    """
    if not home:
        return None
    from hermes_constants import get_default_hermes_root, get_routing_process_hermes_home
    from tools.environments.local import served_profile_child_env
    target = Path(home).resolve()
    if target != get_default_hermes_root().resolve() and target == get_routing_process_hermes_home().resolve():
        return None
    env = served_profile_child_env(target_home=str(target), inherit_credentials=True)
    env.pop("_HERMES_GATEWAY", None)
    return env


def restart_watcher_env(run_argv: list[str], *, host: bool, home: Optional[str]) -> Optional[dict[str, str]]:
    """The watcher's ``env=`` (None: inherit this process's)."""
    if host:
        return _host_gateway_watcher_env()
    return _profile_gateway_watcher_env(home or _argv_profile_home(run_argv))
