"""Environment for the detached gateway-restart watcher (``hermes_cli.gateway._spawn_gateway_restart_watcher``).

The watcher hands its own ``os.environ`` to the gateway it respawns, so its env IS that gateway's env.
The updater that spawns it has already loaded its OWN profile's ``.env`` into ``os.environ``: a
sibling profile respawned with it starts with the updater's bot tokens, and a profile that declares
none of its own (an API-only profile) polls the root bot and steals its token lock (#135792).
"""

from __future__ import annotations

import os
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


def _launch_bot_token_names() -> set[str]:
    """Messaging-adapter secrets (bot tokens, app secrets) the LAUNCH profile supplied: its ``.env``,
    any dotenv this process loaded, its secret sources. Those are the root bot's identity and must not
    follow a sibling (#135792). A provider key, a proxy, or a token the user's SHELL exported is not
    the launch profile's: a manual ``hermes -p X gateway run`` from that shell has it too, so it stays."""
    from agent.secret_scope import load_env_file
    from hermes_cli.env_loader import launch_dotenv_keys, source_supplied_names
    from hermes_constants import get_routing_process_hermes_home
    from tools.environments.local_env_policy import _ADAPTER_SECRET_ENV, _home_adapter_secret_env
    from gateway.restart import is_supervised_gateway_launch
    adapter_secrets = _ADAPTER_SECRET_ENV | _home_adapter_secret_env()
    if is_supervised_gateway_launch():
        # Under systemd/launchd (``/update`` sent to a service gateway) there is no user shell: an
        # ``Environment=`` token is the unit's own profile's, never recorded in any .env.
        return set(adapter_secrets)
    supplied = (set(load_env_file(get_routing_process_hermes_home() / ".env")) | set(launch_dotenv_keys())
                | set(source_supplied_names()))
    return {name.upper() for name in supplied if name.upper() in adapter_secrets}


def profile_gateway_restart_env(home: Optional[str]) -> Optional[dict[str, str]]:
    """The env of a profile gateway an update restarts (the watcher's respawn, and the abort recovery's
    ``gateway restart`` child in ``update_restart_recovery``): the updater's minus everything the LAUNCH profile supplied
    (its ``.env``, loaded dotenvs, secret sources, bridged ``TERMINAL_*``, authorization gates:
    ``strip_launch_profile_env``) and, under a service manager, every adapter secret; then the target
    profile's own secrets on top. A key only the user's shell exported survives: a blanket credential
    scrub dropped it, and the sibling came back from ``hermes update`` unable to call its model.

    A gateway respawned on the launcher's OWN named profile keeps the launcher's environment (an
    exported bot token may be its only credential): None.
    """
    if not home:
        return None
    from agent.secret_scope import build_profile_secret_scope
    from hermes_constants import apply_scratch_tmp_env, get_default_hermes_root, get_routing_process_hermes_home
    from tools.environments.local import strip_launch_profile_env
    target = Path(home).resolve()
    if target != get_default_hermes_root().resolve() and target == get_routing_process_hermes_home().resolve():
        return None
    strip = _launch_bot_token_names()
    env = {k: v for k, v in os.environ.items() if k.upper() not in strip and k != "_HERMES_GATEWAY"}
    strip_launch_profile_env(env, target)  # what the launch .env/sources supplied, and its gates (#113270)
    env["HERMES_HOME"] = str(target)
    apply_scratch_tmp_env(env)
    env.update((k, v) for k, v in build_profile_secret_scope(target).items() if v is not None)
    return env


def restart_watcher_env(run_argv: list[str], *, host: bool, home: Optional[str]) -> dict[str, str]:
    """The watcher's ``env=``. Every respawn sheds the supervisor markers, the launcher's own profile
    included: an update armed under a systemd unit carries its ``INVOCATION_ID``, and a respawn holding
    it believes the unit launched it and skips the supervised-gateway guard (#73480)."""
    from hermes_cli.gateway_runtime_start import strip_supervisor_markers
    if host:
        env = _host_gateway_watcher_env()
    else:
        env = profile_gateway_restart_env(home or _argv_profile_home(run_argv))
        # None = the launcher's own profile: its env, exported tokens included, still minus the markers.
        env = dict(os.environ) if env is None else env  # health: allow PS-P05 -- respawns the same profile's gateway that armed it
    return strip_supervisor_markers(env)
