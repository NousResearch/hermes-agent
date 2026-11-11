"""Env selection for the detached gateway-restart watcher (``gateway._spawn_gateway_restart_watcher``).

The watcher copies its own environ into the respawned gateway, so a respawn that must not see
the launcher's env gets a scrubbed environ of its own. Moved out of the ``gateway`` facade for
the file-size ratchet; stdlib-only at module scope so the bare store Python running the tail of
``hermes update`` can import it.
"""

from __future__ import annotations

from pathlib import Path


def host_gateway_watcher_env() -> dict[str, str]:
    """Scrubbed default-profile env for a detached host-gateway respawn watcher."""
    from tools.environments.local import host_gateway_child_env

    env = host_gateway_child_env()
    env.pop("_HERMES_GATEWAY", None)
    return env


def routed_home_watcher_env(home: str) -> dict[str, str] | None:
    """Scrubbed env for a watcher respawning a gateway on a ROUTED home — ``hermes update``'s
    fleet restart of sibling profiles. The watcher copies its own environ into the respawned
    gateway, so the updater's launch-profile env must not flow through: the updater already
    loaded the ROOT profile's ``.env`` into ``os.environ``, and a sibling whose own dotenv
    lacks a platform token would resolve the inherited one (``get_env_value_prefer_dotenv``'s
    environ fallback) and steal that bot's gateway lock (#135792). ``served_profile_child_env``
    strips the launch residue, scrubs every credential, and overlays the target home's own
    secrets instead — the same env ``_spawn_detached`` and the POSIX replay already use.
    ``None`` when ``home`` IS the launcher's own (a same-profile replay keeps the launcher's
    exported env — an exported bot token may be its only credential) or when the env helpers
    are unavailable, so the respawn degrades to the plain inherited env instead of failing."""
    try:
        from hermes_constants import get_routing_process_hermes_home

        if Path(home).resolve() == get_routing_process_hermes_home().resolve():
            return None
        from tools.environments.local import served_profile_child_env

        env = served_profile_child_env(target_home=home, inherit_credentials=True)
    except Exception:  # health: allow BLE001 -- degrade-to-inherited-env boundary: an unavailable helper or unreadable home must not kill the restart
        return None
    env.pop("_HERMES_GATEWAY", None)
    return env


def watcher_env_for_restart(
    run_argv: list[str], host: bool | None, home: str | None
) -> dict[str, str] | None:
    """The detached restart watcher's own environ: ``None`` (inherit the launcher's) unless the
    respawn must not see it. Host respawns must not inherit a named launcher's dotenv; a routed
    home (a sibling profile's, or an unmapped gateway recorded on another home) gets the same
    treatment — the watcher's environ is the respawned gateway's env base (#135792)."""
    from hermes_cli.gateway import (
        _restart_argv_is_host_gateway,
    )  # late: the facade imports this module

    if _restart_argv_is_host_gateway(run_argv) if host is None else host:
        return host_gateway_watcher_env()
    if home:
        return routed_home_watcher_env(home)
    return None


def profile_home_or_none(profile: str) -> str | None:
    """The profile's HERMES_HOME dir, or ``None`` when unresolvable (a bare profile id is
    enough to keep today's home-less respawn)."""
    try:
        from hermes_cli.profiles import get_profile_dir

        return str(get_profile_dir(profile))
    except Exception:  # health: allow BLE001 -- degrade boundary: an unresolvable profile keeps the pre-home respawn
        return None
