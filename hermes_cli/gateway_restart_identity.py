"""Who a detached gateway respawn relaunches: the host multiplexer or one named profile.

The restart watcher (``hermes_cli.gateway._spawn_gateway_restart_watcher``) respawns a captured or
profile-derived ``gateway run`` after the old PID exits. The respawn has no supervisor marker, so a
selector-less argv follows the sticky ``active_profile`` (#22502). The host is therefore settled
here (from live evidence) and pinned with ``--profile default`` before the spawn (#132645).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def restart_argv_is_host_gateway(argv: list[str]) -> bool:
    """True when *argv* relaunches the host multiplexer, not a named profile's own gateway.

    ``--profile <name>`` (other than default) is that profile. A selector-less argv is
    decided by ALREADY-SETTLED identity, never ambient coordinates alone (#93943):

    1. this process's own settled multiplex verdict (``is_multiplex_active`` — set by
       boot after ``resolve_multiplex_mode``; the gateway replaying its own restart);
    2. the live host gateway's published rendezvous record (proof the RUNNING owner
       settled what it serves — the update/fleet process replaying a foreign gateway's
       captured argv has no settled flag of its own): a multiplex roster, or exactly
       the default profile (a single-profile host is still the host);
    3. only then the compatibility default-root comparison.

    Steps 2-3 only hold while the gateway is alive or the caller sits on the default
    root, so callers that replay after the process is gone settle this before stopping
    it and pass ``host=`` explicitly.
    """
    if not argv or "gateway" not in argv:
        return False
    for flag in ("--profile", "-p"):
        if flag in argv:
            idx = argv.index(flag)
            name = argv[idx + 1] if idx + 1 < len(argv) else ""
            return name == "default"
    if any(part == "--profile=default" for part in argv):
        return True
    if any(part.startswith("--profile=") and not part.endswith("=default") for part in argv):
        return False
    from agent.secret_scope import is_multiplex_active
    from gateway import host_rendezvous as hr
    from hermes_constants import get_default_hermes_root, get_hermes_home

    if is_multiplex_active():
        return True
    # The publishing gateway's SETTLED served set, not this process's ambient home:
    # a host launched from a named profile must be replayed as the host even though
    # the replaying process (the updater) sits on the named profile's home.
    # ``read_record`` already maps unreadable/corrupt records to None; OSError is the state dir.
    try:
        record = hr.read_record(hr.ROLE_GATEWAY)
        if record is not None and hr.liveness_is_proven(record):
            served = tuple(record.profiles)
            if len(served) > 1 or served == ("default",):
                return True
    except OSError as exc:
        logger.debug("host rendezvous record unreadable while classifying a restart argv: %s", exc)
    try:
        return get_hermes_home().resolve() == get_default_hermes_root().resolve()
    except OSError:
        return False


def host_gateway_watcher_env() -> dict[str, str]:
    """Scrubbed default-profile env for a detached host-gateway respawn watcher."""
    from tools.environments.local import host_gateway_child_env
    env = host_gateway_child_env()
    env.pop("_HERMES_GATEWAY", None)
    return env


def pin_host_profile_selector(argv: list[str]) -> list[str]:
    """Insert ``--profile default`` before the ``gateway`` subcommand when *argv* names no profile.

    A selector-less ``gateway run`` resolves its home from the sticky ``active_profile`` file
    (#22502) unless a supervisor marker is set, and a detached restart watcher sets none. With
    ``hermes profile use <named>`` in effect, the respawned host gateway re-homed into that profile
    and was refused ("Profile '<named>' does not get a gateway of its own"), so every ``hermes
    update`` left the multiplex host down while reporting the restart as done. Pinning the
    selector keeps the respawn on the host identity the caller already settled.
    """
    if any(part in ("--profile", "-p") or part.startswith("--profile=") for part in argv):
        return list(argv)
    try:
        idx = argv.index("gateway")
    except ValueError:
        return list(argv)
    return [*argv[:idx], "--profile", "default", *argv[idx:]]
