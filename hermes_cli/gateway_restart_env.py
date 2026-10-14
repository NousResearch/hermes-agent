"""Restart-watcher env and argv helpers, split out of ``hermes_cli.gateway`` (that module
is past its FILE_LINES cap; moved code keeps its cap — see AGENTS.md). ``gateway`` re-imports
the two moved names so ``from hermes_cli.gateway import _restart_argv_is_host_gateway`` and
module-level monkeypatching keep working.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _restart_argv_is_host_gateway(argv: list[str]) -> bool:
    """True when *argv* relaunches the host multiplexer, not a named profile's own gateway.

    ``--profile <name>`` (other than default) is that profile. A selector-less argv is
    decided by ALREADY-SETTLED identity, never ambient coordinates alone (#93943):

    1. this process's own settled multiplex verdict (``is_multiplex_active`` — set by
       boot after ``resolve_multiplex_mode``; the gateway replaying its own restart);
    2. the live host gateway's published rendezvous record (proof the RUNNING owner
       settled multiplex — the update/fleet process replaying a foreign gateway's
       captured argv has no settled flag of its own);
    3. only then the compatibility default-root comparison.
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
    try:
        from agent.secret_scope import is_multiplex_active
        if is_multiplex_active():
            return True
    except Exception:
        pass
    # The publishing gateway's SETTLED served set, not this process's ambient home:
    # a host launched from a named profile must be replayed as the host even though
    # the replaying process (the updater) sits on the named profile's home.
    try:
        from gateway import host_rendezvous as hr
        record = hr.read_record(hr.ROLE_GATEWAY)
        if record is not None and hr.liveness_is_proven(record) and len(record.profiles) > 1:
            return True
    except Exception:
        pass
    try:
        from hermes_constants import get_default_hermes_root, get_hermes_home
        return get_hermes_home().resolve() == get_default_hermes_root().resolve()
    except Exception:
        return False


def _host_gateway_watcher_env() -> dict[str, str]:
    """Scrubbed default-profile env for a detached host-gateway respawn watcher."""
    from tools.environments.local import host_gateway_child_env
    env = host_gateway_child_env()
    env.pop("_HERMES_GATEWAY", None)
    return env


def scrub_delegate_child_env_markers(environ) -> bool:
    """Strip the delegate-child marker and any stale kanban task id — a gateway is never a
    delegate child. Returns True when something was stripped. An inherited marker fences the
    gateway's embedded dispatcher: every board write would fail with PermissionError."""
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
    stripped = False
    for key in (DELEGATED_CHILD_ENV_MARKER, "HERMES_KANBAN_TASK"):
        if environ.get(key):
            stripped = True
            environ.pop(key, None)
    if stripped:
        logger.warning(
            "gateway started with delegate-child env marker(s) (HERMES_DELEGATED_CHILD_CONTEXT / "
            "HERMES_KANBAN_TASK); stripping them — a gateway is never a delegate child"
        )
    return stripped
