"""Optional, profile-scoped GitHub Issue ingress on the Gateway owner socket."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import threading

logger = logging.getLogger(__name__)
_SOURCE_LOCK = threading.Lock()


def _configuration():
    """Return the enabled source configuration, or ``None`` when the feature is off."""
    raw_path = os.environ.get("GITHUB_BRIDGE_CONFIG", "").strip()
    if not raw_path:
        return None
    from github_bridge.config import load_config
    path = Path(raw_path).expanduser().resolve(strict=True)
    config = load_config(path)
    if config.issue_ingress_enabled is not True:
        return None
    profile = (config.issue_ingress_executor_profile or "").strip()
    if not profile:
        return None
    from hermes_cli.profiles import resolve_profile_env
    from hermes_constants import get_hermes_home
    executor_home = Path(resolve_profile_env(profile)).resolve()
    active_home = Path(get_hermes_home()).resolve()
    if executor_home != active_home:
        return None
    return path


def _source_handlers():
    """Build handlers only in the configured executor profile and under explicit opt-in."""
    path = _configuration()
    if path is None:
        return None
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli.issue_source_handlers import build_source_handlers
    admit, poll = build_source_handlers(
        connect_owner_db=lambda: connect(board="missions"),
        config_path=path,
        enabled=True,
    )
    return admit, poll


def control_handlers(connection):
    """Handlers for the authenticated native owner control socket.

    GitHub API requests and SQLite work run outside the Gateway event loop.
    Non-owner sockets cannot trigger external reads or task creation.
    """
    async def admit(_ref, params):
        return await asyncio.to_thread(invoke_sync, "admit", connection, params)

    async def poll(_ref, params):
        return await asyncio.to_thread(invoke_sync, "poll", connection, params)

    return {"issue22-admit": admit, "issue22-poll": poll}


def invoke_sync(which, connection, params):
    if not connection.native_owner:
        return {"verdict": "REJECT", "reason": "OWNER_REQUIRED"}
    try:
        with _SOURCE_LOCK:
            handlers = _source_handlers()
            if handlers is None:
                return {"verdict": "REJECT", "reason": "INGRESS_DISABLED"}
            return handlers[0 if which == "admit" else 1](params)
    except Exception as exc:
        logger.warning("Issue 22 %s refused: %s", which, exc)
        return {"verdict": "REJECT", "reason": "OWNER_ERROR"}


def poll_enabled_source() -> None:
    """One bounded polling tick, called from profile-scoped Gateway housekeeping."""
    try:
        with _SOURCE_LOCK:
            handlers = _source_handlers()
            if handlers is None:
                return
            result = handlers[1]({})
        if result.get("verdict") not in {"NOOP", "BUSY", "ACK"}:
            logger.warning("Issue 22 poll returned %s", result.get("reason", result.get("verdict")))
    except Exception:
        logger.warning("Issue 22 poll failed; the next Gateway tick will retry", exc_info=True)
