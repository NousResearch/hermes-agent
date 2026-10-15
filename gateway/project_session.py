"""Durable project workspace for a gateway session started by the Telegram project picker.

The workspace is stored on the session ROW (``model_config["project_session"]``), not on the
routing entry: ``switch_session`` / compression advance / ``/resume`` replace the entry and drop its
metadata, while the row marker is carried across compression rotations
(``hermes_state_compression._CONVERSATION_MODEL_CONFIG_KEYS``) and survives a cold restart.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_MARKER_KEY = "project_session"


class ProjectWorkspaceError(Exception):
    """The session's project workspace cannot be honored; the turn must not run elsewhere."""


def project_marker(project_id: str, cwd: str) -> dict[str, str]:
    return {"project_id": project_id, "cwd": cwd}


def require_local_backend() -> None:
    """The picker registers HOST paths; only the local terminal backend can honor them.

    A docker/ssh/modal backend would reinterpret (or remap) the host path, so it is rejected
    explicitly instead of silently running in a different directory.
    """
    from tools.terminal_scope import terminal_env

    backend = (terminal_env("TERMINAL_ENV", "local") or "local").strip().lower()
    if backend != "local":
        raise ProjectWorkspaceError(
            f"Project workspaces need the local terminal backend; this profile uses '{backend}'."
        )


def route_transition_lock(runner: Any, route: str) -> asyncio.Lock:
    """One asyncio lock per route serializing session-id transitions that stop/clean a turn."""
    locks = runner.__dict__.setdefault("_route_transition_locks", {})
    return locks.setdefault(route, asyncio.Lock())


async def project_cwd_for_turn(runner: Any, session_entry: Any, source: Any) -> str:
    """The validated project workspace of this turn's session, ``""`` for an ordinary session.

    Checked on EVERY turn: a folder deleted (or a backend switched) after the session was created
    fails closed here rather than letting the runtime quietly fall back to another cwd.
    """
    from gateway.config import Platform
    from gateway.project_picker import validate_workspace

    if source.platform != Platform.TELEGRAM:
        return ""
    store = runner.session_store
    db = store._db_for_key(session_entry.session_key)
    hinted = bool(session_entry.metadata.get("project_id"))
    if db is None:
        if hinted:
            raise ProjectWorkspaceError("The project session database is unavailable.")
        return ""
    try:
        marker = await asyncio.to_thread(
            db.get_session_model_config_value, session_entry.session_id, PROJECT_MARKER_KEY
        )
    except Exception as exc:
        logger.warning("Could not read project marker for %s: %s", session_entry.session_id, exc)
        raise ProjectWorkspaceError("Could not read this session's project workspace.") from exc
    if not marker:
        return ""
    cwd = str((marker or {}).get("cwd") or "")
    if not cwd:
        raise ProjectWorkspaceError("This session's project workspace record is incomplete.")
    require_local_backend()
    try:
        return await asyncio.to_thread(validate_workspace, cwd)
    except (OSError, ValueError) as exc:
        raise ProjectWorkspaceError(
            f"The project workspace for this session is no longer available ({cwd}). "
            "Start a new session with /new or /new -project."
        ) from exc


def bind_workspace(session_key: str, session_id: str, cwd: str) -> str:
    """Pin (or release) the terminal/file-tool anchor for this turn; returns the agent's root cwd.

    Terminal calls key their cwd record on the route (``session_key``) while file tools receive the
    session id as ``task_id``, so the pin is registered under both. A ``cd`` made earlier in the SAME
    session is preserved (it lives in the route's recorded cwd); a pin left by a previous session on
    this route is dropped, so an ordinary ``/new`` never inherits the old project's directory.
    """
    from tools.terminal_tool import (
        clear_task_session_cwd, get_task_session_cwd, register_task_env_overrides,
    )

    if not cwd:
        clear_task_session_cwd(session_key)
        return ""
    pin = {"cwd_source": "session", "cwd_session_id": session_id}
    terminal_cwd = get_task_session_cwd(session_key, session_id) or cwd
    register_task_env_overrides(session_key, {"cwd": terminal_cwd, **pin})
    if session_id and session_id != session_key:
        register_task_env_overrides(session_id, {"cwd": terminal_cwd, **pin})
    return cwd


def release_workspace(session_key: str, session_id: str | None) -> None:
    """Drop a finished session's pins so a replacement session on the route starts clean."""
    from tools.terminal_tool import clear_task_session_cwd

    clear_task_session_cwd(session_key)
    if session_id:
        clear_task_session_cwd(session_id)
