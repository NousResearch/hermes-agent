"""Resolve logical working directories for Hermes-owned Relay scopes."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_CWD_SENTINELS = frozenset({"", ".", "./", "auto", "cwd"})


def _clean_cwd(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return "" if value.lower() in _CWD_SENTINELS else value


def _recorded_cwd(key: str) -> str:
    if not key:
        return ""
    from tools.terminal_tool import get_session_cwd

    return _clean_cwd(get_session_cwd(key))


def resolve_relay_scope_cwds(
    agent: Any, task_id: str, session_id: str, platform: str
) -> tuple[str, str]:
    """Return logical ``(session_cwd, turn_cwd)`` for Relay scope input.

    A task may use a worktree distinct from its owning session. Preserve remote paths as
    declared and omit unknown paths instead of substituting the Hermes host's cwd.
    """
    try:
        task_cwd = _recorded_cwd(task_id)
    except Exception:
        logger.debug("Unable to read the Relay turn cwd", exc_info=True)
        task_cwd = ""

    session_cwd = ""
    try:
        from agent.runtime_cwd import scoped_session_cwd

        session_cwd = _clean_cwd(scoped_session_cwd())
    except Exception:
        logger.debug("Unable to read the scoped Relay session cwd", exc_info=True)

    if not session_cwd:
        try:
            from gateway.session_context import get_session_env

            session_key = get_session_env("HERMES_SESSION_KEY", "")
            for key in dict.fromkeys(key for key in (session_key, session_id) if key):
                if recorded := _recorded_cwd(key):
                    session_cwd = recorded
                    break
        except Exception:
            logger.debug("Unable to read the recorded Relay session cwd", exc_info=True)

    if not session_cwd:
        session_cwd = _clean_cwd(getattr(agent, "session_cwd", None))

    backend = ""
    if not session_cwd:
        try:
            from tools.terminal_scope import terminal_env

            backend = terminal_env("TERMINAL_ENV", "local").strip().lower()
            session_cwd = _clean_cwd(terminal_env("TERMINAL_CWD", ""))
        except Exception:
            logger.debug(
                "Unable to read the configured Relay session cwd", exc_info=True
            )

    if not session_cwd and platform == "cli" and backend in {"", "local"}:
        try:
            from agent.runtime_cwd import resolve_agent_cwd

            session_cwd = _clean_cwd(str(resolve_agent_cwd()))
        except Exception:
            logger.debug("Unable to resolve the local Relay session cwd", exc_info=True)

    turn_cwd = task_cwd or session_cwd
    if platform in {"subagent", "cron"} and task_cwd:
        session_cwd = task_cwd
    return session_cwd or turn_cwd, turn_cwd
