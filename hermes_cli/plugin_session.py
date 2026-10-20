"""The session a plugin hook or tool is running for, read from the turn's task-local context.

Plugin hooks and tools run in the gateway process, whose own cwd is the install root and is shared
by every session (it is never ``os.chdir``-ed). These read the calling session's key, client surface
and recorded working directory instead.
"""
from __future__ import annotations

import os


def _active_policy():
    from gateway.session_policy import active_policy
    return active_policy()


def session_key() -> str:
    """Route key of the session whose turn is running in this context, ``""`` outside one."""
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_KEY", "")


def session_surface() -> str:
    """``cli``/``tui``/``desktop``/``acp`` for a local session's turn, the messaging platform
    (``telegram``, ``discord``, ...) for a chat's turn, ``""`` outside any session."""
    policy = _active_policy()
    if policy is not None:
        return policy.platform
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_PLATFORM", "")


def session_cwd() -> str:
    """The calling session's recorded working directory (a local session's workspace, else the
    profile's terminal cwd), ``""`` when it has none. Never the gateway process's own cwd.
    In ingress hooks (``pre_gateway_dispatch``), which run before any session, it is the profile's
    terminal cwd, not the workspace of the session the message will reach."""
    from agent.runtime_cwd import scope_terminal_cwd, scoped_session_cwd
    policy = _active_policy()
    return scoped_session_cwd() or (policy.cwd if policy is not None else "") or scope_terminal_cwd().strip()


def context_surface(manager) -> str:
    """``session_surface()`` for a plugin context: the classic CLI process itself is ``cli``."""
    return "cli" if manager._cli_ref is not None else session_surface()


def context_cwd(manager) -> str:
    """``session_cwd()`` for a plugin context: the classic CLI process keeps its launch directory."""
    return os.getcwd() if manager._cli_ref is not None else session_cwd()
