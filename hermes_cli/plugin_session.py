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


def session_client() -> dict | None:
    """The client that created the calling local session, as it declared itself on ``session.create``:
    a copy of ``{"name": ..., "version": ...}`` (``version`` only when sent). ``None`` outside a local
    session, for a messaging chat, and for a client that sent no ``client`` (Ink, Desktop, the CLI and
    ACP today). A label the client chose, not an authentication: never grant anything on it.
    Under ``plugins.isolation: host`` the plugin's own process has no session context; read
    ``info["client"]`` in a prompt section there."""
    from gateway.session_client import policy_client
    return policy_client(_active_policy())


def session_cwd() -> str:
    """The calling session's recorded working directory (a local session's workspace, else the
    profile's terminal cwd), ``""`` when it has none. Never the gateway process's own cwd.
    In ingress hooks (``pre_gateway_dispatch``), which run before any session, it is the profile's
    terminal cwd, not the workspace of the session the message will reach."""
    from agent.runtime_cwd import scope_terminal_cwd, scoped_session_cwd
    policy = _active_policy()
    return scoped_session_cwd() or (policy.cwd if policy is not None else "") or scope_terminal_cwd().strip()


_LOCAL_SURFACES = frozenset({"cli", "tui", "desktop", "acp"})


def _gateway_live_sessions(runner, home) -> list[dict]:
    from gateway.session_authorities import authority_for_home
    from gateway.session_policy import policy_for_source
    authority = authority_for_home(runner, home)
    if authority is None:
        return []
    rows = []
    for session_id, live in authority.sessions.copy().items():
        try:
            policy = policy_for_source(runner, live.source)  # None for a messaging chat or history mount
            if policy is None or policy.platform not in _LOCAL_SURFACES:
                continue
            row = authority.persistence.sessions.list_row(session_id)
            if row is not None and row.ended_at is not None:
                continue
            rows.append({"session_key": live.route, "session_id": session_id, "title": (row.title if row else "") or "",
                         "cwd": policy.cwd or "", "surface": policy.platform,
                         "busy": live.task is not None and not live.task.done()})
        except Exception:  # a session ending under us is simply not live
            continue
    return rows


def live_sessions(manager) -> list[dict]:
    """Live interactive local sessions this process hosts for *manager*'s profile: the gateway's,
    then a TUI/desktop host's, deduplicated by ``session_key`` (first wins). Reads only; ``[]`` when
    this process hosts none."""
    rows = []
    if (gateway := manager._gateway_message_injector) is not None:
        rows += _gateway_live_sessions(gateway[0], manager.home_path)
    if manager.has_tui_message_injector:
        from tui_gateway import server
        rows += server.live_tui_sessions(manager.home_path)
    unique = {}
    for row in rows:
        unique.setdefault(row["session_key"], row)
    return list(unique.values())


def context_surface(manager) -> str:
    """``session_surface()`` for a plugin context: the classic CLI process itself is ``cli``."""
    return "cli" if manager._cli_ref is not None else session_surface()


def context_cwd(manager) -> str:
    """``session_cwd()`` for a plugin context: the classic CLI process keeps its launch directory."""
    return os.getcwd() if manager._cli_ref is not None else session_cwd()
