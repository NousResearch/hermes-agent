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


class PluginSessionContextMixin:
    """Public task-local session and cancellation helpers for PluginContext."""

    def session_key(self) -> str:
        """The calling conversation's routing key, or ``""`` outside a session.

        Capture this in a hook or tool before starting background work, then pass it
        to ``inject_message(session_key=...)``. A session id is not a routing key.
        """
        from gateway.session_context import bound_session_key
        return bound_session_key()

    def session_cwd(self) -> str:
        """The calling session's recorded cwd, never the gateway's own (``plugin_session``)."""
        from hermes_cli.plugin_session import context_cwd
        return context_cwd(self._manager)

    def session_supports_async_delivery(self) -> bool:
        """Whether the calling surface can receive input after its current turn.

        This is a surface capability, not permission to inject a message. Stateless
        API and finite automation contexts can explicitly opt out of delivery.
        """
        from gateway.session_context import async_delivery_supported
        from gateway.session_finite import finite_turn_required
        return async_delivery_supported() and finite_turn_required() is not True

    def is_interrupted(self) -> bool:
        """Whether the originating tool worker was interrupted, including in a plugin host.

        Long waits should poll this on the executing handler and release any
        pending external wait before returning. A fresh background task has no
        authority to borrow another worker's cancellation state.
        """
        from tools.interrupt import is_interrupted
        return is_interrupted()

    def session_surface(self) -> str:
        """The calling session's client (``cli``/``tui``/``desktop``/``acp``) or messaging platform."""
        from hermes_cli.plugin_session import context_surface
        return context_surface(self._manager)
