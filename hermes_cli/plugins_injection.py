"""Host routing for ``PluginContext.inject_message``: classic CLI REPL, Ink TUI / desktop, messaging
gateway (existing ``session_key`` or a new session at ``origin``)."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

logger = logging.getLogger("hermes_cli.plugins")


def inject_plugin_message(
    ctx: Any, content: str, role: str, *, session_key: Optional[str],
    origin: Optional[Mapping[str, Any]],
) -> bool:
    manager = ctx._manager
    msg = content if role == "user" else f"[{role}] {content}"
    cli = manager._cli_ref
    # An origin names a messaging chat; the local REPL is never that chat.
    if cli is not None and origin is None:
        queue_ = cli._interrupt_queue if getattr(cli, "_agent_running", False) else cli._pending_input
        queue_.put(msg)
        return True
    if session_key and origin is not None:
        logger.warning("inject_message: pass session_key or origin, not both")
        return False
    # Neither: the live gateway picks the chat (gateway/plugin_keyless_inject.py) from the session this
    # call works for, read here on the caller's own context; no gateway keeps main's refusal.
    keyless = not session_key and origin is None
    if keyless and not manager.has_gateway_message_injector:
        logger.warning("inject_message: gateway mode requires an existing session_key or an origin")
        return False
    if origin is not None and not isinstance(origin, Mapping):
        logger.warning("inject_message: origin must be a mapping in the SessionSource.to_dict() shape")
        return False
    if not ctx._gateway_injection_allowed():
        logger.warning("inject_message: gateway injection denied for plugin %s; set "
                       "plugins.entries.%s.allow_gateway_injection: true to allow it",
                       ctx.plugin_id, ctx.plugin_id)
        return False
    # TUI/desktop host is a different slot. It accepts only when it owns this
    # session_key; a miss falls through so a co-resident messaging gateway
    # still receives its own keys. An exception fails closed — do not also
    # hand the same text to the gateway.
    if session_key and manager.has_tui_message_injector:
        try:
            if manager.inject_tui_message(session_key=session_key, content=msg, plugin_id=ctx.plugin_id):
                return True
        except Exception:
            logger.warning("inject_message: TUI scheduling failed for plugin %s", ctx.plugin_id, exc_info=True)
            return False
    if not manager.has_gateway_message_injector:
        logger.warning("inject_message: no live gateway is available")
        return False
    # The manager's immutable home is the plugin's profile: the gateway creates an origin's session
    # there or nowhere, and a keyed local session must belong to it (never the ambient HERMES_HOME
    # of whatever thread called us).
    if keyless:
        from gateway.session_context import bound_session_key
        target: dict[str, Any] = {"keyless": True, "calling_session_key": bound_session_key(),
                                  "plugin_home": manager.home_path}
    elif origin is None:
        target = {"session_key": session_key, "plugin_home": manager.home_path}
    else:
        target = {"origin": dict(origin), "plugin_home": manager.home_path}
    try:
        return bool(manager.inject_gateway_message(**target, content=msg, plugin_id=ctx.plugin_id))
    except Exception:
        logger.warning("inject_message: gateway scheduling failed for plugin %s", ctx.plugin_id, exc_info=True)
        return False
