"""Resolve per-turn stream retention from all registered delivery policies."""
import logging
from typing import Any

logger = logging.getLogger("agent.conversation_loop")


def retain_stream_for_turn(agent: Any, *, user_message: Any, conversation_history: Any, turn_id: str) -> bool:
    try:
        from hermes_cli.plugins import get_plugin_manager, has_hook, invoke_hook
        if not has_hook("pre_delivery"):
            return False
        manager = get_plugin_manager()
        if not manager.pre_delivery_scope_covers_gates():
            return True
        results = invoke_hook(
            "pre_delivery_scope", session_id=getattr(agent, "session_id", "") or "",
            user_message=user_message, conversation_history=list(conversation_history or []),
            platform=getattr(agent, "platform", "") or "", turn_id=turn_id,
        )
        return any(value is True for value in results) if results and all(isinstance(value, bool) for value in results) else True
    # health: allow BLE001 -- Any policy lookup failure retains the stream; tracebacks may expose clinical hook inputs.
    except Exception as exc:
        logger.warning("pre_delivery scope lookup failed (%s); retaining stream", type(exc).__name__)
        return True
