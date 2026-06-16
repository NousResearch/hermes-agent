"""Minimal dashboard-auth mobile bridge helpers.

This module intentionally does not revive the older ``/mobile-native`` route
stack.  It provides a small BFF-style helper that can be called by the
dashboard API after the existing dashboard auth middleware has accepted the
request.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional


_log = logging.getLogger(__name__)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _best_effort_cleanup(label: str, cleanup: Callable[[], Any]) -> None:
    try:
        await _maybe_await(cleanup())
    except Exception:
        _log.debug("Mobile bridge cleanup failed: %s", label, exc_info=True)


async def _release_agent_clients(agent: Any) -> None:
    release_clients = getattr(agent, "release_clients", None)
    if callable(release_clients):
        await _best_effort_cleanup("agent.release_clients", release_clients)


async def _disconnect_adapter(adapter: Any) -> None:
    disconnect = getattr(adapter, "disconnect", None)
    if callable(disconnect):
        await _best_effort_cleanup("adapter.disconnect", disconnect)


async def _close_adapter_session_db(adapter: Any) -> None:
    session_db = getattr(adapter, "_session_db", None)
    close = getattr(session_db, "close", None)
    if callable(close):
        await _best_effort_cleanup("adapter._session_db.close", close)


def _default_adapter_factory() -> Any:
    """Create an API-server adapter instance without starting an HTTP server."""
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    return APIServerAdapter(PlatformConfig(enabled=True))


def _redacted_mobile_response(
    result: Any,
    usage: Optional[Dict[str, Any]],
    *,
    fallback_session_id: str,
) -> Dict[str, Any]:
    if isinstance(result, dict):
        session_id = result.get("session_id") or fallback_session_id
        final_response = result.get("final_response") or ""
    else:
        session_id = fallback_session_id
        final_response = ""

    if not isinstance(final_response, str):
        final_response = str(final_response)

    return {
        "object": "hermes.mobile.chat.completion",
        "session_id": session_id,
        "message": {
            "role": "assistant",
            "content_redacted": True,
            "content_length": len(final_response),
        },
        "usage": usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


async def run_mobile_dashboard_chat_turn(
    *,
    session_id: str,
    user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    system_message: Optional[str] = None,
    gateway_session_key: Optional[str] = None,
    adapter_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Run one dashboard-auth mobile chat turn with soft cleanup only.

    The owned adapter/agent lifetime is per call, but the cleanup boundary is
    deliberately soft for the agent: call ``agent.release_clients()`` and never
    call ``agent.close()`` or ``shutdown_memory_provider()``.  Adapter-owned
    resources (notably ``ResponseStore``) are disconnected after the turn.
    """
    factory = adapter_factory or _default_adapter_factory
    adapter = factory()
    agent_ref: list[Any] = [None]

    try:
        result, usage = await adapter._run_agent(
            user_message=user_message,
            conversation_history=conversation_history or [],
            ephemeral_system_prompt=system_message,
            session_id=session_id,
            gateway_session_key=gateway_session_key or session_id,
            agent_ref=agent_ref,
        )
        return _redacted_mobile_response(
            result,
            usage,
            fallback_session_id=session_id,
        )
    finally:
        agent = agent_ref[0]
        if agent is not None:
            await _release_agent_clients(agent)
        await _disconnect_adapter(adapter)
        await _close_adapter_session_db(adapter)
