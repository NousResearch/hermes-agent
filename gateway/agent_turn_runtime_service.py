"""Shared sync-turn helpers for gateway foreground agent execution."""

from __future__ import annotations

import asyncio
from typing import Any, Callable


def reuse_or_create_gateway_agent(
    *,
    session_key: str | None,
    signature: str,
    cache: dict[str, tuple[Any, str]] | None,
    cache_lock: Any,
    create_agent: Callable[[], Any],
    logger: Any,
) -> tuple[Any, bool]:
    """Return a cached agent for the session/signature or create and cache one."""

    if cache_lock and cache is not None:
        with cache_lock:
            cached = cache.get(session_key)
            if cached and cached[1] == signature:
                logger.debug("Reusing cached agent for session %s", session_key)
                return cached[0], False

    agent = create_agent()
    if cache_lock and cache is not None:
        with cache_lock:
            cache[session_key] = (agent, signature)
    logger.debug("Created new agent for session %s (sig=%s)", session_key, signature)
    return agent, True


def configure_gateway_agent_for_turn(
    *,
    agent: Any,
    progress_runtime: Any,
    stream_delta_callback: Callable[[str | None], None] | None,
    reasoning_config: dict[str, Any] | None,
    background_review_callback: Callable[[str], None],
) -> None:
    """Apply per-turn callback wiring onto an existing or cached agent."""

    agent.tool_progress_callback = progress_runtime.progress_callback
    agent.step_callback = progress_runtime.step_callback
    agent.stream_delta_callback = stream_delta_callback
    agent.status_callback = progress_runtime.status_callback
    agent.reasoning_config = reasoning_config
    agent.background_review_callback = background_review_callback


def build_gateway_background_review_callback(
    *,
    status_adapter: Any,
    status_chat_id: str | None,
    status_thread_metadata: dict[str, Any] | None,
    loop_for_step: Any,
    logger: Any,
) -> Callable[[str], None]:
    """Build the per-turn background review delivery callback for gateway chats."""

    def _background_review_send(message: str) -> None:
        if not status_adapter:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                status_adapter.send(
                    status_chat_id,
                    message,
                    metadata=status_thread_metadata,
                ),
                loop_for_step,
            )
        except Exception as exc:
            logger.debug("background_review_callback error: %s", exc)

    return _background_review_send
