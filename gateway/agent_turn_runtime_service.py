"""Shared sync-turn helpers for gateway foreground agent execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from gateway.agent_execution_service import setup_gateway_stream_consumer
from gateway.agent_runtime import (
    GatewayAgentRuntimeSpec,
    agent_config_signature,
)


@dataclass(slots=True)
class GatewayPreparedTurnAgent:
    """Prepared cached-or-fresh agent plus optional stream consumer."""

    agent: Any
    stream_consumer: Any | None


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


def prepare_gateway_cached_turn_agent(
    *,
    runtime_spec: GatewayAgentRuntimeSpec,
    session_key: str | None,
    session_id: str,
    source: Any,
    progress_runtime: Any,
    reasoning_config: dict[str, Any] | None,
    streaming_config: Any,
    adapter: Any,
    thread_metadata: dict[str, Any] | None,
    stream_consumer_holder: list[Any | None] | None,
    cache: dict[str, tuple[Any, str]] | None,
    cache_lock: Any,
    create_agent: Callable[[], Any],
    status_adapter: Any,
    status_chat_id: str | None,
    status_thread_metadata: dict[str, Any] | None,
    loop_for_step: Any,
    logger: Any,
) -> GatewayPreparedTurnAgent:
    """Set up streaming, cache reuse, and per-turn agent callbacks."""

    del session_id  # Reserved for future diagnostics/tracing.

    stream_consumer, stream_delta_callback = setup_gateway_stream_consumer(
        streaming_config=streaming_config,
        adapter=adapter,
        chat_id=getattr(source, "chat_id", None),
        thread_metadata=thread_metadata,
        stream_consumer_holder=stream_consumer_holder,
        logger=logger,
    )

    signature = agent_config_signature(
        runtime_spec.turn_route["model"],
        runtime_spec.turn_route["runtime"],
        runtime_spec.enabled_toolsets,
        runtime_spec.combined_ephemeral or "",
    )
    agent, _ = reuse_or_create_gateway_agent(
        session_key=session_key,
        signature=signature,
        cache=cache,
        cache_lock=cache_lock,
        create_agent=create_agent,
        logger=logger,
    )
    configure_gateway_agent_for_turn(
        agent=agent,
        progress_runtime=progress_runtime,
        stream_delta_callback=stream_delta_callback,
        reasoning_config=reasoning_config,
        background_review_callback=build_gateway_background_review_callback(
            status_adapter=status_adapter,
            status_chat_id=status_chat_id,
            status_thread_metadata=status_thread_metadata,
            loop_for_step=loop_for_step,
            logger=logger,
        ),
    )

    return GatewayPreparedTurnAgent(
        agent=agent,
        stream_consumer=stream_consumer,
    )
