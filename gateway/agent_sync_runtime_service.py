"""Shared sync worker assembly for gateway foreground agent turns."""

from __future__ import annotations

import os
from typing import Any, Callable


def execute_gateway_sync_turn_for_runner(
    runner: Any,
    *,
    message: str,
    context_prompt: str,
    history: list[dict[str, Any]],
    source: Any,
    session_id: str,
    session_key: str | None,
    user_config: dict[str, Any],
    enabled_toolsets: list[str],
    progress_runtime: Any,
    status_adapter: Any,
    status_chat_id: str | None,
    status_thread_metadata: dict[str, Any] | None,
    loop_for_step: Any,
    admin_user_ids: list[str] | None,
    is_admin_user: bool | None,
    raw_message: Any,
    event: Any,
    env_path: Any,
    load_dotenv_fn: Callable[..., Any],
    resolve_runtime_agent_kwargs_fn: Callable[[], dict[str, Any]],
    resolve_gateway_model_fn: Callable[[dict[str, Any] | None], str],
    prepare_sync_runtime_fn: Callable[..., Any],
    prepare_cached_turn_agent_fn: Callable[..., Any],
    create_gateway_agent_fn: Callable[..., Any],
    execute_sync_turn_fn: Callable[..., Any],
    empty_response_fallback: Callable[[str], str | None],
    agent_holder: list[Any | None],
    result_holder: list[dict[str, Any] | None],
    tools_holder: list[list[Any] | None],
    stream_consumer_holder: list[Any | None],
    logger: Any,
) -> dict[str, Any]:
    """Run the synchronous foreground-turn execution path for GatewayRunner."""

    os.environ["HERMES_SESSION_KEY"] = session_key or ""

    try:
        prepared_runtime = prepare_sync_runtime_fn(
            env_path=env_path,
            load_dotenv_fn=load_dotenv_fn,
            resolve_runtime_agent_kwargs_fn=resolve_runtime_agent_kwargs_fn,
            load_reasoning_config_fn=runner._load_reasoning_config,
            source=source,
            user_message=message,
            context_prompt=context_prompt,
            gateway_ephemeral_system_prompt=getattr(runner, "_ephemeral_system_prompt", ""),
            provider_routing=getattr(runner, "_provider_routing", {}),
            fallback_model=getattr(runner, "_fallback_model", None),
            smart_model_routing=getattr(runner, "_smart_model_routing", {}),
            user_config=user_config,
            model=resolve_gateway_model_fn(user_config),
            enabled_toolsets=enabled_toolsets,
        )
    except Exception as exc:
        return {
            "final_response": f"⚠️ Provider authentication failed: {exc}",
            "messages": [],
            "api_calls": 0,
            "tools": [],
        }

    runtime_spec = prepared_runtime.runtime_spec
    reasoning_config = prepared_runtime.reasoning_config
    max_iterations = prepared_runtime.max_iterations
    runner._reasoning_config = reasoning_config

    prepared_agent = prepare_cached_turn_agent_fn(
        runtime_spec=runtime_spec,
        session_key=session_key,
        session_id=session_id,
        source=source,
        progress_runtime=progress_runtime,
        reasoning_config=reasoning_config,
        streaming_config=getattr(getattr(runner, "config", None), "streaming", None),
        adapter=runner.adapters.get(source.platform),
        thread_metadata=status_thread_metadata,
        stream_consumer_holder=stream_consumer_holder,
        cache=getattr(runner, "_agent_cache", None),
        cache_lock=getattr(runner, "_agent_cache_lock", None),
        create_agent=lambda: create_gateway_agent_fn(
            runtime_spec=runtime_spec,
            session_id=session_id,
            source=source,
            session_db=runner._session_db,
            prefill_messages=runner._prefill_messages or None,
            max_iterations=max_iterations,
            enabled_toolsets=runtime_spec.enabled_toolsets,
            quiet_mode=True,
            verbose_logging=False,
        ),
        status_adapter=status_adapter,
        status_chat_id=status_chat_id,
        status_thread_metadata=status_thread_metadata,
        loop_for_step=loop_for_step,
        logger=logger,
    )
    agent = prepared_agent.agent
    stream_consumer = prepared_agent.stream_consumer

    agent_holder[0] = agent
    tools_holder[0] = agent.tools if hasattr(agent, "tools") else None

    outcome = execute_sync_turn_fn(
        agent=agent,
        message=message,
        history=history,
        session_id=session_id,
        session_key=session_key,
        admin_user_ids=admin_user_ids,
        is_admin_user=is_admin_user,
        status_adapter=status_adapter,
        status_chat_id=status_chat_id,
        status_thread_metadata=status_thread_metadata,
        loop_for_step=loop_for_step,
        logger=logger,
        admin_only_message_builder=lambda action: runner._admin_only_message(
            source,
            action,
        ),
        stream_consumer=stream_consumer,
        session_store=getattr(runner, "session_store", None),
        session_db=runner._session_db,
        empty_response_fallback=lambda empty_kind: empty_response_fallback(empty_kind),
        pending_model_notes=getattr(runner, "_pending_model_notes", {}),
    )
    result_holder[0] = outcome.result
    return outcome.final_result
