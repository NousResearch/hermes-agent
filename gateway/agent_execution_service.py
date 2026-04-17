"""Shared gateway agent execution helpers."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import re
from typing import Any, Callable, Iterator

from gateway.agent_runtime import GatewayAgentRuntimeSpec


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GatewaySyncTurnOutcome:
    """Normalized output from one sync gateway agent turn."""

    result: dict[str, Any]
    final_result: dict[str, Any]
    tools: list[dict[str, Any]] | None


def create_gateway_agent(
    *,
    runtime_spec: GatewayAgentRuntimeSpec,
    session_id: str,
    source: Any,
    session_db: Any = None,
    prefill_messages: list[dict[str, Any]] | None = None,
    max_iterations: int | None = None,
    enabled_toolsets: list[str] | None = None,
    quiet_mode: bool = True,
    verbose_logging: bool = False,
    skip_memory: bool = False,
    skip_context_files: bool = False,
    persist_session: bool | None = None,
) -> Any:
    """Construct an AIAgent from a shared runtime spec."""
    from run_agent import AIAgent

    agent_kwargs = {
        "model": runtime_spec.turn_route["model"],
        **runtime_spec.turn_route["runtime"],
        "max_iterations": int(max_iterations or runtime_spec.max_iterations),
        "quiet_mode": quiet_mode,
        "verbose_logging": verbose_logging,
        "enabled_toolsets": list(enabled_toolsets or runtime_spec.enabled_toolsets),
        "ephemeral_system_prompt": runtime_spec.combined_ephemeral,
        "reasoning_config": runtime_spec.reasoning_config,
        "providers_allowed": runtime_spec.provider_routing.get("only"),
        "providers_ignored": runtime_spec.provider_routing.get("ignore"),
        "providers_order": runtime_spec.provider_routing.get("order"),
        "provider_sort": runtime_spec.provider_routing.get("sort"),
        "provider_require_parameters": runtime_spec.provider_routing.get("require_parameters", False),
        "provider_data_collection": runtime_spec.provider_routing.get("data_collection"),
        "session_id": session_id,
        "platform": getattr(getattr(source, "platform", None), "value", "unknown") if getattr(source, "platform", None) else "unknown",
        "user_id": getattr(source, "user_id", None),
        "fallback_model": runtime_spec.fallback_model,
    }
    if session_db is not None:
        agent_kwargs["session_db"] = session_db
    if prefill_messages is not None:
        agent_kwargs["prefill_messages"] = prefill_messages
    if skip_memory:
        agent_kwargs["skip_memory"] = True
    if skip_context_files:
        agent_kwargs["skip_context_files"] = True
    if persist_session is not None:
        agent_kwargs["persist_session"] = persist_session
    return AIAgent(**agent_kwargs)


def setup_gateway_stream_consumer(
    *,
    streaming_config: Any,
    adapter: Any,
    chat_id: str | None,
    thread_metadata: dict[str, Any] | None,
    stream_consumer_holder: list[Any | None] | None,
    logger,
) -> tuple[Any | None, Callable[[str | None], None] | None]:
    """Create the per-turn stream consumer if gateway streaming is enabled."""

    stream_consumer = None
    stream_delta_callback = None
    streaming_cfg = streaming_config
    if streaming_cfg is None:
        from gateway.config import StreamingConfig

        streaming_cfg = StreamingConfig()

    if getattr(streaming_cfg, "enabled", False) and getattr(streaming_cfg, "transport", "") != "off":
        try:
            from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

            if adapter:
                consumer_cfg = StreamConsumerConfig(
                    edit_interval=streaming_cfg.edit_interval,
                    buffer_threshold=streaming_cfg.buffer_threshold,
                    cursor=streaming_cfg.cursor,
                )
                stream_consumer = GatewayStreamConsumer(
                    adapter=adapter,
                    chat_id=chat_id,
                    config=consumer_cfg,
                    metadata=thread_metadata,
                )
                stream_delta_callback = stream_consumer.on_delta
                if stream_consumer_holder is not None:
                    stream_consumer_holder[0] = stream_consumer
        except Exception as exc:
            logger.debug("Could not set up stream consumer: %s", exc)

    return stream_consumer, stream_delta_callback


def normalize_conversation_history(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert stored gateway transcript items into agent-ready messages."""
    agent_history: list[dict[str, Any]] = []
    for msg in list(history or []):
        role = msg.get("role")
        if not role or role in {"session_meta", "system"}:
            continue

        has_tool_calls = "tool_calls" in msg
        has_tool_call_id = "tool_call_id" in msg
        is_tool_message = role == "tool"
        if has_tool_calls or has_tool_call_id or is_tool_message:
            clean_msg = {k: v for k, v in msg.items() if k != "timestamp"}
            agent_history.append(clean_msg)
            continue

        content = msg.get("content")
        if not content:
            continue
        if msg.get("mirror"):
            mirror_src = msg.get("mirror_source", "another session")
            content = f"[Delivered from {mirror_src}] {content}"
        entry = {"role": role, "content": content}
        if role == "assistant":
            for reasoning_key in ("reasoning", "reasoning_details", "codex_reasoning_items"):
                reasoning_value = msg.get(reasoning_key)
                if reasoning_value:
                    entry[reasoning_key] = reasoning_value
        agent_history.append(entry)
    return agent_history


def collect_history_media_paths(agent_history: list[dict[str, Any]]) -> set[str]:
    """Collect already-seen MEDIA: tags from prior tool messages."""
    history_media_paths: set[str] = set()
    for msg in agent_history:
        if msg.get("role") not in {"tool", "function"}:
            continue
        content = str(msg.get("content", "") or "")
        if "MEDIA:" not in content:
            continue
        for match in re.finditer(r"MEDIA:(\S+)", content):
            path = match.group(1).strip().rstrip('",}')
            if path:
                history_media_paths.add(path)
    return history_media_paths


def prepend_pending_model_switch_note(
    message: str,
    pending_model_note: str | None,
) -> str:
    """Prepend a pending model-switch note to the next user message."""

    if not pending_model_note:
        return message
    return f"{pending_model_note}\n\n{message}"


def build_gateway_btw_prompt(question: str) -> str:
    """Build the ephemeral side-question prompt used by /btw background jobs."""

    return (
        "[Ephemeral /btw side question. Answer using the conversation "
        "context. No tools available. Be direct and concise.]\n\n"
        + str(question or "")
    )


def build_gateway_approval_notify_sync(
    *,
    status_adapter: Any,
    status_chat_id: str,
    status_thread_metadata: dict[str, Any] | None,
    loop_for_step: Any,
    approval_session_key: str,
    logger,
    admin_only_message_builder: Callable[[str], str | None],
) -> Callable[[dict[str, Any]], None]:
    """Create the sync approval notifier used by gateway agent runs."""

    from tools.approval import build_gateway_approval_message

    def _approval_notify_sync(approval_data: dict) -> None:
        if status_adapter is None:
            return

        status_adapter.pause_typing_for_chat(status_chat_id)

        cmd = approval_data.get("command", "")
        desc = approval_data.get("description", "dangerous command")
        title = approval_data.get("prompt_title", "Dangerous command requires approval")
        approver_name = approval_data.get("approver_name", "管理员")
        allow_persistence = bool(approval_data.get("allow_persistence", True))

        if getattr(type(status_adapter), "send_exec_approval", None) is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    status_adapter.send_exec_approval(
                        chat_id=status_chat_id,
                        command=cmd,
                        session_key=approval_session_key,
                        description=desc,
                        metadata=status_thread_metadata,
                    ),
                    loop_for_step,
                ).result(timeout=15)
                return
            except Exception as exc:
                logger.warning(
                    "Button-based approval failed, falling back to text: %s",
                    exc,
                )

        msg = build_gateway_approval_message(
            command=cmd,
            description=desc,
            prompt_title=title,
            approver_name=approver_name,
            allow_persistence=allow_persistence,
        )
        admin_note = admin_only_message_builder("approve dangerous commands")
        if admin_note:
            msg = f"{msg}\n\n{admin_note}"
        try:
            asyncio.run_coroutine_threadsafe(
                status_adapter.send(
                    status_chat_id,
                    msg,
                    metadata=status_thread_metadata,
                ),
                loop_for_step,
            ).result(timeout=15)
        except Exception as exc:
            logger.error("Failed to send approval request: %s", exc)

    return _approval_notify_sync


def run_gateway_approved_conversation(
    *,
    agent: Any,
    message: str,
    pending_model_note: str | None,
    conversation_history: list[dict[str, Any]] | None,
    task_id: str,
    session_key: str | None,
    admin_user_ids: list[str] | None,
    is_admin_user: bool | None,
    status_adapter: Any,
    status_chat_id: str,
    status_thread_metadata: dict[str, Any] | None,
    loop_for_step: Any,
    logger,
    admin_only_message_builder: Callable[[str], str | None],
    external_backend: Any = None,
) -> dict[str, Any]:
    """Run one gateway foreground conversation under approval wiring."""

    from tools.approval import register_gateway_notify, unregister_gateway_notify

    effective_message = prepend_pending_model_switch_note(
        message,
        pending_model_note,
    )
    approval_session_key = session_key or ""
    approval_notify_sync = build_gateway_approval_notify_sync(
        status_adapter=status_adapter,
        status_chat_id=status_chat_id,
        status_thread_metadata=status_thread_metadata,
        loop_for_step=loop_for_step,
        approval_session_key=approval_session_key,
        logger=logger,
        admin_only_message_builder=admin_only_message_builder,
    )

    with gateway_approval_context(
        session_key=approval_session_key,
        admin_user_ids=list(admin_user_ids or []),
        is_admin_user=is_admin_user,
        external_backend=external_backend,
    ):
        approval_notify_handle = register_gateway_notify(
            approval_session_key,
            approval_notify_sync,
        )
        try:
            return agent.run_conversation(
                effective_message,
                conversation_history=conversation_history,
                task_id=task_id,
            )
        finally:
            try:
                unregister_gateway_notify(approval_session_key, approval_notify_handle)
            except TypeError:
                unregister_gateway_notify(approval_session_key)


def run_gateway_background_conversation(
    *,
    runtime_spec: GatewayAgentRuntimeSpec,
    session_id: str,
    source: Any,
    message: str,
    conversation_history: list[dict[str, Any]] | None,
    session_key: str | None,
    admin_user_ids: list[str] | None,
    is_admin_user: bool | None,
    status_adapter: Any,
    status_chat_id: str | None,
    status_thread_metadata: dict[str, Any] | None,
    loop_for_step: Any,
    logger,
    admin_only_message_builder: Callable[[str], str | None],
    session_db: Any = None,
    external_backend: Any = None,
    on_agent_created: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Create and run one background conversation under the shared approval flow."""

    agent = create_gateway_agent(
        runtime_spec=runtime_spec,
        session_id=session_id,
        source=source,
        session_db=session_db,
        max_iterations=runtime_spec.max_iterations,
        quiet_mode=True,
        verbose_logging=False,
        enabled_toolsets=runtime_spec.enabled_toolsets,
    )
    if on_agent_created is not None:
        on_agent_created(agent)
    return run_gateway_approved_conversation(
        agent=agent,
        message=message,
        pending_model_note=None,
        conversation_history=conversation_history,
        task_id=session_id,
        session_key=session_key,
        admin_user_ids=admin_user_ids,
        is_admin_user=is_admin_user,
        status_adapter=status_adapter,
        status_chat_id=status_chat_id,
        status_thread_metadata=status_thread_metadata,
        loop_for_step=loop_for_step,
        logger=logger,
        admin_only_message_builder=admin_only_message_builder,
        external_backend=external_backend,
    )


def run_gateway_btw_conversation(
    *,
    runtime_spec: GatewayAgentRuntimeSpec,
    session_id: str,
    source: Any,
    question: str,
    conversation_history: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Run one ephemeral /btw side-question conversation."""

    agent = create_gateway_agent(
        runtime_spec=runtime_spec,
        session_id=session_id,
        source=source,
        max_iterations=min(int(runtime_spec.max_iterations), 8),
        enabled_toolsets=[],
        quiet_mode=True,
        verbose_logging=False,
        skip_memory=True,
        skip_context_files=True,
        persist_session=False,
    )
    return agent.run_conversation(
        user_message=build_gateway_btw_prompt(question),
        conversation_history=conversation_history,
        task_id=session_id,
    )


def execute_gateway_sync_turn(
    *,
    agent: Any,
    message: str,
    history: list[dict[str, Any]] | None,
    session_id: str,
    session_key: str | None,
    admin_user_ids: list[str] | None,
    is_admin_user: bool | None,
    status_adapter: Any,
    status_chat_id: str,
    status_thread_metadata: dict[str, Any] | None,
    loop_for_step: Any,
    logger,
    admin_only_message_builder: Callable[[str], str | None],
    stream_consumer: Any | None,
    session_store: Any | None,
    session_db: Any = None,
    empty_response_fallback: Callable[[str], str | None] | None = None,
    pending_model_notes: dict[str, str] | None = None,
) -> GatewaySyncTurnOutcome:
    """Run one fully wired sync gateway turn and normalize its output."""

    tools = agent.tools if hasattr(agent, "tools") else None
    agent_history = normalize_conversation_history(history)
    history_media_paths = collect_history_media_paths(agent_history)
    pending_model_note = (
        pending_model_notes.pop(session_key, None)
        if pending_model_notes is not None and session_key
        else None
    )
    result = run_gateway_approved_conversation(
        agent=agent,
        message=message,
        pending_model_note=pending_model_note,
        conversation_history=agent_history,
        task_id=session_id,
        session_key=session_key,
        admin_user_ids=admin_user_ids,
        is_admin_user=is_admin_user,
        status_adapter=status_adapter,
        status_chat_id=status_chat_id,
        status_thread_metadata=status_thread_metadata,
        loop_for_step=loop_for_step,
        logger=logger,
        admin_only_message_builder=admin_only_message_builder,
    )

    if stream_consumer is not None:
        stream_consumer.finish()

    return GatewaySyncTurnOutcome(
        result=result,
        final_result=finalize_gateway_agent_conversation_result(
            result=result,
            agent=agent,
            tools=tools or [],
            message=message,
            session_id=session_id,
            session_key=session_key,
            history_media_paths=history_media_paths,
            agent_history_len=len(agent_history),
            session_store=session_store,
            session_db=session_db,
            logger=logger,
            empty_response_fallback=empty_response_fallback,
        ),
        tools=tools,
    )


def extract_gateway_agent_token_counts(agent: Any) -> tuple[int, int, int, str | None]:
    """Extract prompt/completion token counters and resolved model from the agent."""

    last_prompt_tokens = 0
    input_tokens = 0
    output_tokens = 0
    if agent and hasattr(agent, "context_compressor"):
        last_prompt_tokens = getattr(agent.context_compressor, "last_prompt_tokens", 0)
        input_tokens = getattr(agent, "session_prompt_tokens", 0)
        output_tokens = getattr(agent, "session_completion_tokens", 0)
    resolved_model = getattr(agent, "model", None) if agent else None
    return last_prompt_tokens, input_tokens, output_tokens, resolved_model


def append_missing_media_tags_to_response(
    final_response: str,
    *,
    messages: list[dict[str, Any]] | None,
    history_media_paths: set[str] | None,
) -> str:
    """Append unseen MEDIA tags from tool outputs back onto the final response."""

    if "MEDIA:" in final_response:
        return final_response

    from gateway.platforms.base import BasePlatformAdapter

    media_tags: list[str] = []
    has_voice_directive = False
    for msg in list(messages or []):
        if msg.get("role") not in ("tool", "function"):
            continue
        content = msg.get("content", "")
        if "MEDIA:" in content:
            extracted_media, _ = BasePlatformAdapter.extract_media(content)
            for path, is_voice in extracted_media:
                if path and path not in set(history_media_paths or set()):
                    media_tags.append(f"MEDIA:{path}")
                if is_voice:
                    has_voice_directive = True
        elif "[[audio_as_voice]]" in content:
            has_voice_directive = True

    if not media_tags:
        return final_response

    seen: set[str] = set()
    unique_tags: list[str] = []
    for tag in media_tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    if has_voice_directive:
        unique_tags.insert(0, "[[audio_as_voice]]")
    return final_response + "\n" + "\n".join(unique_tags)


def sync_gateway_execution_session_split(
    *,
    agent: Any,
    session_id: str,
    session_key: str | None,
    session_store: Any | None,
    agent_history_len: int,
    logger,
) -> tuple[str, int]:
    """Persist session split state and return effective session metadata."""

    session_was_split = False
    if agent and session_key and hasattr(agent, "session_id") and agent.session_id != session_id:
        session_was_split = True
        logger.info(
            "Session split detected: %s → %s (compression)",
            session_id,
            agent.session_id,
        )
        if session_store is not None:
            entry = session_store._entries.get(session_key)
            if entry:
                entry.session_id = agent.session_id
                session_store._save()

    effective_session_id = getattr(agent, "session_id", session_id) if agent else session_id
    effective_history_offset = 0 if session_was_split else agent_history_len
    return effective_session_id, effective_history_offset


def finalize_gateway_agent_conversation_result(
    *,
    result: dict[str, Any],
    agent: Any,
    tools: list[dict[str, Any]] | None,
    message: str,
    session_id: str,
    session_key: str | None,
    history_media_paths: set[str] | None,
    agent_history_len: int,
    session_store: Any | None,
    session_db: Any = None,
    logger=None,
    empty_response_fallback: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Normalize one completed gateway agent run into the gateway return shape."""

    final_response = result.get("final_response")
    suppress_reply = False
    empty_placeholder = False

    (
        last_prompt_tokens,
        input_tokens,
        output_tokens,
        resolved_model,
    ) = extract_gateway_agent_token_counts(agent)

    if isinstance(final_response, str) and final_response.strip() in {"(empty)", "[[NO_REPLY]]"}:
        empty_kind = "no_reply" if final_response.strip() == "[[NO_REPLY]]" else "empty"
        fallback = empty_response_fallback(empty_kind) if empty_response_fallback else None
        if fallback:
            final_response = fallback
        else:
            empty_placeholder = True
            final_response = ""

    if not final_response and not empty_placeholder:
        error_msg = f"⚠️ {result['error']}" if result.get("error") else "(No response generated)"
        return {
            "final_response": error_msg,
            "suppress_reply": False,
            "messages": result.get("messages", []),
            "api_calls": result.get("api_calls", 0),
            "tools": tools or [],
            "history_offset": agent_history_len,
            "last_prompt_tokens": last_prompt_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": resolved_model,
        }

    if isinstance(final_response, str):
        final_response = append_missing_media_tags_to_response(
            final_response,
            messages=result.get("messages", []),
            history_media_paths=history_media_paths,
        )

    if isinstance(final_response, str) and final_response.strip() == "[[NO_REPLY]]":
        suppress_reply = True
        final_response = ""
    elif empty_placeholder and isinstance(final_response, str) and not final_response.strip():
        suppress_reply = True

    effective_session_id, effective_history_offset = sync_gateway_execution_session_split(
        agent=agent,
        session_id=session_id,
        session_key=session_key,
        session_store=session_store,
        agent_history_len=agent_history_len,
        logger=logger,
    )

    if final_response and session_db:
        try:
            from agent.title_generator import maybe_auto_title

            maybe_auto_title(
                session_db,
                effective_session_id,
                message,
                final_response,
                result.get("messages", []),
            )
        except Exception:
            pass

    return {
        "final_response": final_response,
        "suppress_reply": suppress_reply,
        "last_reasoning": result.get("last_reasoning"),
        "messages": result.get("messages", []),
        "api_calls": result.get("api_calls", 0),
        "tools": tools or [],
        "history_offset": effective_history_offset,
        "last_prompt_tokens": last_prompt_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": resolved_model,
        "session_id": effective_session_id,
    }


@contextmanager
def gateway_approval_context(
    *,
    session_key: str,
    admin_user_ids: list[str] | None,
    is_admin_user: bool | None,
    external_backend: Any = None,
) -> Iterator[None]:
    """Apply approval/session context for one foreground/background execution."""
    from tools.approval import (
        reset_current_admin_policy,
        reset_current_session_key,
        reset_external_approval_backend,
        set_current_admin_policy,
        set_current_session_key,
        set_external_approval_backend,
    )

    session_token = set_current_session_key(str(session_key or ""))
    admin_tokens = set_current_admin_policy(
        list(admin_user_ids or []),
        is_admin_user,
    )
    backend_token = None
    if external_backend is not None:
        backend_token = set_external_approval_backend(external_backend)
    try:
        yield
    finally:
        if backend_token is not None:
            reset_external_approval_backend(backend_token)
        reset_current_admin_policy(admin_tokens)
        reset_current_session_key(session_token)
