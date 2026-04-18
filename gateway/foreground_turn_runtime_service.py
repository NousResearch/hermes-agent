"""Shared runtime helpers for gateway foreground turn preparation/finalization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from gateway.agent_completion_runtime_service import (
    prepare_gateway_agent_completion,
    stop_gateway_typing_indicator,
)
from gateway.agent_delivery_runtime_service import finalize_gateway_agent_delivery
from gateway.agent_prelude_runtime_service import (
    build_agent_start_hook_context,
    run_gateway_agent_prelude,
)
from gateway.attachment_message_runtime_service import (
    collect_audio_paths,
    has_visible_image_attachments,
    prepend_document_context_notes,
)
from gateway.message_preprocessing_runtime_service import (
    is_shared_thread_session,
    prepend_reply_context_if_missing,
    prepend_shared_thread_sender,
)
from gateway.platforms.base import MessageType
from gateway.transcript_persistence_runtime_service import (
    persist_gateway_agent_transcript,
)


_STT_FAILURE_MARKERS = (
    "No STT provider",
    "STT is disabled",
    "can't listen",
    "VOICE_TOOLS_OPENAI_KEY",
)


@dataclass(slots=True)
class GatewayPreparedForegroundMessage:
    """Prepared foreground message state ready for _run_agent()."""

    message_text: str
    hook_ctx: dict[str, Any]
    blocked: bool = False


async def _maybe_send_stt_unavailable_notice(
    *,
    message_text: str,
    source: Any,
    adapters: dict[Any, Any],
    has_setup_skill: Callable[[], bool],
) -> None:
    """Send a direct STT setup hint when audio transcription is unavailable."""

    if not any(marker in message_text for marker in _STT_FAILURE_MARKERS):
        return

    adapter = adapters.get(source.platform)
    if not adapter:
        return

    metadata = {"thread_id": source.thread_id} if getattr(source, "thread_id", None) else None
    notice = (
        "🎤 I received your voice message but can't transcribe it — "
        "no speech-to-text provider is configured.\n\n"
        "To enable voice: install faster-whisper "
        "(`pip install faster-whisper` in the Hermes venv) "
        "and set `stt.enabled: true` in config.yaml, "
        "then /restart the gateway."
    )
    if has_setup_skill():
        notice += "\n\nFor full setup instructions, type: `/skill hermes-agent-setup`"

    try:
        await adapter.send(
            source.chat_id,
            notice,
            metadata=metadata,
        )
    except Exception:
        pass


async def prepare_gateway_foreground_message(
    *,
    event: Any,
    source: Any,
    session_id: str,
    history: list[dict[str, Any]],
    thread_sessions_per_user: bool,
    hooks: Any,
    adapters: dict[Any, Any],
    image_vision_inputs_from_event: Callable[[Any], list[str]],
    enrich_message_with_vision: Callable[[str, list[str]], Awaitable[str]],
    auto_vision_degraded_note: Callable[[str, bool], str],
    enrich_message_with_transcription: Callable[[str, list[str]], Awaitable[str]],
    has_setup_skill: Callable[[], bool],
    expand_context_references: Callable[[str], Awaitable[Any]],
) -> GatewayPreparedForegroundMessage:
    """Prepare the visible user message before invoking the foreground agent."""

    raw_message_text = str(getattr(event, "text", "") or "")
    message_text = raw_message_text
    shared_thread = is_shared_thread_session(
        source=source,
        thread_sessions_per_user=thread_sessions_per_user,
    )
    attachments = event.ensure_attachments()

    if has_visible_image_attachments(attachments):
        image_paths = image_vision_inputs_from_event(event)
        if image_paths:
            message_text = await enrich_message_with_vision(
                raw_message_text,
                image_paths,
            )
            if not raw_message_text.strip() and not str(message_text or "").strip():
                message_text = auto_vision_degraded_note("", False)
        elif not raw_message_text.strip():
            message_text = auto_vision_degraded_note("", False)

    if attachments:
        audio_paths = collect_audio_paths(
            attachments,
            message_type=event.message_type,
            voice_type=MessageType.VOICE,
            audio_type=MessageType.AUDIO,
        )
        if audio_paths:
            message_text = await enrich_message_with_transcription(
                message_text,
                audio_paths,
            )
            await _maybe_send_stt_unavailable_notice(
                message_text=message_text,
                source=source,
                adapters=adapters,
                has_setup_skill=has_setup_skill,
            )

    message_text = prepend_document_context_notes(
        message_text,
        attachments=attachments,
        message_type=event.message_type,
        document_type=MessageType.DOCUMENT,
    )
    message_text = prepend_reply_context_if_missing(
        message_text=message_text,
        reply_to_text=getattr(event, "reply_to_text", None),
        reply_to_message_id=getattr(event, "reply_to_message_id", None),
        history=history,
    )
    message_text = prepend_shared_thread_sender(
        message_text=message_text,
        user_name=source.user_name,
        shared_thread=shared_thread,
    )

    hook_ctx = build_agent_start_hook_context(
        platform=source.platform,
        user_id=source.user_id,
        session_id=session_id,
        message_text=message_text,
    )
    adapter = adapters.get(source.platform)
    prelude = await run_gateway_agent_prelude(
        hooks=hooks,
        hook_ctx=hook_ctx,
        message_text=message_text,
        should_expand_context_references="@" in message_text,
        expand_context_references=lambda: expand_context_references(message_text),
        send_blocked_warning=(
            (lambda warning: adapter.send(source.chat_id, warning))
            if adapter is not None
            else None
        ),
    )
    return GatewayPreparedForegroundMessage(
        hook_ctx=prelude.hook_ctx,
        message_text=prelude.message_text,
        blocked=prelude.blocked,
    )


def _load_process_registry(logger: Any) -> Any | None:
    """Best-effort process registry loader for background watcher resumption."""

    try:
        from tools.process_registry import process_registry

        return process_registry
    except Exception as exc:
        logger.error("Process watcher setup error: %s", exc)
        return None


async def finalize_gateway_foreground_success(
    *,
    agent_result: dict[str, Any],
    history: list[dict[str, Any]],
    message_text: str,
    hook_ctx: dict[str, Any],
    session_entry: Any,
    session_store: Any,
    session_db_present: bool,
    source: Any,
    event: Any,
    adapters: dict[Any, Any],
    hooks: Any,
    logger: Any,
    msg_start_time: float,
    platform_name: str,
    show_reasoning: bool,
    empty_response_fallback: Callable[[str], str | None],
    resolve_gateway_model: Callable[[], str],
    sync_visible_final_response: Callable[..., list[dict[str, Any]]],
    run_process_watcher: Callable[[dict[str, Any]], Awaitable[None]],
    should_send_voice_reply: Callable[..., bool],
    send_voice_reply: Callable[[Any, str], Awaitable[None]],
    deliver_media_from_response: Callable[[str, Any, Any], Awaitable[None]],
) -> str | None:
    """Run the success-path completion, transcript persistence, and delivery flow."""

    await stop_gateway_typing_indicator(
        adapters=adapters,
        platform=source.platform,
        chat_id=source.chat_id,
    )

    prepared_completion = await prepare_gateway_agent_completion(
        agent_result=agent_result,
        history_len=len(history),
        empty_response_fallback=empty_response_fallback,
        session_entry=session_entry,
        show_reasoning=show_reasoning,
        hook_ctx=hook_ctx,
        hooks=hooks,
        logger=logger,
        platform_name=platform_name,
        chat_id=source.chat_id,
        msg_start_time=msg_start_time,
        process_registry=_load_process_registry(logger),
        run_process_watcher=run_process_watcher,
        create_task=asyncio.create_task,
    )
    response = prepared_completion.response
    suppress_reply = prepared_completion.suppress_reply
    agent_messages = prepared_completion.agent_messages

    persist_gateway_agent_transcript(
        session_store=session_store,
        session_id=session_entry.session_id,
        session_key=session_entry.session_key,
        platform=source.platform.value if source.platform else "",
        history=history,
        agent_result=agent_result,
        agent_messages=agent_messages,
        message_text=message_text,
        visible_final_response=response,
        resolve_gateway_model=resolve_gateway_model,
        sync_visible_final_response=sync_visible_final_response,
        session_db_present=session_db_present,
        logger=logger,
    )

    return await finalize_gateway_agent_delivery(
        agent_result=agent_result,
        suppress_reply=suppress_reply,
        response=response,
        agent_messages=agent_messages,
        event=event,
        platform=source.platform,
        adapters=adapters,
        should_send_voice_reply=should_send_voice_reply,
        send_voice_reply=send_voice_reply,
        deliver_media_from_response=deliver_media_from_response,
    )
