"""Shared helpers for queued follow-up handling after a gateway agent turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from gateway.agent_response_runtime_service import normalize_gateway_agent_response


@dataclass(slots=True)
class GatewayPendingFollowup:
    """A pending user follow-up captured while the previous turn was running."""

    event: Any | None
    text: str
    was_interrupted: bool


async def process_gateway_pending_followup(
    *,
    result: dict[str, Any] | None,
    adapter: Any,
    session_key: str | None,
    dequeue_pending_event_text: Callable[[Any, str], tuple[Any | None, str | None]],
    logger: Any,
    interrupt_depth: int,
    max_interrupt_depth: int,
    source: Any,
    fallback_event: Any | None,
    chat_id: str | None,
    stream_consumer: Any,
    history: list[dict[str, Any]] | None,
    current_response_fallback: dict[str, Any],
    empty_response_fallback: Callable[[str], str | None],
    recurse_followup: Callable[[GatewayPendingFollowup, list[dict[str, Any]] | None], Awaitable[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Process any queued follow-up after a foreground turn completes."""

    pending_followup = extract_gateway_pending_followup(
        result=result,
        adapter=adapter,
        session_key=session_key,
        dequeue_pending_event_text=dequeue_pending_event_text,
        logger=logger,
    )
    if not pending_followup:
        return None

    logger.debug("Processing pending message: '%s...'", pending_followup.text[:40])
    clear_gateway_pending_interrupt(
        adapter=adapter,
        session_key=session_key,
    )

    if interrupt_depth >= max_interrupt_depth:
        logger.warning(
            "Interrupt recursion depth %d reached for session %s — queueing message instead of recursing.",
            interrupt_depth,
            session_key,
        )
        queue_gateway_pending_followup_for_later(
            adapter=adapter,
            session_key=session_key,
            pending_text=pending_followup.text,
            source=source,
            pending_event=pending_followup.event,
            fallback_event=fallback_event,
        )
        return current_response_fallback

    await deliver_gateway_first_response_before_followup(
        result=result,
        adapter=adapter,
        chat_id=chat_id,
        pending_event=pending_followup.event,
        fallback_event=fallback_event,
        stream_consumer=stream_consumer,
        history_len=len(history or []),
        empty_response_fallback=empty_response_fallback,
        logger=logger,
    )
    updated_history = result.get("messages", history) if result else history
    return await recurse_followup(pending_followup, updated_history)


def extract_gateway_pending_followup(
    *,
    result: dict[str, Any] | None,
    adapter: Any,
    session_key: str | None,
    dequeue_pending_event_text: Callable[[Any, str], tuple[Any | None, str | None]],
    logger: Any,
) -> GatewayPendingFollowup | None:
    """Load and sanitize any queued follow-up that should run next."""

    if not result or not adapter or not session_key:
        return None

    pending_event = None
    pending_text = None
    was_interrupted = bool(result.get("interrupted"))
    if was_interrupted:
        pending_event, pending_text = dequeue_pending_event_text(adapter, session_key)
        if not pending_text and result.get("interrupt_message"):
            pending_text = result.get("interrupt_message")
    else:
        pending_event, pending_text = dequeue_pending_event_text(adapter, session_key)
        if pending_text:
            logger.debug(
                "Processing queued message after agent completion: '%s...'",
                pending_text[:40],
            )

    if not pending_text:
        return None

    stripped = pending_text.strip()
    if stripped.startswith("/"):
        pending_parts = stripped.split(None, 1)
        pending_cmd_word = pending_parts[0][1:].lower() if pending_parts else ""
        if pending_cmd_word:
            try:
                from hermes_cli.commands import resolve_command

                if resolve_command(pending_cmd_word):
                    logger.info(
                        "Discarding command '/%s' from pending queue — commands must not be passed as agent input",
                        pending_cmd_word,
                    )
                    return None
            except Exception:
                pass

    return GatewayPendingFollowup(
        event=pending_event,
        text=pending_text,
        was_interrupted=was_interrupted,
    )


def clear_gateway_pending_interrupt(
    *,
    adapter: Any,
    session_key: str | None,
) -> None:
    """Clear any sticky adapter interrupt event before recursing into the next turn."""

    if (
        adapter
        and hasattr(adapter, "_active_sessions")
        and session_key
        and session_key in adapter._active_sessions
    ):
        adapter._active_sessions[session_key].clear()


def queue_gateway_pending_followup_for_later(
    *,
    adapter: Any,
    session_key: str | None,
    pending_text: str,
    source: Any,
    pending_event: Any | None,
    fallback_event: Any | None,
) -> None:
    """Re-queue a pending follow-up as a normal message when recursion is capped."""

    if not adapter or not hasattr(adapter, "queue_message"):
        return

    from gateway.platforms.base import MessageEvent, MessageType

    seed_event = pending_event or fallback_event
    event_kwargs: dict[str, Any] = {
        "text": pending_text,
        "message_type": MessageType.TEXT,
        "source": source,
        "raw_message": getattr(seed_event, "raw_message", None),
        "message_id": getattr(pending_event, "message_id", None)
        or getattr(fallback_event, "message_id", None),
        "metadata": dict(getattr(seed_event, "metadata", None) or {}) or None,
        "reply_to_message_id": getattr(seed_event, "reply_to_message_id", None),
        "reply_to_text": getattr(seed_event, "reply_to_text", None),
        "auto_skill": getattr(seed_event, "auto_skill", None),
    }

    attachments = []
    if seed_event is not None:
        if hasattr(seed_event, "ensure_attachments"):
            try:
                attachments = list(seed_event.ensure_attachments() or [])
            except Exception:
                attachments = list(getattr(seed_event, "attachments", None) or [])
        else:
            attachments = list(getattr(seed_event, "attachments", None) or [])
    if attachments:
        event_kwargs["attachments"] = attachments
    else:
        event_kwargs["media_urls"] = list(getattr(seed_event, "media_urls", None) or [])
        event_kwargs["media_sources"] = list(getattr(seed_event, "media_sources", None) or [])
        event_kwargs["media_types"] = list(getattr(seed_event, "media_types", None) or [])
    timestamp = getattr(seed_event, "timestamp", None)
    if timestamp is not None:
        event_kwargs["timestamp"] = timestamp

    adapter.queue_message(
        session_key,
        MessageEvent(**event_kwargs),
    )


async def deliver_gateway_first_response_before_followup(
    *,
    result: dict[str, Any] | None,
    adapter: Any,
    chat_id: str | None,
    pending_event: Any | None,
    fallback_event: Any | None,
    stream_consumer: Any,
    history_len: int,
    empty_response_fallback: Callable[[str], str | None],
    logger: Any,
) -> None:
    """Send the first response before a queued non-interrupt follow-up is processed."""

    if not result or result.get("interrupted") or not adapter:
        return

    normalized_response = normalize_gateway_agent_response(
        agent_result=result,
        history_len=history_len,
        empty_response_fallback=empty_response_fallback,
    )
    already_streamed = bool(stream_consumer and getattr(stream_consumer, "already_sent", False))
    first_response = str(normalized_response.response or "")
    suppress_first_response = bool(normalized_response.suppress_reply)
    response_event = fallback_event or pending_event
    if normalized_response.synthetic_fallback and response_event is not None:
        metadata = dict(getattr(response_event, "metadata", None) or {})
        metadata["skip_successful_response_context"] = True
        response_event.metadata = metadata
    if first_response and not already_streamed and not suppress_first_response:
        try:
            send_result = await adapter.send(
                chat_id,
                first_response,
                metadata=getattr(fallback_event, "metadata", None)
                or getattr(pending_event, "metadata", None),
            )
            sent_message_id = str(getattr(send_result, "message_id", "") or "").strip()
            if (
                sent_message_id
                and response_event is not None
                and not normalized_response.synthetic_fallback
                and hasattr(adapter, "_record_successful_response_context")
            ):
                try:
                    adapter._record_successful_response_context(
                        response_event,
                        [sent_message_id],
                    )
                except Exception:
                    logger.debug(
                        "Failed to record successful-response context for pre-followup send",
                        exc_info=True,
                    )
        except Exception as exc:
            logger.warning(
                "Failed to send first response before queued message: %s",
                exc,
            )
