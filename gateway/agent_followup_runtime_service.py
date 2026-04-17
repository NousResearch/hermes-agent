"""Shared helpers for queued follow-up handling after a gateway agent turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


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

    adapter.queue_message(
        session_key,
        MessageEvent(
            text=pending_text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=getattr(pending_event, "message_id", None)
            or getattr(fallback_event, "message_id", None),
        ),
    )


async def deliver_gateway_first_response_before_followup(
    *,
    result: dict[str, Any] | None,
    adapter: Any,
    chat_id: str | None,
    pending_event: Any | None,
    fallback_event: Any | None,
    stream_consumer: Any,
    logger: Any,
) -> None:
    """Send the first response before a queued non-interrupt follow-up is processed."""

    if not result or result.get("interrupted") or not adapter:
        return

    already_streamed = bool(stream_consumer and getattr(stream_consumer, "already_sent", False))
    first_response = str(result.get("final_response", "") or "")
    suppress_first_response = bool(result.get("suppress_reply"))
    if first_response and first_response.strip() == "[[NO_REPLY]]":
        suppress_first_response = True
        first_response = ""
    if first_response and not already_streamed and not suppress_first_response:
        try:
            await adapter.send(
                chat_id,
                first_response,
                metadata=getattr(pending_event, "metadata", None)
                or getattr(fallback_event, "metadata", None),
            )
        except Exception as exc:
            logger.warning(
                "Failed to send first response before queued message: %s",
                exc,
            )
