"""Typing indicator for an inline clarify-bypass dispatch that runs a fresh turn.

``BasePlatformAdapter._handle_message_while_active`` dispatches a message that arrives while its
chat is busy inline, without ``_process_message_background`` — and therefore without the typing
refresh that path brackets a turn with. For the clarify *tool* that is harmless: the agent turn is
already running, so its own refresh owns the indicator. A plugin command's ``ask_choice`` card holds
the chat with no agent turn behind it, so the prose that cancels the card becomes the chat's next
turn and runs with no indicator for its whole duration (#154524).

The bypass is only bracketed when the runner holds no running-turn slot for the session, which is
exactly the state a plugin card leaves behind; an answer to a running turn's clarify is dispatched
as before so the two refreshes never overlap.
"""
import asyncio
import logging

from gateway.platforms.helpers import typing_metadata_for_turn

logger = logging.getLogger(__name__)


def _turn_is_live(adapter, session_key: str) -> bool:
    """True when the runner holds a running-turn slot for ``session_key`` (agent or pending
    sentinel). Fail-safe: an adapter with no runner, or a runner that cannot answer, reads as live,
    so an unknown topology keeps the pre-existing (no extra indicator) behaviour."""
    probe = getattr(getattr(adapter, "gateway_runner", None), "_is_session_running", None)
    if not callable(probe):
        return True
    try:
        return bool(probe(session_key))
    except Exception:  # health: allow BLE001 -- a broken probe must not cost the reply
        logger.debug("Running-turn probe failed for %s; assuming a live turn", session_key, exc_info=True)
        return True


def _card_only_text_reply(event, session_key: str) -> bool:
    """Accepted or retryable choice replies do not start an agent turn.

    Use the clarify resolver's own pure classifier; invalid selections also keep
    the card open rather than falling through to a turn.
    """
    from gateway.platforms.event import MessageType
    if event.message_type != MessageType.TEXT or not isinstance(event.text, str):
        return False
    from tools import clarify_gateway

    pending = clarify_gateway.get_pending_for_session(session_key, include_choice_prompts=True)
    if pending is None:
        return False
    _, reason = clarify_gateway._coerce_text_response_detailed(pending, event.text)
    return reason != "prose"


async def dispatch_clarify_bypass(adapter, event, session_key: str) -> None:
    """Send the clarify text-intercept's reply inline, bracketed by the typing refresh a turn it
    starts would otherwise get from ``_process_message_background``."""
    if _turn_is_live(adapter, session_key) or _card_only_text_reply(event, session_key):
        await adapter._dispatch_inline_reply(event)
        return
    from gateway.platforms.base import _reply_anchor_for_event, _thread_metadata_for_event
    metadata = typing_metadata_for_turn(  # fixed at dispatch: a later inbound must not move it
        adapter, event.source.chat_id, _reply_anchor_for_event(event), _thread_metadata_for_event(event))
    stop = asyncio.Event()
    typing = adapter._start_typing_refresh(event, stop, metadata)  # None for typing_indicator: false
    try:
        await adapter._dispatch_inline_reply(event)
    finally:
        if typing is not None:
            stop.set()
            await adapter._stop_typing_refresh(event.source.chat_id, typing, metadata=metadata)
