"""The delivery-ledger bracket around a final reply (``BasePlatformAdapter.send_final_ledgered`` and
``ledger_attempted_final``): record the obligation before the send, settle it from the result, so a
refused or unconfirmed final leaves a row the boot sweep / timed redelivery can act on.

Calls go through the adapter's ``_record_delivery_obligation`` / ``_finalize_delivery_obligation``
methods so a subclass (the local session adapter) can opt out of ledgering.
"""
import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("gateway.platforms.base")


async def ledger_bracket(adapter: Any, event: Any, session_key: str, text_content: str,
                         send: Callable[[Any, Optional[str]], Awaitable[Any]], *,
                         is_ephemeral_response: bool = False) -> "tuple[Any, Any]":
    """Resolve the CURRENT delivery adapter, record, ``send(delivery_adapter, obligation_id)``, finalize.
    Returns the send result with the adapter that owns it."""
    delivery_adapter = adapter._final_delivery_adapter(event.source)
    obligation_id = await adapter._record_delivery_obligation(
        event, session_key, text_content, delivery_adapter, is_ephemeral_response)
    result = await send(delivery_adapter, obligation_id)
    if obligation_id is not None:
        await adapter._finalize_delivery_obligation(obligation_id, result, event, delivery_adapter)
    return result, delivery_adapter


async def ledger_attempted(adapter: Any, event: Any, session_key: str, text: str, error: str) -> bool:
    """``BasePlatformAdapter.ledger_attempted_final``: ledger ``text`` on the CURRENT transport as a failed
    row instead of sending it again, so the timed redelivery sends it once more with the duplicate marker
    and a lost send is not silence. False when nothing was saved (ledger off, a refused reply)."""
    saved: list[str] = []

    async def _attempted(_delivery_adapter: Any, obligation_id: Optional[str]) -> Any:
        from gateway.platforms.base import SendResult
        if obligation_id:
            saved.append(obligation_id)
            logger.warning("Final for session %s may not have landed; the delivery ledger "
                           "redelivers it with the duplicate marker", session_key)
        return SendResult(success=False, error=error)
    await ledger_bracket(adapter, event, session_key, text, _attempted)
    return bool(saved)


async def record_final_obligation(adapter: Any, event: Any, session_key: str, text_content: str,
                            delivery_adapter: Any, is_ephemeral_response: bool) -> Optional[str]:
    """Ledger the final response BEFORE the send so a crash before platform ACK redelivers on
    next boot; best-effort, skips slash-command and ephemeral replies. Returns the obligation id
    or None."""
    if is_ephemeral_response or str(event.text or "").lstrip().startswith(
        ("/", adapter.typed_command_prefix or "!")):
        return None
    try:
        from gateway.delivery_ledger import (
            compute_obligation_id, ledger_enabled, mark_attempting, record_obligation)
        if not await asyncio.to_thread(ledger_enabled):
            return None
        source = event.source
        # ``ledger_message_id`` wins when set: a queued chain's final answers the last message
        # of the chain, not the event that opened it (see ``MessageEvent.ledger_message_id``).
        _ledger_id = getattr(event, "ledger_message_id", None)
        if _ledger_id is None:
            _ledger_id = getattr(event, "message_id", "")
        obligation_id = compute_obligation_id(
            session_key, str(_ledger_id or ""), text_content)
        await asyncio.to_thread(
            record_obligation, obligation_id=obligation_id, session_key=session_key,
            platform=str(getattr(source.platform, "value", source.platform)),
            chat_id=source.chat_id, thread_id=getattr(source, "thread_id", None),
            content=text_content,
            adapter_profile=getattr(delivery_adapter, "_owner_profile", None))
        await asyncio.to_thread(mark_attempting, obligation_id)
        return obligation_id
    except Exception:
        logger.debug("delivery ledger record failed", exc_info=True)
        return None


async def finalize_final_obligation(adapter: Any, obligation_id: str, result: Any, event: Any,
                              delivery_adapter: Any) -> None:
    """Mark the ledger row delivered/failed (best-effort). On ``send_path_degraded`` with a
    replacement adapter live, trigger another redelivery sweep (the watcher's may have run
    before this failure landed; atomic claiming keeps it idempotent). On any other rejection arm
    the runner's timed redelivery, so the reply goes out once the flood penalty or the retry
    backoff has passed instead of waiting for the next restart (#91653)."""
    try:
        from gateway.dead_targets import classify_dead_error
        from gateway.delivery_ledger import (
            is_reconnect_only, mark_delivered, mark_failed, is_terminal_message_error,
        )
        if getattr(result, "success", False):
            await asyncio.to_thread(mark_delivered, obligation_id)
            return
        error = str(getattr(result, "error", "") or "")
        await asyncio.to_thread(mark_failed, obligation_id, error)
        if is_reconnect_only(error):
            redeliver = getattr(
                adapter.gateway_runner, "_redeliver_failed_obligations_for_platform", None)
            live = adapter._final_delivery_adapter(event.source)
            if live is not delivery_adapter and callable(redeliver):
                await redeliver(event.source.platform,
                                profile=getattr(delivery_adapter, "_owner_profile", None))
        elif classify_dead_error(error) is None and not is_terminal_message_error(error):  # a dead chat or terminal error is never retried: no timer to wake
            schedule = getattr(adapter.gateway_runner, "_schedule_flood_redelivery", None)
            if callable(schedule):
                schedule(event.source.platform,
                         profile=getattr(delivery_adapter, "_owner_profile", None))
    except Exception:
        logger.debug("delivery ledger update failed", exc_info=True)
