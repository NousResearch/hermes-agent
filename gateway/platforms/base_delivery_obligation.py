"""Best-effort final-response delivery ledger recording for the base adapter."""

import asyncio
import logging


logger = logging.getLogger("gateway.platforms.base")


async def record_delivery_obligation(
    adapter, event, session_key, text_content, delivery_adapter, is_ephemeral_response,
):
    if is_ephemeral_response or str(event.text or "").lstrip().startswith(
        ("/", adapter.typed_command_prefix or "!")):
        return None
    try:
        from gateway.delivery_ledger import compute_obligation_id, ledger_enabled
        ledger = adapter.delivery_ledger
        if ledger is None and not await asyncio.to_thread(ledger_enabled):
            return None
        source = event.source
        # ``ledger_message_id`` wins when set: a queued chain's final answers the last message
        # of the chain, not the event that opened it (see ``MessageEvent.ledger_message_id``).
        _ledger_id = getattr(event, "ledger_message_id", None)
        if _ledger_id is None:
            _ledger_id = getattr(event, "message_id", "")
        obligation_id = compute_obligation_id(
            session_key, str(_ledger_id or ""), text_content)
        if ledger is None:
            from gateway.delivery_ledger_adapter import selected_delivery_ledger
            ledger = selected_delivery_ledger()
        if ledger is None:
            from gateway.delivery_ledger_adapter import SqliteDeliveryLedger
            ledger = SqliteDeliveryLedger()
        receipt = await asyncio.to_thread(
            ledger.record_obligation, obligation_id=obligation_id, session_key=session_key,
            platform=str(getattr(source.platform, "value", source.platform)),
            chat_id=source.chat_id, thread_id=getattr(source, "thread_id", None),
            content=text_content,
            adapter_profile=getattr(delivery_adapter, "_owner_profile", None))
        receipt = await asyncio.to_thread(ledger.mark_attempting, receipt)
        return receipt
    except Exception:
        logger.debug("delivery ledger record failed", exc_info=True)
        return None
