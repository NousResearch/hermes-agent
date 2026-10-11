"""Gateway turn session resolution, including internal completion ownership receipts."""

import asyncio
from contextlib import suppress
import dataclasses
import logging


logger = logging.getLogger("gateway.run")


async def _entry_for_route(runner, event, source, metadata, expected_key):
    strict = bool(metadata.get("gateway_session_strict"))
    pin = str(metadata.get("gateway_session_id") or "").strip()
    if strict:
        entry = await runner.async_session_store.lookup_by_session_key(expected_key)
        if entry is None or not pin or entry.session_id != pin:
            logger.warning(
                "Dropping internally routed event: expected session id=%s is no longer current for key=%s",
                pin or "missing", expected_key or "missing",
            )
            return None
    else:
        entry = await runner.async_session_store.get_or_create_session(
            source, touch_activity=not bool(getattr(event, "internal", False)),
        )
    if not strict and pin:
        return await runner._resolve_async_delegation_session(entry, pin)
    return entry


async def resolve_session(runner, event, source):
    """Resolve ``source`` to its session entry (topic recovery, internal-route guards, Telegram
    topic-binding heal). Returns ``(source, session_entry, session_key)`` or ``None`` to drop
    the event."""
    # Topic-mode DMs: rewrite a stale/foreign thread_id to the user's last-active topic so a
    # cross-topic Reply doesn't fragment the conversation.
    event_metadata = getattr(event, "metadata", None) or {}
    expected_session_key = str(event_metadata.get("gateway_session_key") or "").strip()
    recovered = (await asyncio.to_thread(runner._recover_telegram_topic_thread_id, source)
                 if not expected_session_key else None)
    if recovered is not None:
        logger.info(
            "telegram topic recovery: chat=%s user=%s %r -> %s",
            source.chat_id, source.user_id, source.thread_id, recovered,
        )
        source = dataclasses.replace(source, thread_id=recovered)
        with suppress(Exception):
            event.source = source

    if expected_session_key:
        derived_session_key = runner._session_key_for_source(source)
        if derived_session_key != expected_session_key:
            logger.warning(
                "Dropping internally routed event after route recovery: expected session=%s derived=%s",
                expected_session_key, derived_session_key,
            )
            return

    pinned_session_id = str(event_metadata.get("gateway_session_id") or "").strip()
    from gateway.run_notifications_receipts import CompletionOwnerReceipt, resolve_prepared_completion_owner
    receipt = getattr(event, "_completion_owner_receipt", None)
    if isinstance(receipt, CompletionOwnerReceipt):
        if (receipt.session_key != expected_session_key
                or receipt.pinned_session_id != pinned_session_id):
            return
        session_entry = await resolve_prepared_completion_owner(runner, event, receipt)
        if session_entry is None:
            return
    else:
        session_entry = await _entry_for_route(runner, event, source, event_metadata, expected_session_key)
        if session_entry is None:
            return
    session_key = session_entry.session_key
    runner._cache_session_source(session_key, source)
    if not isinstance(receipt, CompletionOwnerReceipt) and await asyncio.to_thread(runner._is_telegram_topic_lane, source):
        session_entry = await runner._hmwa_heal_telegram_topic_binding(source, session_entry, session_key)
    from gateway.run_heartbeat_acceptance import resolve_heartbeat_owner
    if not await resolve_heartbeat_owner(runner, event, session_entry):
        return
    return source, session_entry, session_key
