"""Unchanged Telegram edits do not become additional model turns."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("telegram")
from telegram import Update

from tests.plugins.test_telegram_update_admission import connected, update


async def dispatch(adapter, app, *, uid, text=None, edited=False, chat=42, message_id=472, kind="text", group=False, entities=None):
    payload = update(app.bot, uid, kind,
                     text="hello" if text is None else text, edited=edited, chat=chat, group=group).to_dict()
    message = payload["edited_message" if edited else "message"]
    message["message_id"] = message_id
    if kind == "command" and text is not None:
        message["text"] = text
    if entities is not None:
        message["entities"] = entities
    await app.update_queue.put(Update.de_json(payload, app.bot))
    await app.update_queue.join()
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())


async def persist_receipts(adapter):
    # Disconnect has an intentionally short deadline; restart tests need the actual write.
    writer = adapter._update_receipt_flush
    assert writer is not None
    await asyncio.wait_for(asyncio.shield(writer), 15)


@pytest.mark.asyncio
@pytest.mark.parametrize("semantic", ["text", "text_link", "text_mention"])
@pytest.mark.parametrize("kind", ["text", "command"])
async def test_only_semantic_changes_and_reversions_reach_core(monkeypatch, semantic, kind):
    import hermes_cli.lifecycle

    monkeypatch.setattr(hermes_cli.lifecycle, "has_hook", lambda name: True)
    async with connected(monkeypatch) as (adapter, app, delivered):
        observer = adapter._platform_event_handler = AsyncMock()
        for uid, target in enumerate((1, 1, 2, 2, 1, 1), 10):
            text, entities = str(target), []
            if semantic != "text":
                text = "hello"
                entity = {"type": semantic, "offset": 0, "length": 5}
                if semantic == "text_link":
                    entity["url"] = f"https://example.org/{target}"
                else:
                    entity["user"] = {"id": target, "is_bot": False, "first_name": f"Name {uid}"}
                entities.append(entity)
            if kind == "command":
                text = "/status " + text
                for entity in entities:
                    entity["offset"] += 8
                entities.insert(0, {"type": "bot_command", "offset": 0, "length": 7})
            await dispatch(adapter, app, uid=uid, kind=kind, text=text, edited=uid != 10, entities=entities)
        assert len(delivered) == 3
        assert len({event.edit_revision for event in delivered[1:]}) == 2
        assert observer.await_count == 5
        assert all(call.args[0]["event_type"] == "message_edited" for call in observer.call_args_list)


@pytest.mark.asyncio
async def test_identical_text_in_distinct_messages_chats_and_bots_is_independent(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        await dispatch(adapter, app, uid=10)
        await dispatch(adapter, app, uid=11, edited=True)
        await dispatch(adapter, app, uid=12, message_id=473)
        await dispatch(adapter, app, uid=13, chat=43, edited=True)
        assert len(delivered) == 3
        async with connected(monkeypatch, bot_id=222) as (other, other_app, other_delivered):
            await dispatch(other, other_app, uid=14, edited=True)
            assert len(other_delivered) == 1


@pytest.mark.asyncio
async def test_latest_text_receipt_survives_adapter_restart(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        await dispatch(adapter, app, uid=10, text="private text")
        await persist_receipts(adapter)
        assert len(delivered) == 1
        receipt_path = adapter._update_receipt_dir / "telegram_update_receipts_111.json"
    payload = json.loads(receipt_path.read_text())
    assert "private text" not in json.dumps(payload)
    async with connected(monkeypatch) as (adapter, app, delivered):
        await dispatch(adapter, app, uid=11, text="private text", edited=True)
        assert not delivered
        await dispatch(adapter, app, uid=12, text="changed", edited=True)
        assert [event.text for event in delivered] == ["changed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "command"])
async def test_failed_text_preparation_does_not_poison_latest_receipt(monkeypatch, kind):
    import hermes_cli.lifecycle

    monkeypatch.setattr(hermes_cli.lifecycle, "has_hook", lambda name: True)
    async with connected(monkeypatch) as (adapter, app, delivered):
        observer = adapter._platform_event_handler = AsyncMock()
        error_observer = AsyncMock()
        assert app is not None
        app.add_error_handler(error_observer)
        with monkeypatch.context() as broken:
            broken.setattr(adapter, "_cache_replied_media", AsyncMock(side_effect=OSError("preparation failed")))
            await dispatch(adapter, app, uid=11, kind=kind, edited=True)
        assert error_observer.await_count == observer.await_count == 1
        assert not adapter._latest_text_receipts
        await dispatch(adapter, app, uid=12, kind=kind, edited=True)
        await dispatch(adapter, app, uid=13, kind=kind, edited=True)
        assert len(delivered) == 1


@pytest.mark.asyncio
async def test_native_handoff_cannot_accept_another_tasks_text_preparation(monkeypatch):
    from telegram.ext import TypeHandler
    from gateway.platforms.event import MessageType

    async with connected(monkeypatch) as (adapter, app, delivered):
        staged, native_done = asyncio.Event(), asyncio.Event()

        async def native(incoming, context):
            if incoming.update_id != 10:
                return
            await asyncio.wait_for(staged.wait(), 5)
            event = adapter._build_message_event(incoming.effective_message, MessageType.TEXT,
                                                 update_id=incoming.update_id)
            event.text = "native forwarded event"
            try:
                await adapter.handle_message(event)
            finally:
                native_done.set()

        async def fail_preparation(message, event):
            staged.set()
            await asyncio.wait_for(native_done.wait(), 5)
            raise OSError("core failed before handoff")

        app.add_handler(TypeHandler(Update, native, block=False), group=-1)
        with monkeypatch.context() as broken:
            broken.setattr(adapter, "_cache_replied_media", fail_preparation)
            await dispatch(adapter, app, uid=10)
        assert [event.text for event in delivered] == ["native forwarded event"]
        assert not adapter._latest_text_receipts
        await dispatch(adapter, app, uid=11, edited=True)
        await dispatch(adapter, app, uid=12, edited=True)
        assert [event.text for event in delivered] == ["native forwarded event", "hello"]
        assert delivered[-1].edit_revision == "u11"


@pytest.mark.asyncio
async def test_passive_chatter_still_reaches_observation(monkeypatch):
    async with connected(monkeypatch, extra={"observe_unmentioned_group_messages": True, "require_mention": True,
                                            "allowed_chats": ["42"], "group_allowed_chats": ["42"]}) as (adapter, app, delivered):
        observed = []
        monkeypatch.setattr(adapter, "_observe_unmentioned_group_message", lambda *args, **kwargs: observed.append(args[0].text))
        await dispatch(adapter, app, uid=10, group=True)
        await dispatch(adapter, app, uid=11, group=True, edited=True)
        assert not delivered
        assert observed == ["hello", "hello"]


@pytest.mark.asyncio
async def test_accepted_text_receipt_survives_post_enqueue_failure(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        enqueue = adapter._enqueue_text_event

        def enqueue_then_fail(event):
            enqueue(event)
            raise OSError("failed after handoff")

        with monkeypatch.context() as broken:
            broken.setattr(adapter, "_enqueue_text_event", enqueue_then_fail)
            await dispatch(adapter, app, uid=10)
        await dispatch(adapter, app, uid=11, edited=True)
        assert len(delivered) == 1


@pytest.mark.asyncio
async def test_text_receipts_are_bounded_and_eviction_allows_unknown_edit(monkeypatch):
    from plugins.platforms.telegram import update_admission

    monkeypatch.setattr(update_admission, "_SEEN_CAP", 2)
    async with connected(monkeypatch) as (adapter, app, delivered):
        for message_id in (472, 473, 474):
            await dispatch(adapter, app, uid=message_id, message_id=message_id)
        await persist_receipts(adapter)
        assert len(adapter._latest_text_receipts) == 2
        receipt_path = adapter._update_receipt_dir / "telegram_update_receipts_111.json"
    assert len(json.loads(receipt_path.read_text())["message_texts"]) == 2
    async with connected(monkeypatch) as (adapter, app, delivered):
        await dispatch(adapter, app, uid=475, message_id=474, edited=True)
        assert not delivered
        await dispatch(adapter, app, uid=476, message_id=472, edited=True)
        assert len(delivered) == 1


@pytest.mark.asyncio
async def test_held_text_establishes_baseline_at_queue_handoff(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        adapter._mark_disconnected()
        await dispatch(adapter, app, uid=10)
        await dispatch(adapter, app, uid=11, edited=True)
        assert len(adapter._held_inbound_events) == 1
        assert not delivered
        adapter._mark_connected()
        redispatch = adapter._held_inbound_redispatch_task
        assert redispatch is not None
        await redispatch
        assert len(delivered) == 1


@pytest.mark.asyncio
async def test_expired_memory_baseline_fails_open_then_is_remembered(monkeypatch):
    import time
    from plugins.platforms.telegram import update_admission

    async with connected(monkeypatch) as (adapter, app, delivered):
        await dispatch(adapter, app, uid=10)
        now = time.time()
        with monkeypatch.context() as clock:
            clock.setattr(time, "time", lambda: now + update_admission.RECEIPT_TTL_SECONDS + 1)
            await dispatch(adapter, app, uid=11, edited=True)
            await dispatch(adapter, app, uid=12, edited=True)
        assert len(delivered) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["expired", "preupgrade", "profile"])
async def test_unknown_restart_baseline_fails_open_then_is_remembered(monkeypatch, tmp_path, mode):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from plugins.platforms.telegram import update_admission

    async with connected(monkeypatch) as (adapter, app, delivered):
        await dispatch(adapter, app, uid=10)
        await persist_receipts(adapter)
        receipt_path = adapter._update_receipt_dir / "telegram_update_receipts_111.json"
    if mode != "profile":
        payload = json.loads(receipt_path.read_text())
        if mode == "preupgrade":
            payload.pop("message_texts")
        else:
            for value in payload["message_texts"].values():
                value[1] -= update_admission.RECEIPT_TTL_SECONDS + 1
        receipt_path.write_text(json.dumps(payload))
    token = set_hermes_home_override(tmp_path / "other-profile") if mode == "profile" else None
    try:
        async with connected(monkeypatch) as (adapter, app, delivered):
            await dispatch(adapter, app, uid=11, edited=True)
            await dispatch(adapter, app, uid=12, edited=True)
            assert len(delivered) == 1
    finally:
        if token is not None:
            reset_hermes_home_override(token)
