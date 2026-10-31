"""Todo callback → real Telegram adapter → real SDK → in-memory Bot API (#31106)."""
import asyncio
from contextlib import suppress

import pytest

from gateway.platforms.event import MessageEvent
from tests.plugins._telegram_todo_api import snapshot, turn_for
from tests.waiting import async_wait_until

pytest_plugins = ["tests.plugins._telegram_todo_api"]


@pytest.mark.asyncio
@pytest.mark.parametrize("grouping", ["accumulate", "separate"])
async def test_topics_reuse_ids_and_deliver_separate_finals(telegram_api, grouping):
    adapter, api = telegram_api

    async def handler(event):
        topic = event.source.thread_id
        turn = turn_for(adapter, topic, grouping=grouping)
        sender = asyncio.create_task(turn.send_progress_messages())
        try:
            turn.progress_callback("tool.completed", "todo", result=snapshot(topic, "pending"))
            await async_wait_until(lambda: any(p.get("message_thread_id") == int(topic)
                                              for p in api.requests("sendMessage")))
            turn._ctx.progress_queue.put(("__reset__",))
            turn.progress_callback("tool.completed", "todo_list", result=snapshot(topic, "completed"))
            await async_wait_until(lambda: any(f"Work {topic}" in p["text"]
                                              for p in api.requests("editMessageText")))
        finally:
            sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender
        return f"Final {topic}"

    adapter.set_message_handler(handler)
    events = [MessageEvent(text="offline smoke", message_id=str(200 + int(topic)),
                           source=turn_for(adapter, topic)._ctx.source) for topic in ("11", "22")]
    await asyncio.gather(*(adapter._process_message_background(event, f"session-{event.source.thread_id}")
                           for event in events))
    checklist_ids = []
    for topic in (11, 22):
        initial = [p for p in api.requests("sendMessage") if p.get("message_thread_id") == topic
                   and "Task List" in p["text"]]
        assert len(initial) == 1
        assert initial[0]["chat_id"] == -100123
        assert initial[0]["disable_notification"] is True
        assert initial[0]["parse_mode"] == "MarkdownV2"
        edits = [p for p in api.requests("editMessageText") if f"Work {topic}" in p["text"]]
        assert edits and all(p["chat_id"] == -100123 for p in edits)
        mid = edits[0]["message_id"]
        checklist_ids.append(mid)
        assert api.messages[-100123, mid]["message_thread_id"] == topic
        assert api.messages[-100123, mid]["text"] == f"📋 Task List\n\\[x\\] Work {topic} \\({topic}\\)"
        assert all(p["message_id"] == mid for p in edits)
        final = [p for p in api.requests("sendMessage") if p["text"] == f"Final {topic}"]
        assert len(final) == 1 and final[0]["message_thread_id"] == topic
        assert not final[0].get("disable_notification", False)
    assert checklist_ids[0] != checklist_ids[1]


@pytest.mark.asyncio
async def test_plaintext_fallback_keeps_one_binding_and_clears_list(telegram_api):
    adapter, api = telegram_api
    turn = turn_for(adapter)
    sender = asyncio.create_task(turn.send_progress_messages())
    api.reject("sendMessage", 400, "Bad Request: can't parse entities")
    try:
        turn.progress_callback("tool.completed", "todo", result=snapshot("11", "pending"))
        await async_wait_until(lambda: len(api.messages) == 1)
        turn.progress_callback("tool.completed", "todo", result={"todos": []})
        api.reject("editMessageText", 400, "Bad Request: can't parse entities")
    finally:
        sender.cancel()
        with suppress(asyncio.CancelledError):
            await sender
    assert len(api.messages) == 1
    calls = api.requests("sendMessage")
    assert len(calls) == 2 and calls[0]["parse_mode"] == "MarkdownV2"
    assert "parse_mode" not in calls[1]
    edits = api.requests("editMessageText")
    assert len(edits) == 2 and "parse_mode" not in edits[-1]
    assert edits[-1]["text"] == "📋 Task List\nNo tasks."


@pytest.mark.asyncio
async def test_real_sdk_cancelled_ack_cannot_duplicate_status(telegram_api):
    adapter, api = telegram_api
    api.hold_receipt = True
    turn = turn_for(adapter)
    sender = asyncio.create_task(turn.send_progress_messages())
    try:
        turn.progress_callback("tool.completed", "todo", result=snapshot("11", "pending"))
        await asyncio.wait_for(api.accepted.wait(), 5)
        turn.progress_callback("tool.completed", "todo", result=snapshot("11", "completed"))
    finally:
        sender.cancel()
        await asyncio.wait_for(sender, 5)
    assert len(api.requests("sendMessage")) == 1
    assert len(api.messages) == 1 and api.inflight == 0


@pytest.mark.asyncio
async def test_status_transitions_and_new_run_get_distinct_bindings(telegram_api, monkeypatch):
    from tests.plugins._telegram_todo_api import status_for
    from gateway.todo_progress import TodoChecklist, TodoProgress, _render_todo_checklist

    adapter, api = telegram_api
    status, clock = status_for(adapter, monkeypatch)
    for index, (state, glyph) in enumerate([("pending", "[ ]"), ("in_progress", "[>]"),
                                             ("completed", "[x]"), ("cancelled", "[-]")]):
        clock[0] = 10 + index * 2
        status.absorb(TodoChecklist(_render_todo_checklist(snapshot("11", state))))
        await status.flush()
        assert len(api.messages) == 1
        visible = next(iter(api.messages.values()))["text"].replace("\\", "")
        assert visible == f"📋 Task List\n{glyph} Work 11 (11)"
    old_id = status.state.progress_msg_id
    second = TodoProgress(turn_for(adapter), adapter)
    second.absorb(TodoChecklist(_render_todo_checklist(snapshot("11", "pending"))))
    await second.flush()
    assert second.state.progress_msg_id != old_id
    assert len(api.requests("sendMessage")) == 2
    assert len({p["message_id"] for p in api.requests("editMessageText")}) == 1
