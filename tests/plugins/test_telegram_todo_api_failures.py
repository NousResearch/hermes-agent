"""Telegram SDK error responses must preserve checklist IDs and normal replies."""
import asyncio
from contextlib import suppress

import pytest

from tests.plugins._telegram_todo_api import snapshot, status_for, turn_for
from telegram.error import NetworkError, TimedOut

from gateway.platforms.event import MessageEvent
from gateway.todo_progress import TodoChecklist, _render_todo_checklist
from tests.waiting import async_wait_until

pytest_plugins = ["tests.plugins._telegram_todo_api"]


async def initial_status(adapter, monkeypatch):
    status, clock = status_for(adapter, monkeypatch)
    status.absorb(TodoChecklist(_render_todo_checklist(snapshot("11", "pending"))))
    await status.flush()
    status.absorb(TodoChecklist(_render_todo_checklist(snapshot("11", "completed"))))
    clock[0] = 12
    return status, clock


@pytest.mark.asyncio
async def test_flood_response_defers_even_forced_flush_and_reuses_id(telegram_api, monkeypatch):
    adapter, api = telegram_api
    status, clock = await initial_status(adapter, monkeypatch)
    mid = next(iter(api.messages))[1]
    api.reject("editMessageText", 429, "Too Many Requests: retry after 30", retry_after=30, times=2)
    await status.flush()
    attempts = len(api.requests("editMessageText"))
    assert attempts == 2  # Markdown and plain fallback both see the real SDK RetryAfter.
    await status.flush(force=True)
    assert len(api.requests("editMessageText")) == attempts
    clock[0] = 43
    await status.flush()
    assert len(api.requests("sendMessage")) == 1
    assert all(p["message_id"] == mid for p in api.requests("editMessageText"))
    assert "Work 11" in api.messages[-100123, mid]["text"]
    assert "[x" in api.messages[-100123, mid]["text"]


@pytest.mark.asyncio
async def test_network_failure_retries_the_same_edit(telegram_api, monkeypatch):
    adapter, api = telegram_api
    status, clock = await initial_status(adapter, monkeypatch)
    api.failures["editMessageText"].extend([NetworkError("Network error")] * 2)
    await status.flush()
    assert not status.disabled
    clock[0] = 14
    await status.flush()
    assert len(api.requests("sendMessage")) == 1
    assert len({p["message_id"] for p in api.requests("editMessageText")}) == 1
    assert status.latest == status.delivered


@pytest.mark.asyncio
@pytest.mark.parametrize("code,description", [(403, "Forbidden: bot was blocked by the user"),
                                              (400, "Bad Request: message to edit not found")])
async def test_permanent_edit_refusal_stops_status_updates(telegram_api, monkeypatch, code, description):
    adapter, api = telegram_api
    status, clock = await initial_status(adapter, monkeypatch)
    api.reject("editMessageText", code, description, times=2)
    await status.flush()
    assert status.disabled
    attempts = len(api.requests("editMessageText"))
    clock[0] = 100
    await status.flush(force=True)
    assert len(api.requests("editMessageText")) == attempts
    assert len(api.requests("sendMessage")) == 1


@pytest.mark.asyncio
async def test_not_modified_is_success_without_plain_fallback(telegram_api, monkeypatch):
    adapter, api = telegram_api
    status, _ = await initial_status(adapter, monkeypatch)
    api.reject("editMessageText", 400, "Bad Request: message is not modified")
    await status.flush()
    assert len(api.requests("editMessageText")) == 1
    assert status.latest == status.delivered and not status.disabled


@pytest.mark.asyncio
async def test_ambiguous_send_timeout_is_not_retried(telegram_api, monkeypatch):
    adapter, api = telegram_api
    status, clock = status_for(adapter, monkeypatch)
    api.failures["sendMessage"].append(TimedOut("Timed out"))
    status.absorb(TodoChecklist(_render_todo_checklist(snapshot("11", "pending"))))
    await status.flush()
    clock[0] = 100
    await status.flush(force=True)
    assert status.disabled and len(api.requests("sendMessage")) == 1


@pytest.mark.asyncio
async def test_permission_error_does_not_block_real_final_delivery(telegram_api):
    adapter, api = telegram_api

    async def handler(event):
        turn = turn_for(adapter)
        sender = asyncio.create_task(turn.send_progress_messages())
        try:
            turn.progress_callback("tool.completed", "todo", result=snapshot("11", "pending"))
            await async_wait_until(lambda: len(api.messages) == 1)
            api.reject("editMessageText", 403, "Forbidden: message cannot be edited", times=2)
            turn.progress_callback("tool.completed", "todo", result=snapshot("11", "completed"))
        finally:
            sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender
        return "Final answer"

    adapter.set_message_handler(handler)
    event = MessageEvent(text="test", source=turn_for(adapter)._ctx.source)
    await adapter._process_message_background(event, "session-11")
    assert len(api.requests("sendMessage")) == 2
    assert api.requests("sendMessage")[-1]["text"] == "Final answer"


@pytest.mark.asyncio
async def test_unicode_and_markdown_expansion_fit_one_message(telegram_api):
    adapter, api = telegram_api
    from gateway.todo_progress import TodoProgress

    status = TodoProgress(turn_for(adapter), adapter)
    data = {"todos": [{"id": str(i), "content": "😀[]()!_" * 40} for i in range(256)]}
    status.absorb(TodoChecklist(_render_todo_checklist(data)))
    await status.flush()
    assert len(api.requests("sendMessage")) == 1
    payload = api.requests("sendMessage")[0]
    assert len(payload["text"].encode("utf-16-le")) // 2 <= 4096
    assert "more tasks" in payload["text"]


@pytest.mark.asyncio
async def test_initial_flood_retries_only_after_server_delay(telegram_api, monkeypatch):
    adapter, api = telegram_api
    status, clock = status_for(adapter, monkeypatch)
    api.reject("sendMessage", 429, "Too Many Requests: retry after 30", retry_after=30)
    status.absorb(TodoChecklist(_render_todo_checklist(snapshot("11", "pending"))))
    await status.flush()
    assert not api.messages
    clock[0] = 20
    await status.flush(force=True)
    assert len(api.requests("sendMessage")) == 1
    clock[0] = 41
    # Expire the adapter's separately recorded cooldown as well.
    adapter._telegram_send_cooldown_until.clear()
    await status.flush()
    assert len(api.requests("sendMessage")) == 2
    assert len(api.messages) == 1 and not status.disabled
