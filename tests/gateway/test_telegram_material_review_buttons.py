"""Stateless material buttons authorize the reviewer before a fixed command."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


PAYLOAD = {
    "profile": "max", "title": "Unit 3", "filename": "unit-3.jpg",
    "confidence": 0.94, "summary": "候選單字（2）", "candidate_count": 2,
    "candidates": [{"term": "arrive", "definition": "抵達"},
                   {"term": "borrow", "definition": "借入"}],
    "approve_action_id": "opaque_a1", "reject_action_id": "opaque_r1",
}


def make_adapter(settings=None):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token", extra={
        "material_review": settings if settings is not None else {
            "reviewer_id": 12345, "chat_id": -10099, "thread_id": 37,
        },
    }))
    adapter._bot = AsyncMock()
    adapter._bot.send_message.return_value = SimpleNamespace(message_id=42)
    adapter._message_handler = AsyncMock()
    return adapter


def callback(data="mr:opaque_a1:a", *, user=12345, chat=-10099, thread=37):
    query = SimpleNamespace(
        data=data, from_user=SimpleNamespace(id=user),
        message=SimpleNamespace(chat_id=chat, message_thread_id=thread,
                                message_id=42, chat=SimpleNamespace(type="supergroup")),
        answer=AsyncMock(), edit_message_text=AsyncMock(),
    )
    return SimpleNamespace(callback_query=query)


def process(monkeypatch, *, result=None, code=0):
    child = SimpleNamespace(
        returncode=code,
        communicate=AsyncMock(return_value=(json.dumps(result if result is not None else {
            "action_id": "opaque_a1", "decision": "approve", "status": "approved",
        }).encode(), b"")),
        kill=Mock(), wait=AsyncMock(),
    )
    launch = AsyncMock(return_value=child)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    return child, launch


@pytest.mark.asyncio
async def test_card_sends_plain_text_and_only_opaque_decision_buttons(monkeypatch):
    monkeypatch.setattr("plugins.platforms.telegram.adapter.InlineKeyboardButton",
                        lambda text, callback_data: SimpleNamespace(text=text, callback_data=callback_data))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
                        lambda rows: SimpleNamespace(inline_keyboard=rows))
    adapter = make_adapter()
    result = await adapter.send_material_review_card(PAYLOAD)
    assert result.success and result.message_id == "42"
    sent = adapter._bot.send_message.call_args.kwargs
    assert sent["chat_id"] == -10099 and sent["message_thread_id"] == 37
    assert "arrive" in sent["text"] and "94%" in sent["text"]
    assert sent.get("parse_mode") is None
    buttons = sent["reply_markup"].inline_keyboard[0]
    assert [(button.text, button.callback_data) for button in buttons] == [
        ("批准", "mr:opaque_a1:a"), ("退回", "mr:opaque_r1:r"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("code,decision,status,label", [
    ("a", "approve", "approved", "已批准"),
    ("r", "reject", "rejected", "已退回"),
])
async def test_callback_after_restart_answers_before_fixed_command_then_edits(
        monkeypatch, code, decision, status, label):
    adapter = make_adapter()  # No card/send state exists in this instance.
    update = callback(f"mr:opaque_a1:{code}")
    child, launch = process(monkeypatch, result={"action_id": "opaque_a1", "status": status})

    async def invoke(*args, **kwargs):
        assert update.callback_query.answer.await_count == 1
        return child

    launch.side_effect = invoke
    monkeypatch.setenv("HERMES_ENGLISH_LEARNING_ENV_FILE", "/wrong/profile.env")
    await adapter._handle_callback_query(update, None)
    assert launch.call_args.args == (
        "/home/alan/.hermes/skills/alan-english-review/bin/english-material-review-action",
        decision, "opaque_a1", "12345",
    )
    assert "HERMES_ENGLISH_LEARNING_ENV_FILE" not in launch.call_args.kwargs["env"]
    assert label in update.callback_query.edit_message_text.call_args.kwargs["text"]
    assert update.callback_query.edit_message_text.call_args.kwargs["reply_markup"] is None
    assert adapter._message_handler.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [
    {"user": 777}, {"chat": -10088}, {"thread": 38}, {"thread": None},
    {"user": None}, {"user": "12345"}, {"user": True},
    {"chat": "-10099"}, {"thread": "37"},
])
async def test_wrong_identity_never_executes_even_with_allow_all(kwargs, monkeypatch):
    adapter = make_adapter()
    _, launch = process(monkeypatch)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    update = callback(**kwargs)
    await adapter._handle_callback_query(update, None)
    assert update.callback_query.answer.await_count == 1
    assert launch.await_count == 0
    assert update.callback_query.edit_message_text.await_count == 0
    assert adapter._message_handler.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [
    "mr::a", "mr:opaque_a1:approve", "mr:opaque_a1:a:extra", "mr:../db:a",
    "mr:" + "a" * 60 + ":a", "mr:單字:a", "mr:opaque_a1:a\n", "mr:a1:;",
])
async def test_malformed_callback_is_answered_without_execution(data, monkeypatch):
    adapter = make_adapter()
    _, launch = process(monkeypatch)
    update = callback(data)
    await adapter._handle_callback_query(update, None)
    assert update.callback_query.answer.await_count == 1
    assert launch.await_count == 0
    assert adapter._message_handler.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [{}, {"reviewer_id": 12345, "chat_id": -10099},
    {"reviewer_id": "12345", "chat_id": -10099, "thread_id": 37}])
async def test_missing_or_invalid_config_fails_closed(settings, monkeypatch):
    adapter = make_adapter(settings)
    _, launch = process(monkeypatch)
    update = callback()
    await adapter._handle_callback_query(update, None)
    assert update.callback_query.answer.await_count == 1
    assert launch.await_count == 0
    assert not (await adapter.send_material_review_card(PAYLOAD)).success
    assert adapter._bot.send_message.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("result,code", [
    ({"action_id": "opaque_a1", "status": "approved"}, 1),
    ({"action_id": "other", "status": "approved"}, 0),
    ({"action_id": "opaque_a1", "status": "pending"}, 0),
    (["approved"], 0),
])
async def test_failed_or_invalid_dispatcher_result_keeps_card(result, code, monkeypatch):
    adapter = make_adapter()
    process(monkeypatch, result=result, code=code)
    update = callback()
    await adapter._handle_callback_query(update, None)
    assert update.callback_query.answer.await_count == 1
    assert update.callback_query.edit_message_text.await_count == 0


@pytest.mark.asyncio
async def test_repeated_click_uses_durable_dispatcher_result_after_edit_failure(monkeypatch):
    process(monkeypatch, result={"action_id": "opaque_r1", "status": "approved"})
    first = callback("mr:opaque_r1:r")
    first.callback_query.edit_message_text.side_effect = RuntimeError("network")
    await make_adapter()._handle_callback_query(first, None)
    replay = callback("mr:opaque_r1:r")
    await make_adapter()._handle_callback_query(replay, None)
    assert first.callback_query.answer.await_count == 1
    assert replay.callback_query.answer.await_count == 1
    assert "已批准" in replay.callback_query.edit_message_text.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_dispatcher_timeout_kills_child_and_keeps_buttons(monkeypatch):
    child, _ = process(monkeypatch)
    child.communicate.side_effect = asyncio.TimeoutError
    update = callback()
    await make_adapter()._handle_callback_query(update, None)
    assert child.kill.call_count == 1 and child.wait.await_count == 1
    assert update.callback_query.answer.await_count == 1
    assert update.callback_query.edit_message_text.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {}, {**PAYLOAD, "approve_action_id": "../db"},
    {**PAYLOAD, "reject_action_id": "opaque_a1"},
])
async def test_invalid_card_payload_is_not_sent(payload):
    adapter = make_adapter()
    result = await adapter.send_material_review_card(payload)
    assert not result.success
    assert adapter._bot.send_message.await_count == 0
