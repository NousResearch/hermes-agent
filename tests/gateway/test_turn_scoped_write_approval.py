"""Turn-scoped staged-write events and Telegram approval-card regressions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import os
import shutil
import tempfile
import threading

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource
from plugins.platforms.telegram import adapter as telegram_module
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Button:
    def __init__(self, text, callback_data=None, **_kwargs):
        self.text = text
        self.callback_data = callback_data


class _Markup:
    def __init__(self, rows):
        self.inline_keyboard = rows


@pytest.fixture
def hermes_home(monkeypatch):
    root = tempfile.mkdtemp(prefix="hermes_turn_wa_test_")
    home = os.path.join(root, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(root, ignore_errors=True)


def _source(*, thread_id="417808"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="273403055",
        chat_type="dm",
        user_id="42",
        thread_id=thread_id,
    )


def _make_adapter(monkeypatch) -> TelegramAdapter:
    monkeypatch.setattr(telegram_module, "InlineKeyboardButton", _Button)
    monkeypatch.setattr(telegram_module, "InlineKeyboardMarkup", _Markup)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=77))
    return adapter


def test_stage_write_emits_only_the_current_turn_record(hermes_home):
    from tools import write_approval as wa

    seen = []
    with wa.capture_staged_writes(
        session_key="agent:main:telegram:dm:273403055:417808",
        run_generation=9,
        profile="default",
        callback=seen.append,
    ):
        current = wa.stage_write(
            "memory",
            {"action": "add", "target": "memory", "content": "current fact"},
            summary="add to memory: current fact",
            origin="foreground",
        )

    stale = wa.stage_write(
        "memory",
        {"action": "add", "target": "memory", "content": "stale fact"},
        summary="add to memory: stale fact",
        origin="foreground",
    )

    assert [event["pending_id"] for event in seen] == [current["id"]]
    assert seen[0]["session_key"] == "agent:main:telegram:dm:273403055:417808"
    assert seen[0]["run_generation"] == 9
    assert seen[0]["profile"] == "default"
    assert seen[0]["subsystem"] == "memory"
    assert seen[0]["target"] == "memory"
    assert seen[0]["operation"] == "add"
    assert seen[0]["preview"] == "current fact"
    assert stale["id"] not in [event["pending_id"] for event in seen]


def test_stage_capture_propagates_into_background_review_thread(hermes_home):
    from tools import write_approval as wa
    from tools.thread_context import propagate_context_to_thread

    seen = []
    with wa.capture_staged_writes(
        session_key="agent:main:telegram:dm:273403055:417808",
        run_generation=9,
        profile="default",
        callback=seen.append,
    ):
        worker = threading.Thread(
            target=propagate_context_to_thread(
                lambda: wa.stage_write(
                    "memory",
                    {
                        "action": "add",
                        "target": "memory",
                        "content": "background fact",
                    },
                    summary="add to memory: background fact",
                    origin="background_review",
                )
            )
        )
        worker.start()
        worker.join(timeout=3)

    assert worker.is_alive() is False
    assert len(seen) == 1
    assert seen[0]["preview"] == "background fact"
    assert seen[0]["session_key"] == "agent:main:telegram:dm:273403055:417808"


def test_durable_card_uses_live_adapter_after_turn_generation_is_stale():
    from gateway.run import _staged_write_delivery_adapter
    from gateway.turn_context import TurnContext

    original = object()
    reconnected = object()
    ctx = TurnContext(
        source=_source(),
        _status_adapter=original,
        _run_still_current=lambda: False,
    )
    runner = SimpleNamespace(_adapter_for_source=lambda _source: reconnected)

    assert _staged_write_delivery_adapter(runner, ctx) is reconnected


def test_durable_card_falls_back_to_original_adapter_after_turn_is_stale():
    from gateway.run import _staged_write_delivery_adapter
    from gateway.turn_context import TurnContext

    original = object()
    ctx = TurnContext(
        source=_source(),
        _status_adapter=original,
        _run_still_current=lambda: False,
    )
    runner = SimpleNamespace(_adapter_for_source=lambda _source: None)

    assert _staged_write_delivery_adapter(runner, ctx) is original


def test_approval_card_uses_captured_id_not_global_pending_queue(hermes_home):
    from gateway.write_approval_interactions import approval_card_for_events
    from tools import write_approval as wa

    stale = wa.stage_write(
        "memory",
        {"action": "add", "target": "memory", "content": "stale fact"},
        summary="add to memory: stale fact",
        origin="foreground",
    )
    current = wa.stage_write(
        "memory",
        {"action": "add", "target": "user", "content": "current fact"},
        summary="add to user profile: current fact",
        origin="foreground",
    )

    event = wa.event_for_record(
        current,
        session_key="agent:main:telegram:dm:273403055:417808",
        run_generation=3,
        profile="default",
    )
    card = approval_card_for_events([event])

    assert "current fact" in card.text
    assert "USER.md" in card.text
    assert current["id"] in card.text
    assert stale["id"] not in card.text
    assert card.surface["subsystems"] == ["memory"]
    assert card.surface["items"] == {"memory": [current["id"]]}
    assert card.surface["scope"]["session_key"] == (
        "agent:main:telegram:dm:273403055:417808"
    )


@pytest.mark.asyncio
async def test_foreground_staged_write_gets_card_after_main_reply(monkeypatch):
    from gateway.write_approval_interactions import deliver_staged_write_cards
    from tools import write_approval as wa

    adapter = AsyncMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="91"))
    source = _source()
    record = {
        "id": "abc12345",
        "subsystem": "memory",
        "action": "add",
        "summary": "add to memory: current fact",
        "payload": {"action": "add", "target": "memory", "content": "current fact"},
    }
    event = wa.event_for_record(
        record,
        session_key="agent:main:telegram:dm:273403055:417808",
        run_generation=11,
        profile="default",
    )

    delivered = await deliver_staged_write_cards(
        adapter=adapter,
        source=source,
        reply_to_message_id="55",
        events=[event],
    )

    assert delivered is True
    adapter.send.assert_awaited_once()
    args = adapter.send.call_args.args
    kwargs = adapter.send.call_args.kwargs
    assert args[0] == "273403055"
    assert "current fact" in args[1]
    assert kwargs["metadata"]["write_approval"]["items"]["memory"] == ["abc12345"]
    assert kwargs["metadata"]["thread_id"] == "417808"
    assert kwargs["metadata"]["notify"] is True


@pytest.mark.asyncio
async def test_multiple_staged_records_get_independent_cards():
    from gateway.write_approval_interactions import deliver_staged_write_cards

    adapter = AsyncMock()
    adapter.send = AsyncMock(
        side_effect=[
            SendResult(success=True, message_id="91"),
            SendResult(success=True, message_id="92"),
        ]
    )
    base = {
        "subsystem": "memory",
        "operation": "add",
        "target": "memory",
        "session_key": "agent:main:telegram:dm:273403055:417808",
        "run_generation": 11,
        "profile": "default",
    }
    events = [
        {**base, "pending_id": "abc12345", "preview": "first fact"},
        {**base, "pending_id": "def67890", "preview": "second fact"},
    ]

    delivered = await deliver_staged_write_cards(
        adapter=adapter,
        source=_source(),
        reply_to_message_id="55",
        events=events,
    )

    assert delivered is True
    assert adapter.send.await_count == 2
    first = adapter.send.await_args_list[0]
    second = adapter.send.await_args_list[1]
    assert first.kwargs["metadata"]["write_approval"]["items"]["memory"] == [
        "abc12345"
    ]
    assert second.kwargs["metadata"]["write_approval"]["items"]["memory"] == [
        "def67890"
    ]
    assert "first fact" in first.args[1]
    assert "second fact" in second.args[1]


class _DeliveryAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, reply_to, metadata))
        return SendResult(success=True, message_id=str(len(self.sent)))


@pytest.mark.asyncio
async def test_post_delivery_callback_sends_card_when_main_handler_errors():
    from gateway.write_approval_interactions import deliver_staged_write_cards

    adapter = _DeliveryAdapter()
    source = _source()
    event = MessageEvent(
        text="remember this",
        message_type=MessageType.TEXT,
        source=source,
        message_id="55",
    )
    staged = {
        "pending_id": "abc12345",
        "subsystem": "memory",
        "operation": "add",
        "target": "memory",
        "preview": "current fact",
        "session_key": "agent:main:telegram:dm:273403055:417808",
        "run_generation": 11,
        "profile": "default",
    }

    async def handler(_event):
        adapter.register_post_delivery_callback(
            staged["session_key"],
            lambda: deliver_staged_write_cards(
                adapter=adapter,
                source=source,
                reply_to_message_id="55",
                events=[staged],
            ),
        )
        raise RuntimeError("agent failed after staging")

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, staged["session_key"])

    assert any("agent failed after staging" in content for _, content, _, _ in adapter.sent)
    assert any("current fact" in content for _, content, _, _ in adapter.sent)
    card_metadata = next(
        metadata for _, content, _, metadata in adapter.sent if "current fact" in content
    )
    assert card_metadata["write_approval"]["items"]["memory"] == ["abc12345"]
    assert card_metadata["thread_id"] == "417808"


@pytest.mark.asyncio
async def test_callback_requires_the_original_chat_topic_profile_and_pending_id(
    monkeypatch,
):
    adapter = _make_adapter(monkeypatch)
    adapter._authorization_check = lambda *_args: True
    adapter.set_message_handler(AsyncMock(return_value="should not run"))
    adapter.send = AsyncMock()
    adapter._remember_write_approval_surface(
        "273403055",
        "77",
        {
            "subsystems": ["memory"],
            "items": {"memory": ["abc12345"]},
            "scope": {
                "session_key": "agent:main:telegram:dm:273403055:417808",
                "run_generation": 11,
                "profile": "default",
                "chat_id": "273403055",
                "thread_id": "417808",
            },
        },
    )
    query = SimpleNamespace(
        data="wa:m:a:abc12345",
        from_user=SimpleNamespace(id=42, first_name="Maxim"),
        message=SimpleNamespace(
            chat_id=273403055,
            chat=SimpleNamespace(type="private"),
            message_thread_id=999999,
            message_id=77,
        ),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    adapter._message_handler.assert_not_awaited()
    assert "expired" in query.answer.call_args.kwargs["text"].lower()


@pytest.mark.asyncio
async def test_resolved_callback_refreshes_original_card_and_uses_snackbar(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    adapter._authorization_check = lambda *_args: True
    adapter.set_message_handler(
        AsyncMock(side_effect=[
            "Approved 1 memory write(s).",
            "No pending memory writes.",
        ])
    )
    adapter.send = AsyncMock()
    adapter._remember_write_approval_surface(
        "273403055",
        "77",
        {
            "subsystems": ["memory"],
            "items": {"memory": ["abc12345"]},
            "scope": {
                "session_key": "agent:main:telegram:dm:273403055:417808",
                "run_generation": 11,
                "profile": "default",
                "chat_id": "273403055",
                "thread_id": "417808",
            },
        },
    )
    query = SimpleNamespace(
        data="wa:m:a:abc12345",
        from_user=SimpleNamespace(id=42, first_name="Maxim"),
        message=SimpleNamespace(
            chat_id=273403055,
            chat=SimpleNamespace(type="private"),
            message_thread_id=417808,
            message_id=77,
            text="💾 Memory proposal\nAdd to MEMORY.md:\n“current fact”",
        ),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    assert adapter._message_handler.await_count == 2
    assert [
        call.args[0].text for call in adapter._message_handler.await_args_list
    ] == ["/memory approve abc12345", "/memory pending"]
    adapter.send.assert_not_awaited()
    query.edit_message_text.assert_awaited_once()
    edit_kwargs = query.edit_message_text.call_args.kwargs
    assert edit_kwargs["text"] == "No pending memory writes."
    assert edit_kwargs["reply_markup"] is None
    assert query.answer.call_args.kwargs["text"] == "✅ Approved abc12345"
