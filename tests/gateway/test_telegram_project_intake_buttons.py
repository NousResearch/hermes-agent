"""Tests for Telegram project-intake inline keyboard buttons."""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class _FakeInlineKeyboardButton:
    def __init__(self, text, callback_data=None, **kwargs):
        self.text = text
        self.callback_data = callback_data
        self.kwargs = kwargs


class _FakeInlineKeyboardMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard


class _FakeBadRequest(Exception):
    pass


_fake_telegram = types.ModuleType("telegram")
_fake_telegram.Update = object
_fake_telegram.Bot = object
_fake_telegram.Message = object
_fake_telegram.InlineKeyboardButton = _FakeInlineKeyboardButton
_fake_telegram.InlineKeyboardMarkup = _FakeInlineKeyboardMarkup
_fake_telegram_error = types.ModuleType("telegram.error")
_fake_telegram_error.NetworkError = OSError
_fake_telegram_error.BadRequest = _FakeBadRequest
_fake_telegram_error.TimedOut = OSError
_fake_telegram.error = _fake_telegram_error
_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(
    MARKDOWN_V2="MarkdownV2",
    MARKDOWN="Markdown",
    HTML="HTML",
)
_fake_telegram_constants.ChatType = SimpleNamespace(
    GROUP="group",
    SUPERGROUP="supergroup",
    CHANNEL="channel",
    PRIVATE="private",
)
_fake_telegram.constants = _fake_telegram_constants
_fake_telegram_ext = types.ModuleType("telegram.ext")
_fake_telegram_ext.Application = object
_fake_telegram_ext.CommandHandler = object
_fake_telegram_ext.CallbackQueryHandler = object
_fake_telegram_ext.MessageHandler = object
_fake_telegram_ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
_fake_telegram_ext.filters = object
_fake_telegram_request = types.ModuleType("telegram.request")
_fake_telegram_request.HTTPXRequest = object

for _name, _module in {
    "telegram": _fake_telegram,
    "telegram.error": _fake_telegram_error,
    "telegram.constants": _fake_telegram_constants,
    "telegram.ext": _fake_telegram_ext,
    "telegram.request": _fake_telegram_request,
}.items():
    sys.modules[_name] = _module

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
import gateway.platforms.telegram as telegram_platform
from gateway.platforms.telegram import TelegramAdapter
from gateway.session import SessionSource

# Other Telegram gateway tests may import gateway.platforms.telegram before this
# file's fake telegram modules are installed. Patch the module globals too so
# this test remains order-independent in larger pytest invocations.
telegram_platform.InlineKeyboardButton = _FakeInlineKeyboardButton
telegram_platform.InlineKeyboardMarkup = _FakeInlineKeyboardMarkup


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=77))
    )
    adapter._app = MagicMock()
    return adapter


def _button_callback_data(markup):
    return [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data
    ]


def _seed_flow(adapter, flow_id="abc123", *, step="kind", answers=None, callback=None):
    adapter._project_intake_state[flow_id] = {
        "session_key": "agent:main:telegram:dm:12345",
        "chat_id": "12345",
        "title": "Mobile Telegram intake",
        "description": "Buttons are hidden on Telegram mobile",
        "step": step,
        "answers": dict(answers or {}),
        "metadata": {},
        "on_intake_selected": callback or AsyncMock(return_value="Created t_test"),
    }
    return adapter._project_intake_state[flow_id]


def _make_callback_update(
    data,
    *,
    user_id="12345",
    chat_id=12345,
    chat_type="private",
    thread_id=None,
    message_id=55,
):
    query = SimpleNamespace(
        data=data,
        message=SimpleNamespace(
            chat_id=chat_id,
            chat=SimpleNamespace(type=chat_type),
            message_thread_id=thread_id,
            message_id=message_id,
        ),
        from_user=SimpleNamespace(id=user_id, first_name="Tester"),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )
    return SimpleNamespace(callback_query=query), query


@pytest.mark.asyncio
async def test_send_project_intake_prompt_renders_inline_keyboard():
    adapter = _make_adapter()
    callback = AsyncMock(return_value="Created t_test")

    result = await adapter.send_project_intake_prompt(
        chat_id="12345",
        title="Mobile Telegram intake",
        state={"description": "Buttons are hidden on Telegram mobile"},
        session_key="agent:main:telegram:dm:12345",
        on_intake_selected=callback,
    )

    assert result.success is True
    assert result.message_id == "77"
    send_kwargs = adapter._bot.send_message.call_args.kwargs
    assert send_kwargs["chat_id"] == 12345
    assert "Mobile Telegram intake" in send_kwargs["text"]
    markup = send_kwargs["reply_markup"]
    assert isinstance(markup, _FakeInlineKeyboardMarkup)
    callback_data = _button_callback_data(markup)
    assert callback_data
    assert all(value.startswith("pi:") for value in callback_data)
    assert all(len(value.encode("utf-8")) <= 64 for value in callback_data)

    flow_ids = {value.split(":")[1] for value in callback_data}
    assert len(flow_ids) == 1
    flow_id = flow_ids.pop()
    stored = adapter._project_intake_state[flow_id]
    assert stored["session_key"] == "agent:main:telegram:dm:12345"
    assert stored["step"] == "kind"
    assert stored["answers"] == {}
    assert stored["prompt_message_id"] == "77"
    assert stored["on_intake_selected"] is callback


@pytest.mark.asyncio
async def test_project_intake_flow_id_uses_random_nonce(monkeypatch):
    values = iter(["nonce_one", "nonce_two"])
    monkeypatch.setattr(telegram_platform.secrets, "token_urlsafe", lambda nbytes: next(values))
    adapter = _make_adapter()

    assert adapter._new_project_intake_flow_id() == "nonce_one"
    assert adapter._new_project_intake_flow_id() == "nonce_two"


@pytest.mark.asyncio
async def test_project_intake_callback_rejects_wrong_prompt_message_id():
    adapter = _make_adapter()
    state = _seed_flow(
        adapter,
        step="board",
        answers={"kind": "feature"},
        callback=AsyncMock(),
    )
    state["prompt_message_id"] = "55"
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    update, query = _make_callback_update(
        "pi:abc123:board:control",
        chat_id=12345,
        message_id=999,
    )

    await adapter._handle_callback_query(update, SimpleNamespace())

    query.answer.assert_called_once_with(text="Project intake belongs to a different prompt.")
    query.edit_message_text.assert_not_called()
    assert state["step"] == "board"
    assert state["answers"] == {"kind": "feature"}


@pytest.mark.asyncio
async def test_send_project_intake_prompt_uses_dm_topic_reply_fallback():
    adapter = _make_adapter()

    await adapter.send_project_intake_prompt(
        chat_id="12345",
        title="Mobile Telegram intake",
        state={"description": "Buttons are hidden on Telegram mobile"},
        session_key="agent:main:telegram:dm:12345:20197",
        on_intake_selected=AsyncMock(),
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "462",
        },
    )

    send_kwargs = adapter._bot.send_message.call_args.kwargs
    assert send_kwargs["reply_to_message_id"] == 462
    assert send_kwargs["message_thread_id"] == 20197
    assert "direct_messages_topic_id" not in send_kwargs


@pytest.mark.asyncio
async def test_project_intake_callback_rejects_unauthorized_user():
    adapter = _make_adapter()
    callback = AsyncMock()
    _seed_flow(adapter, callback=callback)
    adapter._is_callback_user_authorized = MagicMock(return_value=False)
    update, query = _make_callback_update("pi:abc123:kind:feature", user_id="999")

    await adapter._handle_callback_query(update, SimpleNamespace())

    query.answer.assert_called_once_with(
        text="⛔ You are not authorized to answer this intake prompt."
    )
    query.edit_message_text.assert_not_called()
    callback.assert_not_called()
    assert adapter._project_intake_state["abc123"]["answers"] == {}
    assert adapter._project_intake_state["abc123"]["step"] == "kind"


@pytest.mark.asyncio
async def test_project_intake_callback_advances_step_and_edits_message():
    adapter = _make_adapter()
    _seed_flow(adapter)
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    update, query = _make_callback_update("pi:abc123:kind:feature")

    await adapter._handle_callback_query(update, SimpleNamespace())

    state = adapter._project_intake_state["abc123"]
    assert state["answers"] == {"kind": "feature"}
    assert state["step"] == "board"
    query.edit_message_text.assert_called_once()
    edit_kwargs = query.edit_message_text.call_args.kwargs
    assert "Where should it land" in edit_kwargs["text"]
    assert isinstance(edit_kwargs["reply_markup"], _FakeInlineKeyboardMarkup)
    assert any(data == "pi:abc123:board:control" for data in _button_callback_data(edit_kwargs["reply_markup"]))
    query.answer.assert_called_once()


@pytest.mark.asyncio
async def test_project_intake_callback_submit_preview_does_not_create_card():
    adapter = _make_adapter()
    callback = AsyncMock()
    _seed_flow(
        adapter,
        step="confirm",
        answers={
            "kind": "feature",
            "board": "control",
            "scope": "impl_after_spec",
            "risk": "safe",
        },
        callback=callback,
    )
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    update, query = _make_callback_update("pi:abc123:confirm:preview")

    await adapter._handle_callback_query(update, SimpleNamespace())

    callback.assert_not_called()
    query.edit_message_text.assert_called_once()
    edit_kwargs = query.edit_message_text.call_args.kwargs
    assert "Project intake preview" in edit_kwargs["text"]
    assert "Feature / UX improvement" in edit_kwargs["text"]
    assert edit_kwargs["reply_markup"] is None


@pytest.mark.asyncio
async def test_project_intake_callback_submit_create_invokes_callback_once():
    adapter = _make_adapter()
    callback = AsyncMock(return_value="Created triage card t_abc123")
    _seed_flow(
        adapter,
        step="confirm",
        answers={
            "kind": "feature",
            "board": "control",
            "scope": "impl_after_spec",
            "risk": "safe",
        },
        callback=callback,
    )
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    update, query = _make_callback_update("pi:abc123:confirm:create")

    await adapter._handle_callback_query(update, SimpleNamespace())

    callback.assert_awaited_once()
    payload = callback.await_args.args[0]
    assert payload["title"] == "Mobile Telegram intake"
    assert payload["description"] == "Buttons are hidden on Telegram mobile"
    assert payload["answers"] == {
        "kind": "feature",
        "board": "control",
        "scope": "impl_after_spec",
        "risk": "safe",
    }
    assert payload["source"]["platform"] == "telegram"
    assert "token" not in repr(payload).lower()
    assert "abc123" not in adapter._project_intake_state
    edit_kwargs = query.edit_message_text.call_args.kwargs
    assert "Created triage card t_abc123" in edit_kwargs["text"]
    assert edit_kwargs["reply_markup"] is None


@pytest.mark.asyncio
async def test_project_intake_callback_rejects_stale_step_buttons():
    adapter = _make_adapter()
    state = _seed_flow(
        adapter,
        step="board",
        answers={"kind": "bug"},
        callback=AsyncMock(),
    )
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    update, query = _make_callback_update("pi:abc123:kind:feature")

    await adapter._handle_callback_query(update, SimpleNamespace())

    query.answer.assert_called_once_with(text="Project intake moved on — use the latest buttons.")
    query.edit_message_text.assert_not_called()
    assert state["step"] == "board"
    assert state["answers"] == {"kind": "bug"}


@pytest.mark.asyncio
async def test_project_intake_callback_rejects_wrong_chat_or_thread():
    adapter = _make_adapter()
    state = _seed_flow(
        adapter,
        step="board",
        answers={"kind": "feature"},
        callback=AsyncMock(),
    )
    state["metadata"] = {"thread_id": "20197"}
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    update, query = _make_callback_update(
        "pi:abc123:board:control",
        chat_id=54321,
        thread_id="20197",
    )

    await adapter._handle_callback_query(update, SimpleNamespace())

    query.answer.assert_called_once_with(text="Project intake belongs to a different chat/thread.")
    query.edit_message_text.assert_not_called()
    assert state["step"] == "board"
    assert state["answers"] == {"kind": "feature"}

    update, query = _make_callback_update(
        "pi:abc123:board:control",
        chat_id=12345,
        thread_id="99999",
    )

    await adapter._handle_callback_query(update, SimpleNamespace())

    query.answer.assert_called_once_with(text="Project intake belongs to a different chat/thread.")
    query.edit_message_text.assert_not_called()
    assert state["step"] == "board"
    assert state["answers"] == {"kind": "feature"}


@pytest.mark.asyncio
async def test_project_intake_callback_submit_create_requires_complete_answers():
    adapter = _make_adapter()
    callback = AsyncMock(return_value="Created triage card t_abc123")
    state = _seed_flow(
        adapter,
        step="confirm",
        answers={
            "kind": "feature",
            "board": "control",
            "scope": "impl_after_spec",
        },
        callback=callback,
    )
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    update, query = _make_callback_update("pi:abc123:confirm:create")

    await adapter._handle_callback_query(update, SimpleNamespace())

    callback.assert_not_called()
    query.answer.assert_called_once_with(text="Project intake is incomplete — choose missing answers first.")
    query.edit_message_text.assert_called_once()
    edit_kwargs = query.edit_message_text.call_args.kwargs
    assert "Stop-gate sensitivity" in edit_kwargs["text"]
    assert isinstance(edit_kwargs["reply_markup"], _FakeInlineKeyboardMarkup)
    assert state["step"] == "risk"
    assert "abc123" in adapter._project_intake_state


@pytest.mark.asyncio
async def test_project_intake_callback_submit_create_reserves_flow_before_await():
    adapter = _make_adapter()
    callback_calls = []
    callback_entered = asyncio.Event()
    callback_release = asyncio.Event()

    async def slow_create(payload):
        callback_calls.append(payload)
        callback_entered.set()
        await callback_release.wait()
        return "Created triage card t_abc123"

    _seed_flow(
        adapter,
        step="confirm",
        answers={
            "kind": "feature",
            "board": "control",
            "scope": "impl_after_spec",
            "risk": "safe",
        },
        callback=slow_create,
    )
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    update1, query1 = _make_callback_update("pi:abc123:confirm:create")
    first_task = asyncio.create_task(adapter._handle_callback_query(update1, SimpleNamespace()))

    try:
        await asyncio.wait_for(callback_entered.wait(), timeout=1)
        assert "abc123" not in adapter._project_intake_state

        update2, query2 = _make_callback_update("pi:abc123:confirm:create")
        await asyncio.wait_for(adapter._handle_callback_query(update2, SimpleNamespace()), timeout=1)

        query2.answer.assert_called_once_with(text="Project intake expired — use /project again.")
        query2.edit_message_text.assert_not_called()
        assert len(callback_calls) == 1
    finally:
        callback_release.set()
        await asyncio.wait_for(first_task, timeout=1)

    query1.answer.assert_called_once_with(text="Submitted")
    assert len(callback_calls) == 1


@pytest.mark.asyncio
async def test_project_intake_callback_expired_flow_is_harmless():
    adapter = _make_adapter()
    update, query = _make_callback_update("pi:expired:kind:feature")

    await adapter._handle_callback_query(update, SimpleNamespace())

    query.answer.assert_called_once_with(text="Project intake expired — use /project again.")
    query.edit_message_text.assert_not_called()


class _FakeProjectAdapter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def send_project_intake_prompt(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_project_command_is_gateway_known():
    from hermes_cli.commands import is_gateway_known_command, resolve_command

    assert is_gateway_known_command("project") is True
    assert resolve_command("project").name == "project"


@pytest.mark.asyncio
async def test_project_command_calls_adapter_with_thread_metadata():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = _FakeProjectAdapter(SendResult(success=True, message_id="77"))
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_key_for_source = lambda source: "session-key"

    event = MessageEvent(
        text="/project mobile buttons hidden",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            thread_id="20197",
            user_id="12345",
        ),
        message_id="462",
    )

    result = await runner._handle_project_command(event)

    assert result is None
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call["chat_id"] == "12345"
    assert call["title"] == "mobile buttons hidden"
    assert call["state"]["description"] == "mobile buttons hidden"
    assert call["session_key"] == "session-key"
    assert call["metadata"] == {
        "thread_id": "20197",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "462",
    }


@pytest.mark.asyncio
async def test_project_command_returns_text_fallback_when_adapter_unsupported():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: _FakeProjectAdapter(SendResult(success=False, error="Not supported"))}
    runner._session_key_for_source = lambda source: "session-key"

    event = MessageEvent(
        text="/project",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )

    result = await runner._handle_project_command(event)

    assert result is not None
    assert "Project intake" in result
    assert "reply with /project" in result
