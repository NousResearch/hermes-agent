"""Tests for Telegram inline-keyboard /reasoning and /fast pickers."""

import sys
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

import gateway.run as gateway_run
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class _RecordingButton:
    def __init__(self, text, callback_data=None, **_kwargs):
        self.text = text
        self.callback_data = callback_data


class _RecordingMarkup:
    def __init__(self, rows):
        self.inline_keyboard = rows


def _callback_data(markup):
    return [button.callback_data for row in markup.inline_keyboard for button in row]


def test_send_reasoning_picker_renders_effort_buttons(monkeypatch):
    import plugins.platforms.telegram.adapter as tg

    monkeypatch.setattr(tg, "InlineKeyboardButton", _RecordingButton)
    monkeypatch.setattr(tg, "InlineKeyboardMarkup", _RecordingMarkup)

    adapter = _make_adapter()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=101))
    callback = AsyncMock()

    result = asyncio.run(adapter.send_reasoning_picker(
        chat_id="12345",
        current_effort="medium",
        display_enabled=False,
        session_key="sk1",
        on_reasoning_selected=callback,
    ))

    assert result.success is True
    kwargs = adapter._bot.send_message.call_args[1]
    assert "Reasoning" in kwargs["text"]
    data = _callback_data(kwargs["reply_markup"])
    assert "rp:none" in data
    assert "rp:medium" in data
    assert "rp:high" in data
    assert "rp:xhigh" in data
    assert "rp:max" in data
    assert "rp:ultra" in data
    assert "rp:show" in data
    assert "rp:reset" in data
    assert adapter._reasoning_picker_state[("12345", 101)]["session_key"] == "sk1"


def test_reasoning_picker_state_does_not_collide_within_chat(monkeypatch):
    import plugins.platforms.telegram.adapter as tg

    monkeypatch.setattr(tg, "InlineKeyboardButton", _RecordingButton)
    monkeypatch.setattr(tg, "InlineKeyboardMarkup", _RecordingMarkup)

    adapter = _make_adapter()
    adapter._bot.send_message = AsyncMock(side_effect=[
        SimpleNamespace(message_id=101),
        SimpleNamespace(message_id=102),
    ])

    for session_key in ("topic-a", "topic-b"):
        asyncio.run(adapter.send_reasoning_picker(
            chat_id="12345",
            current_effort="medium",
            display_enabled=False,
            session_key=session_key,
            on_reasoning_selected=AsyncMock(),
        ))

    assert adapter._reasoning_picker_state[("12345", 101)]["session_key"] == "topic-a"
    assert adapter._reasoning_picker_state[("12345", 102)]["session_key"] == "topic-b"


def test_reasoning_picker_prunes_expired_and_superseded_state(monkeypatch):
    import plugins.platforms.telegram.adapter as tg

    adapter = _make_adapter()
    adapter._reasoning_picker_state = {
        ("12345", 100): {"session_key": "same-session", "created_at": 995.0},
        ("12345", 101): {"session_key": "expired-session", "created_at": 0.0},
        ("12345", 102): {"session_key": "active-session", "created_at": 999.0},
    }
    monkeypatch.setattr(tg.time, "monotonic", lambda: 1000.0)

    adapter._prune_settings_picker_state(
        adapter._reasoning_picker_state,
        session_key="same-session",
    )

    assert set(adapter._reasoning_picker_state) == {("12345", 102)}


def test_reasoning_picker_callback_invokes_selection_and_clears_state():
    adapter = _make_adapter()
    callback = AsyncMock(return_value="Reasoning effort set to high.")
    adapter._reasoning_picker_state[("12345", 101)] = {
        "session_key": "sk1",
        "on_reasoning_selected": callback,
    }

    query = AsyncMock()
    query.data = "rp:high"
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.message_id = 101
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    asyncio.run(adapter._handle_reasoning_picker_callback(query, "rp:high", "12345"))

    callback.assert_awaited_once_with("12345", "high")
    query.edit_message_text.assert_awaited()
    assert "Reasoning effort set to high" in query.edit_message_text.call_args[1]["text"]
    assert ("12345", 101) not in adapter._reasoning_picker_state


def test_reasoning_picker_callback_is_single_use():
    adapter = _make_adapter()
    callback = AsyncMock(return_value="Reasoning effort set to high.")
    adapter._reasoning_picker_state[("12345", 101)] = {
        "session_key": "sk1",
        "on_reasoning_selected": callback,
    }

    query = AsyncMock()
    query.message = MagicMock(chat_id=12345, message_id=101)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    asyncio.run(adapter._handle_reasoning_picker_callback(query, "rp:high", "12345"))
    asyncio.run(adapter._handle_reasoning_picker_callback(query, "rp:low", "12345"))

    callback.assert_awaited_once_with("12345", "high")
    assert "Picker expired" in query.answer.await_args_list[-1].kwargs["text"]


def test_send_fast_picker_renders_mode_buttons(monkeypatch):
    import plugins.platforms.telegram.adapter as tg

    monkeypatch.setattr(tg, "InlineKeyboardButton", _RecordingButton)
    monkeypatch.setattr(tg, "InlineKeyboardMarkup", _RecordingMarkup)

    adapter = _make_adapter()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=102))
    callback = AsyncMock()

    result = asyncio.run(adapter.send_fast_picker(
        chat_id="12345",
        current_mode="normal",
        session_key="sk1",
        on_fast_selected=callback,
    ))

    assert result.success is True
    kwargs = adapter._bot.send_message.call_args[1]
    assert "Fast Mode" in kwargs["text"]
    data = _callback_data(kwargs["reply_markup"])
    assert "fp:normal" in data
    assert "fp:fast" in data
    assert "fp:cancel" in data
    assert adapter._fast_picker_state[("12345", 102)]["session_key"] == "sk1"


def test_fast_picker_callback_invokes_selection_and_clears_state():
    adapter = _make_adapter()
    callback = AsyncMock(return_value="Fast mode set to FAST.")
    adapter._fast_picker_state[("12345", 102)] = {
        "session_key": "sk1",
        "on_fast_selected": callback,
    }

    query = AsyncMock()
    query.data = "fp:fast"
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.message_id = 102
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    asyncio.run(adapter._handle_fast_picker_callback(query, "fp:fast", "12345"))

    callback.assert_awaited_once_with("12345", "fast")
    query.edit_message_text.assert_awaited()
    assert "Fast mode set to FAST" in query.edit_message_text.call_args[1]["text"]
    assert ("12345", 102) not in adapter._fast_picker_state


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_reasoning_overrides = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._reasoning_config = {"enabled": True, "effort": "medium"}
    runner._service_tier = None
    runner._show_reasoning = False
    runner._session_db = None
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)
    runner.session_store = None
    return runner


def test_reasoning_command_without_args_sends_telegram_picker(monkeypatch, tmp_path):
    runner = _make_runner()
    adapter = SimpleNamespace(send_reasoning_picker=AsyncMock(return_value=SimpleNamespace(success=True)))
    runner.adapters[Platform.TELEGRAM] = adapter

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "display": {
                "show_reasoning": False,
                "platforms": {"telegram": {"show_reasoning": True}},
            }
        },
    )
    monkeypatch.setattr(
        runner,
        "_resolve_session_reasoning_config",
        lambda source=None, session_key=None: {"enabled": True, "effort": "medium"},
    )

    response = asyncio.run(runner._handle_reasoning_command(_make_event("/reasoning")))

    assert response is None
    adapter.send_reasoning_picker.assert_awaited_once()
    kwargs = adapter.send_reasoning_picker.call_args.kwargs
    assert kwargs["chat_id"] == "12345"
    assert kwargs["current_effort"] == "medium"
    assert kwargs["display_enabled"] is True

    callback = kwargs["on_reasoning_selected"]
    for effort in ("max", "ultra"):
        result = asyncio.run(callback("12345", effort))
        assert effort in result
        assert runner._reasoning_config == {"enabled": True, "effort": effort}


def test_fast_command_without_args_sends_telegram_picker(monkeypatch, tmp_path):
    runner = _make_runner()
    adapter = SimpleNamespace(send_fast_picker=AsyncMock(return_value=SimpleNamespace(success=True)))
    runner.adapters[Platform.TELEGRAM] = adapter

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(runner, "_load_service_tier", lambda: "priority")
    to_thread_calls = []

    async def _record_to_thread(func, *args):
        to_thread_calls.append(func)
        return func(*args)

    monkeypatch.setattr(asyncio, "to_thread", _record_to_thread)

    response = asyncio.run(runner._handle_fast_command(_make_event("/fast")))

    assert response is None
    adapter.send_fast_picker.assert_awaited_once()
    kwargs = adapter.send_fast_picker.call_args.kwargs
    assert kwargs["chat_id"] == "12345"
    assert kwargs["current_mode"] == "fast"
    assert runner._normalize_source_for_session_key in to_thread_calls
