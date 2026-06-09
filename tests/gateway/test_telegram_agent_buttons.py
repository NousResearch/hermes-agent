"""Tests for agent-emitted inline keyboard buttons (HERMES_INLINE_BUTTONS).

A tap on a button the agent emitted (any callback_data that doesn't match a
built-in prefix) must be fed back to the agent as the pressing user's next
message, so the agent can react to the choice.
"""

import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# --- Minimal telegram mock so TelegramAdapter imports without the SDK --------
_fake = types.ModuleType("telegram")
_fake.error = types.SimpleNamespace(
    NetworkError=type("NetworkError", (OSError,), {}),
    TimedOut=type("TimedOut", (OSError,), {}),
    BadRequest=type("BadRequest", (Exception,), {}),
)
_fake.constants = types.SimpleNamespace(
    ParseMode=types.SimpleNamespace(MARKDOWN="Markdown", MARKDOWN_V2="MarkdownV2", HTML="HTML"),
    ChatType=types.SimpleNamespace(PRIVATE="private", GROUP="group", SUPERGROUP="supergroup", CHANNEL="channel"),
)
_fake.ext = types.SimpleNamespace(ContextTypes=types.SimpleNamespace(DEFAULT_TYPE=type(None)))
_fake.request = types.SimpleNamespace()


@pytest.fixture(autouse=True)
def _inject_fake_telegram(monkeypatch):
    for name, mod in (
        ("telegram", _fake),
        ("telegram.error", _fake.error),
        ("telegram.constants", _fake.constants),
        ("telegram.ext", _fake.ext),
        ("telegram.request", _fake.request),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def _make_adapter():
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra={})
    adapter._config = adapter.config
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._message_handler = None  # force fail-closed auth fallback (env-driven)
    # Group-observe attribution is irrelevant to a deliberate DM button tap.
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    return adapter


def _make_query(data: str, *, label=None):
    chat = SimpleNamespace(id=555, type="private", title=None, full_name="Echo Test", is_forum=False)
    button = SimpleNamespace(text=label or data, callback_data=data)
    markup = SimpleNamespace(inline_keyboard=[[button]])
    message = SimpleNamespace(
        chat=chat,
        chat_id=555,
        message_id=99,
        message_thread_id=None,
        is_topic_message=False,
        text="Pick one:",
        reply_markup=markup,
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return SimpleNamespace(
        data=data,
        message=message,
        from_user=SimpleNamespace(id=12345, first_name="Echo", full_name="Echo Test"),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_button_tap_dispatched_to_agent_as_user_message(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()

    query = _make_query("confirm order", label="✅ Confirm")
    update = SimpleNamespace(callback_query=query)
    await adapter._handle_callback_query(update, context=None)

    # The tap became a user turn carrying the button's callback_data.
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "confirm order"
    # The callback was acknowledged and the keyboard stripped.
    query.answer.assert_awaited()
    query.edit_message_text.assert_awaited()
    assert query.edit_message_text.await_args.kwargs.get("reply_markup") is None


@pytest.mark.asyncio
async def test_button_tap_denied_for_unauthorized_user(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")  # not the caller
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()

    query = _make_query("confirm order")
    update = SimpleNamespace(callback_query=query)
    await adapter._handle_callback_query(update, context=None)

    adapter.handle_message.assert_not_awaited()
    query.answer.assert_awaited()
