"""Tests for gmail-triage command-verb dispatch.

Inline-button callbacks are limited to shelling out to scripts. A command verb
(e.g. ``reset``) must instead reconstruct the caller's SessionSource from the
callback metadata and route a synthetic MessageEvent into the runner's
_handle_message pipeline, because Telegram never delivers a bot's own outgoing
``/reset`` back to the bot.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig, Platform


# -- Fake telegram modules (minimal stubs) --------------------------------

_fake_telegram_error = types.ModuleType("telegram.error")


class _TelegramError(Exception):
    pass


_fake_telegram_error.TelegramError = _TelegramError
_fake_telegram_error.BadRequest = type("BadRequest", (_TelegramError,), {})
_fake_telegram_error.NetworkError = type("NetworkError", (_TelegramError,), {})

_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(HTML="HTML")

_fake_telegram_request = types.ModuleType("telegram.request")
_fake_telegram_request.HTTPXRequest = type("HTTPXRequest", (), {"__init__": lambda *a, **kw: None})

_fake_telegram_ext = types.ModuleType("telegram.ext")
_fake_telegram_ext.ApplicationBuilder = type("ApplicationBuilder", (), {
    "token": lambda self, *a: self,
    "build": lambda self: None,
})

_fake_telegram = types.ModuleType("telegram")
_fake_telegram.error = _fake_telegram_error
_fake_telegram.constants = _fake_telegram_constants
_fake_telegram.ext = _fake_telegram_ext
_fake_telegram.request = _fake_telegram_request


@pytest.fixture(autouse=True)
def _inject_fake_telegram(monkeypatch):
    monkeypatch.setitem(sys.modules, "telegram", _fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", _fake_telegram_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", _fake_telegram_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", _fake_telegram_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", _fake_telegram_request)


def _make_adapter(monkeypatch, *, handler=None):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter._connected = True
    # NOTE: `name` is a read-only property on TelegramAdapter, so leave it out.

    # Stub the runner back-pointer. The adapter resolves the runner via
    # ``getattr(getattr(self, "_message_handler", None), "__self__", None)``,
    # so give _message_handler a bound-method with a __self__ that carries the
    # async _handle_message we want to observe.
    captures = {}

    async def _handle_message(event):
        captures["event"] = event
        return "ok"

    class _Runner:
        pass

    runner = _Runner()
    runner._handle_message = _handle_message

    class _Bound:
        __self__ = runner

    adapter._message_handler = _Bound()
    return adapter, captures


def _fake_query():
    async def _answer(**kw):
        return None

    async def _edit_text(**kw):
        return None

    return SimpleNamespace(
        answer=_answer,
        message=SimpleNamespace(text="original", edit_text=_edit_text),
        from_user=SimpleNamespace(id=4242, first_name="Tester"),
    )


class TestCommandVerbDispatch:
    """gmail-triage command verbs must route in-process through _handle_message."""

    def test_reset_verb_routes_synthetic_event_to_runner(self, monkeypatch):
        adapter, captures = _make_adapter(monkeypatch)
        query = _fake_query()

        command_entry = adapter._GT_COMMAND_DISPATCH["reset"]
        assert command_entry[0] == "/reset"

        import asyncio

        asyncio.run(adapter._run_command_verb(
            command_entry,
            query,
            caller_id="4242",
            chat_id="4242",
            chat_type="private",
            thread_id=None,
            user_name="tester",
        ))

        event = captures.get("event")
        assert event is not None, "runner._handle_message was not called"
        assert event.text == "/reset"
        assert event.source.platform == Platform.TELEGRAM
        assert event.source.chat_id == "4242"
        # "private" must normalize to "dm" exactly as the auth path does.
        assert event.source.chat_type == "dm"
        assert event.source.user_id == "4242"
        assert event.source.user_name == "tester"

    def test_command_verb_normalizes_supergroup_to_group_without_thread(self, monkeypatch):
        adapter, captures = _make_adapter(monkeypatch)
        query = _fake_query()
        import asyncio

        asyncio.run(adapter._run_command_verb(
            adapter._GT_COMMAND_DISPATCH["reset"],
            query,
            caller_id="4242",
            chat_id="-100123",
            chat_type="supergroup",
            thread_id=None,
            user_name="tester",
        ))
        assert captures["event"].source.chat_type == "group"

    def test_command_verb_normalizes_supergroup_to_forum_with_thread(self, monkeypatch):
        adapter, captures = _make_adapter(monkeypatch)
        query = _fake_query()
        import asyncio

        asyncio.run(adapter._run_command_verb(
            adapter._GT_COMMAND_DISPATCH["reset"],
            query,
            caller_id="4242",
            chat_id="-100123",
            chat_type="supergroup",
            thread_id="7",
            user_name="tester",
        ))
        assert captures["event"].source.chat_type == "forum"
        assert captures["event"].source.thread_id == "7"