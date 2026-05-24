"""Telegram quick-action palette behavior."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock
import pytest

from gateway.config import PlatformConfig
from gateway.platforms import telegram as tg
from gateway.platforms.telegram import TelegramAdapter


class Btn:
    def __init__(self, text, callback_data=None): self.text, self.callback_data = text, callback_data


class Markup:
    def __init__(self, inline_keyboard): self.inline_keyboard = inline_keyboard


@pytest.fixture(autouse=True)
def fake_keyboard(monkeypatch):
    monkeypatch.setattr(tg, "InlineKeyboardButton", Btn)
    monkeypatch.setattr(tg, "InlineKeyboardMarkup", Markup)


def adapter():
    a = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    a._bot = AsyncMock()
    a._send_message_with_thread_fallback = AsyncMock(return_value=NS(message_id=77))
    a.handle_message = AsyncMock()
    return a


def cb(data):
    user = NS(id=123, first_name="Merlin", full_name="Merlin")
    chat = NS(id=456, type="private", title=None, full_name="Merlin", is_forum=False)
    msg = NS(chat_id=456, chat=chat, from_user=user, message_id=88, message_thread_id=None,
             is_topic_message=False, text="", caption=None, reply_to_message=None, date=None)
    q = NS(data=data, message=msg, from_user=user, answer=AsyncMock(), edit_message_text=AsyncMock())
    return NS(callback_query=q), q


def cbs(markup):
    return [b.callback_data for row in markup.inline_keyboard for b in row]


async def open_personality_palette(adapter):
    update, query = cb("qa:personality")
    await adapter._handle_callback_query(update, MagicMock())
    return cbs(query.edit_message_text.call_args.kwargs["reply_markup"])


@pytest.mark.asyncio
async def test_send_palette_renders_discord_matching_actions():
    a = adapter()
    assert (await a.send_palette("456")).success is True
    assert cbs(a._send_message_with_thread_fallback.call_args.kwargs["reply_markup"]) == [
        "qa:status", "qa:usage", "qa:help", "qa:model", "qa:agents", "qa:personality",
        "qa:whoami", "qa:insights", "qa:new", "qa:retry", "qa:undo", "qa:stop",
        "qa:compress", "qa:fast", "qa:yolo",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("data", "expected"), [
    ("qa:model", "/model"), ("qaf:normal", "/fast normal"), ("qap:coder", "/personality coder"),
])
async def test_palette_callbacks_dispatch_existing_command_paths(monkeypatch, data, expected):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    a = adapter()
    await a._handle_callback_query(cb(data)[0], MagicMock())
    assert a.handle_message.call_args.args[0].text == expected


@pytest.mark.asyncio
async def test_palette_config_buttons_open_menus(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.cfg_get", lambda _cfg, *_path, default=None: {"cat": {}, "coder": {}})
    a = adapter()
    for data, expected in [("qa:fast", ["qaf:status", "qaf:fast", "qaf:normal"]),
                           ("qa:personality", ["qap:0", "qap:1", "qap:2"]),
                           ("qa:stop", ["qac:confirm:stop", "qac:cancel:stop"] )]:
        update, query = cb(data)
        await a._handle_callback_query(update, MagicMock())
        assert cbs(query.edit_message_text.call_args.kwargs["reply_markup"]) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["x" * 90, "猫" * 40])
async def test_personality_palette_dispatches_long_and_multibyte_names(monkeypatch, name):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.cfg_get", lambda _cfg, *_path, default=None: {name: {}})
    a = adapter()

    callbacks = await open_personality_palette(a)

    assert callbacks == ["qap:0", "qap:1"]
    assert all(len(data.encode("utf-8")) <= 64 for data in callbacks)
    a.handle_message.reset_mock()
    await a._handle_callback_query(cb("qap:1")[0], MagicMock())
    assert a.handle_message.call_args.args[0].text == f"/personality {name}"


@pytest.mark.asyncio
async def test_personality_palette_disambiguates_names_with_same_long_prefix(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    first = ("shared-prefix-" * 5) + "first"
    second = ("shared-prefix-" * 5) + "second"
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.cfg_get", lambda _cfg, *_path, default=None: {
        first: {},
        second: {},
    })
    a = adapter()

    callbacks = await open_personality_palette(a)

    assert callbacks == ["qap:0", "qap:1", "qap:2"]
    assert len(set(callbacks)) == len(callbacks)
    a.handle_message.reset_mock()
    await a._handle_callback_query(cb("qap:2")[0], MagicMock())
    assert a.handle_message.call_args.args[0].text == f"/personality {second}"


@pytest.mark.asyncio
async def test_palette_confirm_and_command_path(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    a = adapter()
    await a._handle_callback_query(cb("qac:confirm:stop")[0], MagicMock())
    assert a.handle_message.call_args.args[0].text == "/stop"
    assert not hasattr(a.handle_message.call_args.args[0], "preconfirmed_destructive")
    a.handle_message.reset_mock()
    a._should_process_message = MagicMock(return_value=True)
    a._ensure_forum_commands = AsyncMock()
    a._clean_bot_trigger_text = lambda text: text
    a._apply_telegram_group_observe_attribution = lambda event: event
    a.send_palette = AsyncMock()
    user = NS(id=123, first_name="Merlin", full_name="Merlin")
    chat = NS(id=456, type="private", title=None, full_name="Merlin", is_forum=False)
    msg = NS(text="/palette", chat=chat, from_user=user, message_id=99, message_thread_id=None,
             is_topic_message=False, caption=None, reply_to_message=None, date=None)
    await a._handle_command(NS(update_id=100, message=msg, edited_message=None, channel_post=None, edited_channel_post=None), MagicMock())
    a.send_palette.assert_awaited_once()
    a.handle_message.assert_not_awaited()
