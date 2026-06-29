"""Tests for inbound Telegram message reactions → agent actions.

When a user reacts to one of the bot's messages with an emoji mapped in
``platforms.telegram.extra.reaction_actions`` (or registered by a plugin via
``ctx.register_reaction``), the adapter synthesizes an agent turn. Unconfigured
reactions, removed reactions, and unmapped emoji are ignored, so the feature is
backward-compatible by default.
"""

import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType


def _build_telegram_stubs():
    telegram_mod = types.ModuleType("telegram")
    telegram_mod.Update = object
    telegram_mod.Bot = object
    telegram_mod.Message = object
    telegram_mod.InlineKeyboardButton = object
    telegram_mod.InlineKeyboardMarkup = object
    telegram_mod.LinkPreviewOptions = object

    telegram_ext_mod = types.ModuleType("telegram.ext")
    telegram_ext_mod.Application = object
    telegram_ext_mod.CommandHandler = object
    telegram_ext_mod.CallbackQueryHandler = object
    telegram_ext_mod.MessageHandler = object
    telegram_ext_mod.MessageReactionHandler = object
    telegram_ext_mod.ContextTypes = SimpleNamespace(DEFAULT_TYPE=type(None))
    telegram_ext_mod.filters = SimpleNamespace()

    telegram_constants_mod = types.ModuleType("telegram.constants")
    telegram_constants_mod.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    telegram_constants_mod.ChatType = SimpleNamespace(
        GROUP="group", SUPERGROUP="supergroup", CHANNEL="channel", PRIVATE="private",
    )

    telegram_request_mod = types.ModuleType("telegram.request")
    telegram_request_mod.HTTPXRequest = object

    telegram_mod.ext = telegram_ext_mod
    telegram_mod.constants = telegram_constants_mod
    telegram_mod.request = telegram_request_mod

    return {
        "telegram": telegram_mod,
        "telegram.ext": telegram_ext_mod,
        "telegram.constants": telegram_constants_mod,
        "telegram.request": telegram_request_mod,
    }


@pytest.fixture
def telegram_adapter_cls(monkeypatch):
    """Import TelegramAdapter without leaking temporary telegram stubs."""
    module_name = "plugins.platforms.telegram.adapter"
    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        yield existing_module.TelegramAdapter
        return

    telegram_pkg = sys.modules.get("telegram")
    installed = isinstance(getattr(telegram_pkg, "__file__", None), str)
    if telegram_pkg is None:
        try:
            installed = importlib.util.find_spec("telegram") is not None
        except ValueError:
            installed = False

    if not installed:
        for name, module in _build_telegram_stubs().items():
            monkeypatch.setitem(sys.modules, name, module)

    module = importlib.import_module(module_name)
    try:
        yield module.TelegramAdapter
    finally:
        if not installed:
            sys.modules.pop(module_name, None)


def _make_adapter(telegram_adapter_cls, extra=None):
    return telegram_adapter_cls(
        PlatformConfig(enabled=True, token="***", extra=extra or {})
    )


def _make_reaction_update(emoji, *, chat_type="private", old=()):
    chat = SimpleNamespace(id=7409767798, type=chat_type, title=None, full_name="Sean")
    user = SimpleNamespace(id=7409767798, full_name="Sean")
    new = tuple(SimpleNamespace(emoji=e) for e in ([emoji] if emoji else []))
    old_reaction = tuple(SimpleNamespace(emoji=e) for e in old)
    mr = SimpleNamespace(
        chat=chat,
        user=user,
        message_id=42,
        old_reaction=old_reaction,
        new_reaction=new,
        date=None,
    )
    return SimpleNamespace(message_reaction=mr)


@pytest.mark.asyncio
async def test_recognized_reaction_triggers_agent_turn(telegram_adapter_cls):
    adapter = _make_adapter(
        telegram_adapter_cls, extra={"reaction_actions": {"🔥": "escalate to P1"}}
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_message_reaction(_make_reaction_update("🔥"), MagicMock())

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.message_type == MessageType.TEXT
    assert "🔥" in event.text
    assert "escalate to P1" in event.text
    assert event.source.chat_type == "dm"
    assert event.source.chat_id == "7409767798"
    assert event.message_id == "42"


@pytest.mark.asyncio
async def test_unconfigured_reactions_are_noop(telegram_adapter_cls):
    adapter = _make_adapter(telegram_adapter_cls, extra={})  # no reaction_actions
    adapter.handle_message = AsyncMock()

    await adapter._handle_message_reaction(_make_reaction_update("🔥"), MagicMock())

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_unmapped_emoji_ignored(telegram_adapter_cls):
    adapter = _make_adapter(
        telegram_adapter_cls, extra={"reaction_actions": {"🔥": "escalate"}}
    )
    adapter.handle_message = AsyncMock()

    # 👍 is not in the map → ignored.
    await adapter._handle_message_reaction(_make_reaction_update("👍"), MagicMock())

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_removed_reaction_ignored(telegram_adapter_cls):
    adapter = _make_adapter(
        telegram_adapter_cls, extra={"reaction_actions": {"🔥": "escalate"}}
    )
    adapter.handle_message = AsyncMock()

    # new_reaction empty (user removed their reaction) → nothing added → no-op.
    await adapter._handle_message_reaction(
        _make_reaction_update(None, old=("🔥",)), MagicMock()
    )

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_newly_added_emoji_trigger(telegram_adapter_cls):
    adapter = _make_adapter(
        telegram_adapter_cls, extra={"reaction_actions": {"🔥": "escalate", "👍": "done"}}
    )
    adapter.handle_message = AsyncMock()

    # 🔥 was already present; only 👍 is newly added this update.
    update = _make_reaction_update("👍", old=("🔥",))
    # Add 🔥 back into new_reaction so both are present, but 🔥 is not "new".
    update.message_reaction.new_reaction = (
        SimpleNamespace(emoji="🔥"),
        SimpleNamespace(emoji="👍"),
    )
    await adapter._handle_message_reaction(update, MagicMock())

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert "done" in event.text  # 👍 action, not 🔥
