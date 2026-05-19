"""Tests for inbound message reactions on the Telegram adapter."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure repo root importable
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


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

from gateway.config import Platform, PlatformConfig  # noqa: E402
from gateway.platforms.base import ReactionEvent  # noqa: E402
from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _make_update(
    *,
    chat_id: int = -1001,
    chat_type: str = "supergroup",
    chat_title: str = "Test Group",
    message_id: int = 42,
    user_id: int = 100,
    user_full_name: str = "Tester",
    new_emojis: list = None,
    old_emojis: list = None,
):
    new_reactions = [SimpleNamespace(emoji=e) for e in (new_emojis or [])]
    old_reactions = [SimpleNamespace(emoji=e) for e in (old_emojis or [])]

    chat = SimpleNamespace(id=chat_id, type=chat_type, title=chat_title, full_name=None)
    user = SimpleNamespace(id=user_id, full_name=user_full_name)
    mr = SimpleNamespace(
        chat=chat,
        message_id=message_id,
        user=user,
        actor_chat=None,
        new_reaction=new_reactions,
        old_reaction=old_reactions,
        date=datetime(2026, 5, 11, 18, 30, 0),
    )
    return SimpleNamespace(message_reaction=mr)


def test_reaction_drops_when_no_handler_set():
    """Without a registered reaction handler, the callback is a no-op."""
    adapter = _make_adapter()
    update = _make_update(new_emojis=["👍"])
    # Should complete without raising and without crashing.
    asyncio.run(adapter._handle_message_reaction(update, context=None))


def test_reaction_dispatches_added_emoji():
    adapter = _make_adapter()
    received: list[ReactionEvent] = []

    async def handler(event):
        received.append(event)

    adapter.set_reaction_handler(handler)
    update = _make_update(
        chat_id=-1001,
        message_id=99,
        user_id=555,
        user_full_name="Alice",
        new_emojis=["👍"],
        old_emojis=[],
    )
    asyncio.run(adapter._handle_message_reaction(update, context=None))

    assert len(received) == 1
    ev = received[0]
    assert ev.emoji == "👍"
    assert ev.added is True
    assert ev.message_id == "99"
    assert ev.source.platform == Platform.TELEGRAM
    assert ev.source.chat_id == "-1001"
    assert ev.source.user_id == "555"
    assert ev.source.user_name == "Alice"
    assert ev.source.message_id == "99"


def test_reaction_diffs_old_vs_new():
    """User had 👍, swapped to 👎: emit one add for 👎 and one remove for 👍."""
    adapter = _make_adapter()
    received: list[ReactionEvent] = []

    async def handler(event):
        received.append(event)

    adapter.set_reaction_handler(handler)
    update = _make_update(new_emojis=["👎"], old_emojis=["👍"])
    asyncio.run(adapter._handle_message_reaction(update, context=None))

    by_emoji = {(e.emoji, e.added) for e in received}
    assert ("👎", True) in by_emoji
    assert ("👍", False) in by_emoji
    assert len(received) == 2


def test_reaction_no_diff_no_dispatch():
    """If new and old are identical, emit nothing (PTB still delivers idempotent updates)."""
    adapter = _make_adapter()
    received: list[ReactionEvent] = []

    async def handler(event):
        received.append(event)

    adapter.set_reaction_handler(handler)
    update = _make_update(new_emojis=["👍"], old_emojis=["👍"])
    asyncio.run(adapter._handle_message_reaction(update, context=None))

    assert received == []


def test_reaction_handler_exception_swallowed():
    """A raising handler should not crash the dispatcher."""
    adapter = _make_adapter()

    async def boom(event):
        raise RuntimeError("nope")

    adapter.set_reaction_handler(boom)
    update = _make_update(new_emojis=["🔥"])
    # Should not raise.
    asyncio.run(adapter._handle_message_reaction(update, context=None))


def test_reaction_handles_missing_message_reaction():
    """Update without message_reaction attribute is a no-op."""
    adapter = _make_adapter()

    async def handler(event):
        raise AssertionError("should not be called")

    adapter.set_reaction_handler(handler)
    update = SimpleNamespace(message_reaction=None)
    asyncio.run(adapter._handle_message_reaction(update, context=None))


def test_reaction_event_dataclass_shape():
    """Sanity: ReactionEvent has the fields the gateway logger expects."""
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="group")
    ev = ReactionEvent(
        emoji="👍",
        added=True,
        message_id="1",
        source=source,
    )
    assert ev.emoji == "👍"
    assert ev.added is True
    assert ev.message_id == "1"
    assert ev.source is source
    assert ev.timestamp is not None
