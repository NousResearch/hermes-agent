"""Tests for Telegram bot-to-bot loop prevention."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Telegram mock so TelegramAdapter can be imported
# ---------------------------------------------------------------------------
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

from gateway.config import Platform, PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    return adapter


def _message(
    text="hello",
    *,
    chat_id=-100,
    sender_id=1,
    is_bot=True,
    thread_id=None,
):
    return SimpleNamespace(
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        chat=SimpleNamespace(id=chat_id, type="group"),
        from_user=SimpleNamespace(id=sender_id, is_bot=is_bot, full_name=f"user-{sender_id}"),
        sender_chat=None,
        message_thread_id=thread_id,
        media_group_id=None,
        photo=[],
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
    )


@pytest.fixture()
def adapter():
    return _make_adapter()


class TestTelegramLoopPrevention:
    def test_self_authored_messages_are_dropped(self, adapter):
        msg = _message(text="hi", sender_id=999, is_bot=True)

        assert adapter._should_accept_bot_loop_message(msg) is False

    def test_bot_messages_are_deduplicated_and_rate_limited(self, adapter, monkeypatch):
        now = [100.0]
        monkeypatch.setattr("gateway.platforms.telegram.time.monotonic", lambda: now[0])

        first = _message(text="ping", sender_id=123, is_bot=True)
        duplicate = _message(text="ping", sender_id=123, is_bot=True)
        different_but_too_fast = _message(text="pong", sender_id=123, is_bot=True)
        later = _message(text="pong", sender_id=123, is_bot=True)

        assert adapter._should_accept_bot_loop_message(first) is True
        assert adapter._should_accept_bot_loop_message(duplicate) is False

        now[0] += 1.0
        assert adapter._should_accept_bot_loop_message(different_but_too_fast) is False

        now[0] += 5.0
        assert adapter._should_accept_bot_loop_message(later) is True

    def test_max_depth_blocks_runaway_bot_chains_and_human_messages_reset_scope(self, adapter, monkeypatch):
        now = [0.0]
        monkeypatch.setattr("gateway.platforms.telegram.time.monotonic", lambda: now[0])

        for i in range(6):
            msg = _message(text=f"bot turn {i}", sender_id=123, is_bot=True)
            assert adapter._should_accept_bot_loop_message(msg) is True
            now[0] += 6.0

        blocked = _message(text="bot turn 6", sender_id=123, is_bot=True)
        assert adapter._should_accept_bot_loop_message(blocked) is False

        human = _message(text="human interruption", sender_id=456, is_bot=False)
        assert adapter._should_accept_bot_loop_message(human) is True

        now[0] += 1.0
        reset_msg = _message(text="bot after reset", sender_id=123, is_bot=True)
        assert adapter._should_accept_bot_loop_message(reset_msg) is True
