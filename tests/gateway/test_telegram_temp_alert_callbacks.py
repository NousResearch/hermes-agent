"""Tests for Telegram temperature alert callback handlers.

Verifies:
- Auth gate rejects unauthorized users before processing
- temp:sleep is accepted for authorized users
- temp:avoid is accepted for authorized users
- Unknown temp: subcommands return error answer
"""
import os
import subprocess  # noqa: F811 — pre-import so patch.object works for local imports in handler
import sys
import time  # noqa: F811 — pre-import so patch.object works for local imports in handler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
    """Wire up the minimal mocks required to import TelegramAdapter."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.config import Platform, PlatformConfig


def _make_adapter(extra=None, callback_auth=None):
    """Create a TelegramAdapter with mocked internals suitable for callback tests."""
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    if callback_auth is not None:
        adapter._is_callback_user_authorized = callback_auth
    return adapter


def _make_callback_query(data="temp:sleep", *, from_user_id=111, chat_id=-100, message_id=42):
    """Build a minimal callback query SimpleNamespace."""
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=from_user_id, first_name="Test"),
        message=SimpleNamespace(
            message_id=message_id,
            chat_id=chat_id,
            chat=SimpleNamespace(id=chat_id, type="group"),
            message_thread_id=None,
            is_topic_message=False,
        ),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )


class _AuthRunner:
    """Minimal runner shim for testing _is_callback_user_authorized delegation."""

    def __init__(self, authorized: bool):
        self.authorized = authorized

    async def _handle_message(self, event):
        return None

    def _is_user_authorized(self, source):
        return self.authorized


# ===========================================================================
# Auth gate tests
# ===========================================================================


@pytest.mark.asyncio
async def test_unauthorized_user_rejected():
    """An unauthorized user clicking a temp: callback should be rejected."""
    adapter = _make_adapter()
    # Wire a runner that rejects everyone
    adapter._message_handler = _AuthRunner(authorized=False)._handle_message

    query = _make_callback_query(data="temp:sleep", from_user_id=999)
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, SimpleNamespace())

    # Should have answered with auth error, NOT processed
    query.answer.assert_awaited_once()
    answer_text = query.answer.call_args[1].get("text", "")
    assert "not authorized" in answer_text.lower()
    # edit_message_text should NOT have been called — we never got past auth
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_user_avoid_rejected():
    """An unauthorized user clicking temp:avoid should be rejected."""
    adapter = _make_adapter()
    adapter._message_handler = _AuthRunner(authorized=False)._handle_message

    query = _make_callback_query(data="temp:avoid", from_user_id=999)
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, SimpleNamespace())

    query.answer.assert_awaited_once()
    answer_text = query.answer.call_args[1].get("text", "")
    assert "not authorized" in answer_text.lower()
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_user_gets_blocked_before_subprocess():
    """Ensure unauthorized users never reach the subprocess/spawn path."""
    adapter = _make_adapter()
    adapter._message_handler = _AuthRunner(authorized=False)._handle_message

    query = _make_callback_query(data="temp:sleep", from_user_id=999)
    update = SimpleNamespace(callback_query=query)

    with patch.object(subprocess, "Popen") as mock_popen:
        await adapter._handle_callback_query(update, SimpleNamespace())

        # subprocess.Popen should never be called for unauthorized user
        mock_popen.assert_not_called()


# ===========================================================================
# Success path tests
# ===========================================================================


@pytest.mark.asyncio
async def test_temp_sleep_accepted():
    """Authorized user clicking temp:sleep should execute the sleep command."""
    adapter = _make_adapter()
    adapter._message_handler = _AuthRunner(authorized=True)._handle_message

    query = _make_callback_query(data="temp:sleep", from_user_id=111)
    update = SimpleNamespace(callback_query=query)

    with patch.object(subprocess, "Popen") as mock_popen:
        await adapter._handle_callback_query(update, SimpleNamespace())

        # Should answer with sleep message
        query.answer.assert_awaited_once()
        answer_text = query.answer.call_args[1].get("text", "")
        assert "sleep" in answer_text.lower()

        # Should edit the message
        query.edit_message_text.assert_awaited_once()
        edit_text = query.edit_message_text.call_args[1].get("text", "")
        assert "sleeping" in edit_text.lower()

        # Should spawn rtcwake
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert "rtcwake" in args
        assert "-s" in args
        assert "900" in args  # 15 min = 900 seconds


@pytest.mark.asyncio
async def test_temp_avoid_accepted():
    """Authorized user clicking temp:avoid should snooze alerts for 1 hour."""
    adapter = _make_adapter()
    adapter._message_handler = _AuthRunner(authorized=True)._handle_message

    query = _make_callback_query(data="temp:avoid", from_user_id=111)
    update = SimpleNamespace(callback_query=query)

    with patch("builtins.open", MagicMock()) as mock_open, \
         patch.object(time, "time") as mock_time:
        mock_time.return_value = 1000  # Fixed "now"

        fake_file = MagicMock()
        mock_open.return_value.__enter__.return_value = fake_file

        await adapter._handle_callback_query(update, SimpleNamespace())

        # Should answer with snooze message
        query.answer.assert_awaited_once()
        answer_text = query.answer.call_args[1].get("text", "")
        assert "alert" in answer_text.lower() or "avoid" in answer_text.lower()

        # Should edit the message
        query.edit_message_text.assert_awaited_once()
        edit_text = query.edit_message_text.call_args[1].get("text", "")
        assert "snoozed" in edit_text.lower() or "alert" in edit_text.lower()

        # Should write the avoid-until file
        mock_open.assert_called_once_with("/tmp/temp_avoid_until", "w")
        fake_file.write.assert_called_once()
        written_value = fake_file.write.call_args[0][0]
        assert str(1000 + 3600) == written_value  # now + 1 hour


@pytest.mark.asyncio
async def test_temp_sleep_edit_failure_nonfatal():
    """If edit_message_text fails for temp:sleep, the rtcwake still runs."""
    adapter = _make_adapter()
    adapter._message_handler = _AuthRunner(authorized=True)._handle_message

    query = _make_callback_query(data="temp:sleep", from_user_id=111)
    query.edit_message_text = AsyncMock(
        side_effect=Exception("edit failed")
    )
    update = SimpleNamespace(callback_query=query)

    with patch.object(subprocess, "Popen") as mock_popen:
        await adapter._handle_callback_query(update, SimpleNamespace())

        # answer should still happen
        query.answer.assert_awaited_once()
        # rtcwake should still run despite edit failure
        mock_popen.assert_called_once()


# ===========================================================================
# Unknown temp command
# ===========================================================================


@pytest.mark.asyncio
async def test_unknown_temp_command_returns_error():
    """An unknown temp:X subcommand should return an error answer."""
    adapter = _make_adapter()
    adapter._message_handler = _AuthRunner(authorized=True)._handle_message

    query = _make_callback_query(data="temp:unknown_verb", from_user_id=111)
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, SimpleNamespace())

    query.answer.assert_awaited_once()
    answer_text = query.answer.call_args[1].get("text", "")
    assert "unknown" in answer_text.lower()
