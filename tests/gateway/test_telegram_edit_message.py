import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def _adapter_with_bot(edit_side_effect=None):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="tok"))
    bot = MagicMock()
    bot.edit_message_text = AsyncMock(side_effect=edit_side_effect)
    adapter._bot = bot
    return adapter, bot


# ---------------------------------------------------------------------------
# "Message is not modified" — inner except (markdown path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_message_not_modified_inner_returns_success():
    """BadRequest raised during markdown edit is treated as success."""
    err = Exception("Message is not modified: specified new message content and reply markup are exactly the same")
    adapter, bot = _adapter_with_bot(edit_side_effect=err)

    result = await adapter.edit_message("123", "456", "hello")

    assert result.success is True
    assert result.message_id == "456"
    # Only one call — no fallback attempt
    bot.edit_message_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_edit_message_not_modified_case_insensitive():
    """Check is case-insensitive ('Not Modified', 'NOT MODIFIED', etc.)."""
    for variant in ("Not Modified", "NOT MODIFIED", "message Not Modified"):
        err = Exception(variant)
        adapter, bot = _adapter_with_bot(edit_side_effect=err)
        result = await adapter.edit_message("1", "2", "x")
        assert result.success is True, f"Failed for variant: {variant!r}"


# ---------------------------------------------------------------------------
# "Message is not modified" — outer except (plain-text fallback path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_message_not_modified_on_plain_fallback_returns_success():
    """If the plain-text fallback raises 'not modified', outer handler returns success.

    Flow: markdown edit raises a non-"not modified" error (e.g. bad formatting),
    triggering the plain-text fallback, which in turn raises "not modified".
    The outer except catches it and treats it as success.
    """
    call_count = 0

    async def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call (markdown): raise a generic formatting error
            raise Exception("Can't parse entities: bad markdown")
        # Second call (plain-text fallback): raise "not modified"
        raise Exception("Message is not modified")

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="tok"))
    bot = MagicMock()
    bot.edit_message_text = AsyncMock(side_effect=side_effect)
    adapter._bot = bot

    result = await adapter.edit_message("1", "2", "hello")

    assert result.success is True
    assert call_count == 2  # markdown attempt + plain-text fallback


# ---------------------------------------------------------------------------
# Flood control / RetryAfter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_message_flood_control_retries_and_succeeds(monkeypatch):
    """RetryAfter error triggers a wait and plain-text retry."""
    monkeypatch.setattr("gateway.platforms.telegram.asyncio.sleep", AsyncMock())

    call_count = 0

    async def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            err = Exception("Flood control: retry after 3")
            err.retry_after = 3.0
            raise err
        # Second call (retry) succeeds

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="tok"))
    bot = MagicMock()
    bot.edit_message_text = AsyncMock(side_effect=side_effect)
    adapter._bot = bot

    result = await adapter.edit_message("1", "2", "hello")

    assert result.success is True
    assert call_count == 2


@pytest.mark.asyncio
async def test_edit_message_flood_control_retry_fails_returns_error(monkeypatch):
    """If the retry after flood wait also fails, return failure."""
    monkeypatch.setattr("gateway.platforms.telegram.asyncio.sleep", AsyncMock())

    call_count = 0

    async def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            err = Exception("Flood control: retry after 1")
            err.retry_after = 1.0
            raise err
        raise Exception("still failing")

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="tok"))
    bot = MagicMock()
    bot.edit_message_text = AsyncMock(side_effect=side_effect)
    adapter._bot = bot

    result = await adapter.edit_message("1", "2", "hello")

    assert result.success is False
    assert "still failing" in result.error


# ---------------------------------------------------------------------------
# Other errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_message_other_error_returns_failure():
    """Unrelated errors propagate as failure."""
    adapter, bot = _adapter_with_bot(edit_side_effect=Exception("Chat not found"))
    result = await adapter.edit_message("1", "2", "hello")
    assert result.success is False
    assert "Chat not found" in result.error


@pytest.mark.asyncio
async def test_edit_message_no_bot_returns_failure():
    """Returns failure immediately when not connected."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="tok"))
    # _bot is None by default
    result = await adapter.edit_message("1", "2", "hello")
    assert result.success is False
    assert "Not connected" in result.error


@pytest.mark.asyncio
async def test_edit_message_success():
    """Normal edit succeeds and returns the message_id."""
    adapter, bot = _adapter_with_bot(edit_side_effect=None)
    result = await adapter.edit_message("123", "789", "updated content")
    assert result.success is True
    assert result.message_id == "789"
