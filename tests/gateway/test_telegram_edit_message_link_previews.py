"""
Regression tests for issue #61018 - Telegram edit_message paths must
honor platforms.telegram.extra.disable_link_previews.

The bug: `edit_message` (streaming finalize=False) and the legacy
MarkdownV2 fallbacks in `edit_message` do not pass `**self._link_preview_kwargs()`
when calling `self._bot.edit_message_text`. When the response is delivered
through the edit path (streaming updates, final fallback), Telegram sees
content with URLs and auto-generates a link preview because the suppression
kwarg was never set.

`sender.send()` already does it correctly (line ~3678); edit_message
needs the same treatment.

These tests exercise the three edit_message_text call sites in
`edit_message` and assert that link_preview kwargs are propagated when
`disable_link_previews=True`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_adapter(extra=None) -> TelegramAdapter:
    """Build a TelegramAdapter with the bot replaced by an AsyncMock.

    `_bot` is set so the `if not self._bot` early-return is bypassed.
    `_try_edit_rich` is monkey-patched to return None so the rich path
    is skipped and we drive straight to the legacy edit_message_text calls.
    """
    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra=extra or {},
    )
    a = TelegramAdapter(config)
    a._bot = MagicMock()
    a._bot.edit_message_text = AsyncMock(
        return_value=SimpleNamespace(message_id=42)
    )
    a._try_edit_rich = AsyncMock(return_value=None)
    a.handle_message = AsyncMock()
    return a


@pytest.fixture
def adapter_disabled() -> TelegramAdapter:
    return _make_adapter(extra={"disable_link_previews": "true"})


@pytest.fixture
def adapter_enabled() -> TelegramAdapter:
    return _make_adapter(extra={})


def _last_call_kwargs(bot: MagicMock) -> dict:
    assert bot.edit_message_text.await_count >= 1, (
        "expected at least one edit_message_text call; none recorded"
    )
    return bot.edit_message_text.await_args_list[-1].kwargs


def _has_link_preview_suppression(kwargs: dict) -> bool:
    """link_preview_kwargs() emits ONE of two keys depending on the
    python-telegram-bot version. Either is acceptable; what matters is
    that SOME suppression kwarg reaches Telegram.
    """
    return ("disable_web_page_preview" in kwargs) or (
        "link_preview_options" in kwargs
    )


# ---------------------------------------------------------------------------
# Tests for the bug surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_edit_finalize_false_carries_link_preview_kwarg(
    adapter_disabled: TelegramAdapter,
):
    """edit_message(finalize=False) is the streaming-edit branch (lines ~3982
    in adapter.py). On unfixed code the suppress kwarg is missing.
    """
    # Short content so _edit_overflow_split doesn't fire. _try_edit_rich is
    # patched to None to skip the rich finalize path.
    result = await adapter_disabled.edit_message(
        chat_id="123", message_id="42", content="short update", finalize=False,
    )

    assert result.success is True
    kwargs = _last_call_kwargs(adapter_disabled._bot)
    assert _has_link_preview_suppression(kwargs), (
        f"streaming edit omits link-preview kwargs; edit_message_text called "
        f"with kwargs={kwargs!r}. Issue #61018: link-preview appears in "
        f"the streaming-edit message even when disable_link_previews=True."
    )


@pytest.mark.asyncio
async def test_finalize_edit_finalize_true_carries_link_preview_kwarg(
    adapter_disabled: TelegramAdapter,
):
    """The finalize branch (lines ~3993 in adapter.py) also lacks the kwarg
    on unfixed code."""
    result = await adapter_disabled.edit_message(
        chat_id="123", message_id="42", content="final content", finalize=True,
    )

    assert result.success is True
    kwargs = _last_call_kwargs(adapter_disabled._bot)
    assert _has_link_preview_suppression(kwargs), (
        f"finalize edit omits link-preview kwargs; kwargs={kwargs!r}"
    )


@pytest.mark.asyncio
async def test_plain_text_fallback_edit_carries_link_preview_kwarg(
    adapter_disabled: TelegramAdapter,
):
    """When MarkdownV2 parsing fails the path falls through to plain text
    edit_message_text (lines ~4011 in adapter.py). Same kwarg omission here
    too on unfixed code.
    """
    from telegram.error import BadRequest
    err = BadRequest(
        "[test] can't parse entities: unsupported markdown tag"
    )
    # First edit_message_text (the formatted try-block) raises the
    # markdown error. Second call (the plain fallback) succeeds.
    adapter_disabled._bot.edit_message_text = AsyncMock(
        side_effect=[err, SimpleNamespace(message_id=42)]
    )

    result = await adapter_disabled.edit_message(
        chat_id="123", message_id="42", content="<bad markdown>",
        finalize=True,
    )

    assert result.success is True
    # Both edit_message_text calls (the formatted try and the plain fallback)
    # must carry the suppress kwarg.
    assert adapter_disabled._bot.edit_message_text.await_count == 2, (
        f"expected 2 edit_message_text calls (formatted + plain fallback); "
        f"got {adapter_disabled._bot.edit_message_text.await_count}"
    )
    for idx, call in enumerate(
        adapter_disabled._bot.edit_message_text.await_args_list
    ):
        assert _has_link_preview_suppression(call.kwargs), (
            f"plain-text fallback call #{idx} omits link-preview kwargs; "
            f"kwargs={call.kwargs!r}"
        )


@pytest.mark.asyncio
async def test_edit_when_disable_link_previews_disabled_emits_no_suppression(
    adapter_enabled: TelegramAdapter,
):
    """Regression guard: when disable_link_previews is NOT configured,
    the kwargs helper returns {} and Telegram must NOT receive a suppression
    flag (it would be a no-op but we shouldn't pollute the request body)."""
    result = await adapter_enabled.edit_message(
        chat_id="123", message_id="42", content="fine", finalize=False,
    )
    assert result.success is True
    kwargs = _last_call_kwargs(adapter_enabled._bot)
    assert "disable_web_page_preview" not in kwargs, (
        f"emitted disable_web_page_preview despite disable_link_previews=False; "
        f"kwargs={kwargs!r}"
    )
    assert "link_preview_options" not in kwargs
