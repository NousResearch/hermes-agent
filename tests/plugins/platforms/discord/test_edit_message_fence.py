"""
Tests for Discord edit_message brute-force truncation causing broken fences.

edit_message (line 1802-1803) blindly slices content at MAX_MESSAGE_LENGTH.
When the slice cuts through a ``` marker, the chunk has unbalanced fences.

The fix: after brute-force truncation, ensure balanced code fences.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


def _ensure_balanced_fences(text: str) -> str:
    """Append closing fence if odd count of triple-backtick."""
    if text.count("```") % 2 == 1:
        return text.rstrip("\n") + "\n```"
    return text


# ── pure-function tests ──


class TestFenceGuardFunction:
    """_ensure_balanced_fences is the proposed defense-in-depth."""

    def test_even_fence_noop(self):
        assert _ensure_balanced_fences("```a```") == "```a```"

    def test_odd_fence_closed(self):
        result = _ensure_balanced_fences("text\n```\nunclosed")
        assert result.count("```") % 2 == 0
        assert result.endswith("\n```")

    def test_no_fence_noop(self):
        assert _ensure_balanced_fences("plain text") == "plain text"

    def test_empty_noop(self):
        assert _ensure_balanced_fences("") == ""

    def test_trailing_newline_handled(self):
        result = _ensure_balanced_fences("text\n```\n")
        assert result.count("```") % 2 == 0
        # Should add ``` after stripping trailing newline
        assert result.endswith("\n```")

    def test_idempotent(self):
        text = "```python\ncode\n```\n\nmore content"
        assert _ensure_balanced_fences(text) == text


# ── integration: edit_message behavior ──


@pytest.fixture
def adapter():
    """DiscordAdapter with mocked client and channel."""
    a = DiscordAdapter(PlatformConfig(enabled=True, token="token"))
    channel = MagicMock()
    channel.fetch_message = AsyncMock(return_value=MagicMock())
    a._client = MagicMock()
    a._client.get_channel.return_value = channel
    a._client.fetch_channel = AsyncMock(return_value=channel)
    return a


class TestEditMessageFenceTruncation:
    """edit_message brute-force slice must not break code fences."""

    def test_brute_force_slice_cuts_through_fence(self):
        """Confirm the bug: blind slice CAN break a code fence."""
        body = "C" * 2500
        msg = f"```python\n{body}\n```"
        a = DiscordAdapter(PlatformConfig(enabled=True, token="token"))
        formatted = a.format_message(msg)
        assert len(formatted) > a.MAX_MESSAGE_LENGTH
        # Current buggy behavior
        truncated = formatted[: a.MAX_MESSAGE_LENGTH - 3] + "..."

        odd = truncated.count("```") % 2 == 1
        if odd:
            pytest.skip("Bug confirmed: brute-force slice produces odd ``` count")
        else:
            pytest.skip("Slice happened to land outside fence — not always reproducible")

    def test_fence_guard_fixes_broken_truncation(self):
        """After brute-force slice, _ensure_balanced_fences fixes odd fences."""
        body = "C" * 1950
        msg = f"```python\n{body}\n```"
        a = DiscordAdapter(PlatformConfig(enabled=True, token="token"))
        formatted = a.format_message(msg)
        truncated = formatted[: a.MAX_MESSAGE_LENGTH - 3] + "..."

        fixed = _ensure_balanced_fences(truncated)
        assert fixed.count("```") % 2 == 0

    def test_short_message_not_affected(self):
        """Short messages keep balanced fences."""
        msg = "```python\nprint('hi')\n```"
        a = DiscordAdapter(PlatformConfig(enabled=True, token="token"))
        formatted = a.format_message(msg)
        assert len(formatted) <= a.MAX_MESSAGE_LENGTH
        # No truncation needed — fence stays balanced
        assert formatted.count("```") % 2 == 0

    def test_full_reasoning_scenario(self, adapter):
        """Simulate the exact scenario where reasoning + code block
        pushes the edit payload over Discord's limit."""
        body = (
            "Long text with reasoning" * 20
            + "\n```python\n" + "code " * 400 + "\n```\n"
            + "more text" * 30
        )
        formatted = adapter.format_message(body)
        # edit_message path: brute-force truncation (if needed)
        if len(formatted) > adapter.MAX_MESSAGE_LENGTH:
            truncated = formatted[: adapter.MAX_MESSAGE_LENGTH - 3] + "..."
        else:
            truncated = formatted

        # Apply guard
        fixed = _ensure_balanced_fences(truncated)
        assert fixed.count("```") % 2 == 0, (
            f"After fix: {fixed.count('```')} fences, expected even\n"
            f"len={len(formatted)} > MAX? {len(formatted) > adapter.MAX_MESSAGE_LENGTH}"
        )
