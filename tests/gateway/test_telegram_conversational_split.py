"""Tests for Telegram opt-in conversational blank-line splitting (#4445).

Ports the WhatsApp ``split_outgoing_*`` pattern (PR #46314) to Telegram:
replies are optionally split on blank lines outside fenced code blocks and
delivered as separate bubbles, before the existing 4096-char
``truncate_message()`` chunking runs on every part.
"""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    """Mock the telegram package if it's not installed."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _make_adapter(**extra):
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="test-token", extra=extra)
    )
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    return adapter


def _sent_texts(adapter):
    return [c.kwargs["text"] for c in adapter._bot.send_message.call_args_list]


class TestSplitConfig:
    """split_outgoing_* extra keys are opt-in with safe fallbacks."""

    def test_defaults_are_opt_out_with_documented_values(self):
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t"))

        assert adapter._split_outgoing_on_blank_lines is False
        assert adapter._split_outgoing_delay_seconds == 0.6
        assert adapter._split_outgoing_max_parts == 4

    def test_values_are_config_backed_and_invalid_values_fall_back_safely(self):
        configured = TelegramAdapter(
            PlatformConfig(
                enabled=True,
                token="t",
                extra={
                    "split_outgoing_on_blank_lines": "yes",
                    "split_outgoing_delay_seconds": "1.25",
                    "split_outgoing_max_parts": "7",
                },
            )
        )
        invalid = TelegramAdapter(
            PlatformConfig(
                enabled=True,
                token="t",
                extra={
                    "split_outgoing_on_blank_lines": "sometimes",
                    "split_outgoing_delay_seconds": "not-a-number",
                    "split_outgoing_max_parts": -2,
                },
            )
        )
        malformed_max = TelegramAdapter(
            PlatformConfig(
                enabled=True,
                token="t",
                extra={"split_outgoing_max_parts": "many"},
            )
        )

        assert configured._split_outgoing_on_blank_lines is True
        assert configured._split_outgoing_delay_seconds == 1.25
        assert configured._split_outgoing_max_parts == 7
        assert invalid._split_outgoing_on_blank_lines is False
        assert invalid._split_outgoing_delay_seconds == 0.6
        assert invalid._split_outgoing_max_parts == 4
        assert malformed_max._split_outgoing_max_parts == 4


class TestSendConversationalSplit:
    """send() bubble splitting behavior."""

    @pytest.mark.asyncio
    async def test_blank_line_split_is_opt_in(self):
        adapter = _make_adapter()

        await adapter.send("12345", "one\n\ntwo")

        assert _sent_texts(adapter) == ["one\n\ntwo"]

    @pytest.mark.asyncio
    async def test_blank_lines_split_bubbles_but_single_newlines_do_not(self):
        adapter = _make_adapter(split_outgoing_on_blank_lines=True)

        await adapter.send("12345", "one\n\ntwo\nthree")

        assert _sent_texts(adapter) == ["one", "two\nthree"]

    @pytest.mark.asyncio
    async def test_blank_line_split_merges_remainder_at_max_parts(self):
        adapter = _make_adapter(
            split_outgoing_on_blank_lines=True,
            split_outgoing_max_parts=3,
        )

        await adapter.send("12345", "one\n\ntwo\n\nthree\n\nfour")

        assert _sent_texts(adapter) == ["one", "two", "three\n\nfour"]

    @pytest.mark.asyncio
    async def test_blank_lines_inside_fenced_code_stay_in_one_bubble(self):
        adapter = _make_adapter(split_outgoing_on_blank_lines=True)
        content = "intro\n\n```python\nfirst\n\nsecond\n```\n\noutro"

        await adapter.send("12345", content)

        assert _sent_texts(adapter) == [
            "intro",
            "```python\nfirst\n\nsecond\n```",
            "outro",
        ]

    @pytest.mark.asyncio
    async def test_reply_to_only_threads_on_first_bubble(self):
        adapter = _make_adapter(split_outgoing_on_blank_lines=True)

        await adapter.send("12345", "one\n\ntwo\n\nthree", reply_to="999")

        calls = adapter._bot.send_message.call_args_list
        assert len(calls) == 3
        assert calls[0].kwargs.get("reply_to_message_id") == 999
        assert calls[1].kwargs.get("reply_to_message_id") is None
        assert calls[2].kwargs.get("reply_to_message_id") is None

    @pytest.mark.asyncio
    async def test_configured_split_delay_is_used_when_opted_in(self, monkeypatch):
        adapter = _make_adapter(
            split_outgoing_on_blank_lines=True,
            split_outgoing_delay_seconds=1.25,
        )
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)

        await adapter.send("12345", "one\n\ntwo", metadata={"notify": True})

        assert len(_sent_texts(adapter)) == 2
        assert [c.args[0] for c in sleep.await_args_list] == [1.25]

    @pytest.mark.asyncio
    async def test_legacy_multi_chunk_path_stays_delay_free_when_opted_out(
        self, monkeypatch
    ):
        adapter = _make_adapter(split_outgoing_delay_seconds=1.25)
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)

        await adapter.send("12345", "a " * 3000, metadata={"notify": True})

        assert len(_sent_texts(adapter)) > 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_length_chunking_still_applies_to_each_bubble(self):
        adapter = _make_adapter(split_outgoing_on_blank_lines=True)

        await adapter.send("12345", "intro\n\n" + "a " * 3000)

        texts = _sent_texts(adapter)
        assert texts[0] == "intro"
        assert len(texts) == 3
        assert texts[1].endswith(" \\(1/2\\)")
        assert texts[2].endswith(" \\(2/2\\)")
