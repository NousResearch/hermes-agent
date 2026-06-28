"""
Tests for the Slack adapter's native Block Kit ``markdown`` block feature.

The Slack adapter wraps agent replies in Block Kit ``markdown`` blocks (added
to Block Kit in 2025) so tables, fenced code, and other CommonMark constructs
render natively in Slack instead of being force-converted to the proprietary
``mrkdwn`` mini-language. The legacy ``text=`` + ``mrkdwn=true`` path stays
available behind ``gateway.slack.markdown_blocks: false``.

Covers:
  * ``_markdown_blocks_enabled()`` config flag (default-on, opt-out)
  * ``_build_markdown_blocks()`` structure, 12k-char chunking, empty input
  * ``_strip_markdown_for_fallback()`` sigil stripping
  * ``send()`` payload shape in blocks vs. legacy mrkdwn mode
  * markdown table round-trip (the motivating use case for the feature)
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Reuse the slack-bolt import shim from the sibling test module so this file
# can be collected standalone without requiring the real slack-bolt package.
# ---------------------------------------------------------------------------


def _ensure_slack_mock() -> None:
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return  # Real library installed

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)

    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures (mirror the patterns in tests/gateway/test_slack.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point document cache at tmp_path so tests don't touch ~/.hermes."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )
    monkeypatch.setattr(
        "gateway.platforms.base.VIDEO_CACHE_DIR", tmp_path / "video_cache"
    )


# ---------------------------------------------------------------------------
# _markdown_blocks_enabled()
# ---------------------------------------------------------------------------


class TestMarkdownBlocksEnabled:
    def test_defaults_to_true(self, adapter):
        # No explicit config flag -> feature is on.
        assert "markdown_blocks" not in adapter.config.extra
        assert adapter._markdown_blocks_enabled() is True

    @pytest.mark.parametrize("value", [True, "true", "True", "1", "yes", "on"])
    def test_truthy_values_enable(self, adapter, value):
        adapter.config.extra["markdown_blocks"] = value
        assert adapter._markdown_blocks_enabled() is True

    @pytest.mark.parametrize("value", [False, "false", "0", "no", "off", ""])
    def test_falsy_values_disable(self, adapter, value):
        adapter.config.extra["markdown_blocks"] = value
        assert adapter._markdown_blocks_enabled() is False

    def test_bare_new_instance_defaults_to_true(self):
        # ``_standalone_send`` builds a SlackAdapter via __new__ with no config
        # attribute — the enabled-check must not crash and must default on.
        bare = SlackAdapter.__new__(SlackAdapter)
        assert not hasattr(bare, "config")
        assert bare._markdown_blocks_enabled() is True


# ---------------------------------------------------------------------------
# _build_markdown_blocks()
# ---------------------------------------------------------------------------


class TestBuildMarkdownBlocks:
    def test_wraps_markdown_in_block_structure(self, adapter):
        blocks = adapter._build_markdown_blocks("hello world")
        assert blocks == [{"type": "markdown", "text": "hello world"}]

    def test_empty_content_returns_empty_list(self, adapter):
        assert adapter._build_markdown_blocks("") == []
        assert adapter._build_markdown_blocks("   \n  \t ") == []
        assert adapter._build_markdown_blocks(None) == []

    def test_chunks_long_content_at_12000_char_limit(self, adapter):
        # Content that fits within a single 12k block stays a single block.
        short = "a" * 11_999
        assert len(adapter._build_markdown_blocks(short)) == 1

        # Content just over the limit gets split into multiple blocks, proving
        # the chunk boundary is _MARKDOWN_BLOCK_CHAR_LIMIT (12000) and NOT the
        # default MAX_MESSAGE_LENGTH (39000).
        long_content = "a" * 12_001
        blocks = adapter._build_markdown_blocks(long_content)
        assert len(blocks) > 1, "expected >1 block for content over the 12k limit"
        # Every emitted block is a properly typed markdown block.
        for block in blocks:
            assert block["type"] == "markdown"
            assert isinstance(block["text"], str) and block["text"]

    def test_emits_at_most_50_blocks(self, adapter):
        # Construct content large enough to produce (without the cap) far more
        # than 50 chunks at the 12k limit. ~3,000,000 chars would naively
        # produce ~250 chunks; the cap must clamp it to 50 blocks and append a
        # truncation marker.
        huge = ("line of stuff\n" * 300_000)  # ~4.2M chars -> many chunks
        blocks = adapter._build_markdown_blocks(huge)
        assert len(blocks) <= SlackAdapter._MARKDOWN_BLOCK_MAX_BLOCKS
        assert len(blocks) == SlackAdapter._MARKDOWN_BLOCK_MAX_BLOCKS
        # The final block carries a truncation marker the user can see.
        assert "truncated" in blocks[-1]["text"]


# ---------------------------------------------------------------------------
# _strip_markdown_for_fallback()
# ---------------------------------------------------------------------------


class TestStripMarkdownForFallback:
    def test_strips_bold_italic_strikethrough(self):
        out = SlackAdapter._strip_markdown_for_fallback(
            "**bold** and _italic_ and ~~strike~~"
        )
        assert out == "bold and italic and strike"

    def test_strips_headers_and_lists_and_quotes(self):
        out = SlackAdapter._strip_markdown_for_fallback(
            "# Title\n- item one\n> quoted text\nplain"
        )
        assert "Title" in out
        assert "item one" in out
        assert "quoted text" in out
        assert "plain" in out
        # No markdown sigils survive for the constructs we strip.
        assert "#" not in out
        assert ">" not in out

    def test_strips_links_to_label(self):
        out = SlackAdapter._strip_markdown_for_fallback(
            "see [the docs](https://example.com/docs) for more"
        )
        assert out == "see the docs for more"
        assert "https" not in out

    def test_strips_inline_code_backticks(self):
        out = SlackAdapter._strip_markdown_for_fallback("use `fmt.Println` here")
        assert out == "use fmt.Println here"
        assert "`" not in out

    def test_replaces_fenced_code_block_with_marker(self):
        out = SlackAdapter._strip_markdown_for_fallback(
            "before\n```python\nprint('hi')\n```\nafter"
        )
        assert "[code block]" in out
        assert "print('hi')" not in out
        assert "before" in out and "after" in out

    def test_empty_input_returns_empty(self):
        assert SlackAdapter._strip_markdown_for_fallback("") == ""
        assert SlackAdapter._strip_markdown_for_fallback(None) == ""

    def test_bounded_to_3000_chars(self):
        out = SlackAdapter._strip_markdown_for_fallback("x" * 50_000)
        assert len(out) <= 3000


# ---------------------------------------------------------------------------
# send() payload shape (blocks mode vs. legacy mrkdwn mode)
# ---------------------------------------------------------------------------


class TestSendPayloadModes:
    @pytest.mark.asyncio
    async def test_send_blocks_mode_calls_postmessage_with_blocks_no_mrkdwn(
        self, adapter
    ):
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "123.000"}
        )

        result = await adapter.send("C123", "# Title\n\nbody **bold**")

        assert result.success
        assert adapter._app.client.chat_postMessage.await_count == 1
        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert call_kwargs["channel"] == "C123"
        # The whole point: a single markdown block carries the raw markdown,
        # un-escaped, so tables/code/etc. render natively.
        assert call_kwargs["blocks"] == [
            {"type": "markdown", "text": "# Title\n\nbody **bold**"}
        ]
        # ``text=`` is the stripped plain-text fallback, not the mrkdwn form.
        assert call_kwargs["text"] == "Title\n\nbody bold"
        # ``mrkdwn`` must NOT be passed in blocks mode.
        assert "mrkdwn" not in call_kwargs

    @pytest.mark.asyncio
    async def test_send_legacy_mode_still_uses_mrkdwn(self, adapter):
        adapter.config.extra["markdown_blocks"] = False
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "123.000"}
        )

        result = await adapter.send("C123", "**bold** plain")

        assert result.success
        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert call_kwargs["channel"] == "C123"
        # Legacy path: format_message converts ** -> * (Slack bold).
        assert call_kwargs["text"] == "*bold* plain"
        assert call_kwargs["mrkdwn"] is True
        assert "blocks" not in call_kwargs

    @pytest.mark.asyncio
    async def test_send_blocks_mode_no_mrkdwn_kwarg_whatsoever(self, adapter):
        # Belt-and-suspenders: even when a thread ts is involved, blocks mode
        # must not add ``mrkdwn``.
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "123.000"}
        )

        await adapter.send(
            "C123",
            "hello",
            metadata={"thread_id": "111.222"},
        )

        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert call_kwargs.get("thread_ts") == "111.222"
        assert "mrkdwn" not in call_kwargs
        assert "blocks" in call_kwargs


# ---------------------------------------------------------------------------
# Round-trip: a markdown table survives unchanged as a single markdown block
# (the motivating use case for the feature — mrkdwn cannot render tables).
# ---------------------------------------------------------------------------


class TestMarkdownTableRoundTrip:
    TABLE = (
        "| Command | Effect |\n"
        "|---------|--------|\n"
        "| `/q`    | quit   |\n"
        "| `/stop` | halt   |"
    )

    @pytest.mark.asyncio
    async def test_table_becomes_single_preserved_markdown_block(self, adapter):
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "123.000"}
        )

        await adapter.send("C123", self.TABLE)

        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        blocks = call_kwargs["blocks"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "markdown"
        # The table syntax is passed through verbatim — Slack's CommonMark
        # renderer, not format_message(), handles it.
        assert blocks[0]["text"] == self.TABLE
        assert "mrkdwn" not in call_kwargs

    def test_table_block_built_directly_preserved(self, adapter):
        blocks = adapter._build_markdown_blocks(self.TABLE)
        assert len(blocks) == 1
        assert blocks[0] == {"type": "markdown", "text": self.TABLE}
