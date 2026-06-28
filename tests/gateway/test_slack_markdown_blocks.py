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


# ---------------------------------------------------------------------------
# Hardening tests — address tw0316 review on PR #53893 (items 1-4 + 6).
# Each test is a vertical tracer bullet: one behavior, real code path, fails
# today (RED), passes once the fix lands (GREEN).
# ---------------------------------------------------------------------------


class TestYamlConfigDisableBridgesToExtra:
    """Item 1: documented config opt-out must actually disable the feature.

    `gateway.slack.markdown_blocks: false` in config.yaml is the documented
    user-facing switch, but `_apply_yaml_config()` returns None and never
    bridges the key into `PlatformConfig.extra`. Result: the flag is silently
    dropped, `_markdown_blocks_enabled()` still defaults to True, and the
    documented opt-out doesn't disable anything.
    """

    def test_yaml_markdown_blocks_false_disables_feature(self, tmp_path, monkeypatch):
        import os
        from gateway.config import load_gateway_config

        # Write a real config.yaml with the documented opt-out path.
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "slack:\n"
            "  enabled: true\n"
            "  markdown_blocks: false\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Clear any cached env so the fresh config takes effect.
        for k in list(os.environ):
            if k.startswith("SLACK_"):
                monkeypatch.delenv(k, raising=False)

        cfg = load_gateway_config()

        # Find the Slack platform config (keyed by Platform enum in GatewayConfig).
        slack_cfg = None
        platforms = getattr(cfg, "platforms", None) or {}
        for key, value in platforms.items():
            if str(key).endswith("slack") or getattr(key, "value", None) == "slack":
                slack_cfg = value
                break
        assert slack_cfg is not None, "Slack platform config not loaded"

        # The documented user intent is: markdown_blocks is OFF.
        # Today extra == {} (dropped), so _markdown_blocks_enabled() returns True.
        # After the bridge fix, extra will carry the falsy value and the
        # SlackAdapter will report disabled.
        from plugins.platforms.slack.adapter import SlackAdapter
        adapter = SlackAdapter(slack_cfg)
        assert adapter._markdown_blocks_enabled() is False, (
            "slack.markdown_blocks: false in config.yaml must disable the "
            "feature — _apply_yaml_config must bridge it into extra"
        )


class TestCumulativeMarkdownBlockLimit:
    """Item 2: per-payload cumulative markdown-block text must stay <= 12,000.

    Slack's Block Kit spec: "The cumulative limit for all markdown blocks in
    a single payload is 12,000 characters." The current code chunks only by
    block count (max 50 blocks/msg), so a 12,001-char message produces one
    `chat_postMessage` call with two blocks totaling >12k -> Slack rejects
    with `markdown_text_too_long`. The fix: chunk by cumulative text length
    across the whole payload (or one block per message).
    """

    @pytest.mark.asyncio
    async def test_send_over_12k_chars_emits_no_payload_over_limit(self, adapter):
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "123.000"}
        )

        # 12,001 chars — one char over the cumulative payload limit.
        big = "a" * 12_001
        await adapter.send("C123", big)

        # Inspect every call: no single chat_postMessage payload may carry
        # more than 12,000 chars of cumulative markdown-block text.
        for call in adapter._app.client.chat_postMessage.call_args_list:
            kwargs = call.kwargs
            blocks = kwargs.get("blocks") or []
            cumulative = sum(len(b.get("text", "")) for b in blocks if b.get("type") == "markdown")
            assert cumulative <= SlackAdapter._MARKDOWN_BLOCK_CHAR_LIMIT, (
                f"chat_postMessage payload carries {cumulative} chars of "
                f"markdown-block text; Slack rejects payloads over "
                f"{SlackAdapter._MARKDOWN_BLOCK_CHAR_LIMIT}. "
                f"send() must split across multiple messages, not one."
            )

    @pytest.mark.asyncio
    async def test_send_over_12k_chars_produces_multiple_postmessage_calls(self, adapter):
        # Belt-and-suspenders: 12,001 chars must produce >1 API call, proving
        # we did NOT pile multiple sub-12k blocks into one over-limit payload.
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "123.000"}
        )

        await adapter.send("C123", "a" * 12_001)

        assert adapter._app.client.chat_postMessage.await_count >= 2, (
            "12,001-char content must be split across multiple "
            "chat_postMessage calls, not one cumulative-over-limit payload"
        )


class TestLiveFallbackOnSlackRejection:
    """Item 3: when Slack rejects markdown blocks, retry with legacy mrkdwn.

    Today send() catches any exception from chat_postMessage and returns
    SendResult(success=False) with no retry. Slack errors like
    `invalid_blocks`, `markdown_text_too_long`, `msg_too_long` are
    recoverable by retrying the legacy `text=` + `mrkdwn=true` path. The
    fix: a helper that recognizes markdown-block rejection errors and
    retries once with legacy formatting.
    """

    @pytest.mark.asyncio
    async def test_send_retries_legacy_when_slack_rejects_blocks(self, adapter):
        # First call: Slack rejects the markdown-block payload with
        # `invalid_blocks` (a real Slack error for malformed blocks).
        # Second call: legacy retry succeeds.
        adapter._app.client.chat_postMessage = AsyncMock(
            side_effect=[
                SlackRejectedBlocks("invalid_blocks"),
                {"ts": "123.000"},
            ]
        )

        result = await adapter.send("C123", "# Hi\n\nbody")

        assert result.success, "send must retry legacy on Slack block rejection"
        assert adapter._app.client.chat_postMessage.await_count == 2, (
            "expected exactly two chat_postMessage calls: the rejected "
            "blocks call, then the legacy retry"
        )
        # The retry call must use the legacy mrkdwn path, not blocks.
        retry_kwargs = adapter._app.client.chat_postMessage.call_args_list[1].kwargs
        assert "blocks" not in retry_kwargs, (
            "retry must fall back to plain text, not re-send blocks"
        )
        assert retry_kwargs.get("mrkdwn") is True, (
            "retry must re-enable Slack mrkdwn rendering on the text-only payload"
        )

    @pytest.mark.asyncio
    async def test_send_does_not_retry_on_unrelated_errors(self, adapter):
        # A non-markdown-block error (e.g. network failure, auth error)
        # must NOT trigger the legacy fallback — only Slack's block-rejection
        # family of errors should. Other errors surface as SendResult(success=False).
        adapter._app.client.chat_postMessage = AsyncMock(
            side_effect=ConnectionError("network down")
        )

        result = await adapter.send("C123", "hi")

        assert not result.success
        assert adapter._app.client.chat_postMessage.await_count == 1, (
            "non-block errors must not trigger a legacy retry"
        )

    @pytest.mark.asyncio
    async def test_edit_message_retries_legacy_and_clears_stale_blocks(self, adapter):
        # edit_message has an extra concern: when falling back from blocks
        # to plain text on chat.update, Slack can retain the old blocks
        # unless we explicitly pass blocks=[] to clear them.
        adapter._app.client.chat_update = AsyncMock(
            side_effect=[
                SlackRejectedBlocks("invalid_blocks"),
                {"ok": True},
            ]
        )

        await adapter.edit_message("C123", "123.000", "# Hi\n\nbody")

        assert adapter._app.client.chat_update.await_count == 2
        retry_kwargs = adapter._app.client.chat_update.call_args_list[1].kwargs
        assert retry_kwargs.get("blocks") == [], (
            "edit retry must pass blocks=[] to clear stale blocks when "
            "falling back to plain text (Slack otherwise retains old blocks)"
        )
        assert retry_kwargs.get("mrkdwn") is True


class TestEntityMentionSafety:
    """Item 4: Slack entity syntax in assistant output must not ping users.

    Assistant output can contain literal strings like `<@U123>`, `<!channel>`,
    `<#C123>`. If those reach Slack as active entity syntax (in either the
    markdown block text or the top-level `text` fallback), the bot will
    actually ping users, channels, or special groups. Slack parses these in
    mrkdwn-enabled `text` fields AND in markdown-block content. The fix:
    neutralize entity patterns without breaking CommonMark link syntax like
    `[label](https://...)`.
    """

    @pytest.mark.parametrize("entity", ["<@U123>", "<@U123|name>", "<!channel>", "<!here>", "<#C123>", "<#C123|general>"])
    def test_neutralize_entity_in_markdown_block(self, adapter, entity):
        blocks = adapter._build_markdown_blocks(f"hey {entity} look")
        # Slack entity syntax is `<@...>`, `<!...>`, `<#...>`. Each must NOT
        # appear verbatim in the emitted markdown-block text, because Slack
        # will parse it as an active mention even inside a markdown block.
        for b in blocks:
            txt = b.get("text", "")
            assert "<@" not in txt and "<!" not in txt and "<#" not in txt, (
                f"active entity syntax {entity!r} must not reach Slack — "
                f"neutralize before emitting markdown block"
            )

    @pytest.mark.parametrize("entity", ["<@U123>", "<!channel>", "<#C123>"])
    def test_neutralize_entity_in_fallback_text(self, entity):
        # `text=` is consumed by Slack for notifications and on clients that
        # don't render blocks; mrkdwn is enabled there by default, so the same
        # entity syntax pings. Must be neutralized too.
        out = SlackAdapter._strip_markdown_for_fallback(f"hey {entity} look")
        assert "<@" not in out and "<!" not in out and "<#" not in out, (
            f"fallback text must not preserve active entity syntax {entity!r}"
        )

    def test_commonmark_links_preserved_in_markdown_block(self, adapter):
        # Critical: entity neutralization must NOT mangle normal CommonMark
        # link syntax — Slack's markdown block IS CommonMark, so
        # `[label](https://example.com)` must round-trip verbatim.
        md = "see [the docs](https://example.com/docs) for more"
        blocks = adapter._build_markdown_blocks(md)
        assert blocks[0]["text"] == md, (
            "entity neutralization must not break CommonMark link syntax"
        )


class TestFallbackTextCapInvariant:
    """Item 6: lock in the existing invariant — fallback text stays <= 3000 chars.

    `_strip_markdown_for_fallback` already caps at 3000 chars (existing test
    `test_bounded_to_3000_chars` only checks an extreme 50k input). This test
    locks in the boundary case: a final response just under the 12k
    markdown-block limit but over the fallback-text cap must preserve the
    full markdown block while capping only the top-level `text` fallback.
    The invariant protects `chat.update` from `msg_too_long` and keeps
    notifications readable.
    """

    def test_content_over_3000_under_12000_caps_fallback_only(self, adapter):
        # 6,000 chars: under the 12k markdown-block cap (full content visible
        # in the markdown block) but over the 3,000-char fallback cap.
        content = "word " * 1200  # 6,000 chars
        assert 3000 < len(content) < 12000

        # The markdown block carries the full content unchanged.
        blocks = adapter._build_markdown_blocks(content)
        block_total = sum(len(b["text"]) for b in blocks)
        assert block_total >= len(content), (
            "markdown block must carry full content; only the fallback text caps"
        )

        # The standalone fallback helper caps at 3000.
        fallback = SlackAdapter._strip_markdown_for_fallback(content)
        assert len(fallback) <= 3000, (
            "top-level text fallback must stay <= 3000 chars to avoid msg_too_long "
            "on chat.update and to keep notifications readable"
        )

    @pytest.mark.asyncio
    async def test_send_payload_invariants_hold_for_long_response(self, adapter):
        # End-to-end: build the real `_send_payload` output for content between
        # 3000 and 12000 chars. The blocks list carries the full content; the
        # `text` fallback is independently capped.
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "123.000"}
        )

        content = "word " * 1200  # 6,000 chars
        await adapter.send("C123", content)

        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        blocks = call_kwargs.get("blocks") or []
        block_total = sum(len(b["text"]) for b in blocks if b.get("type") == "markdown")
        assert block_total >= len(content), (
            "visible markdown-block content must not be truncated by the "
            "fallback-cap invariant"
        )
        assert len(call_kwargs.get("text", "")) <= 3000, (
            "top-level text fallback must stay <= 3000 chars independently"
        )


# ---------------------------------------------------------------------------
# Test helper — a Slack SDK-shaped error for "blocks rejected."
# Slack's Python SDK raises SlackApiError with a `response` dict containing
# an `error` string; we model the family of block-rejection error codes the
# adapter should recognise when deciding to retry legacy.
# ---------------------------------------------------------------------------


class SlackRejectedBlocks(Exception):
    """Simulates Slack rejecting a markdown-block payload.

    SlackApiError carries a `.response` dict with an `error` field set to one
    of `invalid_blocks`, `markdown_text_too_long`, or `msg_too_long`. For the
    test, a thin subclass is enough — the adapter's recognition helper should
    match on the error message/name string, not the exception class itself.
    """

    def __init__(self, error_code: str):
        super().__init__(error_code)
        self.error_code = error_code
        # Mirror slack_sdk's SlackApiError.response['error'] shape so the
        # adapter can introspect it the same way SlackApiError is introspected.
        self.response = {"ok": False, "error": error_code}


# ---------------------------------------------------------------------------
# Item 5 (tw0316 review): streaming finalization uses a fresh markdown-block
# final when the adapter has markdown blocks enabled, instead of final-editing
# a raw legacy-mrkdwn preview. The adapter opts in via the
# ``prefers_fresh_final_streaming`` hook (see gateway/platforms/base.py) so the
# stream consumer delivers the completed answer as a fresh Block Kit message
# and best-effort deletes the stale preview. This avoids the failure mode
# where a raw streaming preview is treated as the delivered final content
# after a rich final edit fails (#53893 item 5).
# ---------------------------------------------------------------------------


class TestPrefersFreshFinalStreaming:
    """The Slack adapter must opt into the stream consumer's fresh-final path
    when markdown blocks are enabled, so the final answer renders as Block Kit
    rather than as a final-edit of a raw mrkdwn preview."""

    def test_hook_returns_true_when_markdown_blocks_enabled(self, adapter):
        # Default adapter is markdown-blocks-enabled.
        assert adapter._markdown_blocks_enabled() is True
        assert adapter.prefers_fresh_final_streaming("any content") is True

    def test_hook_returns_false_when_markdown_blocks_disabled(self, adapter):
        # Item 1 opt-out path must also gate the fresh-final hook — otherwise
        # users who disable blocks would still get fresh-final sends that go
        # legacy-mrkdwn anyway, paying the delete-and-resend cost for nothing.
        adapter.config.extra["markdown_blocks"] = False
        assert adapter._markdown_blocks_enabled() is False
        assert adapter.prefers_fresh_final_streaming("any content") is False


class TestStreamConsumerFreshFinalPath:
    """End-to-end through ``GatewayStreamConsumer``: when the Slack adapter
    has markdown blocks enabled, the final answer is delivered as a fresh
    Block Kit ``chat.postMessage`` (not an edit of the preview), and the
    stale preview is best-effort deleted."""

    @pytest.mark.asyncio
    async def test_final_renders_as_fresh_markdown_block_not_edit(self, adapter):
        from gateway.stream_consumer import (
            GatewayStreamConsumer,
            StreamConsumerConfig,
        )

        # The first chat_postMessage streams the legacy preview; the second
        # delivers the fresh final answer as a Block Kit payload.
        post_results = [
            {"ok": True, "ts": "100.0"},
            {"ok": True, "ts": "200.0"},
        ]
        adapter._app.client.chat_postMessage = AsyncMock(side_effect=post_results)
        adapter._app.client.chat_update = AsyncMock(
            return_value={"ok": True, "ts": "100.0"}
        )
        adapter.delete_message = AsyncMock(return_value=True)

        cfg = StreamConsumerConfig(
            transport="auto",
            chat_type="dm",
            edit_interval=0.001,
            buffer_threshold=1,
            cursor="",
            fresh_final_after_seconds=0.0,  # adapter hook alone drives fresh-final
        )
        consumer = GatewayStreamConsumer(adapter, "C123", cfg)

        consumer.on_delta("# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # The fresh final must be a *new* chat.postMessage, NOT a chat_update
        # of the preview. The edit path must not be the finalizer.
        assert adapter._app.client.chat_postMessage.await_count >= 2, (
            "fresh-final path must emit a new chat.postMessage for the final; "
            f"got {adapter._app.client.chat_postMessage.await_count} call(s)"
        )

        # The fresh final chat_postMessage must carry Block Kit ``blocks``,
        # not just legacy ``text`` + ``mrkdwn``. A table is the canonical case
        # that only renders correctly in a markdown block.
        final_call = adapter._app.client.chat_postMessage.call_args_list[-1].kwargs
        assert "blocks" in final_call, (
            "fresh final must be delivered as Block Kit blocks, not legacy mrkdwn"
        )
        assert any(
            b.get("type") == "markdown" for b in final_call["blocks"]
        ), f"expected a markdown block in fresh final, got {final_call['blocks']}"

        # The final answer must mark delivery complete so the gateway doesn't
        # double-send it.
        assert consumer.final_response_sent is True


class TestFreshFinalRejectionTriggersLegacyRetry:
    """tw0316 item 5 layered fallback: if the fresh-final markdown-block send
    itself is rejected by Slack (invalid_blocks etc.), the adapter's item-3
    retry path must fire (legacy mrkdwn), so the user still gets the final
    answer rather than a silent send failure after the preview was deleted."""

    @pytest.mark.asyncio
    async def test_fresh_final_markdown_block_rejection_retries_legacy(self, adapter):
        from gateway.stream_consumer import (
            GatewayStreamConsumer,
            StreamConsumerConfig,
        )

        # First chat_postMessage: streaming preview (legacy text, succeeds).
        # Second chat_postMessage: fresh final as blocks (REJECTED).
        # Third chat_postMessage: legacy mrkdwn retry (item 3 wiring) succeeds.
        adapter._app.client.chat_postMessage = AsyncMock(side_effect=[
            {"ok": True, "ts": "100.0"},
            SlackRejectedBlocks("invalid_blocks"),
            {"ok": True, "ts": "300.0"},
        ])
        adapter._app.client.chat_update = AsyncMock(
            return_value={"ok": True, "ts": "100.0"}
        )
        adapter.delete_message = AsyncMock(return_value=True)

        cfg = StreamConsumerConfig(
            transport="auto",
            chat_type="dm",
            edit_interval=0.001,
            buffer_threshold=1,
            cursor="",
            fresh_final_after_seconds=0.0,
        )
        consumer = GatewayStreamConsumer(adapter, "C123", cfg)

        consumer.on_delta("Final answer that must reach the user")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # Three chat_postMessage calls: preview, rejected-blocks, legacy retry.
        assert adapter._app.client.chat_postMessage.await_count >= 3, (
            "fresh-final rejection must trigger a legacy mrkdwn retry; "
            f"got {adapter._app.client.chat_postMessage.await_count} calls"
        )

        # The third (retry) call must use legacy text= + mrkdwn=true, NOT blocks.
        retry_call = adapter._app.client.chat_postMessage.call_args_list[-1].kwargs
        assert "blocks" not in retry_call, (
            "legacy retry after block rejection must not re-send blocks"
        )
        assert retry_call.get("mrkdwn") is True, (
            f"legacy retry must set mrkdwn=True, got {retry_call}"
        )

        assert consumer.final_response_sent is True
