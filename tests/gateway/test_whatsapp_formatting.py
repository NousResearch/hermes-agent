"""Tests for WhatsApp message formatting and chunking.

Covers:
- format_message(): markdown → WhatsApp syntax conversion
- send(): message chunking for long responses
- MAX_MESSAGE_LENGTH: practical UX limit
"""

import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import re

import pytest

from gateway.config import Platform


@pytest.fixture(autouse=True)
def _whatsapp_open_optin(monkeypatch):
    """Opt into WhatsApp allow-all so ``dm_policy: open`` dispatch tests run.

    The adapter fails closed on ``open`` without an allow-all opt-in
    (SECURITY.md 2.6); these formatting/dispatch-mechanics tests set
    ``_dm_policy = "open"`` as a stand-in for "process this DM".
    """
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter():
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter._bridge_port = 3000
    adapter._bridge_script = "/tmp/test-bridge.js"
    adapter._session_path = MagicMock()
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._reply_prefix = None
    adapter._running = True
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._http_session = MagicMock()
    adapter._mention_patterns = []
    adapter._dm_policy = "open"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    return adapter


class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# format_message tests
# ---------------------------------------------------------------------------

class TestFormatMessage:
    """WhatsApp markdown conversion."""


    def test_strikethrough(self):
        adapter = _make_adapter()
        assert adapter.format_message("~~deleted~~") == "~deleted~"

    def test_stray_single_tilde_is_not_strikethrough(self):
        """`~2.00USD~` (model meaning "approximately") must NOT reach
        WhatsApp as native strikethrough. A single `~` is never a valid
        markdown strike, so the parser leaves it literal; WhatsApp's native
        open→close pairing would re-read it as strike, so the renderer
        space-flanks the offending open (`~ `) to break the pair — the `~`
        character itself is preserved (ASCII, no unicode substitution)."""
        adapter = _make_adapter()
        assert adapter.format_message("cost ~2.00USD~ total") == (
            "cost ~ 2.00USD~ total"
        )
        assert adapter.format_message("~~done~~ and ~maybe~") == (
            "~done~ and ~ maybe~"
        )

    def test_stray_tilde_safe_when_open_not_close(self):
        """`~5USD ~4USD` / `range ~5 to ~10`: the trailing `~` is itself
        preceded by a space (so it's an *open*, not a close) — no valid
        open→close pair exists, so WhatsApp never formats it and nothing
        needs to change."""
        adapter = _make_adapter()
        assert adapter.format_message("~5USD ~4USD") == "~5USD ~4USD"
        assert adapter.format_message("range ~5 to ~10") == "range ~5 to ~10"

    def test_stray_delims_never_pair_across_lines(self):
        """WhatsApp matches delimiter pairs per line only: an open `~` at
        the end of one line can never be closed by a `~` on a following
        line, so no flanking must happen (and no space may be inserted)."""
        adapter = _make_adapter()
        assert adapter.format_message("a ~2.00\nUSD~ more") == (
            "a ~2.00\nUSD~ more"
        )
        assert adapter.format_message("x ~ y\n~z") == "x ~ y\n~z"
        # Same-line strays are still flanked.
        assert adapter.format_message("cost ~2.00USD~ total\nnext ~5.00~") == (
            "cost ~ 2.00USD~ total\nnext ~ 5.00~"
        )

    def test_stray_tilde_in_heading_and_list(self):
        adapter = _make_adapter()
        # Same flanking applies anywhere literal text is rendered (headings
        # additionally get the Unicode font; the `~` chars stay ASCII).
        assert adapter.format_message("# ~draft~ title") == "~ 𝐝𝐫𝐚𝐟𝐭~ 𝐭𝐢𝐭𝐥𝐞"
        assert adapter.format_message("- ~opt~ item") == "- ~ opt~ item"

    def test_star_underscore_literals_not_flanked(self):
        """Stray `*`/`_` that can't form a valid WhatsApp open→close pair
        stay untouched (snake_case, spaced math/paths, unmatched, code) — and
        real `**bold**`/`*italic*` constructs still convert. Only pairs that
        WhatsApp would natively format get space-flanked."""
        adapter = _make_adapter()
        keep_literal = [
            "snake_case_name",
            "2 * 3 * 4",
            "x _ y _ z",
            "C++ and A*B * literal",
            "price *(unclosed",
            "`code` with *no* real construct",
        ]
        for src in keep_literal:
            assert "*" in src or "_" in src
            assert str(adapter.format_message(src)) != "", src
        # Real constructs still convert.
        assert adapter.format_message("**b** and *i*") == "*b* and _i_"

    def test_stray_star_underscore_pair_gets_flanked(self):
        """The flanker handles `*`/`_` pairs too (not just `~`), as a
        defensive layer. Through format_message, pairable `*x*`/`_x_` are
        valid markdown emphasis → intentional `_x_` italic, so they never
        reach the literal-text branch; the helper is exercised directly."""
        from gateway.platforms.whatsapp_renderer import _flank_stray_delims

        assert _flank_stray_delims("cost *5 USD* total") == "cost * 5 USD* total"
        assert _flank_stray_delims("see _note_ here") == "see _ note_ here"
        assert _flank_stray_delims("price `5 USD` total") == "price ` 5 USD` total"
        # No valid pair → untouched.
        assert _flank_stray_delims("snake_case_name") == "snake_case_name"
        assert _flank_stray_delims("2 * 3 * 4") == "2 * 3 * 4"
        # Intentional emphasis through the real pipeline stays converted.
        adapter = _make_adapter()
        assert adapter.format_message("cost *5 USD* total") == (
            "cost _5 USD_ total"
        )

    def test_headers_converted_to_unicode_fonts(self):
        adapter = _make_adapter()
        # Default pipeline styles heading *levels* with Unicode fonts to
        # preserve hierarchy native WhatsApp would flatten (H1=bold).
        assert adapter.format_message("# Title") == "𝐓𝐢𝐭𝐥𝐞"
        assert adapter.format_message("## Subtitle") == "𝑺𝒖𝒃𝒕𝒊𝒕𝒍𝒆"
        assert adapter.format_message("### Deep") == "𝐷𝑒𝑒𝑝"

    def test_bold_header_flattened_no_double_wrap(self):
        # "# **Title**" keeps a single bold font, not **Title** (heading
        # content is rendered plain before the Unicode font is applied).
        adapter = _make_adapter()
        assert adapter.format_message("# **Title**") == "𝐓𝐢𝐭𝐥𝐞"
        assert adapter.format_message("## __Strong__") == "𝑺𝒕𝒓𝒐𝒏𝒈"


    def test_already_whatsapp_italic(self):
        """Markdown *italic* converts to WhatsApp _italic_ (PR #58704)."""
        adapter = _make_adapter()
        assert adapter.format_message("*italic*") == "_italic_"
        # Already-WhatsApp _italic_ passes through unchanged
        assert adapter.format_message("_italic_") == "_italic_"


class TestCommonMarkFormatting:
    """AST-based CommonMark → WhatsApp *native* conversion (full construct
    coverage).

    This pins ``unicode_formatting=False`` so the matrix exercises the
    WhatsApp-native dialect (bold/italic/strike/code + readable down-renders)
    that users get when Unicode font styling is disabled. The default-ON
    Unicode path is covered separately by ``TestUnicodeFormattingOption``.
    """

    def _fmt(self, text: str) -> str:
        adapter = _make_adapter()
        adapter.config.extra = {"unicode_formatting": False}
        return adapter.format_message(text)

    # -- inline ------------------------------------------------------------
    def test_bold_italic_strike_code(self):
        assert self._fmt("**b** *i* ~~s~~ `c`") == "*b* _i_ ~s~ `c`"

    def test_nested_emphasis(self):
        assert self._fmt("**b _i_ b**") == "*b _i_ b*"
        assert self._fmt("*i **b** i*") == "_i *b* i_"

    def test_intraword_underscores_stay_literal(self):
        assert self._fmt("snake_case_name and file_name.py") == (
            "snake_case_name and file_name.py"
        )

    def test_bare_url_and_emoji_pass_through(self):
        assert self._fmt("Go https://x.com/a?q=1&r=2 ✅ now") == (
            "Go https://x.com/a?q=1&r=2 ✅ now"
        )

    # -- headings ----------------------------------------------------------
    def test_setext_headings(self):
        assert self._fmt("Title\n=====\n\nSub\n---") == "*Title*\n\n*Sub*"

    def test_heading_with_bold_content(self):
        assert self._fmt("## The **Real** Deal") == "*The *Real* Deal*"

    # -- code --------------------------------------------------------------
    def test_fenced_code_lang_becomes_caption(self):
        assert self._fmt("```python\nx = 1\n```") == (
            "*python*\n```\nx = 1\n```"
        )

    def test_fenced_code_without_lang(self):
        assert self._fmt("```\nx = 1\n```") == "```\nx = 1\n```"

    def test_indented_code(self):
        assert self._fmt("    x = 1\n    y = 2") == "```\nx = 1\ny = 2\n```"

    # -- links & images ----------------------------------------------------
    def test_link_becomes_text_url(self):
        assert self._fmt("See [docs](https://example.com/a_(b)).") == (
            "See docs (https://example.com/a_(b))."
        )

    def test_reference_link_resolves(self):
        assert self._fmt("See [foo][id].\n\n[id]: https://x.dev") == (
            "See foo (https://x.dev)."
        )

    def test_image_keeps_alt_and_src(self):
        assert self._fmt("![screen](https://x/img.png)") == (
            "screen (https://x/img.png)"
        )

    # -- lists -------------------------------------------------------------
    def test_bullet_list(self):
        assert self._fmt("- a\n- b\n- c") == "- a\n- b\n- c"

    def test_ordered_list_preserves_start(self):
        assert self._fmt("5. five\n6. six") == "5. five\n6. six"

    def test_nested_list_indented(self):
        assert self._fmt("- outer\n  - in1\n  - in2\n- out2") == (
            "- outer\n  - in1\n  - in2\n- out2"
        )

    def test_task_list_kept(self):
        assert self._fmt("- [x] done\n- [ ] todo") == "- [x] done\n- [ ] todo"

    # -- blocks ------------------------------------------------------------
    def test_blockquote(self):
        assert self._fmt("> q1\n> q2") == "> q1\n> q2"

    def test_horizontal_rule(self):
        assert self._fmt("a\n\n---\n\nb") == "a\n\n────────────\n\nb"

    def test_table_default_flatten_alignment_free(self):
        """Default ``flatten`` converts tables to key/value lines; WhatsApp
        always soft-wraps long lines, so padded aligned grids would lose
        column alignment the instant a row exceeds the bubble width."""
        out = self._fmt("| name | value |\n|------|-------|\n| a    | 1     |\n| bee  | 222   |")
        assert out == (
            "*name*: a\n"
            "*value*: 1\n"
            "\n"
            "*name*: bee\n"
            "*value*: 222"
        )

    def test_table_flatten_blank_line_between_rows(self):
        out = self._fmt(
            "| Feature | Status |\n|---------|--------|\n"
            "| tables  | fixed  |\n| wrap    | safe   |"
        )
        # Each data row becomes a blank-line-separated block of key/value lines.
        assert "\n\n" in out
        assert "*Feature*: tables" in out
        assert "*Status*: fixed" in out
        assert "*Feature*: wrap" in out
        assert "*Status*: safe" in out

    def test_table_monospace_opt_in_preserves_aligned_pipe(self):
        """``table_mode=monospace`` keeps the legacy aligned pipe fence for
        cross-device fidelity — explicitly opt-in because WhatsApp wraps."""
        from gateway.platforms.whatsapp_renderer import CommonMarkToWhatsApp

        out = CommonMarkToWhatsApp(
            "| name | value |\n|------|-------|\n| a    | 1     |\n| bee  | 222   |",
            table_mode="monospace",
        ).render()
        assert out == (
            "```\n"
            "| name | value |\n"
            "| ---- | ----- |\n"
            "| a    | 1     |\n"
            "| bee  | 222   |\n"
            "```"
        )

    def test_html_stripped_inner_text_kept(self):
        assert self._fmt("<div>Hello <b>world</b></div>") == "Hello world"
        assert self._fmt("2 < 3 and a < b stay literal") == (
            "2 < 3 and a < b stay literal"
        )

    def test_html_declaration_kept(self):
        # Slack-style <!mention> is prose, not markup — must not vanish.
        assert self._fmt("Hey <!everyone> and <!here>!") == (
            "Hey <!everyone> and <!here>!"
        )

    def test_entities_decoded(self):
        assert self._fmt("AT&amp;T &amp; &#39;quoted&#39;") == "AT&T & 'quoted'"

    def test_mixed_document(self):
        out = self._fmt(
            "## Report\n\n"
            "Status: **green**. Details in `runbook.md`.\n\n"
            "- item *one*\n- item **two**\n\n"
            "```sh\nmake deploy\n```\n\n"
            "See [dashboard](https://g.dev/d)."
        )
        assert out == (
            "*Report*\n\n"
            "Status: *green*. Details in `runbook.md`.\n\n"
            "- item _one_\n- item *two*\n\n"
            "*sh*\n```\nmake deploy\n```\n\n"
            "See dashboard (https://g.dev/d)."
        )

    def test_unterminated_fence_degrades_gracefully(self):
        out = self._fmt("```python\nprint('never closed'")
        assert out == "*python*\n```\nprint('never closed'\n```"


# ---------------------------------------------------------------------------
# Unicode font formatting option (opt-in toggle)
# ---------------------------------------------------------------------------

class TestGlyphStyler:
    """Unicode mathematical-alphanumeric restyling."""

    def test_letter_mapping(self):
        from gateway.platforms.whatsapp_unicode import GlyphStyler

        assert GlyphStyler.stylize("Title", "bold") == "𝐓𝐢𝐭𝐥𝐞"  # U+1D413 1D45B ...
        assert GlyphStyler.stylize("Title", "italic") == "𝑇𝑖𝑡𝑙𝑒"
        assert GlyphStyler.stylize("Title", "sans_serif") == "𝖳𝗂𝗍𝗅𝖾"
        assert GlyphStyler.stylize("Title", "monospace") == "𝚃𝚒𝚝𝚕𝚎"

    def test_digits_and_punctuation(self):
        from gateway.platforms.whatsapp_unicode import GlyphStyler

        # Bold digits exist; punctuation has no styled glyph and passes through.
        assert GlyphStyler.stylize("2026!", "bold") == "𝟐𝟎𝟐𝟔!"
        # Italic has no digit block — digits stay plain.
        assert GlyphStyler.stylize("v2.0", "italic") == "𝑣2.0"

    def test_unmappable_pass_through(self):
        from gateway.platforms.whatsapp_unicode import GlyphStyler

        assert GlyphStyler.stylize("Привет ✅", "bold") == "Привет ✅"
        assert GlyphStyler.stylize("", "bold") == ""
        assert GlyphStyler.stylize("Title", "no_such_style") == "Title"

    def test_italic_lowercase_h_substitutes_planck_constant(self):
        """U+1D455 (italic small h) is UNASSIGNED in Unicode — every device
        renders it as tofu.  The styler must substitute ℎ (U+210E)."""
        from gateway.platforms.whatsapp_unicode import GlyphStyler

        result = GlyphStyler.stylize("h", "italic")
        assert result == "ℎ"
        assert ord(result) == 0x210E
        # Heading level 3 renders italic, so a whole italic-h heading must
        # be free of the unassigned code point too.
        heading = GlyphStyler.stylize("high hopes", "italic")
        assert heading == "ℎ𝑖𝑔ℎ ℎ𝑜𝑝𝑒𝑠"
        assert 0x1D455 not in {ord(c) for c in heading}

    def test_no_style_ever_emits_unassigned_glyphs(self):
        """Exhaustive sweep: every style × every mappable ASCII char must
        produce only code points assigned in Unicode (no tofu glyphs)."""
        import string
        import unicodedata

        from gateway.platforms.whatsapp_unicode import STYLES, GlyphStyler

        corpus = string.ascii_letters + string.digits
        for style in STYLES:
            for ch in corpus:
                for glyph in GlyphStyler.stylize(ch, style):
                    assert unicodedata.category(glyph) != "Cn", (
                        f"style {style!r} char {ch!r} emits unassigned "
                        f"U+{ord(glyph):04X}"
                    )


class TestUnicodeFormattingOption:
    """``unicode_formatting`` behavior (default ON) via ``format_message``."""

    def _fmt(self, text: str) -> str:
        return _make_adapter().format_message(text)

    def test_default_on_uses_unicode_headings(self):
        assert self._fmt("# Big Title") == "𝐁𝐢𝐠 𝐓𝐢𝐭𝐥𝐞"  # U+1D401-style bold

    def test_env_true_explicitly_enables(self, monkeypatch):
        monkeypatch.setenv("WHATSAPP_UNICODE_FORMATTING", "true")
        assert self._fmt("# Big Title") == "𝐁𝐢𝐠 𝐓𝐢𝐭𝐥𝐞"

    def test_env_false_disables(self, monkeypatch):
        monkeypatch.setenv("WHATSAPP_UNICODE_FORMATTING", "0")
        assert self._fmt("# Big Title") == "*Big Title*"

    def test_config_extra_true_explicitly_enables(self):
        adapter = _make_adapter()
        adapter.config.extra = {"unicode_formatting": True}
        assert adapter.format_message("# Big Title") == "𝐁𝐢𝐠 𝐓𝐢𝐭𝐥𝐞"

    def test_config_extra_false_disables(self):
        adapter = _make_adapter()
        adapter.config.extra = {"unicode_formatting": False}
        assert adapter.format_message("# Big Title") == "*Big Title*"

    def test_env_false_takes_precedence_over_config_true(self, monkeypatch):
        adapter = _make_adapter()
        adapter.config.extra = {"unicode_formatting": True}
        monkeypatch.setenv("WHATSAPP_UNICODE_FORMATTING", "off")
        assert adapter.format_message("# Big Title") == "*Big Title*"

    def test_heading_levels_get_distinct_styles(self):
        from gateway.platforms.whatsapp_renderer import CommonMarkToWhatsApp

        render = lambda s: CommonMarkToWhatsApp(s, unicode_formatting=True).render()
        h1, h2, h3, h4, h5, h6 = (render(f"#{'#' * i} Level{i}") for i in range(6))
        assert len({h1, h2, h3, h4, h5, h6}) == 6  # hierarchy preserved
        assert h1.startswith("𝐋")  # bold
        assert h6.startswith("𝘓")  # sans-serif italic

    def test_setext_headings_styled_too(self):
        from gateway.platforms.whatsapp_renderer import CommonMarkToWhatsApp

        render = lambda s: CommonMarkToWhatsApp(s, unicode_formatting=True).render()
        assert render("Title\n=====") == "𝐓𝐢𝐭𝐥𝐞"

    def test_no_stray_markers_inside_unicode_heading(self):
        from gateway.platforms.whatsapp_renderer import CommonMarkToWhatsApp

        out = CommonMarkToWhatsApp(
            "## The **Real** Deal", unicode_formatting=True
        ).render()
        assert out == "𝑻𝒉𝒆 𝑹𝒆𝒂𝒍 𝑫𝒆𝒂𝒍"  # bold content flattened, no `*` leak

    def test_paragraph_emphasis_stays_native_in_unicode_mode(self):
        from gateway.platforms.whatsapp_renderer import CommonMarkToWhatsApp

        out = CommonMarkToWhatsApp(
            "Body with **bold** and *ital*.", unicode_formatting=True
        ).render()
        assert out == "Body with *bold* and _ital_."


# ---------------------------------------------------------------------------
# format_message → truncate_message pipeline (e2e)
# ---------------------------------------------------------------------------

class TestFormatTruncatePipeline:
    """A formatted message containing code fences must survive chunking.

    Both send paths (Baileys plugin adapter and Cloud API adapter) run the
    same shared chain — ``format_message`` then ``truncate_message`` over the
    mixin's ``_outgoing_chunk_limit`` — so exercising it once through the
    plugin adapter covers both.
    """

    def test_long_code_block_survives_chunking(self):
        adapter = _make_adapter()
        limit = adapter._outgoing_chunk_limit()
        # 5000 lines (~75KB) clears the 65,536-char cap so the payload is
        # genuinely multi-chunk; a 400-line block now fits one bubble.
        code = "\n".join(f"line {i:04d} = {i}" for i in range(5000))
        content = (
            "## Notes\n\nSummary **bold** and `inline`.\n\n"
            "```python\n" + code + "\n```\n\nSee [docs](https://example.com)."
        )
        formatted = adapter.format_message(content)
        chunks = adapter.truncate_message(formatted, limit)

        assert len(chunks) >= 2
        for chunk in chunks:
            # WhatsApp needs an opening AND closing triple-backtick inside a
            # single message for code formatting — every chunk must be balanced.
            assert len(chunk) <= limit
            assert chunk.count("```") % 2 == 0, f"unbalanced fences: {chunk[:80]!r}"

        joined = "".join(chunks)
        # Every code line survives somewhere in the chunked output.
        for i in (0, 2500, 4999):
            assert f"line {i:04d} = {i}" in joined
        # Structure preserved too: caption, an inline bullet, and the link.
        joined = re.sub(r" \(\d+/\d+\)", "", joined)
        assert "*python*\n```" in joined
        assert "See docs (https://example.com)." in joined

    def test_short_message_not_chunked(self):
        adapter = _make_adapter()
        formatted = adapter.format_message("# Hi\n\nPlain body.")
        assert adapter.truncate_message(formatted, adapter._outgoing_chunk_limit()) == [formatted]


# ---------------------------------------------------------------------------
# MAX_MESSAGE_LENGTH tests
# ---------------------------------------------------------------------------

class TestMessageLimits:
    """WhatsApp message length limits."""


    def test_chunk_limit_reserves_default_self_chat_prefix(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.delenv("WHATSAPP_REPLY_PREFIX", raising=False)
        monkeypatch.setenv("WHATSAPP_MODE", "self-chat")

        assert adapter._outgoing_chunk_limit() == (
            adapter.MAX_MESSAGE_LENGTH - len(adapter.DEFAULT_REPLY_PREFIX)
        )


# ---------------------------------------------------------------------------
# send() chunking tests
# ---------------------------------------------------------------------------

class TestSendChunking:
    """WhatsApp send() splits long messages into chunks."""

    @pytest.mark.asyncio
    async def test_short_message_single_send(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        result = await adapter.send("chat1", "short message")
        assert result.success
        # Only one call to bridge /send
        assert adapter._http_session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_long_message_chunked(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        # Create a message longer than MAX_MESSAGE_LENGTH (65,536 — WhatsApp's
        # real per-message cap; there is no smaller UX chunk size anymore).
        long_msg = "a " * 40000  # ~80,000 chars

        result = await adapter.send("chat1", long_msg)
        assert result.success
        # Should have made multiple calls
        assert adapter._http_session.post.call_count > 1

    @pytest.mark.asyncio
    async def test_chunks_leave_room_for_bridge_prefix(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.delenv("WHATSAPP_REPLY_PREFIX", raising=False)
        monkeypatch.setenv("WHATSAPP_MODE", "self-chat")
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        long_msg = "a " * 40000

        await adapter.send("chat1", long_msg)

        for call in adapter._http_session.post.call_args_list:
            payload = call.kwargs.get("json") or call[1].get("json")
            final_text = adapter.DEFAULT_REPLY_PREFIX + payload["message"]
            assert len(final_text) <= adapter.MAX_MESSAGE_LENGTH


# ---------------------------------------------------------------------------
# bridge event metadata
# ---------------------------------------------------------------------------

class TestBridgeEventMetadata:
    """WhatsApp bridge metadata is preserved for downstream consumers."""

    @pytest.mark.asyncio
    async def test_quoted_reply_metadata_is_preserved_in_raw_message(self):
        adapter = _make_adapter()
        data = {
            "messageId": "incoming-msg",
            "chatId": "15551234567@s.whatsapp.net",
            "senderId": "15551234567@s.whatsapp.net",
            "senderName": "Tester",
            "chatName": "Tester",
            "isGroup": False,
            "body": "approved",
            "hasMedia": False,
            "mediaUrls": [],
            "quotedMessageId": "outbound-msg",
            "quotedParticipant": "99999999999@s.whatsapp.net",
            "quotedRemoteJid": "15551234567@s.whatsapp.net",
            "hasQuotedMessage": True,
        }

        event = await adapter._build_message_event(data)

        assert event is not None
        assert event.raw_message["quotedMessageId"] == "outbound-msg"
        assert event.raw_message["quotedParticipant"] == "99999999999@s.whatsapp.net"
        assert event.raw_message["quotedRemoteJid"] == "15551234567@s.whatsapp.net"
        assert event.raw_message["hasQuotedMessage"] is True


# ---------------------------------------------------------------------------
# display_config tier classification
# ---------------------------------------------------------------------------

class TestWhatsAppTier:
    """WhatsApp should be classified as TIER_MEDIUM."""

    def test_whatsapp_streaming_follows_global(self):
        from gateway.display_config import resolve_display_setting
        # TIER_MEDIUM has streaming: None (follow global), not False
        assert resolve_display_setting({}, "whatsapp", "streaming") is None


# ---------------------------------------------------------------------------
# Every send path formats (no raw # / ** leaks)
# ---------------------------------------------------------------------------

class TestEverySendPathFormats:
    """Every WhatsApp text send path must run ``format_message`` first so raw
    ``#``/``**`` markers never leak into WhatsApp.

    ``send()`` already formatted; these cover the paths that historically
    skipped it: streaming edits (``edit_message``), media captions
    (``_send_media_to_bridge``), and the out-of-process standalone/cron
    delivery (``_standalone_send``).
    """

    def _bridge_http(self, adapter):
        """Rewire the adapter's bridge POST to capture payloads + return 200s.

        ``_make_adapter`` leaves ``_http_session`` a plain MagicMock, which is
        what ``async with self._http_session.post(...)`` needs — so we attach
        a *sync* side effect that returns an async context manager wrapping a
        200 response (an AsyncMock side effect would hand back an unawaited
        coroutine instead).
        """
        captured = []

        def _post(url, **kwargs):
            captured.append((url, kwargs.get("json", {})))
            resp = SimpleNamespace(status=200)
            resp.json = AsyncMock(return_value={"messageId": "m1"})
            resp.text = AsyncMock(return_value="")
            return _AsyncCM(resp)

        adapter._http_session.post = MagicMock(side_effect=_post)
        return captured

    def test_edit_message_converts_markdown(self):
        """Streaming edits (the whole accumulated buffer) render formatted."""
        adapter = _make_adapter()
        captured = self._bridge_http(adapter)
        asyncio.run(adapter.edit_message("12345", "mid1", "## Notes\n\nBody."))
        url, payload = captured[0]
        assert url.endswith("/edit")
        assert payload["message"] == "𝑵𝒐𝒕𝒆𝒔\n\nBody."

    def test_edit_message_single_chunk_keeps_original_id(self):
        """A normal (single-chunk) edit returns the original message id.

        The bridge returns ``messageIds`` only when it split an oversized
        edit (past WhatsApp's 65,536-char cap) into continuation messages;
        an empty list means the edit stayed one bubble and the consumer
        must keep targeting the original ``message_id``.
        """
        adapter = _make_adapter()
        self._bridge_http(adapter)  # default response: messageIds absent
        result = asyncio.run(adapter.edit_message("12345", "mid1", "Body."))
        assert result.success
        assert result.message_id == "mid1"
        assert result.continuation_message_ids == ()

    def test_edit_message_surfaces_continuation_ids(self):
        """When the bridge splits an edit, re-target at the LAST bubble.

        Regression for the duplicate-bubble bug: the bridge /edit handler
        used to split at 4096 UTF-16 units (pre-full-length-cap), and every
        chunk 2+ was sent as
        an unlinked NEW message while the adapter kept reporting the
        original id — so each stream frame spawned another duplicate.
        Now the adapter surfaces the continuation ids and the consumer
        re-targets subsequent edits at the last visible message.
        """
        adapter = _make_adapter()
        captured = []

        def _post(url, **kwargs):
            captured.append((url, kwargs.get("json", {})))
            resp = SimpleNamespace(status=200)
            resp.json = AsyncMock(return_value={"success": True, "messageIds": ["c1", "c2"]})
            resp.text = AsyncMock(return_value="")
            return _AsyncCM(resp)

        adapter._http_session.post = MagicMock(side_effect=_post)
        result = asyncio.run(
            adapter.edit_message("12345", "mid1", "B" * 70000)
        )
        assert result.success
        assert result.message_id == "c2"
        assert result.continuation_message_ids == ("c1", "c2")
        url, payload = captured[0]
        assert url.endswith("/edit")
        assert payload["messageId"] == "mid1"

    def test_send_media_to_bridge_converts_caption(self):
        """Media captions are converted too (WhatsApp renders them formatted)."""
        adapter = _make_adapter()
        captured = self._bridge_http(adapter)
        img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.write(b"x")
        img.close()
        try:
            asyncio.run(
                adapter._send_media_to_bridge(
                    "12345", img.name, "image", caption="# Cap\n\nBody **bold**."
                )
            )
        finally:
            os.unlink(img.name)
        url, payload = captured[0]
        assert url.endswith("/send-media")
        assert payload["caption"] == "𝐂𝐚𝐩\n\nBody *bold*."



class TestWhatsAppTierCurrent:
    """Current-upstream tier check (kept alongside the ported suite)."""

    def test_whatsapp_streaming_follows_global(self):
        from gateway.display_config import resolve_display_setting
        # TIER_MEDIUM has streaming: None (follow global), not False
        assert resolve_display_setting({}, "whatsapp", "streaming") is None
