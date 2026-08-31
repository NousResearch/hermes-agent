"""QA8 regression: the shared Telegram finalizer (format_message) must
convert HTML formatting tags to MarkdownV2 equivalents so no producer
can leak literal <b>, <i>, or <code> tags that cause MarkdownV2 parse
failures in the Telegram adapter.

The hostile payload reproduces the exact malformed shape observed in
production: GA DailyCockpitProjection text with mixed HTML <b> tags
and Markdown backticks.  JSON, timestamps, hashes, and URLs must pass
through unchanged (verified at the sanitizer level).
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    _sanitize_html_for_telegram,
    _strip_mdv2,
)


# ── Hostile payload (matches production GA cockpit output) ─────────────

_HOSTILE = (
    "\U0001f9ed <b>Daily cockpit</b> \u2014 Monday, 29 Aug 2026\n"
    "\n"
    "<b>\U0001f534 Now</b>\n"
    "\u2022 Review Q3 budget <b>URGENT</b>\n"
    "  \u23f0 `2026-08-30T09:00:00+04:00`\n"
    '\u2022 Call with client re: project {"id":"proj-8b7588","status":"blocked"}\n'
    "  \u23f0 `2026-08-30T14:30:00Z`\n"
    "\n"
    "<b>\U0001f64b Needs you</b>\n"
    "\u2022 Approve design mockups \u2014 <i>waiting since Thu</i>\n"
    "  ref: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6\n"
    "\n"
    "See https://example.com/dashboard for details.\n"
    "Run with: <code>hermes cron list</code>\n"
)

# Markers that MUST appear in the MarkdownV2 output after fix.
# format_message converts **bold** → *bold* and *italic* → _italic_.
_MUST_CONTAIN = [
    "*Daily cockpit*",
    "*\U0001f534 Now*",
    "*URGENT*",
    "_waiting since Thu_",
    "`hermes cron list`",
]

# HTML tags that MUST NOT appear
_HTML_TAG = re.compile(r"</?[bi]>|</?code>", re.IGNORECASE)


# ── Helper: minimal adapter that can call format_message ───────────────

def _make_adapter() -> TelegramAdapter:
    config = MagicMock()
    config.extra = {}
    config.reply_to_mode = "first"
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._app = None
    adapter._bot = None
    adapter._webhook_mode = False
    adapter._reply_to_mode = "first"
    adapter._disable_link_previews = False
    adapter._rich_messages_enabled = False
    adapter._rich_drafts_enabled = False
    return adapter


# ── Sanitizer unit tests ──────────────────────────────────────────────


class TestSanitizeHtmlFunction:
    """The _sanitize_html_for_telegram function converts HTML to standard
    markdown (which format_message then converts to MarkdownV2)."""

    def test_converts_bold(self):
        assert _sanitize_html_for_telegram("<b>Hello</b>") == "**Hello**"

    def test_converts_italic(self):
        assert _sanitize_html_for_telegram("<i>world</i>") == "*world*"

    def test_converts_code(self):
        assert _sanitize_html_for_telegram("<code>cmd</code>") == "`cmd`"

    def test_strips_unknown_tags_keeps_content(self):
        assert _sanitize_html_for_telegram("<span>keep</span>") == "keep"

    def test_no_html_tags_survive(self):
        result = _sanitize_html_for_telegram(_HOSTILE)
        assert _HTML_TAG.search(result) is None

    def test_preserves_json(self):
        result = _sanitize_html_for_telegram(_HOSTILE)
        assert '{"id":"proj-8b7588","status":"blocked"}' in result

    def test_preserves_timestamps(self):
        result = _sanitize_html_for_telegram(_HOSTILE)
        assert "2026-08-30T09:00:00+04:00" in result
        assert "2026-08-30T14:30:00Z" in result

    def test_preserves_hex_hashes(self):
        result = _sanitize_html_for_telegram(_HOSTILE)
        assert "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6" in result

    def test_preserves_urls(self):
        result = _sanitize_html_for_telegram(_HOSTILE)
        assert "https://example.com/dashboard" in result

    def test_preserves_backtick_code_spans(self):
        result = _sanitize_html_for_telegram(_HOSTILE)
        assert "`2026-08-30T09:00:00+04:00`" in result
        assert "`2026-08-30T14:30:00Z`" in result


# ── Product-path tests through format_message ─────────────────────────


class TestFormatMessageThroughAdapter:
    """GREEN: format_message converts HTML to MarkdownV2 via the sanitizer."""

    def test_html_tags_converted(self):
        adapter = _make_adapter()
        result = adapter.format_message(_HOSTILE)
        for marker in _MUST_CONTAIN:
            assert marker in result, f"Expected {marker!r} in output"
        assert _HTML_TAG.search(result) is None, (
            f"HTML tags survived: {_HTML_TAG.findall(result)}"
        )

    def test_plain_fallback_no_html(self):
        adapter = _make_adapter()
        formatted = adapter.format_message(_HOSTILE)
        plain = _strip_mdv2(formatted)
        assert _HTML_TAG.search(plain) is None


class TestPreimageRed:
    """RED: format_message WITHOUT the sanitizer would leave HTML intact.

    This test proves the sanitizer ran inside format_message by checking
    that MarkdownV2 markers appear and HTML tags do not.  If the
    sanitizer were removed, <b> tags would pass through and no *bold*
    markers would appear.
    """

    def test_sanitizer_ran_inside_format_message(self):
        adapter = _make_adapter()
        result = adapter.format_message(_HOSTILE)
        assert "*Daily cockpit*" in result
        assert _HTML_TAG.search(result) is None
