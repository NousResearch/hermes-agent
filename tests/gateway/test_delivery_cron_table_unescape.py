"""Regression tests for #53632: cron-delivered Telegram cron tables with MarkdownV2-escaped pipes.

When an LLM prompt instructs ``"use Telegram MarkdownV2 syntax"`` (or when the
model otherwise emits table bars already-escaped: ``\\| Date \\| Event \\|``),
sending the raw MarkdownV2-escaped string to Telegram causes ``\\|`` to render
literally and the recipient sees a broken table instead of a native Telegram
table render.

The fix in ``gateway/delivery.py`` detects cron-routed Telegram deliveries
(``metadata.job_id``) whose content carries a MarkdownV2-escaped pipe table
shape, then rewinds ``\\|`` back to ``|`` inside those rows so the adapter's
normal MarkdownV2 escape re-applies once and Telegram renders the table
natively.
"""

import pytest

from gateway.delivery import (
    _looks_like_cron_markdownv2_table,
    _unescape_markdownv2_tables,
)


# ---------------------------------------------------------------------------
# Helper-level unit tests (no DeliveryRouter needed)
# ---------------------------------------------------------------------------

class TestUnescapeMarkdownv2Tables:
    """Verify _unescape_markdownv2_tables rewinds ``\\|`` to ``|`` only on
    lines whose shape is a pipe-table row."""

    def test_canonical_reporter_case_rewinds_correctly(self):
        """The exact broken-table payload from issue #53632 is rewound."""
        broken = (
            "**\U0001f4c5 2-Week Forward Calendar**\n"
            "\\| Date \\| Event \\| Category \\| Expected Impact \\|\n"
            "\\|:-----\\|:------\\|:---------\\|:----------------\\|\n"
            "\\| 2026-07-15 \\| Fed Beige Book \\| Macro \\| Regional growth signals \\|\n"
            "\\| 2026-07-30 \\| ECB Meeting \\| Policy \\| Rate path clarification \\|\n"
        )
        out = _unescape_markdownv2_tables(broken)
        # All 5 table rows rewound
        assert "\\|" not in out
        assert "| Date | Event | Category | Expected Impact |" in out
        assert "| 2026-07-15 | Fed Beige Book |" in out
        # Surrounding bold preserved
        assert "**\U0001f4c5 2-Week Forward Calendar**" in out

    def test_only_pipe_table_rows_changed_other_lines_untouched(self):
        """Markdown-text lines without pipe-table shape are kept verbatim."""
        # Mix: 1 prose line, 2 table rows, 1 prose line - enough pipes for gate
        content = (
            "Here is the calendar you requested.\n"
            "\\| Date \\| Event \\| Category \\|\n"
            "\\| 2026-07-15 \\| Fed \\| Macro \\|\n"
            "\\| 2026-07-30 \\| ECB \\| Policy \\|\n"
            "Let me know if you want more context.\n"
        )
        out = _unescape_markdownv2_tables(content)
        # Prose untouched
        assert "Here is the calendar you requested." in out
        assert "Let me know if you want more context." in out
        # Table rows rewound (only the table lines had escaped pipes)
        assert "\\|" not in out
        assert "| Date | Event | Category |" in out

    def test_short_escaped_string_does_not_rewind(self):
        """A single ``\\|`` in prose (escape artifact) is NOT touched.

        Only lines with ``>= 3`` escaped bars AND row-shape (>= 3 cells after
        split) are candidates.  ``Use a \\| b`` has only one escaped bar so
        nothing changes — Telegram MarkdownV2 would render ``\\|`` as ``\\|``
        which is the existing legacy behavior.
        """
        prose = "Compute a \\| b then compare, OK?"
        out = _unescape_markdownv2_tables(prose)
        assert out == prose

    def test_code_block_with_escaped_pipe_is_left_alone(self):
        """``>3`` escaped bars inside code/text (not a row) stays untouched."""
        # Long arithmetic line — many | but no real row structure
        art = "a \\| b \\| c \\| d \\| e \\| f \\| g \\| h"
        out = _unescape_markdownv2_tables(art)
        assert out == art  # unchanged

    def test_empty_and_none_inputs(self):
        assert _unescape_markdownv2_tables("") == ""
        assert _unescape_markdownv2_tables(None) is None  # type: ignore[arg-type]

    def test_idempotent(self):
        """Running unescape twice produces the same output as once."""
        broken = "\\| Date \\| Event \\|\n\\| 2026-07-15 \\| Fed \\|\n"
        once = _unescape_markdownv2_tables(broken)
        twice = _unescape_markdownv2_tables(once)
        # Once has plain pipes, not pipe-table rows anymore — second pass
        # returns the same content (no further rewinds possible).
        assert once == twice


class TestLooksLikeCronMarkdownv2Table:
    """Detection predicate — conservative, never false-positive on prose."""

    def test_real_cron_table_detected(self):
        broken = (
            "\\| Date \\| Event \\| Category \\| Expected Impact \\|\n"
            "\\|:-----\\|:------\\|:---------\\|:----------------\\|\n"
            "\\| 2026-07-15 \\| Fed Beige Book \\| Macro \\| Regional growth signals \\|\n"
            "\\| 2026-07-30 \\| ECB Meeting \\| Policy \\| Rate path clarification \\|\n"
        )
        assert _looks_like_cron_markdownv2_table(broken) is True

    def test_plain_text_not_detected(self):
        plain = "Hello, this is a normal cron message without any tables."
        assert _looks_like_cron_markdownv2_table(plain) is False

    def test_arithmetic_with_pipes_not_detected(self):
        # Plenty of pipes, but no row shape (single line, no bars bracketing)
        art = "Compute a \\| b \\| c \\| d \\| e \\| f \\| g"
        assert _looks_like_cron_markdownv2_table(art) is False

    def test_short_two_row_table_not_detected(self):
        """A 2-row table has < 6 total escaped pipes — under threshold."""
        short = (
            "\\| Date \\| Event \\|\n"
            "\\| 2026-07-15 \\| Fed \\|\n"
        )
        assert _looks_like_cron_markdownv2_table(short) is False

    def test_empty_and_none(self):
        assert _looks_like_cron_markdownv2_table("") is False
        assert _looks_like_cron_markdownv2_table(None) is False

    def test_only_separator_line_no_bars(self):
        # A separator-like line without any pipe escape is NOT a table
        sep = ":-----|:------|:---------|:----------------"
        assert _looks_like_cron_markdownv2_table(sep) is False


# ---------------------------------------------------------------------------
# Gate logic test: simulate the gate in ``_deliver_to_platform`` so we don't
# need a full DeliveryRouter or async machinery. The gate is:
#
#   if (target.platform == TELEGRAM
#       and metadata.get("job_id")   # cron path
#       and _looks_like_cron_markdownv2_table(content)):
#       content = _unescape_markdownv2_tables(content)
#
# This nested test below exercises each of the three boolean predicates
# independently to lock the contract.
# ---------------------------------------------------------------------------

class TestCronGateContract:
    """Simulate the cron+TELEGRAM+table gate without spinning up the router."""

    @staticmethod
    def gate(content, *, target_platform, metadata):
        """Mirror the gate in ``_deliver_to_platform`` minus side-effects."""
        is_cron_path = bool((metadata or {}).get("job_id"))
        if (
            target_platform == "TELEGRAM"
            and is_cron_path
            and _looks_like_cron_markdownv2_table(content)
        ):
            return _unescape_markdownv2_tables(content)
        return content

    def test_cron_telegram_table_rewound(self):
        broken = (
            "\\| Date \\| Event \\| Category \\| Expected Impact \\|\n"
            "\\|:-----\\|:------\\|:---------\\|:----------------\\|\n"
            "\\| 2026-07-15 \\| Fed \\| Macro \\| Growth signals \\|\n"
            "\\| 2026-07-30 \\| ECB \\| Policy \\| Rate path \\|\n"
        )
        out = self.gate(broken, target_platform="TELEGRAM",
                        metadata={"job_id": "daily-news"})
        assert "\\|" not in out
        assert "| Date | Event |" in out

    def test_non_cron_telegram_leaves_table_alone(self):
        """Same payload but no ``job_id`` -> gate stays closed."""
        broken = (
            "\\| Date \\| Event \\| Category \\| Expected Impact \\|\n"
            "\\|:-----\\|:------\\|:---------\\|:----------------\\|\n"
            "\\| 2026-07-15 \\| Fed \\| Macro \\| Growth signals \\|\n"
            "\\| 2026-07-30 \\| ECB \\| Policy \\| Rate path \\|\n"
        )
        out = self.gate(broken, target_platform="TELEGRAM",
                        metadata={"notify": True})
        assert out == broken  # untouched

    def test_cron_but_non_telegram_leaves_table_alone(self):
        """Cron path with a non-Telegram target -> gate stays closed.

        Slack/Discord/WhatsApp adapters may have their own Markdown dialects;
        we don't touch them in this PR.  Scoping rewinds to TELEGRAM keeps
        the blast radius tight.
        """
        broken = (
            "\\| Date \\| Event \\| Category \\| Expected Impact \\|\n"
            "\\|:-----\\|:------\\|:---------\\|:----------------\\|\n"
            "\\| 2026-07-15 \\| Fed \\| Macro \\| Growth signals \\|\n"
            "\\| 2026-07-30 \\| ECB \\| Policy \\| Rate path \\|\n"
        )
        out = self.gate(broken, target_platform="SLACK",
                        metadata={"job_id": "daily-news"})
        assert out == broken
