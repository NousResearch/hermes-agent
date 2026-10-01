"""Tests for unsupported markdown link targets on Telegram (issue #97497).

Models sometimes reformat Hermes' desktop-only ``@session:<profile>/<id>``
references as markdown links with a non-URL destination, e.g.
``[Title](Title)`` or ``[Title](@session:profile/id)``. Telegram cannot
render those as links and exposes the raw bracket-and-parenthesis syntax to
the user instead of readable prose.

Both delivery paths must degrade unsupported targets to their display text
while keeping HTTP(S)/tg:// links clickable:

- the legacy MarkdownV2 formatter (``format_message``), and
- the rich-message path — ``_rich_message_payload`` is the shared payload
  builder for rich sends, finalized edits, and drafts, so scrubbing it
  covers all three rich call sites.

Code spans/blocks and table blocks hold literal content and must be left
verbatim.
"""

from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    _degrade_unsupported_markdown_links,
)


class TestLegacyMarkdownV2LinkDegrade:
    """format_message must not emit raw [label](target) for bad targets."""

    def _adapter(self):
        return object.__new__(TelegramAdapter)

    def test_schemeless_target_degrades_to_display_text(self):
        """The deterministic repro from the issue.

        ``format_message("Already underway in [Example Session](Example
        Session).")`` used to preserve the schemeless destination, so the
        Telegram client showed the raw link syntax.
        """
        result = self._adapter().format_message(
            "Already underway in [Example Session](Example Session)."
        )
        assert "](" not in result
        assert "Example Session" in result

    def test_session_reference_degrades_to_display_text(self):
        """``@session:`` is a Hermes-Desktop-only reference Telegram can't
        resolve — it must degrade to the title, never ship as a link."""
        result = self._adapter().format_message(
            "Continuing [Research](@session:default/abc123)."
        )
        assert "](" not in result
        assert "Research" in result
        assert "@session:" not in result

    def test_https_link_stays_clickable(self):
        result = self._adapter().format_message("See [Docs](https://example.com/x).")
        assert "[Docs](https://example.com/x)" in result

    def test_tg_link_stays_clickable(self):
        result = self._adapter().format_message("Open [settings](tg://settings).")
        assert "[settings](tg://settings)" in result

    def test_explicitly_bracketed_numeric_citation_keeps_complete_marker_visible(self):
        result = self._adapter().format_message(
            "A grounded claim.[[3](https://example.com/source)]"
        )
        assert r"[\[3\]](https://example.com/source)" in result

    def test_ordinary_numeric_commit_label_stays_an_ordinary_link(self):
        result = self._adapter().format_message(
            "Built from [1234567](https://github.com/acme/project/commit/1234567)."
        )
        assert r"[1234567](https://github.com/acme/project/commit/1234567)" in result
        assert r"[\[1234567\]]" not in result

    def test_link_with_title_keeps_its_url(self):
        result = self._adapter().format_message(
            'See [Docs](https://example.com/x "Title").'
        )
        assert "[Docs](https://example.com/x)" in result
        assert "Title" not in result

    def test_angle_bracket_destination_keeps_its_url(self):
        result = self._adapter().format_message("See [Docs](<https://example.com/x>).")
        assert "[Docs](https://example.com/x)" in result

    def test_citation_with_title_does_not_put_the_title_in_the_url(self):
        result = self._adapter().format_message(
            'A grounded claim.[[3](https://example.com/source "Source")]'
        )
        assert r"[\[3\]](https://example.com/source)" in result
        assert "Source" not in result

    def test_citation_with_unsupported_target_degrades_to_its_number(self):
        result = self._adapter().format_message("A claim.[[3](@session:default/abc)]")
        assert "](" not in result
        assert "@session:" not in result


class TestRichMessageLinkDegrade:
    """_rich_message_payload feeds rich sends, final edits, and drafts."""

    def _payload_markdown(self, content):
        adapter = object.__new__(TelegramAdapter)
        return adapter._rich_message_payload(content)["markdown"]

    def test_schemeless_target_degrades_to_display_text(self):
        md = self._payload_markdown(
            "Already underway in [Example Session](Example Session)."
        )
        assert md == "Already underway in Example Session."

    def test_session_reference_degrades_to_display_text(self):
        md = self._payload_markdown("Continuing [Research](@session:default/abc123).")
        assert md == "Continuing Research."

    def test_https_link_stays_clickable(self):
        md = self._payload_markdown("See [Docs](https://example.com/x).")
        assert "[Docs](https://example.com/x)" in md

    def test_explicitly_bracketed_numeric_citation_keeps_complete_marker_visible(self):
        md = self._payload_markdown(
            "A grounded claim.[[3](https://example.com/source)]"
        )
        assert r"A grounded claim.[\[3\]](https://example.com/source)" in md

    def test_ordinary_numeric_pr_label_stays_an_ordinary_link(self):
        md = self._payload_markdown(
            "Reviewed in [97](https://github.com/acme/project/pull/97)."
        )
        assert r"[97](https://github.com/acme/project/pull/97)" in md
        assert r"[\[97\]]" not in md


class TestDegradeHelper:
    """The shared outbound scrub used by both delivery paths."""

    def test_text_without_brackets_untouched(self):
        assert _degrade_unsupported_markdown_links("plain text") == "plain text"

    def test_explicitly_bracketed_numeric_citation_becomes_one_clickable_marker(self):
        assert (
            _degrade_unsupported_markdown_links("[[1](https://example.com/source)]")
            == r"[\[1\]](https://example.com/source)"
        )

    def test_ordinary_numeric_link_does_not_become_a_citation_marker(self):
        text = "[1234567](https://github.com/acme/project/commit/1234567)"
        assert _degrade_unsupported_markdown_links(text) == text

    def test_empty_target_degrades(self):
        assert _degrade_unsupported_markdown_links("see [x]()") == "see x"

    def test_inline_code_span_untouched(self):
        assert _degrade_unsupported_markdown_links("see `[a](b)`") == "see `[a](b)`"

    def test_fenced_code_block_untouched(self):
        text = "intro\n```\n[a](b)\n```\n"
        assert _degrade_unsupported_markdown_links(text) == text

    def test_table_block_untouched(self):
        text = "| a | b |\n|---|---|\n| [x](y) | 2 |\n"
        assert _degrade_unsupported_markdown_links(text) == text

    def test_display_markers_survive_degrade(self):
        """Degrading must hand the raw label back to the renderer, so
        emphasis in the display text still renders."""
        assert (
            _degrade_unsupported_markdown_links("[**Bold**](not-a-url)") == "**Bold**"
        )
