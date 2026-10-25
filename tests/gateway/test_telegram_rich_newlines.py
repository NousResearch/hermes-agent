"""Tests for rich-message newline normalization (issue #46070).

When Bot API 10.1 ``sendRichMessage`` is available, slash-command responses
are sent through the rich path with RAW markdown.  Standard Markdown treats
a lone ``\\n`` as a soft line break (renders as whitespace), so multi-line
command output collapses into a single paragraph on Telegram.

``_rich_message_payload`` must normalize single newlines to Markdown hard
breaks (two trailing spaces + ``\\n``) so they render as visible line breaks.
Paragraph breaks (``\\n\\n``) and fenced code blocks must be preserved.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``, so these
tests construct a real ``TelegramAdapter``.
"""

import pytest
from markdown_it import MarkdownIt

from plugins.platforms.telegram.adapter import TelegramAdapter, _rich_separate_ordered_lists


@pytest.fixture()
def adapter():
    """Bare adapter instance — _rich_message_payload doesn't use self."""
    return object.__new__(TelegramAdapter)


class TestRichMessageNewlineNormalization:
    """Verify _rich_message_payload normalizes single \\n to hard breaks."""

    def test_single_newlines_become_hard_breaks(self, adapter):
        """A lone \\n must gain two trailing spaces (Markdown hard break).

        Standard Markdown soft-break rendering causes Bot API 10.1
        ``sendRichMessage`` to collapse multi-line content into one paragraph.
        """
        content = "Line 1\nLine 2\nLine 3"
        payload = adapter._rich_message_payload(content)
        md = payload["markdown"]
        # Each single \n should now be "  \n" (two spaces + newline)
        assert "  \n" in md, f"Expected hard break '  \\n' in {md!r}"
        assert "Line 1  \nLine 2  \nLine 3" == md

    def test_paragraph_breaks_preserved(self, adapter):
        """Double newlines (paragraph breaks) must NOT gain extra spaces."""
        content = "Paragraph 1\n\nParagraph 2"
        payload = adapter._rich_message_payload(content)
        md = payload["markdown"]
        # \n\n should remain as-is — no trailing spaces injected
        assert "Paragraph 1\n\nParagraph 2" == md



class TestRichMessageTableProtection:
    """Hard-break injection must not corrupt GFM tables (rendered natively)."""

    def test_table_rows_keep_bare_newlines(self, adapter):
        """Table block newlines must stay bare — no '  \\n' inside the table."""
        content = "| Col A | Col B |\n|-------|-------|\n| 1 | 2 |\n| 3 | 4 |"
        md = adapter._rich_message_payload(content)["markdown"]
        assert "  \n" not in md
        assert md == content


class TestRichOrderedListAfterProse:
    """CommonMark lets an ordered list interrupt a paragraph only when it starts at 1, so
    ``Label\\n7. item`` stays one paragraph and the numbers render as literal text (#124552)."""

    @pytest.mark.parametrize(
        "content, items",
        [
            ("**Facts and Evidence**\n7. **Verbs.** Is.\n8. **Dates.** As of.", 2),
            ("Intro\n\nNext steps:\n2) second\n3) third\n4) fourth", 3),
        ],
    )
    def test_ordered_list_after_prose_parses_as_list(self, adapter, content, items):
        md = adapter._rich_message_payload(content)["markdown"]
        types = [t.type for t in MarkdownIt().parse(md)]

        assert types.count("ordered_list_open") == 1
        assert types.count("list_item_open") == items

    def test_ordered_list_keeps_authored_start_number(self, adapter):
        md = adapter._rich_message_payload("**Facts and Evidence**\n7. First\n8. Second")["markdown"]
        lists = [t for t in MarkdownIt().parse(md) if t.type == "ordered_list_open"]

        assert lists and lists[0].attrs.get("start") == 7

    @pytest.mark.parametrize(
        "content",
        [
            "1. a\n2. b\n3. c",
            "- a\n7. b",
            "1. a\n   continued\n2. b",
            "```text\nLabel\n7. literal\n```",
            "Streaming draft\n```text\nLabel\n7. literal",
        ],
    )
    def test_lists_and_code_gain_no_blank_line(self, adapter, content):
        md = adapter._rich_message_payload(content)["markdown"]

        assert md.count("\n\n") == content.count("\n\n")

    def test_separation_is_idempotent(self):
        once = _rich_separate_ordered_lists("Label\n7. item\n8. another")

        assert _rich_separate_ordered_lists(once) == once

