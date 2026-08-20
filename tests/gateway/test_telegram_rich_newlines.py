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

from plugins.platforms.telegram.adapter import TelegramAdapter


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

    def test_mixed_single_and_double_newlines(self, adapter):
        """Content with both list items and paragraph breaks must be handled correctly."""
        content = (
            "Header\n\n"
            "`/new` -- Start\n"
            "`/model` -- Switch\n"
            "`/reset` -- Reset\n\n"
            "Footer"
        )
        payload = adapter._rich_message_payload(content)
        md = payload["markdown"]
        # Paragraph breaks preserved
        assert "Header\n\n" in md
        assert "\n\nFooter" in md
        # Single newlines converted to hard breaks
        assert "`/new` -- Start  \n`/model` -- Switch  \n`/reset` -- Reset" in md


class TestRichMessageEmailEntityProtection:
    """Email-bearing content must auto-enable skip_entity_detection
    (issue #68754 — RICH_MESSAGE_EMAIL_INVALID)."""

    def test_provider_prefixed_email_triggers_skip(self, adapter):
        """Provider-prefixed email like ``openai:test@example.com`` must set
        skip_entity_detection to prevent Telegram rejecting the rich message."""
        content = "OAuth profile: openai:test.user@example.com (test.user@example.com)"
        payload = adapter._rich_message_payload(content)
        assert payload.get("skip_entity_detection") is True

    def test_normal_email_triggers_skip(self, adapter):
        """Even a bare email address (no provider prefix) should trigger
        skip_entity_detection because Telegram auto-detects an email entity."""
        content = "Contact support@example.com for help"
        payload = adapter._rich_message_payload(content)
        assert payload.get("skip_entity_detection") is True

    def test_multiple_emails_triggers_skip(self, adapter):
        """Content with multiple @ signs should still trigger the check."""
        content = "user1@a.com and user2@b.com"
        payload = adapter._rich_message_payload(content)
        assert payload.get("skip_entity_detection") is True

    def test_no_email_does_not_skip(self, adapter):
        """Content without @ should NOT set skip_entity_detection."""
        content = "Hello, this is a normal message."
        payload = adapter._rich_message_payload(content)
        assert payload.get("skip_entity_detection") is None

    def test_skip_still_works_without_email(self, adapter):
        """Explicit skip_entity_detection=True must still work on non-email content."""
        content = "Just plain text"
        payload = adapter._rich_message_payload(content, skip_entity_detection=True)
        assert payload.get("skip_entity_detection") is True

    def test_email_pattern_in_code_block(self, adapter):
        """Content with @ inside a code block should still trigger skip."""
        content = "Run:\n```\npip install package==1.0@beta\n```"
        payload = adapter._rich_message_payload(content)
        assert payload.get("skip_entity_detection") is True

    def test_content_has_email_pattern_static(self, adapter):
        """The static helper should correctly identify @ presence."""
        assert adapter._content_has_email_pattern("test@example.com") is True
        assert adapter._content_has_email_pattern("openai:test@example.com") is True
        assert adapter._content_has_email_pattern("no email here") is False
        assert adapter._content_has_email_pattern("") is False
        assert adapter._content_has_email_pattern("@") is True


class TestRichMessageTableProtection:
    """Hard-break injection must not corrupt GFM tables (rendered natively)."""

    def test_table_rows_keep_bare_newlines(self, adapter):
        """Table block newlines must stay bare — no '  \\n' inside the table."""
        content = "| Col A | Col B |\n|-------|-------|\n| 1 | 2 |\n| 3 | 4 |"
        md = adapter._rich_message_payload(content)["markdown"]
        assert "  \n" not in md
        assert md == content

