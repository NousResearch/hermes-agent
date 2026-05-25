"""Tests for agent.manual_compression_feedback — summarize_manual_compression()."""

from __future__ import annotations

import pytest

from agent.manual_compression_feedback import summarize_manual_compression


# ============================================================================
# No-op path (before == after)
# ============================================================================
class TestNoop:
    """When after_messages == before_messages, noop=True."""

    def test_noop_identical_messages_same_tokens(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = summarize_manual_compression(msgs, msgs, 100, 100)
        assert result["noop"] is True
        assert result["headline"] == "No changes from compression: 1 messages"
        assert "unchanged" in result["token_line"]
        assert result["note"] is None

    def test_noop_identical_messages_different_tokens(self):
        """Tokens differ but messages identical — still noop with token delta shown."""
        msgs = [{"role": "user", "content": "hello"}]
        result = summarize_manual_compression(msgs, msgs, 100, 80)
        assert result["noop"] is True
        assert "unchanged" not in result["token_line"]
        assert "~100 → ~80" in result["token_line"]
        assert result["note"] is None

    def test_noop_empty_messages(self):
        result = summarize_manual_compression([], [], 0, 0)
        assert result["noop"] is True
        assert result["headline"] == "No changes from compression: 0 messages"

    def test_noop_multiple_identical(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        result = summarize_manual_compression(msgs, list(msgs), 500, 500)
        assert result["noop"] is True
        assert "3 messages" in result["headline"]


# ============================================================================
# Compression path (after != before)
# ============================================================================
class TestCompressed:
    """When after_messages != before_messages, noop=False."""

    def test_compressed_fewer_messages_fewer_tokens(self):
        before = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        after = [{"role": "user", "content": "summary"}]
        result = summarize_manual_compression(before, after, 1000, 200)
        assert result["noop"] is False
        assert result["headline"] == "Compressed: 10 → 1 messages"
        assert "~1,000 → ~200" in result["token_line"]
        assert result["note"] is None

    def test_compressed_same_count_different_content(self):
        """Different content but same message count — still 'compressed'."""
        before = [{"role": "user", "content": "long text here"}]
        after = [{"role": "user", "content": "short"}]
        result = summarize_manual_compression(before, after, 50, 20)
        assert result["noop"] is False
        assert result["headline"] == "Compressed: 1 → 1 messages"

    def test_compressed_more_messages_but_fewer_tokens(self):
        """Rare: more messages but fewer tokens total."""
        before = [{"role": "user", "content": "x" * 100}]
        after = [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        result = summarize_manual_compression(before, after, 200, 50)
        assert result["noop"] is False
        assert result["headline"] == "Compressed: 1 → 2 messages"


# ============================================================================
# The "denser summary" note
# ============================================================================
class TestDenserNote:
    """When fewer messages but MORE tokens (compression rewrote into denser summaries)."""

    def test_note_appears_when_fewer_messages_but_more_tokens(self):
        before = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        after = [{"role": "user", "content": "A very long detailed summary of everything"}]
        result = summarize_manual_compression(before, after, 100, 300)
        assert result["noop"] is False
        assert result["note"] is not None
        assert "fewer messages can still raise" in result["note"]

    def test_note_absent_when_fewer_messages_and_fewer_tokens(self):
        before = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        after = [{"role": "user", "content": "short"}]
        result = summarize_manual_compression(before, after, 500, 100)
        assert result["note"] is None

    def test_note_absent_when_same_or_more_messages(self):
        """Note only triggers when after_count < before_count."""
        before = [{"role": "user", "content": "a"}]
        after = [{"role": "user", "content": "b"}, {"role": "user", "content": "c"}]
        result = summarize_manual_compression(before, after, 50, 200)
        # after_count (2) > before_count (1) — note should be None
        assert result["note"] is None

    def test_note_absent_when_same_messages(self):
        msgs = [{"role": "user", "content": "x"}]
        result = summarize_manual_compression(msgs, msgs, 50, 200)
        # noop=True, note stays None
        assert result["noop"] is True
        assert result["note"] is None


# ============================================================================
# Token formatting (thousands separator)
# ============================================================================
class TestTokenFormatting:
    def test_thousands_separator(self):
        result = summarize_manual_compression(
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
            1234567,
            987654,
        )
        assert "~1,234,567 → ~987,654" in result["token_line"]

    def test_zero_tokens(self):
        result = summarize_manual_compression([], [], 0, 0)
        # noop, same tokens → (unchanged) variant
        assert "~0 tokens (unchanged)" in result["token_line"]

    def test_single_digit_tokens(self):
        result = summarize_manual_compression(
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
            5,
            3,
        )
        assert "~5 → ~3" in result["token_line"]


# ============================================================================
# Return structure
# ============================================================================
class TestReturnStructure:
    def test_has_all_keys(self):
        result = summarize_manual_compression([], [], 0, 0)
        assert set(result.keys()) == {"noop", "headline", "token_line", "note"}

    def test_noop_is_bool(self):
        result = summarize_manual_compression([], [{"role": "user", "content": "x"}], 0, 0)
        assert isinstance(result["noop"], bool)

    def test_headline_is_string(self):
        result = summarize_manual_compression([], [], 0, 0)
        assert isinstance(result["headline"], str)

    def test_token_line_is_string(self):
        result = summarize_manual_compression([], [], 0, 0)
        assert isinstance(result["token_line"], str)

    def test_headline_always_contains_message_count(self):
        result = summarize_manual_compression(
            [{"role": "user", "content": "hi"}],
            [{"role": "user", "content": "hi"}],
            0, 0,
        )
        assert "1 messages" in result["headline"]


# ============================================================================
# Edge cases
# ============================================================================
class TestEdgeCases:
    def test_list_equality_not_by_identity(self):
        """Same content, different list objects → still noop."""
        a = [{"role": "user", "content": "hi"}]
        b = [{"role": "user", "content": "hi"}]  # equal but not same object
        result = summarize_manual_compression(a, b, 10, 10)
        assert result["noop"] is True

    def test_different_role_same_content(self):
        """Different role makes messages different → compressed."""
        before = [{"role": "user", "content": "hi"}]
        after = [{"role": "assistant", "content": "hi"}]
        result = summarize_manual_compression(before, after, 10, 10)
        assert result["noop"] is False

    def test_empty_message_dicts(self):
        before = [{}]
        after = [{}]
        result = summarize_manual_compression(before, after, 10, 10)
        assert result["noop"] is True

    def test_extra_key_in_message(self):
        """Extra keys make messages different → noop=False."""
        before = [{"role": "user", "content": "hi"}]
        after = [{"role": "user", "content": "hi", "extra": True}]
        result = summarize_manual_compression(before, after, 10, 10)
        assert result["noop"] is False
