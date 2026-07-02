"""Unit tests for agent.manual_compression_feedback — classic + enhanced modes.

The enhanced mode is the 2026-07-02 honesty fix: the gateway /compress path
compresses a chat-only projection of the stored transcript and (on rewrite)
drops all non-chat rows (tool results / system). The classic helper only
compared the chat axis, so a tool-heavy session got "No changes from
compression: 179 messages" printed directly above a "453,542 → ~32,036"
full-request line — a self-contradicting message.
"""

from agent.manual_compression_feedback import summarize_manual_compression


def _chat(n: int) -> list[dict]:
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"m{i}"})
    return out


# ---------------------------------------------------------------------------
# Classic mode (no kwargs) — byte-identical to original behavior
# ---------------------------------------------------------------------------


def test_classic_noop_headline_unchanged():
    msgs = _chat(4)
    s = summarize_manual_compression(msgs, list(msgs), 100, 100)
    assert s["noop"] is True
    assert s["headline"] == "No changes from compression: 4 messages"
    assert s["token_line"] == "Approx request size: ~100 tokens (unchanged)"
    assert s["note"] is None
    assert s["enhanced"] is False
    assert s["chat_line"] is None
    assert s["dropped_line"] is None


def test_classic_compressed_headline_unchanged():
    before = _chat(4)
    after = [before[0], {"role": "assistant", "content": "summary"}, before[-1]]
    s = summarize_manual_compression(before, after, 100, 60)
    assert s["noop"] is False
    assert s["headline"] == "Compressed: 4 → 3 messages"
    assert s["token_line"] == "Approx request size: ~100 → ~60 tokens"


def test_classic_denser_summary_note_unchanged():
    before = _chat(4)
    after = [before[0], {"role": "assistant", "content": "dense"}, before[-1]]
    s = summarize_manual_compression(before, after, 100, 120)
    assert s["note"] is not None
    assert "denser summaries" in s["note"]


def test_zero_non_chat_rows_falls_back_to_classic():
    """A pure-chat transcript (non_chat_count == 0) must use the classic
    wording even when the enhanced kwargs are supplied — there is no
    dropped-rows story to tell."""
    msgs = _chat(4)
    s = summarize_manual_compression(
        msgs,
        list(msgs),
        100,
        100,
        non_chat_count=0,
        non_chat_tokens=0,
        transcript_rewritten=True,
        full_before_count=4,
    )
    assert s["enhanced"] is False
    assert s["headline"] == "No changes from compression: 4 messages"


# ---------------------------------------------------------------------------
# Enhanced mode
# ---------------------------------------------------------------------------


def test_case_a_rewrite_with_unchanged_chat_reports_compaction():
    """CASE A — the live bug: chat already compact, but the rewrite dropped a
    large stored tool/system tail. Headline must report the compaction, NOT
    'no changes'."""
    msgs = _chat(6)
    s = summarize_manual_compression(
        msgs,
        list(msgs),  # chat unchanged
        31_406,
        31_406,
        non_chat_count=133,
        non_chat_tokens=420_100,
        transcript_rewritten=True,
        full_before_count=139,
    )
    assert s["enhanced"] is True
    assert s["noop"] is False
    assert s["headline"] == "Compacted stored transcript: 139 → 6 messages"
    assert "No changes" not in s["headline"]
    assert s["chat_line"] == (
        "Chat: 6 messages (~31,406 tokens) — already compact, kept verbatim"
    )
    assert s["dropped_line"] == (
        "Dropped: 133 stored tool/system messages (~420,100 tokens reclaimed)"
    )


def test_case_b_rewrite_with_compressed_chat_reports_both_axes():
    before = _chat(6)
    after = [before[0], {"role": "assistant", "content": "summary"}, before[-1]]
    s = summarize_manual_compression(
        before,
        after,
        10_000,
        3_000,
        non_chat_count=50,
        non_chat_tokens=200_000,
        transcript_rewritten=True,
        full_before_count=56,
    )
    assert s["enhanced"] is True
    assert s["noop"] is False
    assert s["headline"] == "Compressed: 56 → 3 stored messages"
    assert "Chat: 6 → 3 messages" in s["chat_line"]
    assert "~10,000 → ~3,000 tokens" in s["chat_line"]
    assert "Dropped: 50 stored tool/system messages" in s["dropped_line"]


def test_case_c_no_rewrite_reports_preserved_transcript():
    """CASE C — nothing was rewritten: the message must NOT imply any rows
    were dropped, and must account for the full stored composition."""
    msgs = _chat(6)
    s = summarize_manual_compression(
        msgs,
        list(msgs),
        31_406,
        31_406,
        non_chat_count=133,
        non_chat_tokens=420_100,
        transcript_rewritten=False,
        full_before_count=139,
    )
    assert s["enhanced"] is True
    assert s["noop"] is True
    assert s["headline"] == (
        "No changes: transcript preserved (139 messages: 6 chat + 133 tool/system)"
    )
    assert s["dropped_line"] is None
    assert "unchanged" in s["chat_line"]


def test_case_c_even_when_chat_would_have_compressed():
    """If the compressor produced a smaller chat list but the store was NOT
    rewritten (abort path), the summary must still be CASE C — the next
    request resends the original transcript."""
    before = _chat(6)
    after = [before[0], {"role": "assistant", "content": "s"}, before[-1]]
    s = summarize_manual_compression(
        before,
        after,
        10_000,
        3_000,
        non_chat_count=20,
        non_chat_tokens=50_000,
        transcript_rewritten=False,
        full_before_count=26,
    )
    assert s["noop"] is True
    assert "transcript preserved" in s["headline"]
    assert s["dropped_line"] is None


def test_case_c_token_line_never_implies_change():
    """Greptile P2 regression: in CASE C the helper itself must force the
    'unchanged' token wording even when the caller passes divergent
    before/after token figures (the compressor produced a smaller in-memory
    list that never reached the store)."""
    before = _chat(6)
    after = [before[0], {"role": "assistant", "content": "s"}, before[-1]]
    s = summarize_manual_compression(
        before,
        after,
        10_000,
        3_000,  # divergent after — must NOT be rendered as a range
        non_chat_count=20,
        non_chat_tokens=50_000,
        transcript_rewritten=False,
        full_before_count=26,
    )
    assert s["token_line"] == "Approx request size: ~10,000 tokens (unchanged)"
    assert s["note"] is None


def test_enhanced_requires_non_chat_tokens():
    """Greptile P2 regression: the all-or-nothing gate must include
    non_chat_tokens — otherwise the dropped line silently reads '~0 tokens
    reclaimed'."""
    msgs = _chat(4)
    s = summarize_manual_compression(
        msgs,
        list(msgs),
        100,
        100,
        non_chat_count=10,
        # non_chat_tokens omitted
        transcript_rewritten=True,
        full_before_count=14,
    )
    assert s["enhanced"] is False
    assert s["dropped_line"] is None
