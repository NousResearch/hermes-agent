"""Behavior contracts for the thinking-channel loop guard (ReasoningLoopGuard).

Regression context: thinking streams degenerated into growing char runs —
「「「「「「実行」」」」」」 with the run length exploding line over line — while the visible
reply stayed healthy. The visible-text repetition guard never saw the shape (it needs
60-char exact repeats on truncation paths), and providers that require a
``reasoning_content`` echo on tool-call replays replayed the looped bytes into the next
request, re-seeding the pattern on every following turn.
"""

from agent.repetition_guard import (
    ReasoningLoopGuard,
    THINKING_LOOP_TRUNCATED,
    sanitize_degenerate_reasoning,
)

# Trimmed tail of a captured degenerate thinking stream.
DEGENERATE_TAIL = (
    "了解した。次のステップを進める。\n\n"
    "**「「「「「「「「「「「「「「「「やる」」」」」」」」」」」」」」」」\n\n"
    "**。\n\n**「「「「「「「「「「「「「「「「「「「「「「「「OK実行」」」」」」」」」」」」」」」」」」」」」」」**。\n"
)


def test_degenerate_run_is_cut_at_run_start():
    text = "手順を確認する。\n" + "「" * 30 + "実行" + "」" * 30 + "\n続き"
    guard = ReasoningLoopGuard()
    assert guard.feed(text) is True
    assert guard.trip_index == len("手順を確認する。\n")

    cleaned = sanitize_degenerate_reasoning(text)
    assert cleaned == "手順を確認する。" + "\n\n" + THINKING_LOOP_TRUNCATED
    # A stream that is degenerate from its first character yields the marker alone.
    assert sanitize_degenerate_reasoning("「" * 20 + "実行") == THINKING_LOOP_TRUNCATED


def test_incremental_feeding_matches_single_feed():
    one = ReasoningLoopGuard()
    one.feed(DEGENERATE_TAIL)

    chunked = ReasoningLoopGuard()
    for i in range(0, len(DEGENERATE_TAIL), 7):
        if chunked.feed(DEGENERATE_TAIL[i : i + 7]):
            break

    assert one.tripped is True
    assert chunked.tripped is True
    assert one.trip_index == chunked.trip_index


def test_healthy_reasoning_never_trips():
    # Shapes that occur in healthy reasoning and must not fire: code rule lines,
    # indentation, placeholder tokens, hex literals, dense quoted-word lists,
    # box-drawing rules, decoration glyph runs.
    normal = (
        "```\n" + "-" * 60 + "\n" + "=" * 40 + "\n" + " " * 26 + "x\n```\n"
        "Bearer " + "A" * 21 + "\n"
        "0x" + "0" * 40 + "\n"
        "引用語リスト: 「神」「天才」「最高」「すごい」「好き」「嬉しい」「楽しい」「面白い」\n"
        "コード: " + "─" * 50 + " 罫線 " + "━" * 30 + "\n"
        + "✗" * 60 + " 採点マーク\n"
    )
    guard = ReasoningLoopGuard()
    assert guard.feed(normal) is False
    assert sanitize_degenerate_reasoning(normal) is normal
