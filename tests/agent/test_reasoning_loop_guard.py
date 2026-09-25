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


# ---- shape 2: quote litter -----------------------------------------------------------------

# Sample of the litter shape: openers between tokens, almost never closed, no run anywhere
# (so only the litter rule can fire). Mirrors the corpus statistics of chronic sessions.
_LITTER_SEGMENT = "**「ワード「の「処理「確認!!!「= 「null「時「の「挙動!!!（「テスト」）**\n"


def _litter_text(segments: int) -> str:
    return _LITTER_SEGMENT * segments


def test_quote_litter_is_cut_near_where_openers_piled_up():
    text = _litter_text(24)
    guard = ReasoningLoopGuard()
    assert guard.feed(text) is True
    # Cut back to where the unmatched openers started piling up, not the littered tail.
    assert 0 < guard.trip_index < 200

    cleaned = sanitize_degenerate_reasoning(text)
    assert cleaned.endswith(THINKING_LOOP_TRUNCATED)
    assert len(cleaned) < len(text) // 2


def test_litter_needs_runway_before_firing():
    guard = ReasoningLoopGuard()
    assert guard.feed(_litter_text(10)) is False  # 450 chars: under the 600-char floor
    guard_2 = ReasoningLoopGuard()
    assert guard_2.feed(_litter_text(10) + _litter_text(6)) is True  # 720 chars


def test_litter_rule_ignores_analysis_quoting():
    # Healthy prose that QUOTES a litter excerpt: low overall unmatched-opener rate and
    # sparse dense bins must keep the guard quiet.
    prose = "観察結果を整理し、原因候補を列挙して切り分けを進める。" * 40
    text = prose + _litter_text(5) + prose
    guard = ReasoningLoopGuard()
    assert guard.feed(text) is False



# ---- shape 3: word loops -------------------------------------------------------------------

# Spread shape: an article wedged into nearly every slot with runs of only 2-4 — the shape
# chronic sessions drift into; the pure-run shape is what a collapsing stream ends in.
_WORD_SEGMENT = "the sts: the the the bot: the the the the (the the run): the "


def _word_text(segments: int) -> str:
    return _WORD_SEGMENT * segments


def test_word_loop_is_cut_near_where_the_word_started_swelling():
    text = _word_text(16)
    guard = ReasoningLoopGuard()
    assert guard.feed(text) is True
    # Cut back to where the dominance began, not the littered tail.
    assert 0 < guard.trip_index < 400

    cleaned = sanitize_degenerate_reasoning(text)
    assert cleaned.endswith(THINKING_LOOP_TRUNCATED)
    assert len(cleaned) < len(text) // 2


def test_word_rule_needs_dominance_and_volume():
    # Repeated non-cue words inside plenty of ordinary prose must not trip: ``Down Down``
    # command sequences, boolean arrays (excluded words) and xx-dumps (excluded placeholders)
    # are all real false-positive shapes found and silenced during corpus calibration.
    text = (
        "xdotool key --clearmodifiers " + "Down " * 12
        + " sleep 2 xdotool key --clearmodifiers " + "Down " * 12
        + " seconds and the review of theta kappa lambda sigma omega upsilon "
        "selected := [" + ", ".join(["False"] * 14) + ", True, True] with pattern "
        "XX XX XX XX XX XX XX XX end"
    )
    guard = ReasoningLoopGuard()
    assert guard.feed(text) is False
    assert sanitize_degenerate_reasoning(text) is text


def test_word_rule_ignores_quoting_a_degenerate_sample():
    # An analysis message embedding a quoted word-loop sample: the quoted run is long, but the
    # words never dominate the analyzing text (the measured healthy-row shape).
    prose = "提供された崩壊サンプルを検討する。原因は推論チャネルの循環であり、記録して対策を検討する。" * 20
    filler = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi "
        "rho sigma tau upsilon phi chi psi omega "
    ) * 4
    text = prose + "サンプル: " + "the " * 30 + filler + "以上のような崩壊が見られる。" + prose
    guard = ReasoningLoopGuard()
    assert guard.feed(text) is False


def test_word_rule_chunked_feeding_matches_single_feed():
    # Word runs crossing delta boundaries must count as one run: a pending word is carried
    # across feeds, so the trip lands at the same offset as a single feed of the whole text.
    text = _word_text(16)
    single = ReasoningLoopGuard()
    assert single.feed(text) is True
    chunked = ReasoningLoopGuard()
    tripped = False
    for k in range(0, len(text), 37):
        if chunked.feed(text[k:k + 37]):
            tripped = True
            break
    assert tripped is True
    assert chunked.trip_index == single.trip_index


def test_word_rule_non_cue_words_need_a_runaway_run():
    # A modest repeat of a non-cue word is routine; only a >=25 run of it trips.
    guard = ReasoningLoopGuard()
    assert guard.feed(
        "await " * 12
        + "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 4) is False
    guard_2 = ReasoningLoopGuard()
    assert guard_2.feed(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 2
        + "hello " * 30) is True
