"""Tests for degenerate token-repetition detection + collapse.

Covers ``agent.message_sanitization.text_has_degenerate_repetition`` and
``collapse_degenerate_repetition`` — the storage-boundary scrubber that removes
"call call call …" style neural-text-degeneration spam from assistant content
before it is persisted (Claude Opus 4.x failure mode; see the leaked-tool-call
lesson #50279 for why persisting the run is harmful — it becomes a bad few-shot
that makes later turns imitate the degeneration).

Design intent under test:
- Conservative TRUE only on a long run of ONE short alphanumeric token
  separated solely by whitespace.
- Real prose, repeated list bullets, code, and short legitimate repeats below
  the threshold must stay FALSE / unchanged.
- Collapse preserves prose that precedes/follows the run and reduces the run to
  a single occurrence of the token.
"""

import pytest

from agent.message_sanitization import (
    text_has_degenerate_repetition,
    collapse_degenerate_repetition,
)


# --------------------------------------------------------------------------- #
# True positives — degenerate runs that MUST be detected and collapsed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text",
    [
        "call " * 8,                              # exact default threshold
        "call " * 100,                            # the real session magnitude
        "ok " + "call " * 60,                     # short legit prefix + spam
        "court " * 50,                            # observed 'court' variant
        "la " * 80,                               # observed 'la' variant
        "\n\n".join(["call"] * 40),               # newline-separated (real shape)
        "word " * 8 + "and then a real sentence",  # spam then prose
    ],
)
def test_detects_degenerate_runs(text):
    assert text_has_degenerate_repetition(text) is True


def test_real_session_shape_is_collapsed():
    """Reconstruction of session 0bb74aa5f045 msg 68307: preamble + 100x 'call'
    + a recovery sentence. After collapse the spam run is gone but both the
    preamble and the trailing sentence survive."""
    real = (
        "完全可以——cron 唤醒和我刚才的重启唤醒是同一个机制。\n\n"
        "courier 验证cron能不能稳定唤醒。\n\n"
        "courier\n\n" + ("call\n\n" * 100)
        + "I keep emitting empty filler. Let me make the actual tool call."
    )
    assert text_has_degenerate_repetition(real) is True
    cleaned, runs = collapse_degenerate_repetition(real)
    assert runs == 1
    # The preamble and the recovery sentence are preserved verbatim.
    assert "完全可以" in cleaned
    assert "Let me make the actual tool call." in cleaned
    # The 100x run is gone — only the single 'call' inside "tool call" plus the
    # one collapsed token remain (i.e. far fewer than the original 100).
    assert cleaned.count("call") <= 3
    assert len(cleaned) < len(real) / 2


def test_collapse_reduces_run_to_single_token():
    cleaned, runs = collapse_degenerate_repetition("alpha " + "call " * 30)
    assert runs == 1
    assert cleaned == "alpha call"


def test_multiple_distinct_runs_each_collapse():
    text = "intro " + "call " * 12 + "middle prose here " + "court " * 12 + "end"
    cleaned, runs = collapse_degenerate_repetition(text)
    assert runs == 2
    assert "intro" in cleaned and "middle prose here" in cleaned and "end" in cleaned
    assert cleaned.count("call") == 1
    assert cleaned.count("court") == 1


# --------------------------------------------------------------------------- #
# False positives — ordinary content that MUST stay untouched
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text",
    [
        "I will call the function and then call it again to verify the result.",
        "- item\n- item\n- item\n- item\n- item",          # repeated list bullets
        "no no no no, that's wrong",                         # short legit repeat
        "this is very very very very very very very good",   # 7 'very' < threshold
        "call " * 7,                                          # one below threshold
        "The word 'buffalo' repeated means buffalo buffalo buffalo here only thrice.",
        "```\nx = 1\nx = 1\nx = 1\nx = 1\nx = 1\nx = 1\nx = 1\nx = 1\n```",
        "",
        "   ",
    ],
)
def test_does_not_flag_ordinary_content(text):
    assert text_has_degenerate_repetition(text) is False
    cleaned, runs = collapse_degenerate_repetition(text)
    assert runs == 0
    assert cleaned == text  # exact no-op, content byte-for-byte preserved


def test_non_string_inputs_are_safe():
    assert text_has_degenerate_repetition(None) is False  # type: ignore[arg-type]
    assert text_has_degenerate_repetition(123) is False   # type: ignore[arg-type]
    cleaned, runs = collapse_degenerate_repetition(None)   # type: ignore[arg-type]
    assert cleaned is None and runs == 0


def test_threshold_boundary_is_exact():
    """7 repeats stays clean, 8 (the default minimum) fires."""
    assert text_has_degenerate_repetition("x " * 7) is False
    assert text_has_degenerate_repetition("x " * 8) is True


def test_custom_threshold_is_honored():
    text = "call " * 5
    assert text_has_degenerate_repetition(text, min_repeats=8) is False
    assert text_has_degenerate_repetition(text, min_repeats=5) is True


def test_different_tokens_interleaved_do_not_collapse():
    """A B A B … is not a single-token run — the backref requires identity."""
    text = "alpha beta " * 20
    assert text_has_degenerate_repetition(text) is False
    cleaned, runs = collapse_degenerate_repetition(text)
    assert runs == 0
    assert cleaned == text
