"""Reasoning dedup carries overlap normalization through main and Relay collectors."""

from agent.chat_completion_helpers_relay import RelayChatAccumulator
from agent.reasoning_summaries import ReasoningDeltaAccumulator


def _relay(deltas):
    accumulator = RelayChatAccumulator()
    for delta in deltas:
        accumulator.observe({"choices": [{"delta": {"reasoning_content": delta}}]})
    return accumulator.finalize()["choices"][0]["message"]["reasoning_content"]


def test_cumulative_snapshots_keep_only_new_suffix():
    parts = ["A sufficiently long provider summary", "A sufficiently long provider summary plus new", "A sufficiently long provider summary plus new"]
    accumulator = ReasoningDeltaAccumulator()
    callbacks = [delta for part in parts if (delta := accumulator.feed(part))]
    expected = "A sufficiently long provider summary plus new"
    assert "".join(callbacks) == expected
    assert "".join(accumulator.parts) == expected
    assert _relay(parts) == expected


def test_long_suffix_echo_dedup_preserves_short_repeated_words():
    repeated = "another substantial reasoning section"
    assert _relay([repeated, repeated[-24:] + " and fresh detail"]) == repeated + " and fresh detail"
    accumulator = ReasoningDeltaAccumulator()
    assert [accumulator.feed(part) for part in ("the", "the")] == ["the", "the"]


def test_summary_echoes_preserve_heading_boundaries_and_new_details():
    first = "**Inspecting the first summary part**"
    second = "**Checking the second summary part**"
    parts = [first, first + second, first + second + " tail"]
    accumulator = ReasoningDeltaAccumulator()
    actual = "".join(accumulator.feed(part) for part in parts)
    assert actual == first + "\n\n" + second + " tail"
    assert _relay(parts) == actual


def test_exact_short_echo_and_genuine_short_repetition_are_preserved():
    assert _relay(["the", "the"]) == "thethe"
    assert _relay(["the", "the so"]) == "thethe so"
    assert _relay(["ordinary prose is not an echo.", "ordinary prose is not an echo."]) == "ordinary prose is not an echo."
