"""A handoff row never stays in the protected tail, so it must not decide where the tail cut lands.

The handoff scan folds every row up to the newest in-transcript handoff into the summary window. When the
tail walk charged that handoff, a summary row that alone pushed the transcript over the soft ceiling ended
the walk on itself: the planned window held only the handoff and compress() returned the
``empty_post_handoff_window`` no-op at every trigger (live: four no-ops in a row, prompt 47k -> 66.7k on a
64k window, after a manual compress followed by one automatic compaction).
"""

from unittest.mock import patch

from agent.context_compressor import SUMMARY_PREFIX, ContextCompressor, _estimate_msg_budget_tokens


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=64_000):
        return ContextCompressor(
            model="test-model", quiet_mode=True, config_context_length=64_000, protect_first_n=3,
            protect_last_n=20, summary_target_ratio=0.20, threshold_percent=0.50, tail_mode="lean",
        )


def _handoff(body_words: int = 1_800) -> dict:
    return {"role": "user", "content": f"{SUMMARY_PREFIX}\n## Goal\n" + " ".join(["earlier"] * body_words),
            "_compressed_summary": True}


def _turns(n: int, words: int = 330) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({"role": "user", "content": f"question {i} " + " ".join(["ask"] * 40)})
        rows.append({"role": "assistant", "content": f"answer {i} " + " ".join(["reply"] * words)})
    return rows


def _over_ceiling_only_with_handoff(cc: ContextCompressor, head: list[dict]) -> list[dict]:
    """Rows after the handoff fit the soft ceiling; adding the handoff breaks it."""
    soft = cc._tail_soft_ceiling(cc.tail_token_budget)
    handoff = _handoff()
    for n in range(4, 80):
        turns = _turns(n)
        after = sum(map(_estimate_msg_budget_tokens, turns))
        if after <= soft < after + _estimate_msg_budget_tokens(handoff):
            return [*head, handoff, *turns]
    raise AssertionError("fixture could not straddle the soft ceiling")


def test_leading_handoff_does_not_collapse_the_window_to_itself():
    cc = _compressor()
    messages = _over_ceiling_only_with_handoff(cc, head=[])
    with patch.object(cc, "_generate_summary", return_value="NEW SUMMARY BODY"):
        result = cc.compress(messages, current_tokens=60_000)

    telemetry = cc._last_compression_telemetry or {}
    assert telemetry.get("failure_class") is None
    assert cc._last_compression_made_progress is True
    assert len(result) < len(messages)
    assert sum(1 for m in result if cc._is_context_summary_message(m)) == 1
    assert result[-1]["content"] == messages[-1]["content"]


def test_handoff_behind_a_protected_head_folds_the_turns_after_it():
    """Gateway manual compress keeps protect_first_n=3: the handoff sits at index 3 of the next transcript."""
    cc = _compressor()
    head = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second request"},
    ]
    messages = _over_ceiling_only_with_handoff(cc, head=head)
    start, end = cc._compress_window(messages)
    handoff_idx = len(head)
    assert start == 0
    # The window reaches past the old handoff into the turns after it, not just the rows in front of it.
    assert end > handoff_idx + 1
    assert cc.has_content_to_compress(messages) is True
