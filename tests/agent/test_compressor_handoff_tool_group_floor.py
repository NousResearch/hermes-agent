"""A tail cut that lands inside the tool group right after a handoff must not collapse the window onto the handoff.

After one compaction the transcript is ``[handoff, assistant(tool_calls) + results, ...]``. When the raw tail budget
breaks inside that first tool group, aligning the cut backward to the group's assistant leaves a window holding only
the handoff, which the handoff scan drops: compress() returned the ``empty_post_handoff_window`` no-op and the session
stayed over the trigger with no warning (C4 e2e ``test_good_summarizer_compacts_and_keeps_every_invariant[11]``).
"""

from unittest.mock import patch

from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, SUMMARY_PREFIX, ContextCompressor


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=64_000):
        cc = ContextCompressor(
            model="test-model", quiet_mode=True, config_context_length=64_000, protect_first_n=3,
            protect_last_n=4, summary_target_ratio=0.20, threshold_percent=0.50, tail_mode="lean",
        )
    cc.tail_token_budget = 6_000
    return cc


def _calls(*ids: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": i, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "f.txt"}'}} for i in ids
    ]}


def _result(call_id: str, tokens: int) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": "w " * (tokens * 2)}


def _transcript() -> list[dict]:
    handoff = {"role": "user", "content": f"{SUMMARY_PREFIX}\n## Goal\n" + "earlier " * 380,
               COMPRESSED_SUMMARY_METADATA_KEY: True}
    return [
        handoff,
        _calls("a", "b", "c"), _result("a", 50), _result("b", 1_160), _result("c", 420),
        _calls("d", "e"), _result("d", 1_340), _result("e", 340),
        {"role": "assistant", "content": "answer for the previous ask"},
        {"role": "user", "content": "please read the next file and report"},
        _calls("f"), _result("f", 1_770),
        _calls("g"), _result("g", 1_670),
    ]


def test_cut_inside_the_first_tool_group_still_compacts_past_the_handoff():
    cc = _compressor()
    messages = _transcript()
    with patch.object(cc, "_generate_summary", return_value="NEW SUMMARY BODY"):
        result = cc.compress(messages, current_tokens=12_800)

    assert (cc._last_compression_telemetry or {}).get("failure_class") is None
    assert cc._last_compression_made_progress is True
    assert len(result) < len(messages)
    assert sum(1 for m in result if cc._is_context_summary_message(m)) == 1
    # The active ask and everything after it stay verbatim in the tail.
    assert [m.get("content") for m in result[-5:]] == [m.get("content") for m in messages[-5:]]


def test_first_tool_group_right_before_the_active_request_still_compacts():
    """The floor lands on the active request; the bridge rule must not pull the cut back onto the handoff."""
    cc = _compressor()
    ask = "please read the next file and report"
    messages = [
        _transcript()[0],
        _calls("a", "b"), _result("a", 1_200), _result("b", 1_300),
        {"role": "user", "content": ask},
        _calls("f"), _result("f", 1_770), _calls("g"), _result("g", 1_670), _calls("h"), _result("h", 1_500),
    ]
    with patch.object(cc, "_generate_summary", return_value="NEW SUMMARY BODY"):
        result = cc.compress(messages, current_tokens=12_800)

    assert (cc._last_compression_telemetry or {}).get("failure_class") is None
    assert cc._last_compression_made_progress is True
    assert len(result) < len(messages)
    assert sum(1 for m in result if cc._is_context_summary_message(m)) == 1
    # The active ask survives verbatim (after the summary's end marker) and the rounds after it stay in the tail.
    assert result[0]["content"].endswith(ask)
    assert [m.get("content") for m in result[1:]] == [m.get("content") for m in messages[5:]]
