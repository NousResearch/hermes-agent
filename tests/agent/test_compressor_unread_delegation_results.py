"""Unread async-delegation results stay verbatim across compaction.

Several background children finishing together append one user-role result row each after the
parent's last reply. None of them has been in a request the model answered yet, so a summary of
them is the only copy the parent would ever see.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools.process_registry_notifications import format_process_notification


@pytest.fixture()
def compressor():
    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
        return ContextCompressor(model="test/model", quiet_mode=True)


def _result_row(k: int, chars: int) -> dict:
    text = format_process_notification({
        "type": "async_delegation", "delegation_id": f"deleg-{k}", "goal": f"audit part {k}", "status": "completed",
        "summary": f"REPORT-{k} " + "finding. " * (chars // 9), "api_calls": 3, "duration_seconds": 10,
        "completed_at": 1.0, "dispatched_at": 0.5,
    })
    return {"role": "user", "content": text, "display_kind": "async_delegation_complete"}


def _transcript(n_results: int, chars: int) -> tuple[list[dict], int]:
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(30):
        msgs += [{"role": "user", "content": f"human turn {i} " + "u" * 3000},
                 {"role": "assistant", "content": f"assistant reply {i} " + "a" * 3000}]
    msgs += [{"role": "user", "content": "fan out the audit"},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"id": "t1", "type": "function", "function": {"name": "delegate_task", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "t1", "content": "dispatched"},
             {"role": "assistant", "content": f"dispatched {n_results} background subagents"}]
    first_result = len(msgs)
    msgs += [_result_row(k, chars) for k in range(n_results)]
    msgs.append({"role": "user", "content": "ok, what did they find?"})
    return msgs, first_result


def test_every_unread_result_at_the_default_summary_cap_stays_in_the_tail(compressor):
    # Six children at the default 24K-char summary cap: the over-ceiling guards used to cut inside the
    # run, so results 0-3 went to the summarizer and only 4-5 reached the parent verbatim.
    msgs, first_result = _transcript(6, 23_000)
    head_end = compressor._align_boundary_forward(msgs, compressor._protect_head_size(msgs))

    cut = compressor._find_tail_cut_by_tokens(msgs, head_end)

    assert cut <= first_result
    assert head_end < cut  # the older conversation still compacts


def test_unread_results_past_the_tail_share_of_the_trigger_are_not_all_pinned(compressor):
    # A pile larger than the tail's share of the trigger keeps the newest results; pinning all of them
    # would leave a tail that alone overflows the threshold and compaction could never make progress.
    from agent.context_compressor_tail import TAIL_MAX_THRESHOLD_FRACTION

    msgs, first_result = _transcript(40, 23_000)
    head_end = compressor._align_boundary_forward(msgs, compressor._protect_head_size(msgs))

    cut = compressor._find_tail_cut_by_tokens(msgs, head_end)

    price = compressor._tail_row_pricer(msgs)
    pinned = sum(map(price, range(cut, len(msgs))))
    assert first_result < cut < len(msgs) - 1
    assert pinned <= compressor.threshold_tokens * TAIL_MAX_THRESHOLD_FRACTION
