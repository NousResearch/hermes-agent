"""A carried tool round must survive or reach the next summarizer (#123625)."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


@pytest.mark.parametrize("tail_mode", ["legacy", "lean"])
def test_back_to_back_compress_accounts_for_every_carried_tool_round(tail_mode):
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(
            model="test/model", quiet_mode=True, protect_first_n=3,
            protect_last_n=4, tail_mode=tail_mode,
        )
    compressor.tail_token_budget = 100
    messages = [
        {"role": "system", "content": "Keep the system prompt."},
        {"role": "user", "content": "Inspect all the files."},
        {"role": "assistant", "content": "I will inspect them."},
        {"role": "user", "content": "Report the findings."},
    ]
    for i in range(16):
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"fixture.txt"}'},
            }]},
            {"role": "tool", "tool_call_id": f"call_{i}", "content": f"finding-{i} " + "data " * 100},
            {"role": "assistant", "content": f"Read file {i}."},
            {"role": "user", "content": f"Continue with file {i + 1}."},
        ])
    seen = []

    def summarize(turns, **kwargs):
        seen.append(deepcopy(turns))
        return compressor._with_summary_prefix("Inspected earlier files.")

    with patch.object(compressor, "_generate_summary", side_effect=summarize):
        first = compressor.compress(messages, force=True)
        assert len(seen) == 1
        carriers = [m for m in first if m.get("tool_calls") and compressor._is_context_summary_message(m)]
        assert carriers, "fixture must exercise an actual merged tool-call carrier"
        first_snapshot = deepcopy(first)
        second = compressor.compress(first, force=True)
    assert len(seen) == 2
    for carrier in carriers:
        for call in carrier["tool_calls"]:
            call_id = call["id"]
            for field in ("call", "result"):
                def has_round(rows):
                    if field == "call":
                        return any(tc["id"] == call_id for row in rows for tc in row.get("tool_calls", []))
                    return any(row.get("tool_call_id") == call_id for row in rows)
                assert has_round(second) or has_round(seen[1]), f"lost {field}: {call_id}"
    assert first == first_snapshot
    assert second[0] == first[0]


@pytest.mark.parametrize("content", ["", []])
@pytest.mark.parametrize("force_user_leading", [False, True])
def test_empty_carrier_retains_calls_but_not_display_scaffolding(content, force_user_leading):
    from agent.compaction_display import project_compaction_message_for_display
    from agent.context_compressor import reference_handoff_would_drive_next_model_call

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test/model", quiet_mode=True)
    carrier = {"role": "assistant", "content": content, "tool_calls": [{
        "id": "pending", "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }]}
    compressor._merge_summary_into_tail_row(
        carrier, compressor._with_summary_prefix("Prior facts."),
        "user" if force_user_leading else "assistant", force_user_leading,
    )
    original = deepcopy(carrier)
    stripped = compressor._strip_context_summary_handoff_message(carrier)
    assert stripped is not None
    assert stripped["tool_calls"] == carrier["tool_calls"]
    assert not compressor._is_context_summary_message(stripped)
    assert reference_handoff_would_drive_next_model_call([carrier]) is False
    assert project_compaction_message_for_display(carrier) is None
    assert carrier == original


def test_standalone_handoff_still_drops_and_plain_call_is_unchanged():
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY

    summary = {"role": "assistant", "content": ContextCompressor._with_summary_prefix("Facts."),
               COMPRESSED_SUMMARY_METADATA_KEY: True}
    assert ContextCompressor._strip_context_summary_handoff_message(summary) is None
    plain = {"role": "assistant", "content": "", "tool_calls": [{"id": "plain"}]}
    assert ContextCompressor._strip_context_summary_handoff_message(plain) == plain
