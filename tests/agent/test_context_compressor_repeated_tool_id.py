"""Regressions: context compression must not collapse repeated provider IDs."""

import json
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.tool_occurrences import tool_result_metadata_by_index


def _assistant(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
        ],
    }


def _reused_id_messages(*, long_results: bool = False) -> list[dict]:
    a = "A" * 600 if long_results else "RESULT_A"
    b = "B" * 600 if long_results else "RESULT_B"
    return [
        _assistant("x", "read_file", {"path": "/a"}),
        {"role": "tool", "tool_call_id": "x", "content": a},
        {"role": "user", "content": "next"},
        _assistant("x", "terminal", {"command": "build"}),
        {"role": "tool", "tool_call_id": "x", "content": b},
    ]


def test_prune_uses_metadata_from_same_repeated_id_occurrence():
    compressor = ContextCompressor(model="test/model", quiet_mode=True)
    messages = [
        *_reused_id_messages(long_results=True),
        {"role": "user", "content": "recent protected tail"},
    ]

    pruned, count = compressor._prune_old_tool_results(
        messages,
        protect_tail_count=1,
    )

    assert count >= 2
    assert pruned[1]["content"].startswith("[read_file]")
    assert not pruned[1]["content"].startswith("[terminal]")
    assert pruned[4]["content"].startswith("[terminal]")


def test_static_fallback_summary_uses_same_occurrence_metadata():
    compressor = ContextCompressor(model="test/model", quiet_mode=True)
    seen: list[str] = []

    def summarize(tool_name: str, tool_args: str, tool_content: str) -> str:
        seen.append(tool_name)
        return f"[{tool_name}]"

    with patch("agent.context_compressor._summarize_tool_result", side_effect=summarize):
        compressor._build_static_fallback_summary(_reused_id_messages())

    assert seen == ["read_file", "terminal"]


def test_pressure_tail_demote_keeps_same_occurrence_metadata():
    compressor = ContextCompressor(model="test/model", quiet_mode=True)
    messages = _reused_id_messages(long_results=True)
    metadata = tool_result_metadata_by_index(messages)

    demoted = compressor._pressure_demote_tail(
        messages,
        prune_boundary=0,
        protect_tail_tokens=1,
        tool_metadata_by_result_idx=metadata,
        min_prune_chars=100,
    )

    assert demoted >= 1
    assert messages[1]["content"].startswith("[read_file]")
    assert not messages[1]["content"].startswith("[terminal]")
