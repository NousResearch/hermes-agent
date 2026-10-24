"""Qualification probe: cost-only pruning must keep the last instruction-file read.

This proposes a narrow invariant, not a universal ban on full compression.
"""
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor, _estimate_msg_budget_tokens
from hermes_state import SessionDB
from tools.file_tools import read_file_tool


def _pair(call_id, name, arguments, content):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


@pytest.mark.parametrize("filename", ["AGENTS.md", "CLAUDE.md", "AGENTS.override.md", "agents.md", "claude.md", ".cursorrules"])
def test_cost_prune_keeps_last_instruction_read(tmp_path, filename):
    instructions = tmp_path / filename
    instructions.write_text("# Project instructions\n" + "Keep changes within the requested scope.\n" * 550)
    content = read_file_tool(str(instructions), task_id=f"retention-{filename}")
    assert not json.loads(content).get("error")
    messages = [{"role": "system", "content": "Synthetic system."},
                {"role": "user", "content": "Begin the synthetic task."},
                {"role": "assistant", "content": "Ready."},
                {"role": "user", "content": "Read instructions and continue."}]
    messages.extend(_pair("instructions", "read_file", {"path": str(instructions)}, content))
    for i in range(10):
        messages.extend(_pair(f"log-{i}", "terminal", {"command": f"synthetic-{i}"}, str(i) * 10000))
    compressor = ContextCompressor(
        model="synthetic-test-model", config_context_length=1_000_000, quiet_mode=True,
        protect_first_n=2, protect_last_n=4, proactive_prune_tokens=48000,
        proactive_prune_min_result_chars=8000,
    )
    assert not compressor.should_compress(prompt_tokens=120000)
    pruned, count = compressor.prune_tool_results_only(messages, current_tokens=120000)
    assert count > 0, "Ordinary old results must still be pruned."
    retained = next(row for row in pruned if row.get("tool_call_id") == "instructions")
    assert retained["content"] == content, "The only successful instruction read was reduced to metadata."


def _compressor():
    return ContextCompressor(
        model="synthetic-test-model", config_context_length=1_000_000, quiet_mode=True,
        protect_first_n=2, protect_last_n=4, proactive_prune_tokens=48000,
        proactive_prune_min_result_chars=8000,
    )


def _body(label="instructions", **fields):
    return json.dumps({"content": label * 5000, "total_lines": 1, "truncated": False, **fields})


def _history(*pairs):
    rows = [{"role": "user", "content": "Read instructions and complete the task."}]
    for pair in pairs:
        rows.extend(pair)
    for i in range(8):
        rows.extend(_pair(f"log-{i}", "terminal", {"command": f"synthetic-{i}"}, str(i) * 10000))
    rows.append({"role": "user", "content": "Continue the task."})
    return rows


def _result(rows, call_id):
    return next(row["content"] for row in rows if row.get("tool_call_id") == call_id)


@pytest.mark.parametrize("later", [
    _body("new"),
    json.dumps({"content_returned": False, "note": "Already read; full content is above."}),
    json.dumps({"success": False, "error": "File not found", "content": "error" * 2500}),
    "[read_file] metadata only",
])
def test_latest_success_survives_without_protecting_superseded_reads(later):
    args = {"path": "/project/AGENTS.md"}
    before = _body("old")
    messages = _history(_pair("old", "read_file", args, before), _pair("new", "read_file", args, later))
    original = deepcopy(messages)
    c = _compressor()
    pruned, count = c.prune_tool_results_only(messages, current_tokens=120000)
    successful = later == _body("new")
    assert (_result(pruned, "old") != before) is successful
    if successful:
        assert _result(pruned, "new") == later
    assert count == sum(a != b for a, b in zip(messages, pruned))
    assert messages == original
    assert [row.get("tool_calls") for row in pruned] == [row.get("tool_calls") for row in messages]
    assert c.prune_tool_results_only(pruned, current_tokens=None) == (pruned, 0)


@pytest.mark.parametrize("args,new_args,fields", [
    ({"path": "/project/AGENTS.md", "offset": 1, "limit": 200}, {"path": "/project/AGENTS.md", "offset": 201, "limit": 200}, {"truncated": True}),
    ({"path": "AGENTS.md"}, {"path": "AGENTS.md"}, {}),
    ({"path": "/project/CLAUDE.md"}, {"path": "/project/CLAUDE.md"}, {"truncated": True, "next_offset": 150}),
])
def test_partial_ranges_and_unknown_relative_cwd_do_not_displace_prior_content(args, new_args, fields):
    first, second = _body("first"), _body("second", **fields)
    messages = _history(_pair("a", "read_file", args, first), _pair("b", "read_file", new_args, second))
    pruned, count = _compressor().prune_tool_results_only(messages, current_tokens=120000)
    assert count > 0
    assert _result(pruned, "a") == first
    assert _result(pruned, "b") == second


def test_reused_call_ids_and_embedded_markers_do_not_relabel_ordinary_output():
    body = _body("[INSTRUCTIONS_PRUNED] /project/AGENTS.md")
    messages = _history(
        _pair("reuse", "terminal", {"command": "synthetic"}, body),
        _pair("reuse", "read_file", {"path": "/project/AGENTS.md"}, body),
        _pair("ordinary", "read_file", {"path": "/project/report.md"}, body),
    )
    pruned, count = _compressor().prune_tool_results_only(messages, current_tokens=120000)
    reused = [row["content"] for row in pruned if row.get("tool_call_id") == "reuse"]
    assert count > 0
    assert reused[0] != body
    assert reused[1] == body
    assert _result(pruned, "ordinary") != body


def test_retained_bytes_count_toward_reclaim_and_full_compression_still_progresses():
    body = _body("large", total_lines=50000)
    messages = [{"role": "user", "content": "Start."}]
    messages.extend(_pair("read", "read_file", {"path": "/project/AGENTS.md"}, body))
    for i in range(8):
        messages.extend([{"role": "user", "content": f"Continue {i}."},
                         {"role": "assistant", "content": f"Step {i}."}])
    c = _compressor()
    same, count = c.prune_tool_results_only(messages, current_tokens=120000)
    assert same is messages and count == 0
    assert c._proactive_prune_rearm_tokens == 0
    c.tail_token_budget = 1000
    with patch.object(c, "_call_summary_llm", return_value="## Goal\nComplete the task.\n## Progress\nEarlier steps complete."):
        compacted = c.compress(messages, current_tokens=120000, force=True)
    assert sum(map(_estimate_msg_budget_tokens, compacted)) < sum(map(_estimate_msg_budget_tokens, messages))
    assert body not in [row.get("content") for row in compacted]


def test_retention_is_in_durable_prune_payload_and_rolls_back_on_write_failure(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("retention", source="cli")
        body = _body()
        rows = _history(_pair("instructions", "read_file", {"path": "/project/AGENTS.md"}, body))
        db.append_messages_batch("retention", rows)
        c = _compressor()
        c._session_db, c._session_id = db, "retention"
        messages = db.get_messages_as_conversation("retention")
        with patch.object(db, "rewrite_pruned_rows", side_effect=RuntimeError("synthetic write failure")):
            same, count = c.prune_tool_results_only(messages, current_tokens=120000)
        assert same is messages and count == 0
        assert db.get_messages_as_conversation("retention") == messages
        pruned, count = c.prune_tool_results_only(messages, current_tokens=120000)
        assert count > 0
        durable = db.get_messages_as_conversation("retention")
        assert _result(durable, "instructions") == body
        assert [row["content"] for row in durable] == [row["content"] for row in pruned]
    finally:
        db.close()


@pytest.mark.parametrize("whole_call", [False, True])
def test_object_shaped_calls_remain_supported(whole_call):
    body = _body()
    pair = _pair("object", "read_file", {"path": "/project/AGENTS.md"}, body)
    call = pair[0]["tool_calls"][0]
    call["function"] = SimpleNamespace(**call["function"])
    if whole_call:
        pair[0]["tool_calls"][0] = SimpleNamespace(**call)
    rows = _history(pair)
    pruned, count = _compressor().prune_tool_results_only(rows, current_tokens=120000)
    assert count > 0
    assert _result(pruned, "object") == body
