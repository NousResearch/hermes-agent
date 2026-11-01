"""Identifiers in pruned tool bodies remain available to the lean Anchor Index."""

import copy
from unittest.mock import patch

import pytest

from agent.compaction_anchor_source import with_pre_prune_tool_bodies
from agent.context_compressor import ContextCompressor
from agent.prepared_compaction import PreparedCompaction
from tests.agent.test_prepared_compaction import _messages, _prepare, _summary_row


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=40960):
        cc = ContextCompressor(
            model="test/model", protect_first_n=3, protect_last_n=5, quiet_mode=True,
        )
        cc.context_length = 40960
    cc.threshold_tokens = 20000
    cc.tail_token_budget = 2500
    cc._session_id = "anchor-session"
    return cc


def _history(identifier):
    messages = _messages(30)
    for index, call_id, body in (
        (2, "head-call", "head output " * 300),
        (10, "middle-call", "old output " * 300 + "\n" + identifier),
    ):
        messages[index] = {
            "role": "assistant", "content": "Reading output.",
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": "terminal", "arguments": '{"command": "inspect"}',
            }}],
        }
        messages.insert(index + 1, {"role": "tool", "tool_call_id": call_id, "content": body})
    return messages


@pytest.mark.parametrize("path", ["inline", "fallback", "prepared"])
def test_pruned_body_anchors_do_not_restore_prompt_or_conversation_bodies(monkeypatch, path):
    cc = _compressor()
    identifier = "artifacts/retained_identifier.json"
    messages = _history(identifier)
    before = copy.deepcopy(messages)
    prompts = []

    def summarize(self, prompt, started_at):
        prompts.append(prompt)
        return "The inspection is complete."

    monkeypatch.setattr(ContextCompressor, "_call_summary_llm", summarize)
    if path == "fallback":
        monkeypatch.setattr(cc, "_fallback_streak_skip", lambda telemetry: True)
    if path == "prepared":
        cc.prepared_compaction = PreparedCompaction()
        entry = _prepare(cc, messages)
        # Another plan on the live compressor cannot change the worker's completed index.
        cc._plan_compaction_window(_history("artifacts/other_window.json"))
        assert cc._previous_summary is None

    compressed = cc.compress(messages, current_tokens=20001)
    summary = _summary_row(compressed)["content"]
    head_result = next(row for row in compressed if row.get("tool_call_id") == "head-call")
    assert head_result["content"].startswith("[terminal]")
    assert len(head_result["content"]) < 200
    assert messages == before
    assert all(identifier not in prompt for prompt in prompts)
    assert len(prompts) == (0 if path == "fallback" else 1)
    if path == "prepared":
        assert entry.summary in summary
        assert "artifacts/other_window.json" not in summary
    assert identifier in summary.split("## Anchor Index", 1)[-1].split("\n\n", 1)[0]

    # The next compaction reuses the call id with a different body; the prior source must not leak.
    cc._previous_summary = None
    cc.compression_count = 0
    next_summary = _summary_row(cc.compress(_history("artifacts/next_window.json"), current_tokens=20001))
    anchors = next_summary["content"].split("## Anchor Index", 1)[-1].split("\n\n", 1)[0]
    assert "artifacts/next_window.json" in anchors
    assert identifier not in anchors


def test_ambiguous_ids_and_non_string_bodies_are_not_substituted():
    cc = _compressor()
    source = [
        {"role": "tool", "tool_call_id": "reused", "content": "artifacts/first.json"},
        {"role": "tool", "tool_call_id": "reused", "content": "artifacts/second.json"},
        {"role": "tool", "tool_call_id": "structured-source", "content": ["artifacts/list.json"]},
        {"role": "tool", "tool_call_id": "structured-target", "content": "artifacts/text.json"},
    ]
    cc._plan_compaction_window(source)
    turns = [
        {"role": "tool", "tool_call_id": "reused", "content": "stub"},
        {"role": "tool", "tool_call_id": "structured-source", "content": "stub"},
        {"role": "tool", "tool_call_id": "structured-target", "content": []},
    ]
    assert "## Anchor Index" not in cc._augment_summary_lean("Summary.", turns)


def test_long_identifier_runs_are_dropped_from_recovered_bodies():
    run = "/".join("ab" for _ in range(40_000))  # the file pattern backtracks quadratically over this
    source = [{"role": "tool", "tool_call_id": "big", "content": f"wrote src/agent/kept.py\n{run}\nend"}]
    (row,) = with_pre_prune_tool_bodies([{"role": "tool", "tool_call_id": "big", "content": "stub"}], source)

    assert "src/agent/kept.py" in row["content"]
    assert len(row["content"]) < 100
