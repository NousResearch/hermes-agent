"""Compaction names content it removed without implying that user quotes are facts."""

import json
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    ContextCompressor, HISTORICAL_TASK_HEADING, _NO_USER_TASK_SENTINEL, _summarize_tool_result,
)


def _summary(turns, mode="lean", path="fallback"):
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(model="test/model", quiet_mode=True, tail_mode=mode)
        _ = compressor.context_length
    compressor._session_id = "test-session"
    if path == "fallback":
        return compressor._build_static_fallback_summary(turns)
    body = f"{HISTORICAL_TASK_HEADING}\n{_NO_USER_TASK_SENTINEL}\n\n## Goal\nSummary."
    with patch.object(compressor, "_call_summary_llm", return_value=body):
        return compressor._generate_summary(turns)


@pytest.mark.parametrize("path", ["fallback", "model"])
def test_removal_notice_names_pruned_outputs_and_skills_without_inventing_other_losses(path):
    turns = [{"role": "tool", "content": _summarize_tool_result(
        name, json.dumps(args), "BODY_BYTES " * 100,
    )} for name, args in (("terminal", {"command": "echo done"}),
                         ("skill_view", {"name": "example-skill"}))]
    summary = _summary(turns, path=path)
    notice = summary.split("\nRemoved from context: ", 1)[1]
    assert notice == (
        "older tool outputs replaced by stubs, skill bodies replaced by reload markers; "
        "quoted user text records what was said, not verified fact."
    )
    assert len("Removed from context: " + notice) <= 300
    assert "BODY_BYTES" not in summary
    assert "Removed from context:" not in _summary(turns, mode="legacy", path=path)


@pytest.mark.parametrize(("turns", "removed"), [
    ([{"role": "user", "content": "Please continue."}], ""),
    ([{"role": "assistant", "content": "Done.", "reasoning": "private draft"}], "reasoning"),
    ([{"role": "assistant", "content": "Done.", "reasoning_content": "draft"}], "reasoning"),
    ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "synthetic-image"}}]}], "older images"),
    ([{"role": "tool", "content": "[screenshot removed to save context]"}], "older images"),
])
def test_removal_notice_only_names_classes_present_in_the_region(turns, removed):
    summary = _summary(turns)
    if not removed:
        assert "Removed from context:" not in summary
    else:
        notice = summary.split("\nRemoved from context: ", 1)[1]
        assert notice == f"{removed}; quoted user text records what was said, not verified fact."


def test_an_echoed_previous_notice_is_replaced_not_repeated():
    from agent.compaction_removal_notice import with_removal_notice
    from agent.context_compressor import _PRUNED_TOOL_PLACEHOLDER

    class _Lean:
        tail_mode = "lean"

    echoed = "## Goal\nwork\n\nRemoved from context: reasoning; quoted user text records what was said, not verified fact."
    out = with_removal_notice(echoed, [{"role": "tool", "content": _PRUNED_TOOL_PLACEHOLDER}], _Lean())
    assert out.count("Removed from context:") == 1
    assert "older tool outputs replaced by stubs" in out and out.startswith("## Goal\nwork")
