"""Tests for tool loop context pruning.

Verifies:
1. Repeated identical call/response pairs are collapsed into a summary
2. The summary message replaces the collapsed pairs
3. Non-loop messages are preserved exactly
4. The most recent call/response pair is kept
5. Corrective guidance includes intended tool name when available
6. Corrective guidance omits intended tool when not available
7. Message roles and structure remain valid after pruning
"""
import json
import pytest

from agent.tool_loop_pruner import prune_tool_loop


def _make_tool_call_msg(tool_name, args, call_id="call_1", reasoning=None):
    msg = {
        "role": "assistant",
        "content": "",
        "finish_reason": "tool_calls",
        "tool_calls": [{
            "id": call_id,
            "call_id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(args)},
        }],
    }
    if reasoning:
        msg["reasoning"] = reasoning
    return msg


def _make_tool_result_msg(content, call_id="call_1"):
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": call_id,
    }


class TestPruneToolLoop:

    def test_collapses_repeated_pairs(self):
        """5 identical call/response pairs -> summary + last pair."""
        messages = [
            {"role": "user", "content": "trigger the workflow"},
        ]
        for i in range(5):
            cid = f"call_{i}"
            messages.append(_make_tool_call_msg("generate", {"desc": "test"}, call_id=cid))
            messages.append(_make_tool_result_msg('{"error": "hosted_only"}', call_id=cid))

        pruned = prune_tool_loop(messages, tool_name="generate", streak=5)

        assert len(pruned) < len(messages)
        assert pruned[0]["role"] == "user"
        summaries = [
            m for m in pruned
            if m.get("role") == "tool" and "LOOP DETECTED" in m.get("content", "")
        ]
        assert len(summaries) == 1
        assert pruned[-1]["role"] == "tool"
        assert pruned[-2]["role"] == "assistant"

    def test_preserves_non_loop_messages(self):
        """Messages before the loop are untouched."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "do the thing"},
        ]
        for i in range(3):
            cid = f"call_{i}"
            messages.append(_make_tool_call_msg("grep", {"pattern": "foo"}, call_id=cid))
            messages.append(_make_tool_result_msg("no matches", call_id=cid))

        pruned = prune_tool_loop(messages, tool_name="grep", streak=3)

        assert pruned[0] == {"role": "user", "content": "hello"}
        assert pruned[1] == {"role": "assistant", "content": "hi there"}
        assert pruned[2] == {"role": "user", "content": "do the thing"}

    def test_includes_intended_tool_in_guidance(self):
        """When intended_tool is provided, the summary mentions it."""
        messages = []
        for i in range(3):
            cid = f"call_{i}"
            messages.append(_make_tool_call_msg(
                "mcp_n8n_n8n_generate_workflow",
                {"description": "test"},
                call_id=cid,
                reasoning="I want to call mcp_n8n_n8n_test_workflow"
            ))
            messages.append(_make_tool_result_msg('{"hosted_only": true}', call_id=cid))

        pruned = prune_tool_loop(
            messages,
            tool_name="mcp_n8n_n8n_generate_workflow",
            streak=3,
            intended_tool="mcp_n8n_n8n_test_workflow",
        )

        summary = next(m for m in pruned if "LOOP DETECTED" in m.get("content", ""))
        assert "mcp_n8n_n8n_test_workflow" in summary["content"]

    def test_no_intended_tool_still_prunes(self):
        """Pruning works without intended_tool."""
        messages = []
        for i in range(3):
            cid = f"call_{i}"
            messages.append(_make_tool_call_msg("grep", {"pattern": "x"}, call_id=cid))
            messages.append(_make_tool_result_msg("no match", call_id=cid))

        pruned = prune_tool_loop(messages, tool_name="grep", streak=3)
        summary = next(m for m in pruned if "LOOP DETECTED" in m.get("content", ""))
        assert "different approach" in summary["content"]

    def test_valid_message_structure_after_pruning(self):
        """After pruning, tool messages follow assistant messages with tool_calls."""
        messages = [{"role": "user", "content": "go"}]
        for i in range(4):
            cid = f"call_{i}"
            messages.append(_make_tool_call_msg("t", {"a": 1}, call_id=cid))
            messages.append(_make_tool_result_msg("r", call_id=cid))

        pruned = prune_tool_loop(messages, tool_name="t", streak=4)
        roles = [m["role"] for m in pruned]
        for i in range(len(roles) - 1):
            assert not (roles[i] == "user" and roles[i + 1] == "user"), \
                f"Consecutive user messages at index {i}"

    def test_single_pair_not_pruned(self):
        """A single call/response pair should not be pruned."""
        messages = [
            {"role": "user", "content": "go"},
            _make_tool_call_msg("t", {"a": 1}, call_id="c1"),
            _make_tool_result_msg("r", call_id="c1"),
        ]
        pruned = prune_tool_loop(messages, tool_name="t", streak=1)
        assert len(pruned) == len(messages)

    def test_does_not_mutate_input(self):
        """prune_tool_loop returns a new list, not modifying the input."""
        messages = [{"role": "user", "content": "go"}]
        for i in range(3):
            cid = f"call_{i}"
            messages.append(_make_tool_call_msg("t", {"a": 1}, call_id=cid))
            messages.append(_make_tool_result_msg("r", call_id=cid))

        original_len = len(messages)
        pruned = prune_tool_loop(messages, tool_name="t", streak=3)
        assert len(messages) == original_len
        assert len(pruned) < original_len
