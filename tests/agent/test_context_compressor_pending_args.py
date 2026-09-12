"""Pending tool-call args survive Pass-3/4 truncation; stale orphans exempt nothing (#105574, #105598)."""

import json
import logging
from unittest.mock import patch

from agent.context_compressor import ContextCompressor, _first_pending_tool_call_index


def _make_compressor(**overrides):
    kwargs = dict(
        model="test/model",
        quiet_mode=True,
        protect_first_n=1,
        protect_last_n=2,
    )
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(**kwargs)


def _delegate_call(call_id, goal_chars):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "delegate_task",
                "arguments": json.dumps(
                    {"goal": "G" * goal_chars, "tasks": ["t1", "t2"]}
                ),
            },
        }],
    }


def _tool_result(call_id, size):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": "R" * size,
    }


def _responses_call(call_id, item_id, goal_chars):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "call_id": call_id,
            "response_item_id": item_id,
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"goal": "G" * goal_chars}),
            },
        }],
    }


def _goal_of(msg):
    args = msg["tool_calls"][0]["function"]["arguments"]
    return json.loads(args)["goal"]


class TestPendingArgsExemptFromPressure:
    def test_pending_call_survives_while_executed_call_shrinks(self):
        """Trailing pending dispatch (true pre-send shape) survives; executed history shrinks."""
        c = _make_compressor()
        old_args_len = len(json.dumps({"goal": "G" * 2000, "tasks": ["t1"]}))
        assert old_args_len > 500
        msgs = [
            {"role": "user", "content": "start " + "x" * 200},
            _delegate_call("call_old", 2000),
            _tool_result("call_old", 20000),
            {"role": "user", "content": "active ask"},
            _delegate_call("call_pending", 2000),
        ]
        pending_before = _goal_of(msgs[4])
        result, _ = c._prune_old_tool_results(
            msgs, protect_tail_count=4, protect_tail_tokens=100
        )
        pending_after = _goal_of(result[4])
        assert pending_after == pending_before
        assert len(pending_after) == 2000
        assert "...[truncated]" not in result[4]["tool_calls"][0]["function"]["arguments"]
        old_after = _goal_of(result[1])
        assert len(old_after) < 2000
        assert "...[truncated]" in result[1]["tool_calls"][0]["function"]["arguments"]

    def test_pressure_truncation_warns_with_tool_name(self, caplog):
        c = _make_compressor()
        msgs = [
            {"role": "user", "content": "start " + "x" * 200},
            _delegate_call("call_old", 2000),
            _tool_result("call_old", 20000),
            {"role": "user", "content": "active ask"},
        ]
        with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
            c._prune_old_tool_results(
                msgs, protect_tail_count=4, protect_tail_tokens=100
            )
        warnings = [
            r for r in caplog.records
            if "truncated tool-call args" in r.getMessage()
            and "delegate_task" in r.getMessage()
        ]
        assert warnings, "expected a warning naming delegate_task and the shrink"

    def test_responses_call_answered_by_item_id_is_not_pending(self):
        c = _make_compressor()
        msgs = [
            {"role": "user", "content": "start " + "x" * 200},
            _responses_call("call_old", "fc_old", 2000),
            _tool_result("fc_old", 20000),
            {"role": "user", "content": "active ask"},
        ]
        result, _ = c._prune_old_tool_results(
            msgs, protect_tail_count=4, protect_tail_tokens=100
        )
        args = result[1]["tool_calls"][0]["function"]["arguments"]
        assert "...[truncated]" in args

    def test_pending_call_in_large_batch_survives_pass3(self):
        """Pass 3 must not truncate a pending call that fell out of the tail floor (#105598)."""
        c = _make_compressor()
        tcs = [
            {
                "id": f"call_{k}",
                "type": "function",
                "function": {
                    "name": "mytool",
                    "arguments": json.dumps({"goal": "G" * 2000, "i": k}),
                },
            }
            for k in range(1, 10)
        ]
        msgs: list = [{"role": "user", "content": "start"}]
        msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
        for k in range(1, 9):
            msgs.append(_tool_result(f"call_{k}", 20000))
        result, _ = c._prune_old_tool_results(
            msgs, protect_tail_count=20, protect_tail_tokens=100
        )
        args9 = result[1]["tool_calls"][8]["function"]["arguments"]
        assert "...[truncated]" not in args9
        assert json.loads(args9)["goal"] == "G" * 2000

    def test_stale_orphan_does_not_exempt_later_completed_call(self):
        """A settled historical orphan must not disable reclamation downstream (#105598)."""
        c = _make_compressor()
        msgs = [
            {"role": "user", "content": "start"},
            _delegate_call("call_orphan", 2000),
            _delegate_call("call_done", 2000),
            _tool_result("call_done", 20000),
            {"role": "user", "content": "later turn"},
        ]
        assert _first_pending_tool_call_index(msgs) == len(msgs)
        result, _ = c._prune_old_tool_results(
            msgs, protect_tail_count=4, protect_tail_tokens=100
        )
        args = result[2]["tool_calls"][0]["function"]["arguments"]
        assert "...[truncated]" in args
