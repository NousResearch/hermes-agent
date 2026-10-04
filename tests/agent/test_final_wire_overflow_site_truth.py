"""Enumerate all real loop compaction sites and exercise remote-overflow owners."""
import ast
import inspect
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from agent.conversation_compression import ProviderBoundRequestOverLimit, compaction_mutation_outcome
from tests.run_agent.test_413_compression import agent, _mock_response  # noqa: F401


# One proactive loop site (and the separate prologue) have dedicated truth
# suites; the five remaining overflow/post-tool sites are exercised here.
# The source assertion prevents new naked compaction calls or post-owner
# identity-based truth assignments.
def test_all_loop_compaction_sites_share_detached_sticky_owner():
    source = Path(__file__).parents[2] / "agent" / "conversation_loop.py"
    tree = ast.parse(source.read_text())
    tracked = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_compress_tracked"]
    naked = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_compress_context"]
    owner = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_compress_tracked")
    assert len(tracked) == 6
    assert len(naked) == 1 and owner.lineno < naked[0].lineno < owner.end_lineno
    writes = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in {"_transcript_rewritten_this_turn", "_mutation_outcome_this_turn"}]
    assert all(n.lineno < owner.lineno or owner.lineno <= n.lineno <= owner.end_lineno for n in writes)


@pytest.mark.parametrize("site", ["long_context", "413", "anthropic_output", "context_overflow", "post_tool"])
@pytest.mark.parametrize("mode", ["replacement", "inplace", "system", "partial", "nochange", "equalcopy", "unknown"])
def test_actual_overflow_compaction_site_snapshot_truth(agent, site, mode):
    a = agent
    a.valid_tool_names = {"inert"}
    a.tools = []
    a.max_tokens = 1
    a.max_compression_attempts = 2
    a._cached_system_prompt = "inert system"
    a.context_compressor.context_length = 10000
    a.context_compressor.threshold_tokens = 9000
    a.max_iterations = 3
    rows = [{"role": "user", "content": "x" * 900}, {"role": "assistant", "content": "previous"}]
    calls, observations, sites = [], [], []
    def dispatch(kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            if site == "413":
                error = Exception("request entity too large")
                error.status_code = 413
                raise error
            if site == "long_context":
                error = Exception("Extra usage is required for long context requests")
                error.status_code = 429
                raise error
            if site == "context_overflow":
                raise Exception("prompt is too long: 12000 tokens > 10000 maximum")
            if site == "anthropic_output":
                raise Exception("max_tokens: 2000 > context_window: 10000 - input_tokens: 9000 = available_tokens: 1000")
            response = _mock_response(content="unfinished", finish_reason="tool_calls", tool_calls=[SimpleNamespace(id="inert-id", type="function", function=SimpleNamespace(name="inert", arguments="{}"))])
            response.usage = SimpleNamespace(prompt_tokens=9000, completion_tokens=1, total_tokens=9001)
            return response
        raise ProviderBoundRequestOverLimit(10001, 10000)
    def compact(current, *args, **kwargs):
        prompt = a._cached_system_prompt
        if mode == "replacement":
            return [{"role": "user", "content": "short"}], prompt
        if mode == "inplace":
            current[0]["content"] = "short"
        if mode == "partial":
            current.pop(0)
        if mode == "system":
            prompt += " changed"
        if mode == "equalcopy":
            return deepcopy(current), prompt
        if mode == "unknown":
            current[0]["content"] = "short"
            current[0]["opaque_test_state"] = object()
        return current, prompt
    def observe(before, after, prompt):
        outcome = compaction_mutation_outcome(before, after, prompt)
        frames = inspect.stack(context=0)
        owner_index = next(i for i, frame in enumerate(frames) if frame.function == "_compress_tracked")
        sites.append(frames[owner_index + 1].lineno)
        del frames
        observations.append(outcome)
        return outcome
    if site == "long_context":
        a.context_compressor.context_length = 300000
        a.context_compressor.threshold_tokens = 290000
    clock = [0.0]
    def now():
        clock[0] += 0.5
        return clock[0]
    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=10),
        patch("time.time", side_effect=now),
        patch.object(a, "_execute_tool_calls"),
        patch.object(a.context_compressor, "update_model", side_effect=lambda **kw: setattr(a.context_compressor, "context_length", kw["context_length"])),
        patch.object(a.context_compressor, "should_compress", side_effect=lambda tokens: site == "post_tool" and tokens >= 9000),
        patch.object(a, "_compress_context", side_effect=compact) as compression,
        patch.object(a, "_interruptible_api_call", side_effect=dispatch),
        patch.object(a, "_try_activate_fallback", return_value=False),
        patch("agent.conversation_loop.compaction_mutation_outcome", side_effect=observe),
        patch.object(a, "_persist_session"), patch.object(a, "_save_trajectory"),
        patch.object(a, "_cleanup_task_resources"), patch.object(a, "_emit_status") as status,
        patch("time.sleep", return_value=None),
    ):
        result = a.run_conversation("hi", conversation_history=rows)
    assert compression.call_count >= 1, site
    assert len(observations) == compression.call_count
    tree = ast.parse((Path(__file__).parents[2] / "agent" / "conversation_loop.py").read_text())
    nodes = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_compress_tracked"), key=lambda n: n.lineno)
    actual = nodes[{"long_context": 1, "413": 2, "anthropic_output": 3, "context_overflow": 4, "post_tool": 5}[site]]
    assert actual.lineno <= sites[0] <= actual.end_lineno
    expected = "no_change" if mode in {"nochange", "equalcopy"} else "unknown" if mode == "unknown" else "rewrite"
    assert observations[0] == expected
    texts = " ".join(str(c.args[0]) for c in status.call_args_list if c.args)
    if "refused locally" in texts:
        assert ("No messages were dropped" in texts) == (expected == "no_change")
        if expected == "unknown":
            assert "mutation outcome is unknown" in texts
    assert result["completed"] is False


@pytest.mark.parametrize("later_mode", ["nochange", "equalcopy"])
def test_actual_output_overflow_unknown_survives_later_proven_no_change(agent, later_mode):
    a = agent
    a.valid_tool_names = {"inert"}
    a.tools = []
    a.max_tokens = 1
    a.max_compression_attempts = 3
    a._cached_system_prompt = "inert system"
    a.context_compressor.context_length = 10000
    a.context_compressor.threshold_tokens = 9000
    dispatches, compactions = [], []
    def dispatch(kwargs):
        dispatches.append(kwargs)
        if len(dispatches) < 3:
            raise Exception("max_tokens: 2000 > context_window: 10000 - input_tokens: 9000 = available_tokens: 1000")
        raise ProviderBoundRequestOverLimit(10001, 10000)
    def compact(rows, *args, **kwargs):
        compactions.append(rows)
        if len(compactions) == 1:
            rows[0]["content"] += " partial change"
            raise RuntimeError("inert partial compression failure")
        return (deepcopy(rows) if later_mode == "equalcopy" else rows), a._cached_system_prompt
    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=10),
        patch.object(a.context_compressor, "should_compress", return_value=False),
        patch.object(a, "_compress_context", side_effect=compact),
        patch.object(a, "_interruptible_api_call", side_effect=dispatch),
        patch.object(a, "_persist_session"), patch.object(a, "_save_trajectory"),
        patch.object(a, "_cleanup_task_resources"), patch.object(a, "_emit_status") as status,
        patch("time.sleep", return_value=None),
    ):
        result = a.run_conversation("hi", conversation_history=[{"role": "user", "content": "prior"}, {"role": "assistant", "content": "previous"}])
    assert len(compactions) == 2
    assert len(dispatches) == 3
    texts = " ".join(str(c.args[0]) for c in status.call_args_list if c.args)
    assert "mutation outcome is unknown" in texts
    assert "No messages were dropped" not in texts
    assert result["failed"] is True
    assert any("partial change" in row.get("content", "") for row in result["messages"])
