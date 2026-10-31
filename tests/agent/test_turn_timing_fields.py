"""Agent timing follows real result/telemetry seams and resets between turns."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.agent.test_turn_finalizer_cleanup_guard import _StubAgent, _run


@pytest.mark.parametrize("model", [None, 1.5])
def test_finalizer_exposes_only_available_timings(model):
    agent = _StubAgent(raise_in=())
    agent._turn_tool_seconds, agent._turn_compaction_seconds = 2.0, 3.0
    if model is not None:
        agent._turn_model_seconds = model
    agent._turn_first_token_at = 12.0
    result = _run(agent)["turn_timing"]
    assert result == {"tools_s": 2.0, "compaction_s": 3.0, **({"model_s": model} if model is not None else {})}


def test_ttfb_requires_first_request_start():
    agent = _StubAgent(raise_in=())
    agent._turn_first_token_at, agent._turn_first_api_request_at = 12.0, 10.0
    assert _run(agent)["turn_timing"]["ttfb_s"] == 2.0


def test_accumulators_reset_each_turn():
    from agent.turn_context import _reset_per_turn_agent_state
    agent = Mock(api_mode="anthropic_messages", _compression_warning=None)
    agent._turn_tool_seconds, agent._turn_compaction_seconds = 20.0, 30.0
    _reset_per_turn_agent_state(agent)
    assert agent._turn_tool_seconds == agent._turn_compaction_seconds == 0.0


@pytest.mark.parametrize("seconds", [2.5, float("nan"), float("inf"), -1.0])
def test_tool_commit_adds_only_finite_nonnegative_duration(monkeypatch, seconds):
    from agent.tool_executor import _commit_tool_result, _ToolCallRef
    agent = Mock(_turn_tool_seconds=1.0, verbose_logging=False, tool_result_metadata_callback=None)
    agent._subdirectory_hints.check_tool_call.return_value = None
    agent._tool_result_content_for_active_model.return_value = "ok"
    monkeypatch.setattr("agent.tool_executor.maybe_persist_tool_result", lambda **kw: "ok")
    monkeypatch.setattr("agent.tool_executor._flush_session_db_after_tool_progress", lambda *a, **kw: True)
    _commit_tool_result(agent, [], _ToolCallRef("read_file", {}, "task", "call", []), "ok",
                        budget=Mock(), tool_duration=seconds, is_error=False, blocked=True, effect_disposition=None)
    assert agent._turn_tool_seconds == (3.5 if seconds == 2.5 else 1.0)


@pytest.mark.parametrize("bypassed", [False, True])
@pytest.mark.parametrize("elapsed", [2.5, float("nan"), float("inf"), -1.0])
def test_compaction_telemetry_accumulates_valid_durations(monkeypatch, bypassed, elapsed):
    from agent import conversation_compression_telemetry as telemetry
    agent = SimpleNamespace(context_compressor=SimpleNamespace(), _turn_compaction_seconds=1.0)
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: elapsed)
    if bypassed:
        telemetry._emit_bypassed_attempt_telemetry(agent, 0.0, commit_status="bypassed", failure_class=None, approx_tokens=0)
    else:
        telemetry._emit_compression_attempt_telemetry(agent, started_at=0.0, commit_status="committed", split_status="not_applicable")
    assert agent._turn_compaction_seconds == (3.5 if elapsed == 2.5 else 1.0)
