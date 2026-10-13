"""Per-row token accounting (#136851): the provider's completion tokens for a response
must land on the assistant row that response produced.

The loop records usage (``record_response_usage``) BEFORE the assistant row is built, so
the figure is staged on the agent and consumed exactly once by
``build_assistant_message`` — the single chokepoint every live assistant row flows
through. A usage-less call clears the stage: its row carries no count, not the previous
call's (the stage is per-response, never per-agent-turn)."""
import logging
from types import SimpleNamespace


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    return AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
                   api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t", platform="cli",
                   quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])


def _resp(completion_tokens):
    if completion_tokens is None:
        return SimpleNamespace(usage=None, model="anthropic/claude-fable-5.1")
    return SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=100, completion_tokens=completion_tokens, total_tokens=100 + completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        completion_tokens_details=None), model="anthropic/claude-fable-5.1")


def _record(agent, resp):
    from agent import turn_usage
    turn_usage.record_response_usage(
        agent, resp, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
        api_duration=0.2, compression_attempts=0, max_compression_attempts=3)


def _build_row(agent, content="hi"):
    from agent.chat_completion_helpers import build_assistant_message
    return build_assistant_message(agent, SimpleNamespace(content=content, tool_calls=None), "stop")


def test_the_row_a_response_builds_carries_its_completion_tokens(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _record(a, _resp(42))
        row = _build_row(a)
        assert row["token_count"] == 42
    finally:
        a.close()


def test_the_stage_is_consumed_once(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _record(a, _resp(42))
        first, second = _build_row(a), _build_row(a)
        assert first["token_count"] == 42
        assert "token_count" not in second  # a re-prompt row must not inherit the count
    finally:
        a.close()


def test_a_usage_less_call_clears_the_stage_not_passes_it_on(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _record(a, _resp(42))
        _record(a, _resp(None))
        row = _build_row(a)
        assert "token_count" not in row  # provider reported nothing: NULL is the honest value
    finally:
        a.close()


def test_two_calls_stamp_their_own_rows_in_loop_order(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _record(a, _resp(50))
        tool_round_row = _build_row(a, "checking...")
        _record(a, _resp(30))
        final_row = _build_row(a, "done")
        assert tool_round_row["token_count"] == 50
        assert final_row["token_count"] == 30
    finally:
        a.close()


def test_zero_completion_tokens_stage_no_count(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _record(a, _resp(0))
        row = _build_row(a)
        assert "token_count" not in row
    finally:
        a.close()
