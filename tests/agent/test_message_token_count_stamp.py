"""messages.token_count finally carries a per-row figure (#136851).

The column shipped with the schema but no runtime path ever populated it. The stamp
lives in build_assistant_message — the one place a row is built from its own response —
reading the figure record_response_usage stashed for THAT response, so a usage-less
response cannot leave the previous response's number behind, and _db_flush_row carries
it into state.db (the db layer has always read it).
"""
from types import SimpleNamespace


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    return AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
                   api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t",
                   platform="cli", quiet_mode=True, skip_context_files=True, skip_memory=True,
                   save_trajectories=False, enabled_toolsets=["file"])


def _response(completion=7, prompt=62_889, cached=34_283):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              total_tokens=prompt + completion,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=0),
                              completion_tokens_details=None),
        id="gen-test", model="anthropic/claude-fable-5.1")


def _fold(agent, resp):
    from agent import turn_usage
    turn_usage.record_response_usage(agent, resp, messages=[{"role": "user", "content": "hi"}],
                                     api_call_count=1, api_duration=0.2,
                                     compression_attempts=0, max_compression_attempts=3)


def _assistant_response(content="hello"):
    return SimpleNamespace(content=content, tool_calls=None, finish_reason="stop",
                           reasoning=None, reasoning_content=None, reasoning_details=None)


def test_row_carries_this_response_completion_tokens(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _fold(a, _response(completion=1_204))
        msg = a._build_assistant_message(_assistant_response(), "stop")
    finally:
        a.close()
    assert msg["token_count"] == 1_204
    # the stamp is additive: the row shape the replay path depends on is unchanged
    assert msg["role"] == "assistant" and msg["content"] == "hello" and msg["finish_reason"] == "stop"


def test_usage_less_response_leaves_no_stale_stamp(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _fold(a, _response(completion=1_204))
        # A provider that omits usage must not let the previous response's figure
        # ride onto the next built row.
        _fold(a, SimpleNamespace(usage=None, id=None, model="anthropic/claude-fable-5.1"))
        msg = a._build_assistant_message(_assistant_response(), "stop")
    finally:
        a.close()
    assert "token_count" not in msg


def test_no_stamp_without_any_recorded_usage(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        msg = a._build_assistant_message(_assistant_response(), "stop")
    finally:
        a.close()
    assert "token_count" not in msg


def test_stamp_survives_the_db_flush_row(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _fold(a, _response(completion=1_204))
        msg = a._build_assistant_message(_assistant_response(), "stop")
        from agent.session_persistence import _db_flush_row
        row = _db_flush_row(a, msg, is_current_turn_user=False)
        user_row = _db_flush_row(a, {"role": "user", "content": "hi", "timestamp": 1.0},
                                 is_current_turn_user=True)
    finally:
        a.close()
    assert row["token_count"] == 1_204
    # rows the stamp never touched stay NULL, exactly as before
    assert user_row["token_count"] is None
