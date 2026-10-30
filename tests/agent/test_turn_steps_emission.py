"""The main-loop pool emits one ``turn_steps`` row per provider response, usage-less included."""

from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


class _Compressor:
    threshold_tokens = 0
    awaiting_real_usage_after_compression = False
    _context_probed = False

    def update_from_response(self, usage):
        self.last_usage = usage

    def note_usage_less_response(self):
        self.usage_less = True


def _agent(db, monkeypatch):
    import agent.turn_usage as tu

    monkeypatch.setattr(
        tu, "estimate_usage_cost",
        lambda *a, **k: SimpleNamespace(amount_usd=0.1, status="ok", source="api"),
    )
    monkeypatch.setattr(
        tu, "_loop_mod",
        lambda: SimpleNamespace(_should_rearm_compression_budget=lambda *a, **k: False),
    )
    agent = SimpleNamespace(
        session_id="s", platform="cli", _session_db=db, _session_db_created=True,
        model="main-model", provider="prov", api_mode="chat_completions", base_url="http://prov",
        api_key="k", client=None, context_compressor=_Compressor(), verbose_logging=False,
        quiet_mode=True, log_prefix="", _vprint=lambda *a, **k: None, _persist_disabled=True,
        _current_turn_id="turn-a", _current_api_request_id="turn-a:api:1",
        _current_iteration_uid="turn-a:it:abc", _user_turn_count=1, _api_call_count=1,
        _turn_route_task="", _api_latency_history=[], _api_output_history=[],
        session_api_calls=0, session_prompt_tokens=0, session_completion_tokens=0,
        session_total_tokens=0, session_input_tokens=0, session_output_tokens=0,
        session_cache_read_tokens=0, session_cache_write_tokens=0, session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0, session_cost_status=None, session_cost_source=None,
    )
    return agent


def test_usage_response_emits_a_ledger_row_with_identity_and_tools(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", "cli", model="main-model")
    try:
        agent = _agent(db, monkeypatch)
        response = SimpleNamespace(
            usage={"input_tokens": 10, "output_tokens": 5},
            id="resp-1",
            choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[SimpleNamespace(function=SimpleNamespace(name="bash"))]))],
        )
        from agent.turn_usage import record_response_usage

        record_response_usage(
            agent, response, messages=[{"role": "user", "content": "hi"}],
            api_call_count=1, api_duration=0.25, compression_attempts=0,
            max_compression_attempts=3, finish_reason="tool_calls",
        )
        assert db.flush_token_counts() is True

        [row] = db.turn_steps_for_session("s")
        assert (row["step_seq"], row["turn_id"], row["turn_seq"], row["step_index"]) == (1, "turn-a", 1, 1)
        assert row["iteration_uid"] == "turn-a:it:abc"
        assert row["request_id"] == "turn-a:api:1"
        assert row["tool_names"] == '["bash"]'
        assert row["tool_call_count"] == 1
        assert row["finish_reason"] == "tool_calls"
        assert (row["input_tokens"], row["output_tokens"], row["total_tokens"]) == (10, 5, 15)
        assert row["usage_status"] == "ok"
        assert row["duration_ms"] == 250
    finally:
        db.close()


def test_usage_less_response_emits_a_missing_row_only(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", "cli", model="main-model")
    try:
        agent = _agent(db, monkeypatch)
        from agent.turn_usage import record_response_usage

        record_response_usage(
            agent, SimpleNamespace(usage=None), messages=[{"role": "user", "content": "hi"}],
            api_call_count=1, api_duration=0.05, compression_attempts=0, max_compression_attempts=3,
        )
        assert db.flush_token_counts() is True

        [row] = db.turn_steps_for_session("s")
        assert (row["usage_status"], row["step_seq"], row["api_calls"]) == ("missing", 1, 1)
        assert (row["input_tokens"], row["output_tokens"]) == (0, 0)
        assert db.get_session("s")["api_call_count"] == 0
    finally:
        db.close()


def test_tool_names_reads_anthropic_and_responses_shapes():
    from agent.turn_usage import _tool_names_from_response

    anthropic = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="x"),
        SimpleNamespace(type="tool_use", name="read_file"),
    ])
    responses = SimpleNamespace(output=[
        SimpleNamespace(type="function_call", name="bash"),
        SimpleNamespace(type="message", name="ignored"),
    ])
    assert _tool_names_from_response(anthropic) == ["read_file"]
    assert _tool_names_from_response(responses) == ["bash"]


def _codex_agent(db):
    return SimpleNamespace(
        session_id="s", _session_db=db, _session_db_created=True,
        model="codex-model", provider="codex", base_url="http://codex", api_key="",
        context_compressor=_Compressor(), session_api_calls=0,
        session_input_tokens=0, session_output_tokens=0, session_cache_read_tokens=0,
        session_cache_write_tokens=0, session_reasoning_tokens=0,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        session_estimated_cost_usd=0.0, session_cost_status=None, session_cost_source=None,
        _current_turn_id="turn-c", _user_turn_count=2, _api_call_count=3,
        _current_iteration_uid="", _agent_name="",
    )


@pytest.mark.parametrize("usage,total", [({"inputTokens": 10, "cachedInputTokens": 2, "outputTokens": 5, "totalTokens": 15}, 15), (None, 0)])
def test_codex_app_server_turn_emits_one_ok_row_even_without_usage(tmp_path, monkeypatch, usage, total):
    import agent.usage_pricing as pricing

    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", "cli", model="codex-model")
    try:
        monkeypatch.setattr(
            pricing, "estimate_usage_cost",
            lambda *a, **k: SimpleNamespace(amount_usd=0.0, status="ok", source="api"),
        )
        turn = SimpleNamespace(
            token_usage_last=usage, tool_iterations=2,
            projected_messages=[{"role": "assistant", "tool_calls": [{"function": {"name": "bash"}}]}],
            interrupted=False, error=None, turn_id="ct", thread_id="th", model_context_window=None,
        )
        from agent.codex_runtime import _record_codex_app_server_usage

        _record_codex_app_server_usage(_codex_agent(db), turn)
        assert db.flush_token_counts() is True

        [row] = db.turn_steps_for_session("s")
        assert (row["step_seq"], row["step_index"], row["turn_seq"]) == (1, 3, 2)
        assert row["usage_status"] == "ok"
        assert row["total_tokens"] == total
        assert row["request_id"] == "codex:th:ct"
        assert row["tool_names"] == '["bash"]'
        assert db.get_session("s")["api_call_count"] == 1
    finally:
        db.close()
