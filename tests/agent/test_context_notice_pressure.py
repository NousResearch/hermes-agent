"""Pressure warnings require distinct attempts and provider-proven relief."""
from types import SimpleNamespace

import pytest

from agent.context_compressor import ContextCompressor
from agent.turn_preflight_gate import run_preflight_gate
from agent.turn_usage import record_response_usage
from hermes_state import SessionDB
from tests.agent.test_context_notices import _agent, _outcome


def _usage(agent, tokens):
    response = SimpleNamespace(usage=None if tokens is None else {"prompt_tokens": tokens, "completion_tokens": 1})
    return record_response_usage(agent, response, messages=[], api_call_count=2,
                                 api_duration=0.1, compression_attempts=2, max_compression_attempts=3)


def _pressure_agent(db, monkeypatch, events):
    agent = _agent(db, monkeypatch, events)
    agent.context_compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    agent.context_compressor.bind_session_state(db, agent.session_id)
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens",
                 "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "api_calls", "estimated_cost_usd"):
        setattr(agent, f"session_{name}", 0)
    agent.model, agent.provider, agent.api_mode, agent.base_url = "test/model", "test", "chat_completions", ""
    agent.compression_enabled, agent.quiet_mode, agent.verbose_logging = True, True, False
    agent._session_db_created = True
    return agent


def _estimated_pressure(agent):
    pressure = agent.context_compressor.threshold_tokens + 100
    return run_preflight_gate(
        agent, request_pressure_tokens=pressure, _last_preflight_pressure=pressure,
        _moa_prepared_request=None, pending_moa_prepared_request=None, messages=[],
        system_message="", user_message="", active_system_prompt="", conversation_history=[],
        api_call_count=1, compression_attempts=1, max_compression_attempts=3,
        effective_task_id=agent.session_id, final_response=None, failed=False,
        _turn_exit_reason=None, _compression_timeout_exhausted=False,
        _preflight_compression_blocked=False, _provider_overflow_recovery_pending=False,
    )


@pytest.mark.parametrize("source", ["estimated", "provider", "would_grow"])
def test_pressure_persists_until_positive_provider_recovery(monkeypatch, tmp_path, source):
    path = tmp_path / "state.db"
    events = []
    db = SessionDB(db_path=path)
    db.create_session("conversation", source="gui")
    try:
        agent = _pressure_agent(db, monkeypatch, events)
        writes = []
        update = db.update_context_notice_state
        with monkeypatch.context() as quiet:
            quiet.setattr(db, "update_context_notice_state", lambda *a, **kw: (writes.append(a), update(*a, **kw))[1])
            _usage(agent, 100)
        assert writes == [], "ordinary responses with no pending notice must not add synchronous DB writes"
        for attempt in ("first", "second"):
            agent = _pressure_agent(db, monkeypatch, events)
            _outcome(agent, attempt, failure="would_grow" if source == "would_grow" else None,
                     committed=source != "would_grow")
            if source == "estimated":
                assert _estimated_pressure(agent)._preflight_compression_blocked
                _estimated_pressure(agent)  # same attempt, not a second strike
            elif source == "provider":
                _usage(agent, agent.context_compressor.threshold_tokens + 100)
                _usage(agent, agent.context_compressor.threshold_tokens + 100)
            if attempt == "first":
                assert not [e for e in events if e[0] == "notification.show"]
            db.close()
            db = SessionDB(db_path=path)
        shows = [e for e in events if e[0] == "notification.show"]
        assert len(shows) == 1, "two distinct unsuccessful pressure attempts must surface a sticky notice"
        assert "pressure" in shows[0][2]["text"].lower()
        if source == "estimated":
            assert "estimated" in shows[0][2]["text"].lower()
        key = shows[0][2]["key"]
        events.clear()
        agent = _pressure_agent(db, monkeypatch, events)
        agent.context_compressor.update_model("test/other", 100_000)
        _outcome(agent, "healthy-summary", failure=None, committed=True)
        for missing in (None, 0):
            _usage(agent, missing)
        assert events == [], "probe/model reset, compacted, and missing usage are not pressure recovery"
        _usage(agent, agent.context_compressor.threshold_tokens - 1)
        assert events[0][:2] == ("notification.clear", "runtime")
        assert events[0][2] == {"key": key, "state_key": key,
                                "state_revision": db.get_context_notice_state("conversation")["revision"]}
        assert len(events) == 2 and events[1][2]["kind"] == "ttl"
    finally:
        db.close()
