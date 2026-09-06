"""Exercise the real execution and normalized post-request seams with local DBs."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import time

import pytest

from agent.turn_api_call import perform_api_call
from agent.turn_response_intake import _fire_post_api_request_hook
from hermes_state import SessionDB


def _timeline(db):
    # Explicitly choose a time AFTER completion; Windows wall-clock ticks can tie.
    latest = db._read_one("SELECT MAX(completed_at_us) FROM usage_events")[0]
    end = max(time.time_ns() // 1000, (latest or 0) + 1)
    with patch("hermes_state_usage_events.time.time_ns", return_value=end * 1000):
        return db.codex_usage_timeline()


def _agent(db, response, streaming):
    return SimpleNamespace(
        _session_db=db, provider="openai-codex", model="executed-model", api_mode="codex_responses",
        session_id="s", platform="cli", base_url="https://example.invalid", _disable_streaming=not streaming,
        _has_stream_consumers=lambda: streaming,
        _get_transport=lambda: SimpleNamespace(preflight_kwargs=lambda kw, **opts: kw),
        _is_copilot_url=lambda: False, _is_codex_backend=lambda: True,
        _interruptible_streaming_api_call=lambda kw, **opts: response,
        _interruptible_api_call=lambda kw: response,
        _has_pending_redirect=lambda: False,
    )


def _perform(agent, retry=0):
    return perform_api_call(agent, api_kwargs={"model": agent.model}, _original_api_kwargs={},
                            _llm_middleware_trace=[], _moa_prepared_request=None, _retry=SimpleNamespace(),
                            thinking_spinner=None, retry_count=retry, api_call_count=0,
                            api_request_id="same-logical-id", effective_task_id="t", turn_id="turn",
                            interrupted=False).response


def _post(agent, response):
    _fire_post_api_request_hook(agent, response, SimpleNamespace(content="do not persist", tool_calls=[]),
                               "stop", api_messages=[], api_call_count=0, api_duration=999,
                               api_start_time=1, api_request_id="same-logical-id",
                               effective_task_id="t", turn_id="turn")


@pytest.mark.parametrize("streaming", [False, True])
def test_attempts_dedupe_retries_and_freeze_executed_route(tmp_path, monkeypatch, streaming):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    usage = dict(input_tokens=100, output_tokens=20, input_tokens_details={"cached_tokens": 70},
                 output_tokens_details={"reasoning_tokens": 15})
    response = SimpleNamespace(usage=usage, status="completed", model="reported-model")
    with SessionDB(tmp_path / "profile-a" / "state.db") as db:
        agent = _agent(db, response, streaming)
        before = time.time_ns() // 1000
        assert _perform(agent) is response
        after = time.time_ns() // 1000
        # Later route/profile mutations cannot relabel or redirect a finished attempt.
        agent.model, agent.provider = "later-model", "anthropic"
        _post(agent, response)
        _post(agent, response)
        rows = db._read_all("SELECT * FROM usage_events")
        assert len(rows) == 1
        row = rows[0]
        assert row["model"] == "executed-model" and row["provider"] == "openai-codex"
        assert before <= row["completed_at_us"] <= after
        assert row["input_tokens"] == 100 and row["output_tokens"] == 20
        assert row["cache_read_tokens"] == 70 and row["cache_write_tokens"] is None
        assert row["reasoning_tokens"] == 15
        assert row["profile"] and str(tmp_path) not in row["profile"]
        agent.provider, agent.model = "openai-codex", "executed-model"
        _perform(agent, retry=1)
        _post(agent, response)
        rows = db._read_all("SELECT * FROM usage_events")
        assert len(rows) == 2 and len({r["attempt_id"] for r in rows}) == 2
        assert {r["retry_count"] for r in rows} == {0, 1}
        assert _timeline(db)["total"]["processed_tokens"] == 240


@pytest.mark.parametrize("usage,state,input_,output", [
    (None, "missing", None, None), ({}, "missing", None, None),
    ({"input_tokens": 0, "output_tokens": 0}, "reported", 0, 0),
    ({"input_tokens": 5}, "partial", 5, None),
    ({"input_tokens": -1, "output_tokens": True}, "invalid", None, None),
])
def test_missing_usage_is_not_measured_zero(tmp_path, monkeypatch, usage, state, input_, output):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    with SessionDB(tmp_path / "state.db") as db:
        response = SimpleNamespace(usage=usage, status="completed")
        agent = _agent(db, response, False)
        _perform(agent)
        _post(agent, response)
        row = db._read_one("SELECT * FROM usage_events")
        assert row is not None
        assert (row["usage_state"], row["input_tokens"], row["output_tokens"]) == (state, input_, output)


def test_recording_failure_never_breaks_post_hook(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    with SessionDB(tmp_path / "state.db") as db:
        response = SimpleNamespace(usage={"input_tokens": 1, "output_tokens": 1}, status="completed")
        agent = _agent(db, response, False)
        assert _perform(agent) is response
        monkeypatch.setattr(db, "record_usage_event", Mock(side_effect=OSError("sensitive-content")))
        _post(agent, response)
        assert _timeline(db)["coverage"]["recording_failures_in_process"] >= 1
        assert "sensitive-content" not in caplog.text



def test_real_codex_loop_continuation_and_child_do_not_double_count(tmp_path, monkeypatch):
    from run_agent import AIAgent
    from tests.run_agent.test_run_agent_codex_responses import (
        _patch_agent_bootstrap, _codex_message_response, _codex_incomplete_message_response,
    )
    from tools.delegate_tool import _open_child_session_db
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *args, **kwargs: 0)
    with SessionDB(tmp_path / "non-launch-profile" / "state.db") as db:
        parent = AIAgent(model="gpt-5-codex", provider="openai-codex", api_mode="codex_responses",
                         api_key="test-only", base_url="https://example.invalid", session_db=db,
                         quiet_mode=True, max_iterations=4, skip_context_files=True, skip_memory=True)
        parent.compression_enabled = False
        parent.save_trajectories = False
        responses = iter([_codex_incomplete_message_response("partial"), _codex_message_response("finished")])
        monkeypatch.setattr(parent, "_run_codex_stream", lambda *args, **kwargs: next(responses))
        result = parent.run_conversation("test")
        assert result["completed"]
        assert _timeline(db)["total"]["events"] == 2
        assert _timeline(db)["total"]["processed_tokens"] == 14
        with _open_child_session_db(parent) as child_db:
            child = _agent(child_db, _codex_message_response("child"), False)
            child.is_subagent = True
            response = _perform(child)
            _post(child, response)
            # A parent-facing delegated summary is not another executed Codex attempt.
            summary_agent = _agent(db, response, False)
            summary_agent.provider = "moa"
            _post(summary_agent, response)
        assert _timeline(db)["total"]["events"] == 3
        assert _timeline(db)["total"]["processed_tokens"] == 22
        parent.close()


def test_error_retry_and_synthetic_response_have_no_invented_usage(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    with SessionDB(tmp_path / "state.db") as db:
        response = SimpleNamespace(usage={"input_tokens": 2, "output_tokens": 1}, status="completed")
        agent = _agent(db, response, False)
        agent._interruptible_api_call = Mock(side_effect=[TimeoutError("test"), response])
        with pytest.raises(TimeoutError):
            _perform(agent)
        assert _timeline(db)["total"]["events"] == 0
        _perform(agent, retry=1)
        _post(agent, response)
        # Windows clock ticks can tie completion and as_of (correctly excluded).
        completed = db._read_one("SELECT completed_at_us FROM usage_events")[0]
        monkeypatch.setattr(time, "time_ns", lambda: (completed + 1) * 1000)
        assert _timeline(db)["total"]["events"] == 1
        monkeypatch.setattr("hermes_cli.middleware.run_llm_execution_middleware",
                            lambda *args, **kwargs: response)
        _perform(agent)
        _post(agent, response)
        assert _timeline(db)["total"]["events"] == 1



@pytest.mark.parametrize("details,expected", [
    ({"cache_write_tokens": 12}, 12), ({"cache_creation_tokens": 9}, 9),
    ({"cache_write_tokens": 0, "cache_creation_tokens": 9}, 0),
])
def test_codex_cache_write_details_preserve_zero(tmp_path, monkeypatch, details, expected):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    with SessionDB(tmp_path / "state.db") as db:
        response = SimpleNamespace(usage={"input_tokens": 100, "output_tokens": 10,
                                         "input_tokens_details": details}, status="completed")
        agent = _agent(db, response, False)
        _perform(agent)
        _post(agent, response)
        row = db._read_one("SELECT * FROM usage_events")
        assert row["cache_write_tokens"] == expected
        assert _timeline(db)["total"]["processed_tokens"] == 110
