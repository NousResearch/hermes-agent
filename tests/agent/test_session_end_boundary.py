"""``on_session_end`` fires exactly once per ``run_conversation``, from every way a run ends.

Only ``finalize_turn`` used to emit it, so a run that ended on a non-retryable provider error,
a turn-start preflight timeout, the codex app-server runtime or a raised exception never told
per-run observers (disk-cleanup, outbound webhooks, shared metrics) that it was over.
"""

from unittest.mock import patch

import pytest

from tests.agent import test_run_agent as _base
from tests.agent.test_run_agent import _mock_response

# Reuse the facade's fixtures by name (an import would shadow-redefine ``agent`` — F811).
agent = _base.agent
_mock_plugin_discovery = _base._mock_plugin_discovery


@pytest.fixture()
def session_ends(monkeypatch):
    ends = []

    def capture(name, **kwargs):
        if name == "on_session_end":
            ends.append(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", capture)
    return ends


@pytest.fixture()
def ready(agent):
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        yield agent


def _forbidden(status_code=403):
    err = Exception(f"Error code: {status_code} - forbidden")
    err.status_code = status_code
    return err


def _assert_own_identity(end, agent, task_id):
    assert end["session_id"] == agent.session_id
    assert end["task_id"] == task_id
    assert end["turn_id"] == agent._current_turn_id
    assert end["turn_id"].startswith(f"{agent.session_id}:{task_id}:")


def test_normal_turn_ends_once_with_its_outcome(ready, session_ends):
    ready.client.chat.completions.create.return_value = _mock_response(content="Final answer")

    result = ready.run_conversation("hello", task_id="task-ok")

    assert result["completed"] is True
    [end] = session_ends
    _assert_own_identity(end, ready, "task-ok")
    assert (end["completed"], end["failed"], end["interrupted"]) == (True, False, False)
    assert end["turn_exit_reason"] == result["turn_exit_reason"]


def test_non_retryable_api_error_ends_once(ready, session_ends):
    ready.client.chat.completions.create.side_effect = _forbidden()

    result = ready.run_conversation("hello", task_id="task-403")

    assert ready.client.chat.completions.create.called
    assert result["failed"] is True
    [end] = session_ends
    _assert_own_identity(end, ready, "task-403")
    assert (end["completed"], end["failed"], end["interrupted"]) == (False, True, False)


def test_turn_start_preflight_timeout_ends_once(ready, session_ends):
    compressor = ready.context_compressor
    ready.compression_enabled = True
    compressor.protect_first_n = compressor.protect_last_n = 0
    compressor.threshold_tokens = compressor._resolved_context_length = 1
    compressor.should_compress = lambda *_a, **_k: True
    compressor.should_defer_preflight_to_real_usage = lambda *_a, **_k: False
    compressor.get_active_compression_failure_cooldown = lambda *_a, **_k: None

    def _timed_out(messages, _system_message, **_kwargs):
        from agent.conversation_compression import (
            mark_context_compression_timed_out, reset_context_compression_timeout_outcome,
        )
        reset_context_compression_timeout_outcome(ready)
        mark_context_compression_timed_out(ready)
        return messages, ready._cached_system_prompt

    ready._compress_context = _timed_out
    history = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "older"}]

    result = ready.run_conversation("continue", conversation_history=history, task_id="task-pf")

    ready.client.chat.completions.create.assert_not_called()
    assert result["turn_exit_reason"] == "context_compression_timeout"
    [end] = session_ends
    _assert_own_identity(end, ready, "task-pf")
    assert (end["completed"], end["failed"]) == (False, True)
    assert end["turn_exit_reason"] == "context_compression_timeout"


def test_codex_app_server_runtime_ends_once(ready, session_ends):
    ready.api_mode = "codex_app_server"

    def _app_server_turn(**kwargs):  # stands in for the codex subprocess
        return {"final_response": "done", "messages": kwargs["messages"], "api_calls": 1,
                "completed": True, "failed": False, "interrupted": False}

    ready._run_codex_app_server_turn = _app_server_turn

    result = ready.run_conversation("hello", task_id="task-codex")

    assert result["final_response"] == "done"
    ready.client.chat.completions.create.assert_not_called()
    [end] = session_ends
    _assert_own_identity(end, ready, "task-codex")
    assert end["completed"] is True


def test_persist_disabled_run_fires_no_session_end(ready, session_ends):
    ready._persist_disabled = True
    ready.client.chat.completions.create.side_effect = _forbidden()

    ready.run_conversation("hello", task_id="task-review")

    assert ready.client.chat.completions.create.called
    assert session_ends == []


def test_context_build_failure_keeps_its_error_and_reports_this_runs_identity(ready, monkeypatch):
    """A cached gateway agent still carries the previous run's ids while this run's context build
    fails before binding its own; a broken observer must not replace the build's error."""
    ends = []

    def broken_observer(name, **kwargs):
        if name == "on_session_end":
            ends.append(kwargs)
            raise RuntimeError("observer broke")
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", broken_observer)
    ready._current_task_id, ready._current_turn_id = "previous-task", "previous-turn"

    # Restoring the primary runtime reloads the credential store before the turn binds its ids.
    with patch.object(ready, "_restore_primary_runtime", side_effect=OSError("credential store unreadable")):
        with pytest.raises(OSError, match="credential store unreadable"):
            ready.run_conversation("hello", task_id="task-now")

    [end] = ends
    assert end["task_id"] == "task-now"
    assert end["turn_id"].startswith(f"{ready.session_id}:task-now:")
    assert (end["completed"], end["failed"], end["interrupted"]) == (False, True, False)

