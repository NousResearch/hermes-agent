"""Each public run ends once, with its own identity and the richer task metrics preserved."""

from unittest.mock import patch

import pytest

from tests.agent import test_run_agent as _base
from tests.agent.test_cross_process_turn_lease import _DB, _agent_with_db
from tests.agent.test_run_agent import _mock_response
from tests.hermes_cli import test_relay_shared_metrics_runtime as _metrics

# Reuse fixtures by name (an import would shadow-redefine them — F811).
agent, _mock_plugin_discovery = _base.agent, _base._mock_plugin_discovery
direct_runtime = _metrics.direct_runtime


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


def _forbidden():
    err = Exception("Error code: 403 - forbidden")
    err.status_code = 403
    return err


def _assert_own_identity(end, agent, task_id):
    assert (end["session_id"], end["task_id"]) == (agent.session_id, task_id)
    assert end["turn_id"] == agent._current_turn_id


def test_normal_turn_ends_once_with_its_outcome(ready, session_ends):
    ready.client.chat.completions.create.return_value = _mock_response(content="Final answer")

    result = ready.run_conversation("hello", task_id="task-ok")

    [end] = session_ends
    _assert_own_identity(end, ready, "task-ok")
    assert (end["completed"], end["failed"], end["interrupted"]) == (True, False, False)
    assert end["turn_exit_reason"] == result["turn_exit_reason"]


def test_non_retryable_api_error_ends_once(ready, session_ends):
    ready.client.chat.completions.create.side_effect = _forbidden()

    result = ready.run_conversation("hello", task_id="task-403")

    assert ready.client.chat.completions.create.called and result["failed"] is True
    [end] = session_ends
    _assert_own_identity(end, ready, "task-403")
    assert (end["completed"], end["failed"], end["interrupted"]) == (False, True, False)


def test_persist_disabled_run_fires_no_session_end(ready, session_ends):
    ready._persist_disabled = True
    ready.client.chat.completions.create.side_effect = _forbidden()

    ready.run_conversation("hello", task_id="task-review")

    assert ready.client.chat.completions.create.called
    assert session_ends == []


def test_refused_turn_lease_ends_once_with_the_runs_ids(session_ends, monkeypatch):
    def never(*_args, **_kwargs):
        raise AssertionError("turn must not start without a lease")

    monkeypatch.setattr("agent.conversation_loop.run_conversation", never)
    held = _agent_with_db(_DB(acquire_result=False))

    result = held.run_conversation("new message", task_id="task-busy")

    [end] = session_ends
    assert result["failed"] is True and end["failed"] is True and end["completed"] is False
    assert end["task_id"] == "task-busy"
    assert end["turn_id"].startswith(f"{held.session_id}:task-busy:")


def test_raising_run_keeps_its_error_and_failure_class(direct_runtime, ready, tmp_path):
    """Through the real façade and lifecycle dispatch: the shared-metrics task close keeps the
    exception class, the end hook fires once, and a raising observer replaces nothing."""
    from hermes_cli import lifecycle, plugins

    ends = []

    def broken_observer(**kwargs):
        ends.append(kwargs)
        raise RuntimeError("observer broke")

    plugins._plugin_manager._hooks["on_session_end"] = [broken_observer]
    # Restoring the primary runtime reloads the credential store at turn start.
    with patch.object(ready, "_restore_primary_runtime", side_effect=OSError("credential store unreadable")):
        with pytest.raises(OSError, match="credential store unreadable"):
            ready.run_conversation("hello", task_id="task-raise")
    lifecycle.finalize_session(session_id=ready.session_id)
    _metrics._join_export_workers()

    [end] = ends
    assert (end["task_id"], end["failed"], end["completed"]) == ("task-raise", True, False)
    [finished] = _metrics._stored_counters(tmp_path)["hermes.task_run.finished"]
    assert finished["failure_class"] == "exception"


def test_end_naming_the_rotated_session_still_closes_the_runs_task(direct_runtime):
    """Compression hands s1 to s1c mid-run, so the end names s1c: it must still close s1's task
    (an end naming an unrelated session must not: test_skipped_turn_does_not_finish_...)."""
    from hermes_cli import lifecycle
    from hermes_cli.observability import relay_shared_metrics as metrics

    metrics.start_task_run(session_id="s1", task_id="t1", platform="cli")
    metrics.rotate_segment("s1", "s1c")
    lifecycle.invoke_hook(
        "on_session_end", session_id="s1c", task_id="t1", completed=True, failed=False,
        interrupted=False, turn_exit_reason="text_response(stop)",
    )

    assert [e for e in direct_runtime.events if e[0] == "scope.pop" and e[1][1] == metrics.TASK_SCOPE]
