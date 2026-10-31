"""``subagent_stop`` releases a delegated child's shared-metrics session through the lifecycle dispatcher."""

from __future__ import annotations

from hermes_cli import lifecycle
from hermes_cli.observability import relay_shared_metrics
from tests.hermes_cli.test_relay_shared_metrics_runtime import direct_runtime

_CHILD = {"session_id": "child", "task_id": "child-task", "parent_session_id": "parent"}


def _sessions() -> dict:
    return relay_shared_metrics._get_runtime(retry_failed=True)._sessions


def _start_child_turn() -> None:
    lifecycle.invoke_hook("pre_llm_call", **_CHILD, platform="subagent")
    lifecycle.invoke_hook("pre_api_request", **_CHILD, api_request_id="r1", provider="p", model="m")


def _task_outcomes(fake) -> list:
    return [event[2]["output"]["outcome"] for event in fake.events
            if event[0] == "scope.pop" and event[1][1] == relay_shared_metrics.TASK_SCOPE]


def _stop_child() -> None:
    lifecycle.invoke_hook("subagent_stop", parent_session_id="parent", child_session_id="child")


def test_subagent_stop_releases_a_finished_child_session(direct_runtime):  # health: allow F811 -- pytest injects the imported direct_runtime fixture by parameter name
    _start_child_turn()
    lifecycle.invoke_hook("post_api_request", **_CHILD, api_request_id="r1", provider="p", model="m", usage=None)
    relay_shared_metrics.finish_task_run(
        session_id="child", task_id="child-task", platform="subagent", result={"completed": True})
    assert "child" in _sessions()

    _stop_child()
    assert "child" not in _sessions()
    _stop_child()  # a repeated stop is a no-op
    assert "child" not in _sessions()


def test_subagent_stop_waits_for_a_child_turn_still_unwinding(direct_runtime):  # health: allow F811 -- pytest injects the imported direct_runtime fixture by parameter name
    # A timed-out child: the parent reports it stopped while its worker is still mid-turn.
    _start_child_turn()
    _stop_child()
    assert "child" in _sessions()

    lifecycle.invoke_hook("post_api_request", **_CHILD, api_request_id="r1", provider="p", model="m", usage=None)
    relay_shared_metrics.finish_task_run(
        session_id="child", task_id="child-task", platform="subagent", result={"interrupted": True})
    assert "child" not in _sessions()
    assert _task_outcomes(direct_runtime) == ["cancelled"]  # the task ends once, with its own outcome
