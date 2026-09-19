"""Skill-lifecycle correlation and privacy tests for the shared-metrics runtime.

Part of the fracture of tests/hermes_cli/test_relay_shared_metrics_runtime.py
(issue #79967, epic #78647): the direct_runtime fixture and its _Relay
double stay in the parent module and are imported here, so the moved test bodies
are byte-identical to their previous location.

Issue numbers live in docstrings, not filenames (tests/test_tests_tree_layout.py).
"""

from __future__ import annotations

import json

from hermes_cli import lifecycle
from hermes_cli.observability import relay_shared_metrics

from tests.hermes_cli.test_relay_shared_metrics_runtime import (  # noqa: F401 — pytest fixture
    direct_runtime,
)


def test_skill_lifecycle_flows_through_relay_to_a_privacy_safe_package(
    direct_runtime,
    tmp_path,
):
    common = {
        "skill_name": "private-skill-name",
        "provenance": "agent_created",
    }
    lifecycle.invoke_hook("on_skill_lifecycle", **common, action="created")
    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        **common,
        action="loaded",
        use_count=1,
        reused=False,
        reuse_after_patch=False,
    )
    lifecycle.invoke_hook("on_skill_lifecycle", **common, action="patched")
    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        **common,
        action="loaded",
        use_count=2,
        reused=True,
        reuse_after_patch=True,
    )

    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    runtime.shutdown()

    marks = [event for event in direct_runtime.events if event[0] == "scope.event"]
    assert [event[1] for event in marks] == [
        "hermes.skill.lifecycle",
        "hermes.skill.load",
        "hermes.skill.lifecycle",
        "hermes.skill.load",
    ]
    assert "private-skill-name" not in json.dumps(marks)

    outbox = tmp_path / "hermes-home" / "telemetry" / "shared_metrics" / "outbox"
    [package_path] = list(outbox.glob("*.json"))
    package = json.loads(package_path.read_text(encoding="utf-8"))
    skill_metrics = [
        metric
        for metric in package["metrics"]
        if metric["name"].startswith("hermes.skill.")
    ]
    assert {metric["name"] for metric in skill_metrics} == {
        "hermes.skill.lifecycle.count",
        "hermes.skill.load.count",
    }
    assert "private-skill-name" not in json.dumps(package)


def test_skill_lifecycle_with_only_task_id_uses_unique_task_scope(direct_runtime):
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    task = runtime.start_task({
        "session_id": "session-1",
        "task_id": "task-1",
        "platform": "cli",
    })
    assert task is not None

    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        action="created",
        skill_name="private-skill-name",
        provenance="agent_created",
        task_id="task-1",
    )

    [mark] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.event" and event[1] == "hermes.skill.lifecycle"
    ]
    assert mark[2]["handle"] == task.handle


def test_skill_task_only_correlation_does_not_guess_across_sessions(direct_runtime):
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    for session_id in ("session-1", "session-2"):
        assert runtime.start_task({
            "session_id": session_id,
            "task_id": "shared-task",
            "platform": "cli",
        }) is not None

    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        action="created",
        skill_name="private-skill-name",
        provenance="agent_created",
        task_id="shared-task",
    )

    [mark] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.event" and event[1] == "hermes.skill.lifecycle"
    ]
    assert "handle" not in mark[2]


def test_late_skill_lifecycle_is_not_reemitted_at_the_root(direct_runtime):
    base = {
        "session_id": "session-1",
        "task_id": "task-1",
        "turn_id": "turn-1",
        "platform": "cli",
    }
    lifecycle.invoke_hook("pre_llm_call", **base)
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )

    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        **base,
        action="loaded",
        skill_name="private-skill-name",
        provenance="local",
        use_count=2,
        reused=True,
        reuse_after_patch=False,
    )

    assert [
        event
        for event in direct_runtime.events
        if event[0] == "scope.event" and event[1] == "hermes.skill.load"
    ] == []


def test_skill_lifecycle_does_not_fallback_across_an_explicit_session(
    direct_runtime,
):
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    assert runtime.start_task({
        "session_id": "session-1",
        "task_id": "task-1",
        "platform": "cli",
    }) is not None

    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        action="created",
        skill_name="private-skill-name",
        provenance="local",
        session_id="wrong-session",
        task_id="task-1",
    )

    assert [
        event
        for event in direct_runtime.events
        if event[0] == "scope.event" and event[1] == "hermes.skill.lifecycle"
    ] == []
