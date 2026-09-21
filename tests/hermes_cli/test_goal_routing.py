from __future__ import annotations

import json


def test_build_goal_kanban_task_keeps_open_risks_visible():
    from hermes_cli.goal_routing import build_goal_kanban_task

    task = build_goal_kanban_task(
        goal="Ship webhook-driven PR review routing",
        subgoals=["Create route safely", "Do not auto-dispatch destructive work"],
        session_id="sid-123",
    )

    assert task.title == "Goal: Ship webhook-driven PR review routing"
    assert task.assignee == "orchestrator"
    assert task.triage is True
    assert "Fleet hygiene" in task.body
    assert "open risk" in task.body.lower()
    assert "live count" in task.body
    assert "GitNexus/LadybugDB" in task.body
    assert "Webhook autonomy" in task.body
    assert "Skill cross-pollination" in task.body
    assert "Dispatcher safety" in task.body
    assert "Kensei owns cron/Kanban dispatch" in task.body
    assert "Do not describe broken items as resolved" in task.body
    assert "Create route safely" in task.body
    assert "Do not auto-dispatch destructive work" in task.body
    assert len(task.body) < 1500


def test_build_goal_child_tasks_from_subgoals_are_linked_to_parent():
    from hermes_cli.goal_routing import build_goal_child_tasks

    children = build_goal_child_tasks(
        parent_task_id="t_parent",
        goal="Ship routed goal",
        subgoals=["Implement backend route", "Run QA review"],
        session_id="sid-child",
    )

    assert len(children) == 2
    assert children[0].title == "Sub-goal 1: Implement backend route"
    assert children[0].assignee == "octacon-backend"
    assert children[0].parents == ("t_parent",)
    assert "Acceptance criteria" in children[0].body
    assert len(children[0].body) < 1500
    assert children[1].assignee == "quan-code"


def test_route_goal_to_kanban_can_decompose_children_after_parent_creation():
    from hermes_cli.goal_routing import route_goal_to_kanban

    calls = []

    def fake_create(args):
        calls.append(args)
        return json.dumps({"ok": True, "task_id": f"task-{len(calls)}", "status": "backlog"})

    result = route_goal_to_kanban(
        "Build route",
        subgoals=["Implement backend", "Write content draft"],
        session_id="sid-decompose",
        create_task=fake_create,
        decompose=True,
    )

    assert result.task_id == "task-1"
    assert result.child_task_ids == ("task-2", "task-3")
    assert len(calls) == 3
    assert calls[1]["parents"] == ["task-1"]
    assert calls[1]["assignee"] == "octacon-backend"
    assert calls[2]["assignee"] == "ceecee-writer"


def test_goal_kanban_task_serialises_to_kanban_create_args():
    from hermes_cli.goal_routing import build_goal_kanban_task

    task = build_goal_kanban_task("Audit fleet", subgoals=[], session_id="sid-456")
    args = task.to_kanban_create_args()

    assert args["title"] == "Goal: Audit fleet"
    assert args["assignee"] == "orchestrator"
    assert args["triage"] is True
    assert args["initial_status"] == "backlog"
    assert args["priority"] == 1
    assert args["theme"] == "goal-routing"
    assert args["session_id"] == "sid-456"
    assert args["idempotency_key"].startswith("goal-route:sid-456:")
    json.dumps(args)


def test_route_goal_to_kanban_uses_injected_creator_and_returns_task_id():
    from hermes_cli.goal_routing import route_goal_to_kanban

    calls = []

    def fake_create(args):
        calls.append(args)
        return json.dumps({"ok": True, "task_id": "task-123", "status": "ready"})

    result = route_goal_to_kanban(
        "Build the routing layer",
        subgoals=["with tests"],
        session_id="sid-789",
        create_task=fake_create,
    )

    assert result.task_id == "task-123"
    assert result.status == "ready"
    assert len(calls) == 1
    assert calls[0]["assignee"] == "orchestrator"
    assert "with tests" in calls[0]["body"]


def test_route_goal_to_kanban_reports_creator_errors_without_hiding_risks():
    from hermes_cli.goal_routing import route_goal_to_kanban

    def fake_create(_args):
        return json.dumps({"ok": False, "error": "kanban unavailable"})

    result = route_goal_to_kanban(
        "Fix knowledge loop",
        subgoals=[],
        session_id="sid-err",
        create_task=fake_create,
    )

    assert result.task_id is None
    assert result.error == "kanban unavailable"
    assert "GitNexus/LadybugDB" in result.task.body
