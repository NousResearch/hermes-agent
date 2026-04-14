"""Tests for agent.coordinator."""

from datetime import datetime
from uuid import UUID

from agent.coordinator import Coordinator
from agent.task_manager import TaskManager


def test_simple_task_single_step():
    coordinator = Coordinator()

    plan = coordinator.plan("Implement D2 coordinator")

    assert plan["source_task"] == "Implement D2 coordinator"
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["title"] == "Implement D2 coordinator"
    assert plan["steps"][0]["execution_mode"] == "local"
    assert plan["steps"][0]["depends_on"] == []
    assert plan["steps"][0]["status"] == "pending"


def test_multi_subtask_plan():
    coordinator = Coordinator()

    plan = coordinator.plan(
        "Ship coordinator",
        subtasks=[
            {"title": "Design coordinator API"},
            {"title": "Add execution flow"},
            {"title": "Write tests"},
        ],
    )

    assert [step["title"] for step in plan["steps"]] == [
        "Design coordinator API",
        "Add execution flow",
        "Write tests",
    ]
    assert [step["execution_mode"] for step in plan["steps"]] == ["local", "local", "local"]
    assert plan["steps"][0]["depends_on"] == []
    assert plan["steps"][1]["depends_on"] == [plan["steps"][0]["step_id"]]
    assert plan["steps"][2]["depends_on"] == [plan["steps"][1]["step_id"]]


def test_delegated_subtask():
    coordinator = Coordinator()

    plan = coordinator.plan(
        "Run mixed plan",
        subtasks=[
            {"title": "Handle locally"},
            {"title": "Ask external reasoner", "needs_external_reasoning": True},
        ],
    )

    assert plan["steps"][0]["execution_mode"] == "local"
    assert plan["steps"][1]["execution_mode"] == "delegated"


def test_execute_plan_creates_tasks(tmp_path):
    coordinator = Coordinator()
    manager = TaskManager(storage_path=tmp_path / "state" / "tasks.json")
    plan = coordinator.plan(
        "Execute plan",
        subtasks=[
            {"title": "Run local step"},
            {"title": "Run delegated step", "needs_external_reasoning": True},
        ],
    )

    executed = coordinator.execute_plan(plan, manager)
    tasks = manager.list()

    assert len(tasks) == 2
    assert [step["status"] for step in executed["steps"]] == ["completed", "completed"]
    assert all(step["task_id"] for step in executed["steps"])
    assert [manager.get(step["task_id"])["kind"] for step in executed["steps"]] == ["local", "delegated"]
    assert [manager.get(step["task_id"])["status"] for step in executed["steps"]] == ["completed", "completed"]


def test_summarize_output():
    coordinator = Coordinator()
    plan = coordinator.plan(
        "Summarize coordinator plan",
        subtasks=[{"title": "First step"}, {"title": "Second step", "needs_external_reasoning": True}],
    )

    summary = coordinator.summarize(plan)

    assert "Summarize coordinator plan" in summary
    assert "First step" in summary
    assert "Second step" in summary
    assert "delegated" in summary


def test_empty_subtasks():
    coordinator = Coordinator()

    plan = coordinator.plan("Fallback single step", subtasks=[])

    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["title"] == "Fallback single step"
    assert plan["steps"][0]["execution_mode"] == "local"


def test_plan_structure():
    coordinator = Coordinator()

    plan = coordinator.plan("Validate structure", subtasks=[{"title": "Only step"}])

    assert set(plan) == {"plan_id", "source_task", "steps", "created_at"}
    UUID(plan["plan_id"])
    datetime.fromisoformat(plan["created_at"])
    assert len(plan["steps"]) == 1

    step = plan["steps"][0]
    assert set(step) == {"step_id", "title", "execution_mode", "depends_on", "status"}
    UUID(step["step_id"])
    assert step["execution_mode"] == "local"
    assert isinstance(step["depends_on"], list)
    assert step["status"] == "pending"
