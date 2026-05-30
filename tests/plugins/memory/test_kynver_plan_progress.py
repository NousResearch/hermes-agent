"""Plan progress projection and read-back for Kynver-first Forge operating tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from plugins.memory.kynver.operating_config import OperatingLinkage, kynver_operating_tools_enabled
from plugins.memory.kynver.plan_progress import (
    project_todo_write,
    reconcile_todos_from_kynver,
    safe_project_todo_write,
)
from plugins.memory.kynver.pre_transition import PreTransitionError, hermes_row_key


def test_hermes_row_key_prefix():
    assert hermes_row_key("abc") == "hermes-todo:abc"
    assert hermes_row_key("hermes-todo:x") == "hermes-todo:x"


def test_project_todo_write_sets_focus_not_running():
    client = MagicMock()
    client.get.return_value = {"items": []}
    linkage = OperatingLinkage(
        plan_id="plan-1",
        task_id="task-1",
        session_id="sess-1",
        executor_ref="hermes:forge",
    )
    todos = [
        {"id": "a", "content": "Step A", "status": "in_progress"},
        {"id": "b", "content": "Step B", "status": "pending"},
    ]
    result = project_todo_write(client, linkage, todos, merge=False)
    assert result["projected"] is True
    client.post.assert_any_call(
        "/plans/plan-1/progress-focus",
        {
            "rowKey": "hermes-todo:a",
            "taskId": "task-1",
            "roleLane": "implementer",
            "executorRef": "hermes:forge",
            "note": "Hermes todo focus: Step A",
        },
    )
    focus_calls = [c for c in client.post.call_args_list if c.args[0].endswith("progress-focus")]
    assert len(focus_calls) == 1
    assert "running" not in str(focus_calls)


def test_running_row_without_focus_maps_to_pending():
    client = MagicMock()
    client.get.side_effect = [
        {"plan": {"id": "plan-1", "inProgressRowKey": None}},
        {
            "items": [
                {"rowKey": "hermes-todo:a", "title": "Executor lease", "status": "running"},
            ]
        },
    ]
    linkage = OperatingLinkage(plan_id="plan-1", task_id=None, session_id=None, executor_ref="hermes:forge")
    merged = reconcile_todos_from_kynver(client, linkage, [])
    assert merged[0]["status"] == "pending"


def test_read_back_merges_kynver_focus():
    client = MagicMock()
    client.get.side_effect = [
        {"plan": {"id": "plan-1", "inProgressRowKey": "hermes-todo:a"}},
        {
            "items": [
                {"rowKey": "hermes-todo:a", "title": "Step A", "status": "todo"},
                {"rowKey": "hermes-todo:b", "title": "Step B", "status": "partial"},
            ]
        },
    ]
    linkage = OperatingLinkage(plan_id="plan-1", task_id=None, session_id=None, executor_ref="hermes:forge")
    merged = reconcile_todos_from_kynver(
        client,
        linkage,
        [{"id": "b", "content": "local b", "status": "pending"}],
    )
    by_id = {item["id"]: item for item in merged}
    assert by_id["a"]["status"] == "in_progress"
    assert by_id["b"]["status"] == "completed"


def test_operating_tools_default_on_with_credentials(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "KYNVER_API_URL=https://example.test\nKYNVER_API_KEY=secret\nKYNVER_AGENT_OS_SLUG=forge\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("plugins.memory.kynver.agentos_bridge._active_env_path", lambda: env_path)
    assert kynver_operating_tools_enabled() is True
    monkeypatch.setenv("KYNVER_OPERATING_TOOLS", "false")
    assert kynver_operating_tools_enabled() is False


def test_safe_project_returns_blocked_without_raise():
    client = MagicMock()
    client.get.return_value = {"items": [{"rowKey": "hermes-todo:a", "status": "running"}]}
    linkage = OperatingLinkage(plan_id="p", task_id=None, session_id=None, executor_ref="hermes:forge")
    out = safe_project_todo_write(
        client,
        linkage,
        [{"id": "a", "content": "x", "status": "in_progress"}],
        merge=False,
    )
    assert out.get("blocked") is True


def test_single_in_progress_guard():
    client = MagicMock()
    client.get.return_value = {"items": []}
    linkage = OperatingLinkage(plan_id="p", task_id=None, session_id=None, executor_ref="hermes:forge")
    with pytest.raises(PreTransitionError):
        project_todo_write(
            client,
            linkage,
            [
                {"id": "1", "status": "in_progress", "content": "a"},
                {"id": "2", "status": "in_progress", "content": "b"},
            ],
            merge=False,
        )


def test_pre_tool_call_blocks_todo_when_projection_blocked(monkeypatch):
    from plugins.memory.kynver.operating_hooks import on_pre_tool_call

    monkeypatch.setenv("KYNVER_API_URL", "https://example.test")
    monkeypatch.setenv("KYNVER_API_KEY", "secret")
    monkeypatch.setenv("KYNVER_AGENT_OS_SLUG", "forge")
    monkeypatch.setenv("KYNVER_PLAN_ID", "plan-1")

    client = MagicMock()
    client.get.return_value = {"items": [{"rowKey": "hermes-todo:a", "status": "running"}]}
    monkeypatch.setattr(
        "plugins.memory.kynver.operating_hooks._client",
        lambda: client,
    )
    monkeypatch.setattr(
        "plugins.memory.kynver.operating_hooks.load_operating_linkage",
        lambda: OperatingLinkage(
            plan_id="plan-1",
            task_id=None,
            session_id=None,
            executor_ref="hermes:forge",
        ),
    )

    block = on_pre_tool_call(
        tool_name="todo",
        args={
            "todos": [{"id": "a", "content": "Step", "status": "in_progress"}],
            "merge": False,
        },
    )

    assert block == {
        "action": "block",
        "message": block["message"],
    }
    assert "running" in block["message"] or "lease" in block["message"]
