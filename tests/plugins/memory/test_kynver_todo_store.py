"""Kynver-first todo store — task plane vs Hermes in_progress semantics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from plugins.memory.kynver.operating_config import OperatingLinkage
from plugins.memory.kynver.operating_context import OperatingContext
from plugins.memory.kynver.todo_store import KynverTodoStore


def _store_with_client(client: MagicMock) -> KynverTodoStore:
    ctx = OperatingContext(plan_id="plan-1", task_id="task-1")
    store = KynverTodoStore(
        client,
        operating_context=ctx,
        allow_fallback=True,
        hermes_session_id="sess-1",
    )
    store._linkage = OperatingLinkage(  # noqa: SLF001 — test seam
        plan_id="plan-1",
        task_id="task-1",
        session_id="sess-1",
        executor_ref="hermes:forge",
    )
    return store


def test_write_maps_in_progress_to_ready_on_task_plane():
    client = MagicMock()
    client.get.return_value = {"tasks": []}
    client.post.return_value = {"id": "agent-task-1"}
    store = _store_with_client(client)

    with patch(
        "plugins.memory.kynver.todo_store.ensure_intake_task",
        return_value=store._ctx,
    ), patch(
        "plugins.memory.kynver.todo_store.safe_project_todo_write",
        return_value={"projected": True},
    ), patch.object(store, "read", return_value=[]):
        store.write(
            [{"id": "t1", "content": "Focus work", "status": "in_progress"}],
            merge=False,
        )

    create_body = client.post.call_args_list[0][0][1]
    assert create_body["status"] == "ready"
    assert "running" not in str(create_body).lower() or create_body["status"] != "running"


def test_degraded_fallback_uses_local_store():
    client = MagicMock()
    client.get.side_effect = RuntimeError("network down")
    store = _store_with_client(client)

    items = store.write(
        [{"id": "t1", "content": "Local only", "status": "pending"}],
        merge=False,
    )
    assert store.degraded is True
    assert items == [{"id": "t1", "content": "Local only", "status": "pending"}]
