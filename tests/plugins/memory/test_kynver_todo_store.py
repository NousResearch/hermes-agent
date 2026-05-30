"""KynverTodoStore — plan progress projection, degraded fallback, no running lease."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from plugins.memory.kynver.operating_config import OperatingLinkage
from plugins.memory.kynver.plan_progress import project_todo_write
from plugins.memory.kynver.pre_transition import PreTransitionError
from plugins.memory.kynver.todo_store import KynverTodoStore


class PlanProgressFakeClient:
    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.focus_key: str | None = None
        self.calls: list[tuple] = []
        self.config = type("C", (), {"enabled": True})()

    def get(self, path, **kwargs):
        self.calls.append(("GET", path))
        if path.endswith("/progress-rows"):
            return {"items": list(self.rows.values())}
        if "/plans/" in path and not path.endswith("progress-rows"):
            return {"plan": {"id": "plan-1", "inProgressRowKey": self.focus_key}}
        return {}

    def post(self, path, body, **kwargs):
        self.calls.append(("POST", path, body))
        if path.endswith("progress-rows"):
            for row in body.get("rows", []):
                self.rows[row["rowKey"]] = dict(row)
        if path.endswith("progress-focus"):
            self.focus_key = body.get("rowKey")
        return {"ok": True}


def test_todo_write_projects_in_progress_focus_not_running():
    client = PlanProgressFakeClient()
    linkage = OperatingLinkage(
        plan_id="plan-1",
        task_id="task-1",
        session_id="sess-1",
        executor_ref="hermes:forge",
    )
    store = KynverTodoStore(client, linkage=linkage)

    items = store.write(
        [{"id": "a", "content": "Step A", "status": "in_progress"}],
        merge=False,
    )

    assert items[0]["status"] == "in_progress"
    focus_calls = [c for c in client.calls if c[0] == "POST" and str(c[1]).endswith("progress-focus")]
    assert focus_calls
    assert focus_calls[0][2]["rowKey"] == "hermes-todo:a"
    row_posts = [c for c in client.calls if c[0] == "POST" and str(c[1]).endswith("progress-rows")]
    assert row_posts
    assert row_posts[0][2]["rows"][0]["status"] == "in_progress"
    assert "running" not in str(row_posts)


def test_running_row_read_back_stays_pending_without_focus():
    client = PlanProgressFakeClient()
    client.rows["hermes-todo:a"] = {
        "rowKey": "hermes-todo:a",
        "title": "Lease row",
        "status": "running",
    }
    client.focus_key = None
    linkage = OperatingLinkage(
        plan_id="plan-1",
        task_id=None,
        session_id=None,
        executor_ref="hermes:forge",
    )
    store = KynverTodoStore(client, linkage=linkage)

    items = store.read()
    assert len(items) == 1
    assert items[0]["status"] == "pending"


def test_degraded_fallback_on_agentos_failure():
    client = PlanProgressFakeClient()

    def fail_get(path, **kwargs):
        raise RuntimeError("network down")

    client.get = fail_get
    linkage = OperatingLinkage(plan_id="plan-1", task_id=None, session_id=None, executor_ref="hermes:forge")
    store = KynverTodoStore(client, linkage=linkage, allow_fallback=True)

    written = store.write([{"id": "x", "content": "local", "status": "pending"}], merge=False)
    assert store.degraded
    assert written[0]["content"] == "local"


def test_idempotent_row_keys_on_repeat_write():
    client = PlanProgressFakeClient()
    linkage = OperatingLinkage(plan_id="plan-1", task_id=None, session_id=None, executor_ref="hermes:forge")
    store = KynverTodoStore(client, linkage=linkage)

    store.write([{"id": "same", "content": "v1", "status": "pending"}], merge=False)
    store.write([{"id": "same", "content": "v2", "status": "completed"}], merge=True)

    assert client.rows["hermes-todo:same"]["title"] == "v2"
    assert client.rows["hermes-todo:same"]["status"] == "partial"


def test_project_todo_write_never_sets_running_status():
    client = MagicMock()
    client.get.return_value = {"items": []}
    linkage = OperatingLinkage(plan_id="p", task_id="t", session_id=None, executor_ref="hermes:forge")
    project_todo_write(
        client,
        linkage,
        [{"id": "1", "content": "x", "status": "in_progress"}],
        merge=False,
    )
    row_body = next(c.args[1] for c in client.post.call_args_list if "progress-rows" in c.args[0])
    assert row_body["rows"][0]["status"] == "in_progress"
    assert all(r["status"] != "running" for r in row_body["rows"])
