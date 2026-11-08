"""Read-only gateway task routes: projection, lookup, and route safety.

The route table deliberately contains GET rows only: writes/control stay on the
dashboard plugin, whose transition rules and tests own them. If this test starts
failing because a POST appeared, that is the design guard working.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from gateway.platforms import api_server_tasks as tasks


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _seed(**overrides):
    with kbc.connect_closing() as conn:
        return kb.create_task(
            conn,
            title=overrides.pop("title", "Gateway task"),
            body=overrides.pop("body", "created by the gateway task test"),
            initial_status=overrides.pop("initial_status", "blocked"),
            **overrides,
        )


def test_list_projects_a_small_stable_row(kanban_home):
    task_id = _seed(priority=3)
    rows = tasks.list_tasks_payload(limit=10)
    assert [row["id"] for row in rows] == [task_id]
    row = rows[0]
    assert row["title"] == "Gateway task"
    assert row["status"] == "blocked"
    assert row["priority"] == 3
    assert row["created_at"].endswith("+00:00")
    assert set(row) >= {"id", "status", "assignee", "project_id", "current_run_id", "created_at"}


def test_get_task_includes_events(kanban_home):
    task_id = _seed()
    payload = tasks.get_task_payload(task_id)
    assert payload is not None
    assert payload["task"]["id"] == task_id
    assert payload["events"], "create should leave at least one event"


def test_unknown_task_returns_none(kanban_home):
    assert tasks.get_task_payload("does-not-exist") is None


def test_status_filter_is_forwarded_and_invalid_status_is_a_value_error(kanban_home):
    _seed(title="blocked one", initial_status="blocked")
    assert len(tasks.list_tasks_payload(status="blocked")) == 1
    with pytest.raises(ValueError):
        tasks.list_tasks_payload(status="not-a-real-status")


def test_route_table_is_read_only_and_orders_static_path_first():
    rows = tasks.http_routes()
    methods = {method for method, _path, _handler in rows}
    assert methods == {"GET"}, "write/control endpoints must stay on the dashboard surface"
    paths = [path for _method, path, _handler in rows]
    assert paths == ["/api/tasks", "/api/tasks/boards", "/api/tasks/{task_id}"]


def test_limit_is_clamped():
    assert tasks._limit(None) is None
    assert tasks._limit("10") == 10
    assert tasks._limit("99999") == tasks.MAX_LIMIT
    with pytest.raises(ValueError):
        tasks._limit("nope")

def test_handler_list_returns_json(kanban_home):
    import asyncio
    import json
    from aiohttp.test_utils import make_mocked_request

    _seed(title="handler task")
    request = make_mocked_request("GET", "/api/tasks")
    response = asyncio.run(tasks._handle_list_tasks(request))
    assert response.status == 200
    body = json.loads(response.text)
    assert [row["title"] for row in body["tasks"]] == ["handler task"]


def test_handler_get_returns_task_and_404(kanban_home):
    import asyncio
    import json
    from aiohttp.test_utils import make_mocked_request

    task_id = _seed(title="lookup task")
    found = make_mocked_request("GET", f"/api/tasks/{task_id}", match_info={"task_id": task_id})
    response = asyncio.run(tasks._handle_get_task(found))
    assert response.status == 200
    assert json.loads(response.text)["task"]["id"] == task_id

    missing = make_mocked_request("GET", "/api/tasks/nope", match_info={"task_id": "nope"})
    response = asyncio.run(tasks._handle_get_task(missing))
    assert response.status == 404
