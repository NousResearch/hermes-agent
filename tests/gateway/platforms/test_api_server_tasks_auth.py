"""The read-only board routes answer only to the API key.

Regression for 2026-11-10: ``/api/tasks*`` were registered without any key check while the
module docstring claimed a middleware covered them. The listener sits behind the relay that
Tailscale Funnel publishes, so the whole board (titles, bodies, errors) was readable by anyone.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest import mock

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from gateway.platforms import api_server_tasks as tasks
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _deny(_request):
    return web.json_response({"error": "unauthorized"}, status=401)


def _rows(check):
    return {path: handler for _method, path, handler in tasks.http_routes(check)}


@pytest.mark.parametrize("path, match_info", [
    ("/api/tasks", {}),
    ("/api/tasks/boards", {}),
    ("/api/tasks/{task_id}", {"task_id": "t_anything"}),
])
def test_every_task_route_returns_the_auth_error_before_reading_the_board(path, match_info):
    handler = _rows(_deny)[path]
    request = make_mocked_request("GET", path.format(**match_info), match_info=match_info)
    with mock.patch.object(tasks, "list_tasks_payload") as list_payload,             mock.patch.object(tasks, "get_task_payload") as get_payload,             mock.patch.object(tasks, "boards_payload") as boards_payload:
        response = asyncio.run(handler(request))
    assert response.status == 401
    list_payload.assert_not_called()
    get_payload.assert_not_called()
    boards_payload.assert_not_called()


def test_an_accepted_request_reaches_the_board(kanban_home):
    handler = _rows(lambda _request: None)["/api/tasks"]
    response = asyncio.run(handler(make_mocked_request("GET", "/api/tasks")))
    assert response.status == 200
    assert json.loads(response.text) == {"tasks": []}


def test_routes_cannot_be_built_without_an_auth_check():
    with pytest.raises(TypeError):
        tasks.http_routes()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        tasks.http_routes(None)  # type: ignore[arg-type]


def test_the_adapter_route_table_puts_the_board_behind_its_own_key_check():
    """Wiring guard: the adapter must hand its ``_check_auth`` to the task rows."""
    from gateway.platforms.api_server import APIServerAdapter

    adapter = mock.Mock()
    adapter._check_auth = mock.Mock(side_effect=_deny)
    rows = APIServerAdapter._http_route_table(adapter)
    task_rows = {path: handler for _method, path, handler in rows if path.startswith("/api/tasks")}
    assert set(task_rows) == {"/api/tasks", "/api/tasks/boards", "/api/tasks/{task_id}"}
    for path, handler in task_rows.items():
        match_info = {"task_id": "t_x"} if "{task_id}" in path else {}
        response = asyncio.run(handler(make_mocked_request("GET", path.format(**match_info), match_info=match_info)))
        assert response.status == 401, path
    assert adapter._check_auth.call_count == 3
