"""Tests for the Kanban dashboard plugin backend (plugins/kanban/dashboard/plugin_api.py).

The plugin mounts as /api/plugins/kanban/ inside the dashboard's FastAPI app,
but here we attach its router to a bare FastAPI instance so we can test the
REST surface without spinning up the whole dashboard.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_plugin_router():
    """Dynamically load plugins/kanban/dashboard/plugin_api.py and return its router."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _presentation_config():
    return {
        "schema": 1,
        "mode": "projection",
        "columns": [
            {
                "id": "queue-view",
                "label": "Queue",
                "helper": "Tasks awaiting verified work.",
                "read_only": True,
                "match": {"status_in": ["triage", "todo", "scheduled", "ready", "blocked"]},
            },
            {
                "id": "work-view",
                "label": "Work",
                "helper": "A current worker is verified.",
                "read_only": True,
                "match": {
                    "all": [
                        {"status_in": ["running"]},
                        {"evidence": "live_worker"},
                    ]
                },
            },
        ],
        "unmatched": {
            "column": "queue-view",
            "diagnostic": "No presentation rule matched; canonical status is unchanged.",
        },
    }


def test_presentation_endpoints_are_board_scoped_and_revision_guarded(client):
    kb.create_board("neutral-board", description="preserve")
    initial = client.get(
        "/api/plugins/kanban/boards/neutral-board/presentation"
    )
    assert initial.status_code == 200
    revision = initial.json()["revision"]

    written = client.put(
        "/api/plugins/kanban/boards/neutral-board/presentation",
        json={"presentation": _presentation_config(), "expected_revision": revision},
    )
    assert written.status_code == 200, written.text
    state = written.json()
    assert state["presentation"] == _presentation_config()
    assert len(state["digest"]) == 64
    assert kb.read_board_metadata("neutral-board")["description"] == "preserve"

    stale = client.delete(
        "/api/plugins/kanban/boards/neutral-board/presentation",
        params={"expected_revision": revision},
    )
    assert stale.status_code == 409
    cleared = client.delete(
        "/api/plugins/kanban/boards/neutral-board/presentation",
        params={"expected_revision": state["revision"]},
    )
    assert cleared.status_code == 200
    assert cleared.json()["presentation"] is None


def test_presentation_validation_endpoint_is_read_only(client):
    kb.create_board("neutral-board", description="unchanged")
    before = kb.board_metadata_path("neutral-board").read_bytes()

    response = client.post(
        "/api/plugins/kanban/boards/neutral-board/presentation/validate",
        json={"presentation": _presentation_config()},
    )

    assert response.status_code == 200, response.text
    assert response.json()["valid"] is True
    assert response.json()["presentation"] == _presentation_config()
    assert len(response.json()["digest"]) == 64
    assert kb.board_metadata_path("neutral-board").read_bytes() == before


def test_presentation_api_rejects_duplicate_json_keys_without_mutation(client):
    kb.create_board("neutral-board", description="unchanged")
    path = kb.board_metadata_path("neutral-board")
    before = path.read_bytes()
    revision = client.get(
        "/api/plugins/kanban/boards/neutral-board/presentation"
    ).json()["revision"]
    presentation = json.dumps(_presentation_config()).replace(
        '"schema": 1', '"schema": 2, "schema": 1', 1
    )

    written = client.put(
        "/api/plugins/kanban/boards/neutral-board/presentation",
        content=(
            f'{{"presentation":{presentation},'
            f'"expected_revision":"{revision}"}}'
        ),
        headers={"content-type": "application/json"},
    )
    assert written.status_code == 400
    assert "duplicate JSON object key" in written.text
    assert path.read_bytes() == before

    validated = client.post(
        "/api/plugins/kanban/boards/neutral-board/presentation/validate",
        content=f'{{"presentation":{presentation}}}',
        headers={"content-type": "application/json"},
    )
    assert validated.status_code == 400
    assert "duplicate JSON object key" in validated.text
    assert path.read_bytes() == before

    unexpected = client.put(
        "/api/plugins/kanban/boards/neutral-board/presentation",
        content=(
            f'{{"presentation":{json.dumps(_presentation_config())},'
            f'"expected_revision":"{revision}","unexpected":true}}'
        ),
        headers={"content-type": "application/json"},
    )
    assert unexpected.status_code == 422
    assert path.read_bytes() == before

    oversized = client.put(
        "/api/plugins/kanban/boards/neutral-board/presentation",
        content=(
            f'{{"presentation":{json.dumps(_presentation_config())},'
            f'"expected_revision":"{revision}",'
            f'"unexpected_padding":"{"x" * 70_000}"}}'
        ),
        headers={"content-type": "application/json"},
    )
    assert oversized.status_code == 413
    assert path.read_bytes() == before


def test_presentation_api_stops_streaming_at_body_limit():
    router = _load_plugin_router()
    route = next(
        route
        for route in router.routes
        if route.path == "/boards/{slug}/presentation" and "PUT" in route.methods
    )
    helper = route.endpoint.__globals__["_strict_presentation_request"]
    model = route.endpoint.__globals__["PresentationWriteBody"]
    chunk = b"x" * 65_536
    received = 0

    async def receive():
        nonlocal received
        received += 1
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": received < 80,
        }

    request = Request(
        {
            "type": "http",
            "method": "PUT",
            "path": "/",
            "headers": [(b"content-length", b"1")],
        },
        receive,
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(helper(request, model))
    assert exc_info.value.status_code == 413
    assert received == 2


def test_board_is_projected_server_side_and_keeps_canonical_status(client):
    revision = client.get(
        "/api/plugins/kanban/boards/default/presentation"
    ).json()["revision"]
    response = client.put(
        "/api/plugins/kanban/boards/default/presentation",
        json={"presentation": _presentation_config(), "expected_revision": revision},
    )
    assert response.status_code == 200, response.text
    created = client.post(
        "/api/plugins/kanban/tasks", json={"title": "Neutral queued task"}
    ).json()["task"]

    with kb.connect() as conn:
        before_task = kb.get_task(conn, created["id"])
        assert before_task is not None
        before_status = before_task.status
        before_events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]

    board = client.get("/api/plugins/kanban/board").json()
    assert board["presentation"]["mode"] == "projection"
    assert [column["id"] for column in board["columns"]] == [
        "queue-view", "work-view"
    ]
    queue = board["columns"][0]
    all_projected_ids = [
        task["id"] for column in board["columns"] for task in column["tasks"]
    ]
    assert all_projected_ids.count(created["id"]) == 1
    assert queue["label"] == "Queue"
    assert queue["helper"] == "Tasks awaiting verified work."
    assert queue["read_only"] is True
    assert queue["tasks"][0]["id"] == created["id"]
    assert queue["tasks"][0]["status"] == "ready"
    with kb.connect() as conn:
        after_task = kb.get_task(conn, created["id"])
        assert after_task is not None
        assert after_task.status == before_status
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == before_events

    rejected = client.patch(
        f"/api/plugins/kanban/tasks/{created['id']}",
        json={"status": "queue-view"},
    )
    assert rejected.status_code == 400
    assert kb.get_task(kb.connect(), created["id"]).status == "ready"

    bulk = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": [created["id"]], "status": "queue-view"},
    )
    assert bulk.status_code == 200
    assert bulk.json()["results"][0]["id"] == created["id"]
    assert bulk.json()["results"][0]["ok"] is False
    assert "queue-view" in bulk.json()["results"][0]["error"]
    assert kb.get_task(kb.connect(), created["id"]).status == "ready"


def test_projection_ignores_stale_result_and_run_summary_markers(client):
    config = _presentation_config()
    config["columns"].insert(
        0,
        {
            "id": "input-view",
            "label": "Input",
            "helper": "Current structured input gate.",
            "read_only": True,
            "match": {
                "all": [
                    {"status_in": ["blocked"]},
                    {"block_kind_in": ["needs_input"]},
                    {"markers_present": ["Question", "Resumes"]},
                ]
            },
        },
    )
    revision = client.get(
        "/api/plugins/kanban/boards/default/presentation"
    ).json()["revision"]
    assert client.put(
        "/api/plugins/kanban/boards/default/presentation",
        json={"presentation": config, "expected_revision": revision},
    ).status_code == 200
    created = client.post(
        "/api/plugins/kanban/tasks", json={"title": "Current input gate"}
    ).json()["task"]
    stale = "**Question:** stale question\n**Resumes:** stale condition"
    with kb.connect() as conn:
        conn.execute(
            "UPDATE tasks SET status = 'blocked', block_kind = 'needs_input', "
            "body = ?, result = ? WHERE id = ?",
            ("Current body has no structured gate.", stale, created["id"]),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, summary, started_at) "
            "VALUES (?, ?, ?, ?)",
            (created["id"], "completed", stale, 1),
        )
        conn.commit()

    board = client.get("/api/plugins/kanban/board").json()
    projected = next(
        task
        for column in board["columns"]
        for task in column["tasks"]
        if task["id"] == created["id"]
    )
    assert projected["display_column"] == "queue-view"
    assert {item["kind"] for item in projected["presentation_diagnostics"]} == {
        "malformed_needs_input",
    }


def test_projection_keeps_included_archived_tasks_visible_as_unmatched(client):
    revision = client.get(
        "/api/plugins/kanban/boards/default/presentation"
    ).json()["revision"]
    assert client.put(
        "/api/plugins/kanban/boards/default/presentation",
        json={"presentation": _presentation_config(), "expected_revision": revision},
    ).status_code == 200
    created = client.post(
        "/api/plugins/kanban/tasks", json={"title": "Archived neutral task"}
    ).json()["task"]
    conn = kb.connect()
    try:
        conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (created["id"],))
        conn.commit()
    finally:
        conn.close()

    board = client.get(
        "/api/plugins/kanban/board", params={"include_archived": True}
    ).json()
    tasks = [task for column in board["columns"] for task in column["tasks"]]
    archived = next(task for task in tasks if task["id"] == created["id"])
    assert archived["status"] == "archived"
    assert {item["kind"] for item in archived["presentation_diagnostics"]} == {
        "unmatched_projection"
    }


def test_malformed_presentation_falls_back_to_canonical_payload_with_error(client):
    kb.write_board_metadata("default", name="Default")
    metadata = kb.read_board_metadata("default")
    metadata.pop("db_path", None)
    metadata["presentation"] = {"schema": 1, "mode": "projection"}
    kb.board_metadata_path("default").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    response = client.get("/api/plugins/kanban/board")
    assert response.status_code == 200
    board = response.json()
    assert {column["name"] for column in board["columns"]} == (
        kb.VALID_STATUSES - {"archived"}
    )
    assert board["presentation"]["mode"] == "canonical"
    assert "unknown or missing fields" in board["presentation"]["error"]


# ---------------------------------------------------------------------------
# GET /board on an empty DB
# ---------------------------------------------------------------------------


def test_board_empty(client):
    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    data = r.json()
    # All canonical columns present (triage + the rest), each empty.
    names = [c["name"] for c in data["columns"]]
    assert set(names) == kb.VALID_STATUSES - {"archived"}
    for expected in ("triage", "todo", "scheduled", "ready", "running", "blocked", "done"):
        assert expected in names, f"missing column {expected}: {names}"
    assert all(len(c["tasks"]) == 0 for c in data["columns"])
    assert data["tenants"] == []
    assert data["assignees"] == []
    assert data["latest_event_id"] == 0
    assert "presentation" not in data


# ---------------------------------------------------------------------------
# POST /tasks then GET /board sees it
# ---------------------------------------------------------------------------


def test_create_task_appears_on_board(client):
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={
            "title": "Research LLM caching",
            "assignee": "researcher",
            "priority": 3,
            "tenant": "acme",
        },
    )
    assert r.status_code == 200, r.text
    task = r.json()["task"]
    assert task["title"] == "Research LLM caching"
    assert task["assignee"] == "researcher"
    assert task["status"] == "ready"  # no parents -> immediately ready
    assert task["priority"] == 3
    assert task["tenant"] == "acme"
    task_id = task["id"]

    # Board now lists it under 'ready'.
    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    data = r.json()
    ready = next(c for c in data["columns"] if c["name"] == "ready")
    assert len(ready["tasks"]) == 1
    assert ready["tasks"][0]["id"] == task_id
    assert "acme" in data["tenants"]
    assert "researcher" in data["assignees"]


def test_patch_board_sets_project_directory(client, tmp_path):
    """Board-level default_workdir must be editable after creation."""
    kb.create_board("late-config")
    project_dir = tmp_path / "late-project"
    project_dir.mkdir()

    response = client.patch(
        "/api/plugins/kanban/boards/late-config",
        json={"default_workdir": str(project_dir)},
    )

    assert response.status_code == 200, response.text
    board = response.json()["board"]
    assert board["default_workdir"] == str(project_dir.resolve())
    # The recommendation flips from scratch to a persistent kind so the
    # create-task dialog's workspace default follows the board setting.
    assert board["default_workspace_kind"] == "dir"
    assert kb.read_board_metadata("late-config")["default_workdir"] == str(
        project_dir.resolve()
    )


def test_patch_board_clears_project_directory(client, tmp_path):
    """Empty string clears default_workdir; omitting it leaves it unchanged."""
    project_dir = tmp_path / "was-configured"
    project_dir.mkdir()
    kb.create_board("clearable", default_workdir=str(project_dir))

    # Omitted key → unchanged.
    r = client.patch(
        "/api/plugins/kanban/boards/clearable",
        json={"name": "Renamed Only"},
    )
    assert r.status_code == 200
    assert r.json()["board"]["default_workdir"] == str(project_dir.resolve())

    # Empty string → cleared, recommendation falls back to scratch.
    r = client.patch(
        "/api/plugins/kanban/boards/clearable",
        json={"default_workdir": ""},
    )
    assert r.status_code == 200
    board = r.json()["board"]
    assert not board.get("default_workdir")
    assert board["default_workspace_kind"] == "scratch"


def test_patch_board_rejects_malformed_metadata_without_overwriting(client):
    kb.create_board("broken-metadata", name="Keep")
    path = kb.board_metadata_path("broken-metadata")
    malformed = b'{"name":"Keep",'
    path.write_bytes(malformed)

    response = client.patch(
        "/api/plugins/kanban/boards/broken-metadata",
        json={"name": "Replacement"},
    )

    assert response.status_code == 400
    assert "metadata" in response.json()["detail"].lower()
    assert path.read_bytes() == malformed


@pytest.mark.parametrize("path", ["relative/project", "~/missing-project"])
def test_patch_board_rejects_invalid_project_directory(client, path):
    """PATCH must validate default_workdir like board creation does."""
    kb.create_board("strict")

    response = client.patch(
        "/api/plugins/kanban/boards/strict",
        json={"default_workdir": path},
    )

    assert response.status_code == 400
    assert "project directory" in response.json()["detail"].lower()


def test_new_board_dialog_collects_project_directory():
    """Board creation should expose the setting that controls safe task defaults."""
    bundle = (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "kanban"
        / "dashboard"
        / "dist"
        / "index.js"
    ).read_text(encoding="utf-8")

    assert 'const [projectDirectory, setProjectDirectory] = useState("");' in bundle
    assert "Project directory" in bundle
    assert "Absolute path to the project folder" in bundle
    assert "default_workdir: projectDirectory.trim() || undefined" in bundle


def test_dashboard_workspace_picker_explains_persistence_contract():
    """Task creation must make scratch deletion visible without a hover."""
    bundle = (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "kanban"
        / "dashboard"
        / "dist"
        / "index.js"
    ).read_text(encoding="utf-8")

    assert "Temporary — deleted on completion" in bundle
    assert "Git worktree — preserved" in bundle
    assert "Directory — preserved" in bundle
    assert "defaultWorkspacePath: (props.boardMeta && props.boardMeta.default_workdir) || \"\"" in bundle
    assert (
        "This workspace and any files left in it are deleted when the task completes."
        in bundle
    )


def test_scheduled_tasks_have_their_own_column_not_todo(client):
    """Scheduled/time-delay tasks must not be silently bucketed into todo."""

    task = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "wait for indexed data", "assignee": "ops"},
    ).json()["task"]

    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'scheduled' WHERE id = ?",
                (task["id"],),
            )
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    columns = {c["name"]: c["tasks"] for c in r.json()["columns"]}
    assert any(t["id"] == task["id"] for t in columns["scheduled"])
    assert not any(t["id"] == task["id"] for t in columns["todo"])


def test_tenant_filter(client):
    client.post("/api/plugins/kanban/tasks", json={"title": "A", "tenant": "t1"})
    client.post("/api/plugins/kanban/tasks", json={"title": "B", "tenant": "t2"})

    r = client.get("/api/plugins/kanban/board?tenant=t1")
    counts = {c["name"]: len(c["tasks"]) for c in r.json()["columns"]}
    total = sum(counts.values())
    assert total == 1

    r = client.get("/api/plugins/kanban/board?tenant=t2")
    total = sum(len(c["tasks"]) for c in r.json()["columns"])
    assert total == 1


def test_dashboard_markdown_html_is_sanitized_before_render():
    """Markdown rendering must sanitize HTML before dangerouslySetInnerHTML."""

    repo_root = Path(__file__).resolve().parents[2]
    bundle = repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    js = bundle.read_text()

    assert "function sanitizeMarkdownHtml(html)" in js
    assert "MARKDOWN_ALLOWED_TAGS" in js
    assert "sanitizeMarkdownHtml(renderMarkdown(props.source || \"\"))" in js
    assert "dangerouslySetInnerHTML: { __html: renderMarkdown(props.source || \"\") }" not in js


# ---------------------------------------------------------------------------
# GET /tasks/:id returns body + comments + events + links
# ---------------------------------------------------------------------------


def test_task_detail_includes_links_and_events(client):
    parent = client.post(
        "/api/plugins/kanban/tasks", json={"title": "parent"},
    ).json()["task"]
    child = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "child", "parents": [parent["id"]]},
    ).json()["task"]
    assert child["status"] == "todo"  # parent not done yet

    # Detail for the child shows the parent link.
    r = client.get(f"/api/plugins/kanban/tasks/{child['id']}")
    assert r.status_code == 200
    data = r.json()
    assert data["task"]["id"] == child["id"]
    assert parent["id"] in data["links"]["parents"]

    # Detail for the parent shows the child.
    r = client.get(f"/api/plugins/kanban/tasks/{parent['id']}")
    assert child["id"] in r.json()["links"]["children"]

    # Events exist from creation.
    assert len(data["events"]) >= 1


# ---------------------------------------------------------------------------
# PATCH /tasks/:id — status transitions
# ---------------------------------------------------------------------------


def test_reopening_parent_demotes_ready_child(client):
    """Reopening a completed parent must invalidate ready children immediately.

    The dispatcher re-checks parent completion on claim, but the dashboard
    should not keep showing a stale child as ready after an operator drags
    its parent back out of done for more work.
    """
    parent = client.post("/api/plugins/kanban/tasks", json={"title": "p"}).json()["task"]
    child = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "c", "parents": [parent["id"]]},
    ).json()["task"]
    assert child["status"] == "todo"

    r = client.patch(
        f"/api/plugins/kanban/tasks/{parent['id']}",
        json={"status": "done"},
    )
    assert r.status_code == 200

    child_after_done = client.get(
        f"/api/plugins/kanban/tasks/{child['id']}"
    ).json()["task"]
    assert child_after_done["status"] == "ready"

    r = client.patch(
        f"/api/plugins/kanban/tasks/{parent['id']}",
        json={"status": "todo"},
    )
    assert r.status_code == 200

    child_after_reopen = client.get(
        f"/api/plugins/kanban/tasks/{child['id']}"
    ).json()["task"]
    assert child_after_reopen["status"] == "todo"


# ---------------------------------------------------------------------------
# DELETE /tasks/:id
# ---------------------------------------------------------------------------

def test_delete_task(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "to-delete"}).json()["task"]
    r = client.delete(f"/api/plugins/kanban/tasks/{t['id']}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert r.json()["task_id"] == t["id"]

    # Gone from board
    board = client.get("/api/plugins/kanban/board").json()
    all_ids = [tt["id"] for col in board["columns"] for tt in col["tasks"]]
    assert t["id"] not in all_ids

    # Gone from detail
    r = client.get(f"/api/plugins/kanban/tasks/{t['id']}")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Comments + Links
# ---------------------------------------------------------------------------


def test_add_comment(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.post(
        f"/api/plugins/kanban/tasks/{t['id']}/comments",
        json={"body": "how's progress?", "author": "teknium"},
    )
    assert r.status_code == 200

    r = client.get(f"/api/plugins/kanban/tasks/{t['id']}")
    comments = r.json()["comments"]
    assert len(comments) == 1
    assert comments[0]["body"] == "how's progress?"
    assert comments[0]["author"] == "teknium"


# ---------------------------------------------------------------------------
# Dispatch nudge
# ---------------------------------------------------------------------------


def test_dispatch_dry_run(client):
    client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "work", "assignee": "researcher"},
    )
    r = client.post("/api/plugins/kanban/dispatch?dry_run=true&max=4")
    assert r.status_code == 200
    body = r.json()
    # DispatchResult is serialized as a dataclass dict.
    assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# Triage column (new v1 status)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Progress rollup (done children / total children)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Auto-init on first board read
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WebSocket auth (query-param token)
# ---------------------------------------------------------------------------


def test_ws_events_rejects_when_token_required(tmp_path, monkeypatch):
    """Loopback mode: a missing or wrong ?token= must be rejected with
    policy-violation; the correct token is accepted. The kanban WS now
    delegates to web_server._ws_auth_ok, so we stub that with the real
    loopback-token semantics (auth_required False → constant-time token
    compare)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    # Stub web_server with a loopback-mode _ws_auth_ok (auth_required False →
    # accept only the correct ?token=). Mirrors the real gate's loopback path.
    import hermes_cli
    import types

    def _fake_ws_auth_ok(ws):
        return ws.query_params.get("token", "") == "secret-xyz"

    stub = types.SimpleNamespace(
        _SESSION_TOKEN="secret-xyz",
        _ws_auth_ok=_fake_ws_auth_ok,
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", stub)
    monkeypatch.setattr(hermes_cli, "web_server", stub, raising=False)

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    c = TestClient(app)

    # No token → policy violation close.
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/plugins/kanban/events"):
            pass
    assert exc.value.code == 1008

    # Wrong token → policy violation close.
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/plugins/kanban/events?token=nope"):
            pass
    assert exc.value.code == 1008

    # Correct token → accepted (connect then close cleanly from our side).
    with c.websocket_connect(
        "/api/plugins/kanban/events?token=secret-xyz"
    ) as ws:
        assert ws is not None  # handshake succeeded


    # The bug symptom was a traceback; we don't assert on stderr because
    # capturing asyncio's internal "exception was never retrieved" logging
    # is flaky. The assertion that matters is: no CancelledError escaped.


# ---------------------------------------------------------------------------
# Bulk actions
# ---------------------------------------------------------------------------


def test_bulk_status_ready(client):
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks", json={"title": "b"}).json()["task"]
    c2 = client.post("/api/plugins/kanban/tasks", json={"title": "c"}).json()["task"]
    # Parent-less tasks land in "ready" already; push them to blocked first.
    for tid in (a["id"], b["id"], c2["id"]):
        client.patch(f"/api/plugins/kanban/tasks/{tid}",
                     json={"status": "blocked", "block_reason": "wait"})

    r = client.post("/api/plugins/kanban/tasks/bulk",
                    json={"ids": [a["id"], b["id"], c2["id"]], "status": "ready"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert all(r["ok"] for r in results)
    # All three are now ready.
    board = client.get("/api/plugins/kanban/board").json()
    ready = next(col for col in board["columns"] if col["name"] == "ready")
    ids = {t["id"] for t in ready["tasks"]}
    assert {a["id"], b["id"], c2["id"]}.issubset(ids)


# ---------------------------------------------------------------------------
# /config endpoint
# ---------------------------------------------------------------------------


def test_config_reads_dashboard_kanban_section(tmp_path, monkeypatch, client):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "dashboard:\n"
        "  kanban:\n"
        "    default_tenant: acme\n"
        "    lane_by_profile: false\n"
        "    include_archived_by_default: true\n"
        "    render_markdown: false\n"
    )
    r = client.get("/api/plugins/kanban/config")
    assert r.status_code == 200
    data = r.json()
    assert data["default_tenant"] == "acme"
    assert data["lane_by_profile"] is False
    assert data["include_archived_by_default"] is True
    assert data["render_markdown"] is False


# ---------------------------------------------------------------------------
# Runs surfacing (vulcan-artivus RFC feedback)
# ---------------------------------------------------------------------------


def test_event_dict_includes_run_id(client):
    """GET /tasks/:id returns events with run_id populated."""
    r = client.post("/api/plugins/kanban/tasks", json={"title": "e", "assignee": "worker"})
    tid = r.json()["task"]["id"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        kb.claim_task(conn, tid)
        run_id = kb.latest_run(conn, tid).id
        kb.complete_task(conn, tid, summary="wss")
    finally:
        conn.close()

    r = client.get(f"/api/plugins/kanban/tasks/{tid}")
    assert r.status_code == 200
    events = r.json()["events"]
    # Every event in the response must have a run_id key (None or int).
    for e in events:
        assert "run_id" in e, f"missing run_id in event: {e}"
    # completed event must have the actual run_id.
    comp = [e for e in events if e["kind"] == "completed"]
    assert comp[0]["run_id"] == run_id


# ---------------------------------------------------------------------------
# Per-task force-loaded skills via REST
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dispatcher-presence warning in POST /tasks response
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _task_dict — outer try/except fallback when task_age raises
#
# Background: kanban_db.task_age was hardened in 061a1830 to return None for
# corrupt timestamp values via _safe_int. The companion fix added a belt-and-
# suspenders try/except in plugin_api._task_dict so that *any future* exception
# from task_age (not just ValueError on '%s') still yields a usable dict
# instead of 500'ing GET /board for the entire org.
#
# kanban_db._safe_int / task_age corruption paths are covered in
# tests/hermes_cli/test_kanban_db.py. The OUTER fallback here is not, which
# means a refactor that drops the try/except would not be caught by CI. The
# tests below pin that contract.
# ---------------------------------------------------------------------------


_FALLBACK_AGE = {
    "created_age_seconds": None,
    "started_age_seconds": None,
    "time_to_complete_seconds": None,
}


# ---------------------------------------------------------------------------
# Home-channel subscription endpoints (#19534 follow-up: GUI opt-in)
# ---------------------------------------------------------------------------
#
# Dashboard surface for per-task, per-platform notification toggles. The
# backend endpoints read the live GatewayConfig, so tests set env vars
# (BOT_TOKEN + HOME_CHANNEL) to simulate a user who has run /sethome on
# telegram and discord.


@pytest.fixture
def with_home_channels(monkeypatch):
    """Simulate a user with home channels set on telegram and discord."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc:fake")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "1234567")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "42")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_NAME", "Main TG")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "disc_fake")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "9999999")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL_NAME", "Main Discord")
    # Slack has a token but NO home — should be excluded from the list.
    monkeypatch.setenv("SLACK_BOT_TOKEN", "slack_fake")


def test_home_channels_lists_only_platforms_with_home(client, with_home_channels):
    """GET /home-channels returns entries only for platforms where the
    user has set a home; untoggled-subscribed bool is false by default."""
    r = client.get("/api/plugins/kanban/home-channels")
    assert r.status_code == 200
    platforms = {h["platform"] for h in r.json()["home_channels"]}
    assert platforms == {"telegram", "discord"}, (
        f"slack has a token but no home — must not appear. got {platforms}"
    )
    for h in r.json()["home_channels"]:
        assert h["subscribed"] is False


# ---------------------------------------------------------------------------
# Recovery endpoints (reclaim + reassign) and warnings field
# ---------------------------------------------------------------------------


def test_reclaim_endpoint_releases_running_claim(client):
    """POST /tasks/<id>/reclaim drops the claim, returns ok, and emits
    a manual reclaimed event."""
    import secrets
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="running", assignee="x")
        lock = secrets.token_hex(8)
        future = int(time.time()) + 3600
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, future, 99999, t),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (t, lock, future, 99999, int(time.time())),
        )
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, t))
        conn.commit()
    finally:
        conn.close()

    r = client.post(
        f"/api/plugins/kanban/tasks/{t}/reclaim",
        json={"reason": "browser recovery"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["task_id"] == t

    # Confirm the task is back to ready.
    conn2 = kb.connect()
    try:
        row = conn2.execute(
            "SELECT status, claim_lock FROM tasks WHERE id=?", (t,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
    finally:
        conn2.close()


def test_reassign_endpoint_switches_profile(client):
    """POST /tasks/<id>/reassign changes the assignee field."""
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="task", assignee="orig")
    finally:
        conn.close()

    r = client.post(
        f"/api/plugins/kanban/tasks/{t}/reassign",
        json={"profile": "newbie", "reclaim_first": False},
    )
    assert r.status_code == 200, r.text
    assert r.json()["assignee"] == "newbie"

    conn2 = kb.connect()
    try:
        row = conn2.execute(
            "SELECT assignee FROM tasks WHERE id=?", (t,),
        ).fetchone()
        assert row["assignee"] == "newbie"
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# Diagnostics endpoint (/api/plugins/kanban/diagnostics)
# ---------------------------------------------------------------------------


def test_diagnostics_endpoint_surfaces_blocked_hallucination(client):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        real = kb.create_task(conn, title="real", assignee="x", created_by="alice")
        import pytest as _pytest
        with _pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent, summary="phantom",
                created_cards=[real, "t_ffff00001234"],
            )
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/diagnostics")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 1
    row = data["diagnostics"][0]
    assert row["task_id"] == parent
    assert row["diagnostics"][0]["kind"] == "hallucinated_cards"
    assert row["diagnostics"][0]["severity"] == "error"
    assert "t_ffff00001234" in row["diagnostics"][0]["data"]["phantom_ids"]


# ---------------------------------------------------------------------------
# POST /tasks/:id/specify — triage specifier endpoint
# ---------------------------------------------------------------------------


def _patch_specifier_response(monkeypatch, *, content, model="test-model"):
    """Helper: install a fake auxiliary client so the specifier endpoint
    can run without hitting any real provider."""
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    # specify_task routes through call_llm now (#35566) — mock it directly.
    fake_call = MagicMock(return_value=resp)
    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call)
    return fake_call


def test_specify_happy_path(client, monkeypatch):
    import json as jsonlib

    # Create a triage task.
    t = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "one-liner", "triage": True},
    ).json()["task"]
    assert t["status"] == "triage"

    _patch_specifier_response(
        monkeypatch,
        content=jsonlib.dumps(
            {"title": "Polished", "body": "**Goal**\nDo the thing."}
        ),
    )

    r = client.post(
        f"/api/plugins/kanban/tasks/{t['id']}/specify",
        json={"author": "ui-tester"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["task_id"] == t["id"]
    assert body["new_title"] == "Polished"

    # Task should have moved off the triage column.
    detail = client.get(f"/api/plugins/kanban/tasks/{t['id']}").json()["task"]
    assert detail["status"] in {"todo", "ready"}
    assert detail["title"] == "Polished"
    assert "**Goal**" in (detail["body"] or "")


# ---------------------------------------------------------------------------
# Final result visibility for Done cards
# ---------------------------------------------------------------------------


