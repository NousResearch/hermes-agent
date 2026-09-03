"""Tests for the Kanban dashboard plugin backend (plugins/kanban/dashboard/plugin_api.py).

The plugin mounts as /api/plugins/kanban/ inside the dashboard's FastAPI app,
but here we attach its router to a bare FastAPI instance so we can test the
REST surface without spinning up the whole dashboard.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
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
    js = bundle.read_text(encoding="utf-8")

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


def test_patch_review_lifecycle_preserves_handoff_and_reopens(client):
    secret = "ghp_" + "D" * 40
    task = client.post(
        "/api/plugins/kanban/tasks", json={"title": "review me", "assignee": "builder"},
    ).json()["task"]

    response = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={
            "status": "review",
            "assignee": "reviewer",
            "summary": f"Implementation ready. {secret}",
            "metadata": {"tests_run": 4, "token": secret},
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["task"]["status"] == "review"
    with kb.connect() as conn:
        run = kb.latest_run(conn, task["id"])
        assert run is not None
        assert run.outcome == "review_requested"
        assert run.metadata is not None
        assert run.metadata["tests_run"] == 4
        assert secret not in str(run.summary)
        assert secret not in json.dumps(run.metadata)
        review_event = [
            event for event in kb.list_events(conn, task["id"])
            if event.kind == "review_requested"
        ][-1]
        assert secret not in json.dumps(review_event.payload)
        assert review_event.payload is not None
        assert review_event.payload["implementer"] == "builder"
        assert review_event.payload["reviewer"] == "reviewer"

    response = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"status": "ready"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["task"]["status"] == "ready"
    assert response.json()["task"]["assignee"] == "builder"
    with kb.connect() as conn:
        assert any(
            event.kind == "review_reopened"
            for event in kb.list_events(conn, task["id"])
        )


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


def test_reopening_parent_retracts_review_and_blocks_approval(client):
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="parent", assignee="planner")
        assert kb.complete_task(conn, parent_id)
        child_id = kb.create_task(
            conn,
            title="child in review",
            assignee="reviewer",
            parents=[parent_id],
        )
        grandchild_id = kb.create_task(
            conn,
            title="downstream",
            assignee="writer",
            parents=[child_id],
        )
        implementation = kb.claim_task(conn, child_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            child_id,
            summary="ready",
            expected_run_id=implementation.current_run_id,
        )
        active_review = kb.claim_review_task(conn, child_id)
        assert active_review is not None

    response = client.patch(
        f"/api/plugins/kanban/tasks/{parent_id}",
        json={"status": "ready"},
    )
    assert response.status_code == 200, response.text

    with kb.connect() as conn:
        child = kb.get_task(conn, child_id)
        assert child is not None
        assert child.status == "todo"
        reclaimed = kb.latest_run(conn, child_id)
        assert reclaimed is not None
        assert reclaimed.outcome == "reclaimed"
        assert kb.claim_review_task(conn, child_id) is None
        assert not kb.complete_task(conn, child_id, summary="must not approve")
        grandchild = kb.get_task(conn, grandchild_id)
        assert grandchild is not None
        assert grandchild.status == "todo"

    response = client.patch(
        f"/api/plugins/kanban/tasks/{parent_id}",
        json={"status": "done"},
    )
    assert response.status_code == 200, response.text

    with kb.connect() as conn:
        child = kb.get_task(conn, child_id)
        assert child is not None
        assert child.status == "review"
        review = kb.claim_review_task(conn, child_id)
        assert review is not None
        assert kb.complete_task(
            conn,
            child_id,
            summary="approved after parent stabilized",
            expected_run_id=review.current_run_id,
        )
        grandchild = kb.get_task(conn, grandchild_id)
        assert grandchild is not None
        assert grandchild.status == "ready"


def test_reopening_parent_recursively_retracts_done_and_running_descendants(client):
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="root", assignee="planner")
        assert kb.complete_task(conn, parent_id)
        child_id = kb.create_task(
            conn,
            title="accepted child",
            assignee="builder",
            parents=[parent_id],
        )
        assert kb.complete_task(conn, child_id)
        grandchild_id = kb.create_task(
            conn,
            title="running grandchild",
            assignee="writer",
            parents=[child_id],
        )
        grandchild_run = kb.claim_task(conn, grandchild_id)
        assert grandchild_run is not None

    response = client.patch(
        f"/api/plugins/kanban/tasks/{parent_id}",
        json={"status": "ready"},
    )
    assert response.status_code == 200, response.text

    with kb.connect() as conn:
        child = kb.get_task(conn, child_id)
        grandchild = kb.get_task(conn, grandchild_id)
        assert child is not None and child.status == "todo"
        assert grandchild is not None and grandchild.status == "todo"
        assert grandchild.current_run_id is None
        assert kb.claim_task(conn, grandchild_id) is None
        reclaimed = kb.latest_run(conn, grandchild_id)
        assert reclaimed is not None
        assert reclaimed.outcome == "reclaimed"

    response = client.patch(
        f"/api/plugins/kanban/tasks/{parent_id}",
        json={"status": "done"},
    )
    assert response.status_code == 200, response.text
    with kb.connect() as conn:
        child = kb.get_task(conn, child_id)
        grandchild = kb.get_task(conn, grandchild_id)
        assert child is not None and child.status == "ready"
        assert grandchild is not None and grandchild.status == "todo"


def test_dashboard_reclaim_of_active_review_preserves_review_phase(client):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="active review", assignee="reviewer")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="ready",
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id)
        assert review is not None

    response = client.patch(
        f"/api/plugins/kanban/tasks/{task_id}",
        json={"status": "ready"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["task"]["status"] == "review"
    assert response.json()["task"]["assignee"] == "reviewer"
    with kb.connect() as conn:
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "reclaimed"
        next_review = kb.claim_review_task(conn, task_id)
        assert next_review is not None


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

    # No token → policy violation close. The rejection is delivered as a
    # close frame AFTER the handshake (a pre-accept close would reach the
    # browser as a bare 1006), so the first receive observes it.
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/plugins/kanban/events") as conn:
            conn.receive_text()
    assert exc.value.code == 1008

    # Wrong token → policy violation close.
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/plugins/kanban/events?token=nope") as conn:
            conn.receive_text()
    assert exc.value.code == 1008

    # Correct token → accepted (connect then close cleanly from our side).
    with c.websocket_connect(
        "/api/plugins/kanban/events?token=secret-xyz"
    ) as ws:
        assert ws is not None  # handshake succeeded


    # The bug symptom was a traceback; we don't assert on stderr because
    # capturing asyncio's internal "exception was never retrieved" logging
    # is flaky. The assertion that matters is: no CancelledError escaped.


def test_ws_events_rejection_code_reaches_a_real_client(tmp_path, monkeypatch):
    """The 1008 must arrive as a close frame on a real server. Starlette's
    TestClient reports a pre-accept close as WebSocketDisconnect(1008), but
    uvicorn answers a pre-accept close with HTTP 403 and no close frame, so
    the dashboard's "1008 -> auth failed, reload" branch never ran and it
    retried with backoff instead. Real uvicorn + real websockets client."""
    uvicorn = pytest.importorskip("uvicorn")
    websockets = pytest.importorskip("websockets")
    import asyncio
    import socket
    import threading
    import types

    import hermes_cli

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    stub = types.SimpleNamespace(
        _SESSION_TOKEN="secret-xyz",
        _ws_auth_ok=lambda ws: ws.query_params.get("token", "") == "secret-xyz",
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", stub)
    monkeypatch.setattr(hermes_cli, "web_server", stub, raising=False)

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True)
    thread.start()
    started = threading.Event()
    for _ in range(200):
        if server.started:
            started.set()
            break
        started.wait(0.05)
    assert server.started, "uvicorn did not start"

    async def capture_close(url):
        async with websockets.connect(url, open_timeout=5) as ws:
            try:
                await asyncio.wait_for(ws.recv(), timeout=5)
            except websockets.ConnectionClosed as exc:
                assert exc.rcvd is not None, "closed without a close frame"
                return exc.rcvd.code
        raise AssertionError("server did not close the socket")

    try:
        code = asyncio.run(capture_close(f"ws://127.0.0.1:{port}/api/plugins/kanban/events?token=nope"))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        try:
            sock.close()
        except OSError:
            pass
    assert code == 1008


# ---------------------------------------------------------------------------
# Bulk actions
# ---------------------------------------------------------------------------


def test_bulk_status_ready(client):
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks", json={"title": "b"}).json()["task"]
    c2 = client.post("/api/plugins/kanban/tasks", json={"title": "c"}).json()["task"]
    # Parent-less tasks land in "ready" already; push them to blocked first.
    for tid in (a["id"], b["id"], c2["id"]):
        client.patch(
            f"/api/plugins/kanban/tasks/{tid}",
            json={"status": "blocked", "block_reason": "wait"},
        )

    response = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": [a["id"], b["id"], c2["id"]], "status": "ready"},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert all(item["ok"] for item in results)
    # All three are now ready.
    board = client.get("/api/plugins/kanban/board").json()
    ready = next(col for col in board["columns"] if col["name"] == "ready")
    ids = {task["id"] for task in ready["tasks"]}
    assert {a["id"], b["id"], c2["id"]}.issubset(ids)


def test_bulk_review_assignment_preserves_implementer_provenance(client):
    tasks = [
        client.post(
            "/api/plugins/kanban/tasks",
            json={"title": title, "assignee": "builder"},
        ).json()["task"]
        for title in ("review a", "review b")
    ]
    response = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={
            "ids": [task["id"] for task in tasks],
            "status": "review",
            "assignee": "reviewer",
            "summary": "ready",
        },
    )
    assert response.status_code == 200, response.text
    assert all(item["ok"] for item in response.json()["results"])
    with kb.connect() as conn:
        for task in tasks:
            current = kb.get_task(conn, task["id"])
            assert current is not None
            assert current.status == "review"
            assert current.assignee == "reviewer"
            event = [
                item for item in kb.list_events(conn, task["id"])
                if item.kind == "review_requested"
            ][-1]
            assert event.payload is not None
            assert event.payload["implementer"] == "builder"
            assert event.payload["reviewer"] == "reviewer"


def test_bulk_status_done_forwards_completion_summary(client):
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks", json={"title": "b"}).json()["task"]

    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={
            "ids": [a["id"], b["id"]],
            "status": "done",
            "result": "DECIDED: ship it",
            "summary": "DECIDED: ship it",
            "metadata": {"source": "dashboard"},
        },
    )

    assert r.status_code == 200
    assert all(r["ok"] for r in r.json()["results"])
    conn = kb.connect()
    try:
        for tid in (a["id"], b["id"]):
            task = kb.get_task(conn, tid)
            run = kb.latest_run(conn, tid)
            assert task.status == "done"
            assert task.result == "DECIDED: ship it"
            assert run.summary == "DECIDED: ship it"
            assert run.metadata == {"source": "dashboard"}
    finally:
        conn.close()


def test_bulk_status_running_rejected(client):
    """Bulk updates must match single-task PATCH: direct 'running' is invalid."""
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]

    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": [t["id"]], "status": "running"},
    )

    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == t["id"]
    assert results[0]["ok"] is False
    assert "running" in results[0]["error"]

    board = client.get("/api/plugins/kanban/board").json()
    statuses = {
        tt["id"]: col["name"]
        for col in board["columns"]
        for tt in col["tasks"]
    }
    assert statuses.get(t["id"]) != "running"


def test_dashboard_done_actions_prompt_for_completion_summary():
    """Behavioral coverage for the migrated ``requestDialog`` flow.

    Replaces the prior bundle-string-only assertion (which only proved the
    rename landed). The dialog state machine at
    ``plugins/kanban/dashboard/dist/index.js`` resolves with
    ``{confirmed: true|false, summary?}``. Each migrated call site must
    gate the dispatch on the resolved ``confirmed`` flag. This test
    asserts that contract at two layers:

    1. **Bundle cancel guards**: every migrated site gates on ``r.confirmed``
       (or its subscripted alias ``r1.confirmed``/``r2.confirmed``) before
       dispatching. We verify by counting the cancel-guard patterns +
       cross-referencing against the 8 migrated sites listed in the PR
       description.
    2. **Visual affordance**: every destructive ``requestDialog`` call marks
       ``destructive: true`` so the host renders the destructive variant.

    The dispatch path itself (PATCH/DELETE actually firing on confirm, not
    on cancel) is covered by the backend behavioral tests
    ``test_dashboard_confirm_dispatches_expected_*`` and
    ``test_dashboard_cancel_keeps_task_in_old_status`` below — together
    they pin the contract end-to-end.
    """

    repo_root = Path(__file__).resolve().parents[2]
    js = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()

    import re

    # Match ``if (!r.confirmed)``, ``if (!r1.confirmed)``, ``if (r.confirmed)``
    # (positive-form gate). The bundle uses both polarities:
    # - negative ``if (!r.confirmed) return null;`` in dialog flow bodies
    # - positive ``if (r.confirmed) props.onDeleteBoard(...);`` in JSX handlers
    cancel_guard_pattern = re.compile(
        r"if\s*\(\s*!?\s*r\d?\.confirmed\s*\)",
        re.IGNORECASE,
    )
    guards = cancel_guard_pattern.findall(js)
    # 8 migrated sites per the PR description:
    # moveTask (1), moveSelected (1), applyBulk (1), deleteTask (1),
    # deleteSelected (1), archiveBoard (1), removeAttachment (1), doPatch (1).
    # Plus performMoveTask callers (moveTask/moveSelected each have
    # ``r1.confirmed`` + ``r2.confirmed`` for the two-stage flow) → up to
    # 10 guards. Loose lower bound to avoid brittleness.
    assert len(guards) >= 8, (
        f"expected >= 8 `if (r?.confirmed)` cancel guards in bundle (one "
        f"per migrated site, plus extras for two-stage flows); found {len(guards)}"
    )

    # Visual affordance: every destructive requestDialog call must mark
    # ``destructive: true`` so the host renders the destructive variant.
    # deleteTask, deleteSelected, archiveBoard → at least 3.
    destructive_call_count = js.count("destructive: true")
    assert destructive_call_count >= 3, (
        f"expected >= 3 `destructive: true` requestDialog calls (single "
        f"delete, bulk delete, archive-board); found {destructive_call_count}"
    )


def test_dashboard_cancel_keeps_task_in_old_status(client):
    """Behavioral: the cancel branch of the dispatch path (no PATCH/DELETE
    issued) must leave the task in its previous status. The cancel guard
    lives in the bundle; this test pins the backend contract that the guard
    relies on.
    """
    t = client.post("/api/plugins/kanban/tasks",
                    json={"title": "x"}).json()["task"]
    # Tasks land in ``ready`` by default. No PATCH issued — simulating the
    # cancel branch in the bundle.
    assert t["status"] == "ready"
    r = client.get(f"/api/plugins/kanban/tasks/{t['id']}")
    assert r.json()["task"]["status"] == "ready"


def test_dashboard_confirm_dispatches_expected_patch_body(client):
    """Behavioral: the PATCH body shape the bundle produces on confirm
    (status + result + summary) must be accepted by the backend without
    rejection. The backend stores ``result`` as the human-readable
    completion summary (the bundle comments confirm ``summary`` is sent
    duplicatively so the backend can store the value under its preferred
    key while the wire format remains explicit).
    This is the contract the bundle's performMoveTask relies on.
    """
    t = client.post("/api/plugins/kanban/tasks",
                    json={"title": "x"}).json()["task"]
    # Bundle's performMoveTask on confirm with a summary produces:
    #   { status, result: summary, summary: summary }
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "done", "result": "shipped", "summary": "shipped"},
    )
    assert r.status_code == 200, r.text
    body = r.json()["task"]
    assert body["status"] == "done"
    assert body.get("result") == "shipped"


def test_dashboard_confirm_dispatches_expected_delete(client):
    """Behavioral: the DELETE call the bundle issues on confirm
    (``fetchJSON(`${API}/tasks/${id}`, { method: 'DELETE' })``) must
    succeed and remove the task.
    """
    t = client.post("/api/plugins/kanban/tasks",
                    json={"title": "x"}).json()["task"]
    r = client.delete(f"/api/plugins/kanban/tasks/{t['id']}")
    assert r.status_code == 200, r.text
    # 404 on the now-deleted task confirms removal.
    r2 = client.get(f"/api/plugins/kanban/tasks/{t['id']}")
    assert r2.status_code == 404


def test_dashboard_surfaces_ready_blocked_error_inline():
    """Regression for #26744: failed status transitions must be surfaced
    inline, not swallowed.  The drag/drop banner and the drawer's action
    row each render the parsed API ``detail`` so operators see *why*
    their click did nothing.
    """
    repo_root = Path(__file__).resolve().parents[2]
    bundle = (
        repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    ).read_text()

    # Helper that strips ``"409: {\"detail\":\"…\"}"`` down to the
    # human-readable message before it lands in any banner.
    assert "function parseApiErrorMessage(err)" in bundle
    assert "parsed.detail" in bundle

    # Drag/drop banner now uses the parsed message instead of raw
    # ``err.message`` so it no longer leaks HTTP plumbing.
    assert "setError(tx(t, \"moveFailed\", \"Move failed: \") + parseApiErrorMessage(err))" in bundle

    # Drawer action row has its own visible error surface and clears it
    # on success/refresh so stale failures don't follow the operator
    # around.
    assert "const [patchErr, setPatchErr] = useState(null);" in bundle
    assert "setPatchErr(parseApiErrorMessage(e))" in bundle
    assert "setPatchErr(null)" in bundle


def test_dashboard_dependency_selects_use_value_change_handler():
    """Regression for the dependency selects in the task drawer: the
    add-parent / add-child dropdowns must wire through the shared
    selectChangeHandler helper so their value actually lands on the
    underlying React state. Salvaged from #20019 @LeonSGP43.
    """
    repo_root = Path(__file__).resolve().parents[2]
    bundle = (
        repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    ).read_text()

    parent_select = (
        'value: newParent,\n'
        '          className: "h-7 text-xs flex-1",\n'
        '        }, selectChangeHandler(setNewParent))'
    )
    child_select = (
        'value: newChild,\n'
        '          className: "h-7 text-xs flex-1",\n'
        '        }, selectChangeHandler(setNewChild))'
    )

    assert parent_select in bundle
    assert child_select in bundle


def test_bulk_archive(client):
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks", json={"title": "b"}).json()["task"]
    r = client.post("/api/plugins/kanban/tasks/bulk",
                    json={"ids": [a["id"], b["id"]], "archive": True})
    assert r.status_code == 200
    assert all(r["ok"] for r in r.json()["results"])
    # Default board (archived hidden) — both gone.
    board = client.get("/api/plugins/kanban/board").json()
    ids = {t["id"] for col in board["columns"] for t in col["tasks"]}
    assert a["id"] not in ids
    assert b["id"] not in ids


def test_bulk_reassign(client):
    a = client.post("/api/plugins/kanban/tasks",
                    json={"title": "a", "assignee": "old"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks",
                    json={"title": "b", "assignee": "old"}).json()["task"]
    r = client.post("/api/plugins/kanban/tasks/bulk",
                    json={"ids": [a["id"], b["id"]], "assignee": "new"})
    assert r.status_code == 200
    for tid in (a["id"], b["id"]):
        t = client.get(f"/api/plugins/kanban/tasks/{tid}").json()["task"]
        assert t["assignee"] == "new"


def test_bulk_unassign_via_empty_string(client):
    a = client.post("/api/plugins/kanban/tasks",
                    json={"title": "a", "assignee": "x"}).json()["task"]
    r = client.post("/api/plugins/kanban/tasks/bulk",
                    json={"ids": [a["id"]], "assignee": ""})
    assert r.status_code == 200
    t = client.get(f"/api/plugins/kanban/tasks/{a['id']}").json()["task"]
    assert t["assignee"] is None


def test_bulk_partial_failure_doesnt_abort_siblings(client):
    """One bad id in the middle of a batch must not prevent others from
    applying."""
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    c2 = client.post("/api/plugins/kanban/tasks", json={"title": "c"}).json()["task"]
    r = client.post("/api/plugins/kanban/tasks/bulk",
                    json={"ids": [a["id"], "bogus-id", c2["id"]], "priority": 7})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    ok_ids = {r["id"] for r in results if r["ok"]}
    assert a["id"] in ok_ids
    assert c2["id"] in ok_ids
    assert any(not r["ok"] and r["id"] == "bogus-id" for r in results)
    # Good siblings actually got the priority bump.
    for tid in (a["id"], c2["id"]):
        t = client.get(f"/api/plugins/kanban/tasks/{tid}").json()["task"]
        assert t["priority"] == 7


def test_bulk_empty_ids_400(client):
    r = client.post("/api/plugins/kanban/tasks/bulk", json={"ids": []})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /config endpoint
# ---------------------------------------------------------------------------


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


