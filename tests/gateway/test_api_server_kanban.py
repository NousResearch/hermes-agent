"""
Tests for the Kanban API endpoints on the API server adapter.

These exercise the real ``hermes_cli.kanban_db`` engine against a
temporary on-disk board (pinned via ``HERMES_KANBAN_DB``), so they verify
that the HTTP layer and the engine agree end-to-end:

- Reads: list (with filters + paging), detail (404), assignees.
- Writes: create (+ idempotency), assign (404 / running -> 409), comment
  (404), patch (title/priority/body + status transitions).
- Auth: 401 when API_SERVER_KEY is set and the bearer is missing/wrong.
"""

from functools import partial

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware
from gateway.platforms import kanban_api
from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def board(tmp_path, monkeypatch):
    """Pin the kanban DB to a temp file and reset the init cache per test."""
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    # connect() caches initialized paths process-wide; clear so each temp DB
    # gets its schema created.
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(db_path)
    return db_path


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/kanban/tasks", partial(kanban_api.handle_list_tasks, adapter))
    app.router.add_post("/api/kanban/tasks", partial(kanban_api.handle_create_task, adapter))
    app.router.add_get("/api/kanban/assignees", partial(kanban_api.handle_assignees, adapter))
    app.router.add_get("/api/kanban/dispatch/state", partial(kanban_api.handle_dispatch_state, adapter))
    app.router.add_get("/api/kanban/tasks/{task_id}", partial(kanban_api.handle_get_task, adapter))
    app.router.add_patch("/api/kanban/tasks/{task_id}", partial(kanban_api.handle_patch_task, adapter))
    app.router.add_post("/api/kanban/tasks/{task_id}/assign", partial(kanban_api.handle_assign_task, adapter))
    app.router.add_post("/api/kanban/tasks/{task_id}/comment", partial(kanban_api.handle_comment_task, adapter))
    return app


def _seed(db_path, **kwargs):
    with kb.connect_closing(db_path) as conn:
        return kb.create_task(conn, **kwargs)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestListTasks:
    @pytest.mark.asyncio
    async def test_list_and_filter(self, board):
        _seed(board, title="A", tenant="rev", assignee="ezra")
        _seed(board, title="B", tenant="other", assignee="ops")
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/api/kanban/tasks")
            assert resp.status == 200
            data = await resp.json()
            assert data["count"] == 2
            assert {t["title"] for t in data["tasks"]} == {"A", "B"}
            # ISO timestamps at the boundary.
            assert data["tasks"][0]["created_at"].endswith("Z")

            resp = await cli.get("/api/kanban/tasks?tenant=rev")
            data = await resp.json()
            assert data["count"] == 1
            assert data["tasks"][0]["title"] == "A"

    @pytest.mark.asyncio
    async def test_paging(self, board):
        for i in range(5):
            _seed(board, title=f"T{i}", priority=i)
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/api/kanban/tasks?limit=2&offset=0")
            page1 = (await resp.json())["tasks"]
            resp = await cli.get("/api/kanban/tasks?limit=2&offset=2")
            page2 = (await resp.json())["tasks"]
            assert len(page1) == 2 and len(page2) == 2
            assert {t["id"] for t in page1}.isdisjoint({t["id"] for t in page2})

    @pytest.mark.asyncio
    async def test_invalid_status_filter(self, board):
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/api/kanban/tasks?status=bogus")
            assert resp.status == 400


class TestGetTask:
    @pytest.mark.asyncio
    async def test_detail_and_404(self, board):
        tid = _seed(board, title="Detail me", body="hello", assignee="ezra")
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get(f"/api/kanban/tasks/{tid}")
            assert resp.status == 200
            task = (await resp.json())["task"]
            assert task["title"] == "Detail me"
            assert task["body"] == "hello"
            assert task["comments"] == []
            assert isinstance(task["events"], list) and task["events"]

            resp = await cli.get("/api/kanban/tasks/nope")
            assert resp.status == 404


class TestAssignees:
    @pytest.mark.asyncio
    async def test_roster_counts(self, board):
        _seed(board, title="A", assignee="ezra")
        _seed(board, title="B", assignee="ezra")
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/api/kanban/assignees")
            assert resp.status == 200
            roster = (await resp.json())["assignees"]
            ezra = next(e for e in roster if e["assignee"] == "ezra")
            assert ezra["total"] == 2


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_then_visible_to_engine(self, board):
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.post("/api/kanban/tasks", json={
                "title": "From API", "tenant": "rev", "assignee": "ops", "priority": 3,
            })
            assert resp.status == 201
            task = (await resp.json())["task"]
            assert task["title"] == "From API"
            assert task["priority"] == 3
            # The dispatcher reads the same engine — task must be there and
            # in 'ready' so a running dispatcher would pick it up.
            with kb.connect_closing(board) as conn:
                row = kb.get_task(conn, task["id"])
            assert row is not None
            assert row.status == "ready"

    @pytest.mark.asyncio
    async def test_missing_title(self, board):
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.post("/api/kanban/tasks", json={"body": "no title"})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_idempotency(self, board):
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            payload = {"title": "Once", "idempotency_key": "k-1"}
            r1 = await cli.post("/api/kanban/tasks", json=payload)
            r2 = await cli.post("/api/kanban/tasks", json=payload)
            id1 = (await r1.json())["task"]["id"]
            id2 = (await r2.json())["task"]["id"]
            assert id1 == id2


class TestAssign:
    @pytest.mark.asyncio
    async def test_assign_and_404(self, board):
        tid = _seed(board, title="Assign me")
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.post(f"/api/kanban/tasks/{tid}/assign", json={"assignee": "ops"})
            assert resp.status == 200
            assert (await resp.json())["task"]["assignee"] == "ops"

            resp = await cli.post("/api/kanban/tasks/nope/assign", json={"assignee": "ops"})
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_running_conflict_409(self, board):
        tid = _seed(board, title="Busy", assignee="ops")
        # Simulate a live claim so assign_task raises RuntimeError.
        with kb.connect_closing(board) as conn:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='running', claim_lock='lock-1' WHERE id=?",
                    (tid,),
                )
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.post(f"/api/kanban/tasks/{tid}/assign", json={"assignee": "ezra"})
            assert resp.status == 409


class TestComment:
    @pytest.mark.asyncio
    async def test_comment_and_404(self, board):
        tid = _seed(board, title="Talk to me")
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.post(f"/api/kanban/tasks/{tid}/comment", json={"author": "alon", "body": "hi"})
            assert resp.status == 201
            comment = (await resp.json())["comment"]
            assert comment["author"] == "alon" and comment["body"] == "hi"
            assert comment["created_at"].endswith("Z")

            resp = await cli.post("/api/kanban/tasks/nope/comment", json={"author": "a", "body": "b"})
            assert resp.status == 404


class TestPatch:
    @pytest.mark.asyncio
    async def test_edit_metadata(self, board):
        tid = _seed(board, title="Old", priority=0)
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.patch(f"/api/kanban/tasks/{tid}", json={
                "title": "New", "priority": 7, "body": "updated",
            })
            assert resp.status == 200
            task = (await resp.json())["task"]
            assert task["title"] == "New" and task["priority"] == 7 and task["body"] == "updated"

    @pytest.mark.asyncio
    async def test_archive_transition(self, board):
        tid = _seed(board, title="Archive me")
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.patch(f"/api/kanban/tasks/{tid}", json={"status": "archived"})
            assert resp.status == 200
            assert (await resp.json())["task"]["status"] == "archived"

    @pytest.mark.asyncio
    async def test_unsupported_status_rejected(self, board):
        tid = _seed(board, title="x")
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.patch(f"/api/kanban/tasks/{tid}", json={"status": "done"})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_unknown_field_rejected(self, board):
        tid = _seed(board, title="x")
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.patch(f"/api/kanban/tasks/{tid}", json={"assignee": "ops"})
            assert resp.status == 400


class TestDispatchState:
    @pytest.mark.asyncio
    async def test_per_profile_running_counts(self, board):
        # Two running for ops, one for ezra; a ready task must not count.
        for title in ("r1", "r2"):
            tid = _seed(board, title=title, assignee="ops")
            with kb.connect_closing(board) as conn:
                with kb.write_txn(conn):
                    conn.execute("UPDATE tasks SET status='running' WHERE id=?", (tid,))
        tid = _seed(board, title="r3", assignee="ezra")
        with kb.connect_closing(board) as conn:
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='running' WHERE id=?", (tid,))
        _seed(board, title="idle", assignee="ops")  # ready, ignored

        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/api/kanban/dispatch/state")
            assert resp.status == 200
            data = await resp.json()
            by_assignee = {e["assignee"]: e for e in data["per_profile"]}
            assert by_assignee["ops"]["in_progress"] == 2
            assert by_assignee["ezra"]["in_progress"] == 1
            # No config in tests -> per-profile cap is null.
            assert by_assignee["ops"]["max_in_progress"] is None

    @pytest.mark.asyncio
    async def test_blocked_circuit_breaker(self, board):
        # Benched: blocked at the default failure limit (2).
        benched = _seed(board, title="benched")
        with kb.connect_closing(board) as conn:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='blocked', consecutive_failures=2, "
                    "last_failure_error='boom' WHERE id=?",
                    (benched,),
                )
        # Blocked but below the limit (e.g. dependency/manual block) -> excluded.
        soft = _seed(board, title="soft-block")
        with kb.connect_closing(board) as conn:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='blocked', consecutive_failures=0 WHERE id=?",
                    (soft,),
                )

        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/api/kanban/dispatch/state")
            data = await resp.json()
            ids = {b["task_id"] for b in data["blocked"]}
            assert benched in ids and soft not in ids
            entry = next(b for b in data["blocked"] if b["task_id"] == benched)
            assert entry["consecutive_failures"] == 2
            assert entry["failure_limit"] == 2
            assert entry["last_failure_error"] == "boom"

    @pytest.mark.asyncio
    async def test_blocked_respects_per_task_max_retries(self, board):
        # max_retries=5 raises the bar: 2 failures is below it -> not benched.
        tid = _seed(board, title="patient")
        with kb.connect_closing(board) as conn:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='blocked', consecutive_failures=2, "
                    "max_retries=5 WHERE id=?",
                    (tid,),
                )
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/api/kanban/dispatch/state")
            data = await resp.json()
            assert tid not in {b["task_id"] for b in data["blocked"]}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    @pytest.mark.asyncio
    async def test_requires_bearer(self, board):
        adapter = _make_adapter(api_key="sk-secret")
        async with TestClient(TestServer(_app(adapter))) as cli:
            assert (await cli.get("/api/kanban/tasks")).status == 401
            ok = await cli.get("/api/kanban/tasks", headers={"Authorization": "Bearer sk-secret"})
            assert ok.status == 200
