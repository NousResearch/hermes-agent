"""Tests for the kanban approval relay (issue #88057).

The relay lets a headless kanban worker route a protected-operation
approval (e.g. a write to a protected agent-instruction file) back to the
originating gateway channel: the worker writes a one-use request to
``kanban_approval_relay``, the gateway's kanban-notifier watcher delivers
the prompt, and the decision is written back for the worker to poll.
"""

import os
import time

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A real kanban DB pinned via HERMES_KANBAN_DB (the env the worker
    process inherits from the dispatcher)."""
    db_path = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    conn = kb.connect()
    yield conn, db_path
    conn.close()


@pytest.fixture
def task_with_origin(db, monkeypatch):
    """A running task with an origin session + a notify subscription."""
    conn, _ = db
    task_id = kb.create_task(
        conn,
        title="Relay test",
        body="body",
        assignee="worker-profile",
        session_id="session-origin-1",
        initial_status="running",
    )
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="discord",
        chat_id="123456789",
        thread_id="111",
        user_id="user-1",
        chat_type="dm",
        delivery_mode="notify",
    )
    return task_id


def _now() -> int:
    return int(time.time())


class TestRelayTableLifecycle:
    def test_create_and_read_back(self, db, task_with_origin):
        conn, _ = db
        request_id = kb.create_approval_relay_request(
            conn,
            task_id=task_with_origin,
            run_id=7,
            session_key="session-origin-1",
            platform="discord",
            chat_id="123456789",
            thread_id="111",
            command="<write to AGENTS.md>",
            description="Write to protected agent-instruction file(s).",
            timeout_seconds=300,
        )
        row = kb.get_approval_relay_request(conn, request_id)
        assert row is not None
        assert row["status"] == "pending"
        assert row["decision"] is None
        assert row["task_id"] == task_with_origin
        assert row["session_key"] == "session-origin-1"
        assert row["command"] == "<write to AGENTS.md>"
        assert row["expires_at"] >= _now() + 299
        assert row["resolved_at"] is None
        # Unguessable id: two requests never collide.
        other = kb.create_approval_relay_request(
            conn, task_id=task_with_origin, run_id=7,
            session_key="session-origin-1", platform="discord",
            chat_id="123456789", command="x",
        )
        assert other != request_id

    def test_list_pending_excludes_resolved_and_expired(self, db, task_with_origin):
        conn, _ = db
        rid_a = kb.create_approval_relay_request(
            conn, task_id=task_with_origin, run_id=1,
            session_key="s", platform="discord", chat_id="1",
            command="a", timeout_seconds=300,
        )
        rid_b = kb.create_approval_relay_request(
            conn, task_id=task_with_origin, run_id=1,
            session_key="s", platform="discord", chat_id="1",
            command="b", timeout_seconds=300,
        )
        rid_c = kb.create_approval_relay_request(
            conn, task_id=task_with_origin, run_id=1,
            session_key="s", platform="discord", chat_id="1",
            command="c", timeout_seconds=300,
        )
        assert kb.resolve_approval_relay_request(conn, rid_b, "once")
        now = _now()
        # Manually expire the third one.
        conn.execute(
            "UPDATE kanban_approval_relay SET expires_at = ? "
            "WHERE request_id = ?",
            (now - 10, rid_c),
        )
        pending = {r["request_id"] for r in kb.list_pending_approval_relay_requests(conn)}
        assert pending == {rid_a}
        assert kb.count_pending_approval_relay_requests(conn) == 1

    def test_resolve_one_use_and_stale_rejection(self, db, task_with_origin):
        conn, _ = db
        rid = kb.create_approval_relay_request(
            conn, task_id=task_with_origin, run_id=1,
            session_key="s", platform="discord", chat_id="1",
            command="x",
        )
        # Approve once: any granted scope maps to one-operation 'approved'.
        assert kb.resolve_approval_relay_request(conn, rid, "once") is True
        row = kb.get_approval_relay_request(conn, rid)
        assert row["status"] == "approved"
        assert row["decision"] == "once"
        assert row["resolved_at"] is not None
        # A duplicate/stale decision must be rejected, never double-counted.
        assert kb.resolve_approval_relay_request(conn, rid, "deny") is False
        assert kb.get_approval_relay_request(conn, rid)["status"] == "approved"

    def test_resolve_deny(self, db, task_with_origin):
        conn, _ = db
        rid = kb.create_approval_relay_request(
            conn, task_id=task_with_origin, run_id=1,
            session_key="s", platform="discord", chat_id="1",
            command="x",
        )
        assert kb.resolve_approval_relay_request(conn, rid, "deny") is True
        row = kb.get_approval_relay_request(conn, rid)
        assert row["status"] == "denied"
        assert row["decision"] == "deny"

    def test_expire_marks_overdue_pending(self, db, task_with_origin):
        conn, _ = db
        rid = kb.create_approval_relay_request(
            conn, task_id=task_with_origin, run_id=1,
            session_key="s", platform="discord", chat_id="1",
            command="x", timeout_seconds=1,
        )
        expired = kb.expire_approval_relay_requests(conn, now=_now() + 60)
        assert expired == 1
        row = kb.get_approval_relay_request(conn, rid)
        assert row["status"] == "expired"
        assert row["decision"] == "deny"
        # Idempotent sweep.
        assert kb.expire_approval_relay_requests(conn, now=_now() + 60) == 0

    def test_unknown_request_never_resolves(self, db, task_with_origin):
        conn, _ = db
        assert kb.resolve_approval_relay_request(conn, "nope", "once") is False
        assert kb.get_approval_relay_request(conn, "nope") is None


class TestLegacyBoardMigration:
    def test_relay_table_created_on_legacy_db(self, tmp_path, monkeypatch):
        """A board DB that predates the relay table gains it on connect."""
        db_path = tmp_path / "legacy.db"
        # Build a legacy-shaped DB WITHOUT the relay table using raw sqlite3
        # (bypassing kb.connect so the path is never marked initialized in
        # _INITIALIZED_PATHS — otherwise the migration pass is skipped on
        # the second open in the same process).
        import sqlite3 as _sqlite3

        raw = _sqlite3.connect(str(db_path))
        raw.execute("CREATE TABLE kanban_board (id INTEGER PRIMARY KEY)")
        raw.execute("CREATE TABLE kanban_tasks (id INTEGER PRIMARY KEY)")
        raw.commit()
        raw.close()
        # A fresh open of the never-initialized path runs the migration
        # pass and creates the relay table.
        conn2 = kb.connect(db_path=db_path)
        try:
            rid = kb.create_approval_relay_request(
                conn2, task_id="t", run_id=None,
                session_key="s", platform="discord", chat_id="1",
                command="x",
            )
            assert kb.get_approval_relay_request(conn2, rid) is not None
        finally:
            conn2.close()
