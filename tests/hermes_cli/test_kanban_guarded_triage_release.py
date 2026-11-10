"""Concurrency contracts for guarded Triage release."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea"):
    return kb.create_task(conn, title=title, triage=True)


def test_guard_observes_protected_comment_committed_before_transaction_acquisition(kanban_home):
    """A writer that wins the lock cannot land a hold between guard and release."""
    with kbc.connect() as setup_conn:
        task_id = _create_triage(setup_conn)

    release_started = threading.Event()
    release_finished = threading.Event()
    outcome = {}

    def guard(conn, guarded_task_id):
        protected = [
            comment
            for comment in kb.list_comments(conn, guarded_task_id)
            if comment.body == "MFA approval required"
        ]
        return kb.TriageReleaseGuardResult(
            allowed=not protected,
            reason="protected hold" if protected else None,
        )

    def release():
        release_started.set()
        try:
            with kbc.connect() as release_conn:
                outcome["receipt"] = kb.guarded_specify_triage_task(
                    release_conn,
                    task_id,
                    guard=guard,
                    author="office",
                )
        finally:
            release_finished.set()

    with kbc.connect() as writer_conn, kb.write_txn(writer_conn):
        writer_conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
            (task_id, "security", "MFA approval required", 1),
        )
        thread = threading.Thread(target=release)
        thread.start()
        assert release_started.wait(timeout=2)
        assert not release_finished.is_set()

    thread.join(timeout=2)
    assert not thread.is_alive()
    receipt = outcome["receipt"]
    assert receipt == kb.TriageReleaseReceipt(
        released=False,
        outcome="precondition_failed",
        status="triage",
        changed_fields=(),
        guard_reason="protected hold",
    )
    with kbc.connect() as verify_conn:
        assert kb.get_task(verify_conn, task_id).status == "triage"
        assert [event.kind for event in kb.list_events(verify_conn, task_id)] == ["created"]


def test_guard_observes_parent_link_committed_before_transaction_acquisition(kanban_home):
    with kbc.connect() as setup_conn:
        parent_id = kb.create_task(setup_conn, title="open parent", triage=True)
        task_id = _create_triage(setup_conn)

    release_started = threading.Event()
    release_finished = threading.Event()
    outcome = {}

    def guard(conn, guarded_task_id):
        blockers = kb.unsatisfied_parents(conn, guarded_task_id)
        return kb.TriageReleaseGuardResult(
            allowed=not blockers,
            reason="parent state changed" if blockers else None,
        )

    def release():
        release_started.set()
        try:
            with kbc.connect() as release_conn:
                outcome["receipt"] = kb.guarded_specify_triage_task(
                    release_conn, task_id, guard=guard
                )
        finally:
            release_finished.set()

    with kbc.connect() as writer_conn, kb.write_txn(writer_conn):
        writer_conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent_id, task_id),
        )
        expected_links = writer_conn.execute(
            "SELECT parent_id, child_id FROM task_links WHERE child_id = ?",
            (task_id,),
        ).fetchall()
        thread = threading.Thread(target=release)
        thread.start()
        assert release_started.wait(timeout=2)
        assert not release_finished.is_set()

    thread.join(timeout=2)
    assert not thread.is_alive()
    assert outcome["receipt"].outcome == "precondition_failed"
    with kbc.connect() as verify_conn:
        assert kb.get_task(verify_conn, task_id).status == "triage"
        actual_links = verify_conn.execute(
            "SELECT parent_id, child_id FROM task_links WHERE child_id = ?",
            (task_id,),
        ).fetchall()
        assert [tuple(row) for row in actual_links] == [tuple(row) for row in expected_links]


def test_allowed_release_promotes_parent_free_task_in_same_transaction(kanban_home):
    with kbc.connect() as conn:
        task_id = _create_triage(conn)
        receipt = kb.guarded_specify_triage_task(
            conn,
            task_id,
            guard=lambda _conn, _task_id: kb.TriageReleaseGuardResult(True),
            title="Refined idea",
            body="Complete body",
            author="office",
        )
        task = kb.get_task(conn, task_id)
        comments = kb.list_comments(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert receipt == kb.TriageReleaseReceipt(
        released=True,
        outcome="released",
        status="ready",
        changed_fields=("title", "body"),
    )
    assert task.status == "ready"
    assert task.title == "Refined idea"
    assert [comment.body for comment in comments] == [
        "Specified — updated title, body and promoted to todo."
    ]
    assert [event.kind for event in events] == ["created", "specified", "promoted"]


def test_allowed_release_preserves_open_parent_gating_and_links(kanban_home):
    with kbc.connect() as conn:
        parent_id = kb.create_task(conn, title="open parent", triage=True)
        task_id = _create_triage(conn)
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent_id, task_id),
        )
        links_before = conn.execute(
            "SELECT parent_id, child_id FROM task_links WHERE child_id = ?",
            (task_id,),
        ).fetchall()
        receipt = kb.guarded_specify_triage_task(
            conn,
            task_id,
            guard=lambda _conn, _task_id: kb.TriageReleaseGuardResult(True),
        )
        links_after = conn.execute(
            "SELECT parent_id, child_id FROM task_links WHERE child_id = ?",
            (task_id,),
        ).fetchall()

    assert receipt.status == "todo"
    assert [tuple(row) for row in links_after] == [tuple(row) for row in links_before]


def test_guard_exception_rolls_back_release(kanban_home):
    with kbc.connect() as conn:
        task_id = _create_triage(conn)

        def fail_guard(_conn, _task_id):
            raise LookupError("authority unavailable")

        with pytest.raises(LookupError, match="authority unavailable"):
            kb.guarded_specify_triage_task(conn, task_id, guard=fail_guard)
        assert kb.get_task(conn, task_id).status == "triage"
        assert [event.kind for event in kb.list_events(conn, task_id)] == ["created"]


def test_guard_cannot_mutate_board(kanban_home):
    with kbc.connect() as conn:
        task_id = _create_triage(conn)

        def mutating_guard(guard_conn, guarded_task_id):
            guard_conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
                (guarded_task_id, "guard", "mutation", 1),
            )
            return kb.TriageReleaseGuardResult(False, "deny")

        with pytest.raises(RuntimeError, match="read-only"):
            kb.guarded_specify_triage_task(conn, task_id, guard=mutating_guard)
        assert kb.get_task(conn, task_id).status == "triage"
        assert kb.list_comments(conn, task_id) == []


def test_guard_cannot_execute_ddl_and_connection_remains_reusable(kanban_home):
    with kbc.connect() as conn:
        task_id = _create_triage(conn)

        def ddl_guard(guard_conn, _task_id):
            guard_conn.execute("CREATE TABLE escaped_guard_write (value TEXT)")
            return kb.TriageReleaseGuardResult(True)

        with pytest.raises(RuntimeError, match="read-only"):
            kb.guarded_specify_triage_task(conn, task_id, guard=ddl_guard)
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'escaped_guard_write'"
        ).fetchone() is None
        assert kb.get_task(conn, task_id).status == "triage"
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_guard_cannot_commit_or_release_transaction_lock(kanban_home):
    with kbc.connect() as conn:
        task_id = _create_triage(conn)

        def commit_guard(guard_conn, _guarded_task_id):
            guard_conn.execute("SELECT 1")
            guard_conn.commit()
            return kb.TriageReleaseGuardResult(True)

        with pytest.raises(RuntimeError, match="read-only"):
            kb.guarded_specify_triage_task(conn, task_id, guard=commit_guard)
        assert kb.list_comments(conn, task_id) == []
        assert kb.get_task(conn, task_id).status == "triage"
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_select_function_cannot_escape_guard_transaction(kanban_home):
    with kbc.connect() as conn:
        task_id = _create_triage(conn)

        def side_effect():
            conn.set_authorizer(None)
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
                (task_id, "guard", "escaped", 1),
            )
            conn.commit()
            return 1

        conn.create_function("guard_side_effect", 0, side_effect)

        def function_guard(guard_conn, _guarded_task_id):
            guard_conn.execute("SELECT guard_side_effect()").fetchone()
            return kb.TriageReleaseGuardResult(True)

        with pytest.raises(sqlite3.DatabaseError):
            kb.guarded_specify_triage_task(conn, task_id, guard=function_guard)
        assert kb.list_comments(conn, task_id) == []
        assert kb.get_task(conn, task_id).status == "triage"
        assert [event.kind for event in kb.list_events(conn, task_id)] == ["created"]
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_guard_restores_existing_connection_authorizer(kanban_home):
    with kbc.connect() as conn:
        task_id = _create_triage(conn)
        observed = []

        def existing_authorizer(action, _arg1, _arg2, _database, _source):
            observed.append(action)
            return sqlite3.SQLITE_OK

        conn.set_authorizer(existing_authorizer)

        def denied_guard(guard_conn, _task_id):
            guard_conn.execute("SELECT 1").fetchone()
            return kb.TriageReleaseGuardResult(False, "hold")

        receipt = kb.guarded_specify_triage_task(conn, task_id, guard=denied_guard)
        observed.clear()
        conn.execute("SELECT 2").fetchone()

    assert receipt.outcome == "precondition_failed"
    assert sqlite3.SQLITE_SELECT in observed


def test_prior_authorizer_cannot_remove_scoped_guard_policy(kanban_home):
    with kbc.connect() as conn:
        task_id = _create_triage(conn)
        assert not hasattr(conn, "_hermes_set_scoped_authorizer")
        policy_restored = []

        def prior_authorizer(action, _arg1, _arg2, _database, _source):
            if getattr(conn, "_hermes_authorizer_locked", False):
                sqlite3.Connection.set_authorizer(conn, None)
            policy_restored.append(action)
            return sqlite3.SQLITE_OK

        conn.set_authorizer(prior_authorizer)
        invoked = False

        def side_effect():
            nonlocal invoked
            invoked = True
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
                (task_id, "guard", "escaped", 1),
            )
            conn.commit()
            return 1

        conn.create_function("hostile_guard_side_effect", 0, side_effect)

        with pytest.raises(sqlite3.DatabaseError):
            kb.guarded_specify_triage_task(
                conn,
                task_id,
                guard=lambda guard_conn, _task_id: (
                    guard_conn.execute("SELECT hostile_guard_side_effect()").fetchone()
                    and kb.TriageReleaseGuardResult(True)
                ),
            )
        assert invoked is False
        assert kb.list_comments(conn, task_id) == []
        assert kb.get_task(conn, task_id).status == "triage"
        policy_restored.clear()
        conn.execute("SELECT 3").fetchone()

    assert sqlite3.SQLITE_SELECT in policy_restored


def test_not_in_triage_does_not_invoke_guard(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="normal task")

        def unexpected_guard(_conn, _task_id):
            raise AssertionError("guard should not run")

        receipt = kb.guarded_specify_triage_task(conn, task_id, guard=unexpected_guard)

    assert receipt.outcome == "not_in_triage"
    assert receipt.released is False


def test_denied_guard_requires_reason(kanban_home):
    with kbc.connect() as conn:
        task_id = _create_triage(conn)
        with pytest.raises(ValueError, match="requires a reason"):
            kb.guarded_specify_triage_task(
                conn,
                task_id,
                guard=lambda _conn, _task_id: kb.TriageReleaseGuardResult(False),
            )
        assert kb.get_task(conn, task_id).status == "triage"
