"""Hermetic regression tests for Kanban completion transitions.

The verifier always uses an explicit temporary database. It rejects all known
live Kanban roots before opening SQLite, then closes every connection before a
reload so persistence assertions cannot be satisfied by one open connection.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _protected_kanban_roots() -> set[Path]:
    """Return canonical live roots that a verifier must never open."""
    default_root = Path.home() / ".hermes"
    configured_home = Path(os.environ.get("HERMES_KANBAN_HOME", "") or default_root)
    active_home = Path(os.environ.get("HERMES_HOME", "") or default_root)
    roots = {
        default_root,
        configured_home.expanduser(),
        active_home.expanduser(),
        kb.kanban_home(),
    }
    protected: set[Path] = set()
    for root in roots:
        # A root can be a default/custom Hermes home or a profile home. Include
        # profiles as a tree so an inactive profile is protected too.
        protected.update(
            {
                root / "kanban",
                root / "kanban.db",
                root / "profiles",
            }
        )
    pinned_db = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if pinned_db:
        pinned_path = Path(pinned_db).expanduser()
        protected.update({pinned_path, pinned_path.parent})
    return {path.resolve() for path in protected}


def _assert_hermetic_verification_path(db_path: Path) -> None:
    """Fail closed before SQLite can open a live/default/profile board path."""
    candidate = db_path.expanduser().resolve()
    for root in _protected_kanban_roots():
        if candidate == root or root in candidate.parents:
            raise AssertionError(
                f"refusing verifier DB under live Hermes Kanban path: {candidate}"
            )


@contextlib.contextmanager
def _closed_db(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Yield one connection, then prove it is closed before the next stage."""
    connection = kb.connect(db_path=db_path)
    try:
        with contextlib.closing(connection) as conn:
            yield conn
    finally:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")


def _task(conn: sqlite3.Connection, task_id: str) -> kb.Task:
    task = kb.get_task(conn, task_id)
    assert task is not None
    return task


def _event_payloads(conn: sqlite3.Connection, task_id: str, kind: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kind),
    ).fetchall()
    return [json.loads(row["payload"]) for row in rows]


def _notification_count(conn: sqlite3.Connection, task_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM kanban_notify_subs WHERE task_id = ?", (task_id,)
    ).fetchone()[0]


@pytest.fixture
def marker() -> str:
    """A fresh marker ensures verification data cannot match a live row."""
    return f"kanban-completion-verify-{secrets.token_hex(16)}"


@pytest.fixture
def hermetic_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Provide an asserted temporary DB despite hostile inherited board pins."""
    hostile_live_db = (
        tmp_path / "live-hermes" / "kanban" / "boards" / "subsidysmart" / "kanban.db"
    )
    monkeypatch.setenv("HERMES_KANBAN_DB", str(hostile_live_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "subsidysmart")

    # This reproduces the unsafe resolver result without opening the hostile
    # path. The verifier must never use this implicit resolution.
    assert kb.kanban_db_path() == hostile_live_db

    with tempfile.TemporaryDirectory(prefix="hermes-kanban-completion-verify-") as directory:
        db_path = Path(directory) / "kanban.db"
        _assert_hermetic_verification_path(db_path)
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        assert kb.init_db(db_path=db_path) == db_path
        assert db_path.exists()
        yield db_path


def test_hermetic_verifier_rejects_default_custom_profile_and_pinned_paths_before_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All profile roots and aliases fail before a verifier could create a DB."""
    home = tmp_path / "home"
    default_root = home / ".hermes"
    active_profile = default_root / "profiles" / "programmer"
    configured_home = tmp_path / "configured-hermes"
    pinned_db = tmp_path / "pinned" / "boards" / "ops" / "kanban.db"
    another_profile_db = default_root / "profiles" / "another-profile" / "kanban.db"
    another_profile_db.parent.mkdir(parents=True)
    alias = tmp_path / "profile-alias"
    alias.symlink_to(another_profile_db.parent, target_is_directory=True)

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(active_profile))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(configured_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "ops")

    protected = (
        default_root / "kanban.db",
        default_root / "kanban" / "boards" / "default" / "kanban.db",
        default_root / "profiles" / "another-profile" / "kanban.db",
        alias / "kanban.db",
        active_profile / "kanban.db",
        active_profile / "kanban" / "boards" / "active" / "kanban.db",
        configured_home / "kanban.db",
        configured_home / "kanban" / "boards" / "custom" / "kanban.db",
        pinned_db,
        pinned_db.parent / "nested.db",
    )
    for candidate in protected:
        with pytest.raises(AssertionError, match="live Hermes Kanban path"):
            _assert_hermetic_verification_path(candidate)

    # This is the independent inactive-profile probe: it is rejected before
    # init_db/connect can create either the directory or database file.
    unopened = default_root / "profiles" / "unopened-profile" / "kanban.db"
    with pytest.raises(AssertionError, match="live Hermes Kanban path"):
        _assert_hermetic_verification_path(unopened)
    assert not unopened.exists()
    assert not unopened.parent.exists()


def test_hermetic_verifier_rejects_unopened_sibling_under_custom_profile_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A custom active profile protects unopened siblings via kanban_home()."""
    custom_root = tmp_path / "custom-root"
    active_profile = custom_root / "profiles" / "active"
    unopened = custom_root / "profiles" / "unopened" / "kanban.db"

    monkeypatch.setenv("HERMES_HOME", str(active_profile))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)

    assert kb.kanban_home() == custom_root
    with pytest.raises(AssertionError, match="live Hermes Kanban path"):
        _assert_hermetic_verification_path(unopened)
    assert not unopened.exists()
    assert not unopened.parent.exists()


def test_blocked_completion_rejection_persists_across_closed_reloads_then_unblocks(
    hermetic_db: Path, marker: str
) -> None:
    """Direct blocked -> done is rejected with task/run/child/event state durable."""
    with _closed_db(hermetic_db) as conn:
        parent_id = kb.create_task(conn, title=f"{marker} parent", assignee="programmer")
        child_id = kb.create_task(
            conn, title=f"{marker} child", assignee="programmer", parents=[parent_id]
        )
        claimed = kb.claim_task(conn, parent_id, claimer="host:worker")
        assert claimed is not None
        run_id = _task(conn, parent_id).current_run_id
        assert run_id is not None
        assert kb.add_comment(conn, parent_id, "programmer", f"{marker} comment")
        conn.execute(
            "INSERT INTO kanban_notify_subs(task_id, platform, chat_id, thread_id, user_id, created_at, last_event_id) "
            "VALUES (?, 'telegram', 'test-chat', '', 'test-user', 0, 0)",
            (parent_id,),
        )
        conn.commit()
        assert kb.block_task(
            conn,
            parent_id,
            reason="review-required: human approval",
            expected_run_id=run_id,
        )

        assert kb.complete_task(conn, parent_id, result=f"{marker} must not persist") is False
        rejected = _task(conn, parent_id)
        assert rejected.status == "blocked"
        assert rejected.completed_at is None
        assert rejected.result is None
        assert _task(conn, child_id).status == "todo"
        assert kb.recompute_ready(conn) == 0
        assert _task(conn, parent_id).status == "blocked"

    # Each reload is after _closed_db proved the preceding connection closed.
    with _closed_db(hermetic_db) as conn:
        reloaded = _task(conn, parent_id)
        assert reloaded.status == "blocked"
        assert reloaded.completed_at is None
        assert reloaded.result is None
        assert _task(conn, child_id).status == "todo"
        blocked_run = kb.get_run(conn, run_id)
        assert blocked_run is not None and blocked_run.status == "blocked"
        assert [comment.body for comment in kb.list_comments(conn, parent_id)] == [f"{marker} comment"]
        assert _notification_count(conn, parent_id) == 1
        assert _event_payloads(conn, parent_id, "completed") == []
        assert _event_payloads(conn, parent_id, "completion_rejected_transition") == [
            {
                "reason": "completion requires status in ['ready', 'running']",
                "status": "blocked",
            }
        ]
        assert [event.kind for event in kb.list_events(conn, parent_id)][-1] == "completion_rejected_transition"

        assert kb.unblock_task(conn, parent_id)
        assert _task(conn, parent_id).status == "ready"
        assert kb.complete_task(conn, parent_id, result=f"{marker} approved") is True

    with _closed_db(hermetic_db) as conn:
        completed = _task(conn, parent_id)
        assert completed.status == "done"
        assert completed.result == f"{marker} approved"
        assert completed.completed_at is not None
        assert _task(conn, child_id).status == "ready"
        closed_run = kb.get_run(conn, run_id)
        assert closed_run is not None and closed_run.status == "blocked"
        assert [run.status for run in kb.list_runs(conn, parent_id)] == [
            "blocked",
            "completed",
        ]
        assert _notification_count(conn, parent_id) == 1
        assert len(_event_payloads(conn, parent_id, "completed")) == 1
        assert any(event.kind == "unblocked" for event in kb.list_events(conn, parent_id))


@pytest.mark.parametrize("status", sorted(kb.VALID_STATUSES - kb.COMPLETABLE_STATUSES))
def test_completion_rejects_every_non_completable_status_with_typed_event(
    hermetic_db: Path, marker: str, status: str
) -> None:
    """The whitelist rejects every current non-ready/running state."""
    with _closed_db(hermetic_db) as conn:
        task_id = kb.create_task(
            conn, title=f"{marker} not completable {status}", assignee="programmer"
        )
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        conn.commit()

        assert kb.complete_task(conn, task_id, result=f"{marker} must not persist") is False
        task = _task(conn, task_id)
        assert task.status == status
        assert task.completed_at is None
        assert task.result is None
        assert _event_payloads(conn, task_id, "completion_rejected_transition") == [
            {
                "reason": "completion requires status in ['ready', 'running']",
                "status": status,
            }
        ]


def test_running_expected_run_id_and_ready_completion_remain_allowed(
    hermetic_db: Path, marker: str
) -> None:
    """The whitelist preserves worker run ownership and manual ready completion."""
    with _closed_db(hermetic_db) as conn:
        running_id = kb.create_task(
            conn, title=f"{marker} worker task", assignee="programmer"
        )
        assert kb.claim_task(conn, running_id, claimer="host:worker") is not None
        run_id = _task(conn, running_id).current_run_id
        assert run_id is not None

        assert kb.complete_task(conn, running_id, expected_run_id=run_id + 1) is False
        assert _task(conn, running_id).status == "running"
        assert _event_payloads(conn, running_id, "completion_rejected_transition") == []
        assert kb.complete_task(conn, running_id, expected_run_id=run_id) is True

        ready_id = kb.create_task(
            conn, title=f"{marker} manual completion", assignee="programmer"
        )
        assert _task(conn, ready_id).status == "ready"
        assert kb.complete_task(conn, ready_id, result=f"{marker} manual") is True

    with _closed_db(hermetic_db) as conn:
        assert _task(conn, running_id).status == "done"
        ready = _task(conn, ready_id)
        assert ready.status == "done"
        assert ready.result == f"{marker} manual"
