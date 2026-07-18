"""Approval-gated Kanban dependency invariants."""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_gate(conn):
    parent = kb.create_task(conn, title="independent review", assignee="otto")
    child = kb.create_task(conn, title="release candidate", assignee="release")
    kb.link_tasks(conn, parent, child, gate_type=kb.APPROVAL_GATE)
    return parent, child


def _complete_review(conn, parent, metadata):
    assert kb.claim_task(conn, parent, claimer="reviewer") is not None
    assert kb.complete_task(
        conn,
        parent,
        summary="structured review verdict",
        metadata=metadata,
    )


def _task(conn, task_id) -> kb.Task:
    task = kb.get_task(conn, task_id)
    assert task is not None
    return task


def _latest_run(conn, task_id) -> kb.Run:
    run = kb.latest_run(conn, task_id)
    assert run is not None
    return run


def test_fresh_schema_declares_typed_task_link_column(kanban_home):
    with kb.connect() as conn:
        columns = {
            row["name"]: row
            for row in conn.execute("PRAGMA table_info(task_links)").fetchall()
        }
    assert columns["gate_type"]["type"] == "TEXT"


def test_connect_migrates_legacy_task_links_idempotently(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = home / "legacy.db"
    conn = sqlite3.connect(db_path)
    legacy_schema = kb.SCHEMA_SQL.replace("    gate_type  TEXT,\n", "")
    conn.executescript(legacy_schema)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('p', 'parent', 'done', 1)"
    )
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('c', 'child', 'todo', 2)"
    )
    conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES ('p', 'c')")
    conn.commit()
    conn.close()

    for _ in range(2):
        with kb.connect(db_path) as migrated:
            columns = {
                row["name"]
                for row in migrated.execute("PRAGMA table_info(task_links)").fetchall()
            }
            link = migrated.execute(
                "SELECT parent_id, child_id, gate_type FROM task_links"
            ).fetchone()
            assert columns >= {"parent_id", "child_id", "gate_type"}
            assert dict(link) == {"parent_id": "p", "child_id": "c", "gate_type": None}


def test_concurrent_connects_migrate_legacy_task_links_once(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = home / "legacy-concurrent.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(kb.SCHEMA_SQL.replace("    gate_type  TEXT,\n", ""))
    conn.commit()
    conn.close()

    def connect_and_read_columns():
        with kb.connect(db_path) as migrated:
            return {
                row["name"]
                for row in migrated.execute("PRAGMA table_info(task_links)")
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        columns = list(pool.map(lambda _: connect_and_read_columns(), range(16)))

    assert all("gate_type" in names for names in columns)


@pytest.mark.parametrize(
    ("metadata", "expected_state"),
    [
        ({"approved": False, "verdict": "FINAL NEEDS_WORK"}, "rejected"),
        ({"verdict": "APPROVED"}, "approval_missing"),
        ({"approved": "true", "verdict": "APPROVED"}, "approval_invalid"),
        (["not", "an", "object"], "metadata_malformed"),
        ({"approved": True, "verdict": "FINAL NEEDS_WORK"}, "verdict_conflict"),
        ({"approved": True, "verdict": 1}, "verdict_invalid"),
    ],
)
def test_approval_gate_holds_rejected_missing_malformed_or_conflicting_review(
    kanban_home, metadata, expected_state
):
    with kb.connect() as conn:
        parent, child = _make_gate(conn)
        _complete_review(conn, parent, metadata)

        assert _task(conn, parent).status == "done"
        assert _task(conn, child).status == "todo"
        assert kb.claim_task(conn, child, claimer="must-not-run") is None
        assert kb.list_runs(conn, child) == []

        blockers = kb.dependency_blockers(conn, child)
        assert blockers == [
            {
                "parent_id": parent,
                "gate_type": kb.APPROVAL_GATE,
                "gate_state": expected_state,
            }
        ]
        held = [event for event in kb.list_events(conn, child) if event.kind == "approval_gate_held"]
        assert held[-1].payload == {
            "parent_id": parent,
            "gate_type": kb.APPROVAL_GATE,
            "gate_state": expected_state,
        }

        # Repeated dispatcher ticks remain held without event spam.
        before = len(held)
        assert kb.recompute_ready(conn) == 0
        assert kb.recompute_ready(conn) == 0
        held = [event for event in kb.list_events(conn, child) if event.kind == "approval_gate_held"]
        assert len(held) == before


def test_approval_gate_holds_ambiguous_reviewer_identity(kanban_home):
    with kb.connect() as conn:
        parent, child = _make_gate(conn)
        _complete_review(conn, parent, {"approved": True, "verdict": "APPROVED"})
        conn.execute("UPDATE tasks SET assignee = 'different-reviewer' WHERE id = ?", (parent,))
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (child,))
        assert kb.recompute_ready(conn) == 0
        assert kb.dependency_blockers(conn, child)[0]["gate_state"] == "identity_ambiguous"
        assert kb.claim_task(conn, child) is None
        assert kb.list_runs(conn, child) == []


def test_approved_gate_promotes_once_and_preserves_review_metadata(kanban_home):
    metadata = {
        "approved": True,
        "verdict": "FINAL APPROVED",
        "exact_head": "abc123",
        "exact_tree": "def456",
    }
    with kb.connect() as conn:
        parent, child = _make_gate(conn)
        _complete_review(conn, parent, metadata)

        assert _task(conn, child).status == "ready"
        assert kb.dependency_blockers(conn, child) == []
        assert kb.recompute_ready(conn) == 0
        assert _latest_run(conn, parent).metadata == metadata

        first = kb.claim_task(conn, child, claimer="winner")
        second = kb.claim_task(conn, child, claimer="loser")
        assert first is not None
        assert second is None
        assert len(kb.list_runs(conn, child)) == 1


def test_archived_approved_parent_remains_satisfied(kanban_home):
    with kb.connect() as conn:
        parent, child = _make_gate(conn)
        _complete_review(conn, parent, {"approved": True, "verdict": "APPROVED"})
        assert kb.archive_task(conn, parent)
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (child,))

        assert kb.recompute_ready(conn) == 1
        task = kb.get_task(conn, child)
        assert task is not None
        assert task.status == "ready"
        assert kb.dependency_blockers(conn, child) == []


def test_untyped_remediation_promotes_while_typed_release_stays_held(kanban_home):
    with kb.connect() as conn:
        review = kb.create_task(conn, title="review", assignee="otto")
        release = kb.create_task(conn, title="release", assignee="release")
        remediation = kb.create_task(conn, title="remediation", assignee="rog")
        kb.link_tasks(conn, review, release, gate_type=kb.APPROVAL_GATE)
        kb.link_tasks(conn, review, remediation)

        _complete_review(
            conn,
            review,
            {"approved": False, "verdict": "FINAL NEEDS_WORK"},
        )

        assert _task(conn, release).status == "todo"
        assert _task(conn, remediation).status == "ready"
        assert kb.claim_task(conn, release) is None
        assert kb.claim_task(conn, remediation) is not None


def test_mixed_parent_fan_in_requires_done_ordinary_and_approved_gate(kanban_home):
    with kb.connect() as conn:
        review = kb.create_task(conn, title="review", assignee="otto")
        build = kb.create_task(conn, title="build", assignee="rog")
        release = kb.create_task(conn, title="release", assignee="release")
        kb.link_tasks(conn, review, release, gate_type=kb.APPROVAL_GATE)
        kb.link_tasks(conn, build, release)

        _complete_review(conn, review, {"approved": True, "verdict": "APPROVED"})
        assert _task(conn, release).status == "todo"
        assert kb.dependency_blockers(conn, release) == [
            {"parent_id": build, "gate_type": None, "gate_state": "parent_pending"}
        ]

        assert kb.complete_task(conn, build, summary="build complete")
        assert _task(conn, release).status == "ready"


def test_claim_and_manual_promote_recheck_gate_after_racy_ready_write(kanban_home):
    with kb.connect() as conn:
        parent, child = _make_gate(conn)
        _complete_review(
            conn,
            parent,
            {"approved": False, "verdict": "FINAL NEEDS_WORK"},
        )
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))

        assert kb.claim_task(conn, child, claimer="racy-dispatch") is None
        assert _task(conn, child).status == "todo"
        assert kb.list_runs(conn, child) == []

        promoted, reason = kb.promote_task(
            conn,
            child,
            actor="operator",
            reason="must remain gated",
        )
        assert promoted is False
        assert reason is not None
        assert parent in reason
        assert "rejected" in reason
        assert kb.list_runs(conn, child) == []


@pytest.mark.parametrize("reclaim_mode", ["manual", "stale"])
def test_reclaim_and_retry_remain_held_if_approval_is_revoked(
    kanban_home, reclaim_mode
):
    with kb.connect() as conn:
        parent, child = _make_gate(conn)
        _complete_review(conn, parent, {"approved": True, "verdict": "APPROVED"})
        assert kb.claim_task(conn, child, claimer="candidate-worker") is not None

        parent_run = kb.latest_run(conn, parent)
        assert parent_run is not None
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (
                json.dumps(
                    {"approved": False, "verdict": "FINAL NEEDS_WORK"},
                    separators=(",", ":"),
                ),
                parent_run.id,
            ),
        )

        if reclaim_mode == "manual":
            assert kb.reclaim_task(conn, child, reason="review corrected") is True
        else:
            conn.execute(
                "UPDATE tasks SET claim_expires = 0 WHERE id = ?",
                (child,),
            )
            conn.execute(
                "UPDATE task_runs SET claim_expires = 0 WHERE task_id = ? "
                "AND ended_at IS NULL",
                (child,),
            )
            assert kb.release_stale_claims(conn) == 1

        task = kb.get_task(conn, child)
        assert task is not None
        assert task.status == "todo"
        assert kb.recompute_ready(conn) == 0
        assert kb.claim_task(conn, child, claimer="retry") is None
        runs = kb.list_runs(conn, child)
        assert len(runs) == 1
        assert runs[0].outcome == "reclaimed"


def test_create_task_supports_explicit_approval_parent(kanban_home):
    with kb.connect() as conn:
        review = kb.create_task(conn, title="review", assignee="otto")
        release = kb.create_task(
            conn,
            title="release",
            assignee="release",
            approval_parents=[review],
        )
        row = conn.execute(
            "SELECT gate_type FROM task_links WHERE parent_id = ? AND child_id = ?",
            (review, release),
        ).fetchone()
        assert row["gate_type"] == kb.APPROVAL_GATE
        assert _task(conn, release).status == "todo"


def test_dispatcher_does_not_spawn_rejected_approval_child(
    kanban_home, monkeypatch
):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _profile: True)
    spawned = []
    with kb.connect() as conn:
        parent, child = _make_gate(conn)
        _complete_review(
            conn,
            parent,
            {"approved": False, "verdict": "FINAL NEEDS_WORK"},
        )
        # Simulate any external writer or stale dispatcher tick that left the
        # task in ready. claim_task is the last invariant boundary before spawn.
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))

        result = kb.dispatch_once(
            conn,
            max_spawn=1,
            spawn_fn=lambda task, workspace: spawned.append((task.id, workspace)),
        )

        assert spawned == []
        assert result.spawned == []
        task = kb.get_task(conn, child)
        assert task is not None
        assert task.status == "todo"
        assert kb.list_runs(conn, child) == []


def test_concurrent_approval_completion_and_claim_never_claims_rejected_child(
    kanban_home,
):
    with kb.connect() as setup:
        parent, child = _make_gate(setup)
        assert kb.claim_task(setup, parent, claimer="reviewer") is not None

    def complete_rejected():
        with kb.connect() as conn:
            return kb.complete_task(
                conn,
                parent,
                summary="rejected",
                metadata={"approved": False, "verdict": "FINAL NEEDS_WORK"},
            )

    def try_claim():
        claimed = 0
        for _ in range(20):
            with kb.connect() as conn:
                conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
                if kb.claim_task(conn, child, claimer="racer") is not None:
                    claimed += 1
        return claimed

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        completed = pool.submit(complete_rejected)
        raced = pool.submit(try_claim)
        assert completed.result() is True
        assert raced.result() == 0

    with kb.connect() as conn:
        # One final post-completion attempt proves the second dispatcher tick
        # also sees the rejected terminal gate rather than the earlier pending
        # state observed by a racing tick.
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
        assert kb.claim_task(conn, child, claimer="post-completion") is None
        assert _task(conn, child).status == "todo"
        assert kb.list_runs(conn, child) == []
        assert json.loads(
            conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? "
                "AND kind = 'claim_rejected' ORDER BY id DESC LIMIT 1",
                (child,),
            ).fetchone()["payload"]
        )["approval_gates"][0]["gate_state"] == "rejected"
