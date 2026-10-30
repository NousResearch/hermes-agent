from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli.kanban_intake import (
    IntakePolicyError,
    exempt_creation,
    reviewing,
)


@pytest.fixture
def intake_board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb.write_board_metadata("default", intake_profile="planner")
    return home


def _state(conn, task_id):
    return conn.execute(
        "SELECT status, intake_state, intake_profile FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()


def test_new_requests_are_pending_even_with_spoofed_creator_skills_and_parent(intake_board):
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", initial_status="blocked")
        task_id = kb.create_task(
            conn, title="request", created_by="planner", skills=["trusted-planner"],
            parents=[parent], triage=False,
        )
        row = _state(conn, task_id)
        assert (row[0], row[1], row[2]) == ("triage", "pending", "planner")
        assert kb.claim_task(conn, task_id) is None
        assert not kb.complete_task(conn, task_id, result="spoofed")
        assert not kb.promote_task(conn, task_id, actor="planner")[0]
        assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)).fetchone()[0] == 0


def test_explicit_block_stays_blocked_until_owner_releases_then_returns_to_triage(intake_board):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="blocked request", initial_status="blocked")
        assert _state(conn, task_id)[0:2] == ("blocked", "pending")
        assert kb.unblock_task(conn, task_id)
        assert _state(conn, task_id)[0:2] == ("triage", "pending")
        assert kb.claim_task(conn, task_id) is None


def test_pending_ready_spoof_cannot_claim_review_or_complete(intake_board):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="request")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        assert kb.claim_task(conn, task_id) is None
        assert kb.request_review(conn, task_id, summary="review it") is False
        assert not kb.complete_task(conn, task_id, result="direct close")
        assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)).fetchone()[0] == 0


def test_manual_author_does_not_admit_pending_but_matching_review_context_does(intake_board):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="request")
        assert not kb.specify_triage_task(conn, task_id, title="Spoof", author="planner")
        with kb.write_txn(conn):
            with pytest.raises(IntakePolicyError):
                with reviewing(conn, task_id, profile="other", source_fingerprint="a" * 64):
                    pass
            with reviewing(conn, task_id, profile="planner", source_fingerprint="a" * 64):
                assert kb.specify_triage_task(conn, task_id, title="Reviewed")
        kb.recompute_ready(conn)
        assert _state(conn, task_id)[0:2] == ("ready", "reviewed")
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'intake_reviewed'", (task_id,),
        ).fetchone()
        assert json.loads(event[0]) == {"profile": "planner", "source_fingerprint": "a" * 64}


def test_reviewed_decomposition_inherits_admission_atomically(intake_board, monkeypatch):
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="request")
        with kb.write_txn(conn):
            with reviewing(conn, root, profile="planner", source_fingerprint="b" * 64):
                children = decompose_triage_task(
                    conn, root, root_assignee="planner", children=[{"title": "child 1"}, {"title": "child 2"}],
                )
        assert children
        assert _state(conn, root)[1] == "reviewed"
        assert [_state(conn, child)[1:] for child in children] == [
            ("inherited", "planner"), ("inherited", "planner"),
        ]
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind = 'intake_inherited'",
        ).fetchone()[0] == 2

    with kbc.connect() as conn:
        root = kb.create_task(conn, title="second request")
        original_insert = __import__("hermes_cli.kanban_db_graph", fromlist=["_insert_decomposed_child"])
        real_insert = original_insert._insert_decomposed_child
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("forced graph failure")
            return real_insert(*args, **kwargs)

        monkeypatch.setattr(original_insert, "_insert_decomposed_child", fail_second)
        with kb.write_txn(conn):
            with reviewing(conn, root, profile="planner", source_fingerprint="c" * 64):
                with pytest.raises(RuntimeError, match="forced graph failure"):
                    decompose_triage_task(
                        conn, root, root_assignee="planner", children=[{"title": "first"}, {"title": "second"}],
                    )
        assert _state(conn, root)[1] == "pending"
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE title IN ('first', 'second')").fetchone()[0] == 0


def test_decomposition_children_are_admitted_and_inherit_root_prerequisites(intake_board):
    metadata_path = kb.board_metadata_path("default")
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("intake_profile")
    metadata_path.write_text(json.dumps(metadata))
    with kbc.connect() as conn:
        prerequisite = kb.create_task(conn, title="legacy prerequisite", initial_status="blocked")
        legacy_root = kb.create_task(
            conn, title="legacy root", parents=[prerequisite], triage=True,
        )

    metadata["intake_profile"] = "planner"
    metadata_path.write_text(json.dumps(metadata))
    with kbc.connect() as conn:
        legacy_children = decompose_triage_task(
            conn, legacy_root, root_assignee=None, children=[{"title": "legacy child"}],
            author="untrusted",
        )
        assert legacy_children
        legacy_child = legacy_children[0]
        assert _state(conn, legacy_child)[0:3] == ("triage", "pending", "planner")
        assert kb.claim_task(conn, legacy_child) is None
        assert conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
            (prerequisite, legacy_child),
        ).fetchone()

    with kbc.connect() as conn:
        root = kb.create_task(conn, title="reviewed root", parents=[prerequisite])
        with kb.write_txn(conn):
            with reviewing(conn, root, profile="planner", source_fingerprint="f" * 64):
                children = decompose_triage_task(
                    conn, root, root_assignee="planner", children=[{"title": "reviewed child"}],
                )
        assert children
        child = children[0]
        assert _state(conn, child)[1:] == ("inherited", "planner")
        assert conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
            (prerequisite, child),
        ).fetchone()
        assert _state(conn, child)[0] == "todo"
        assert kb.claim_task(conn, child) is None


def test_content_exemption_requires_native_context_and_idempotency_preserves_pending(intake_board):
    with kbc.connect() as conn:
        pending = kb.create_task(
            conn, title="proposal", created_by="content-workflow", idempotency_key="proposal-1",
        )
        assert _state(conn, pending)[1] == "pending"
        assert kb.create_task(
            conn, title="duplicate", created_by="planner", idempotency_key="proposal-1",
        ) == pending
        with exempt_creation(conn, reason="validated article proposal workflow"):
            exempt = kb.create_task(conn, title="proposal", created_by="anything")
        assert _state(conn, exempt)[1] == "exempt"
        assert _state(conn, pending)[1] == "pending"


def test_existing_rows_stay_legacy_and_pending_requires_saved_reviewer_after_disable(intake_board):
    metadata_path = kb.board_metadata_path("default")
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("intake_profile")
    metadata_path.write_text(json.dumps(metadata))
    with kbc.connect() as conn:
        legacy = kb.create_task(conn, title="before enrollment")
    metadata["intake_profile"] = "planner"
    metadata_path.write_text(json.dumps(metadata))
    with kbc.connect() as conn:
        assert _state(conn, legacy)[1] is None
        pending = kb.create_task(conn, title="after enable")
        assert _state(conn, pending)[1:] == ("pending", "planner")
        metadata.pop("intake_profile")
        metadata_path.write_text(json.dumps(metadata))
        with kb.write_txn(conn):
            with reviewing(conn, pending, profile="planner", source_fingerprint="d" * 64):
                pass
        metadata["intake_profile"] = "changed"
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(IntakePolicyError):
            with reviewing(conn, pending, profile="planner", source_fingerprint="d" * 64):
                pass


def test_review_does_not_release_independent_dependency_gate(intake_board):
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="unfinished parent", initial_status="blocked")
        task_id = kb.create_task(conn, title="child request", parents=[parent])
        with kb.write_txn(conn):
            with reviewing(conn, task_id, profile="planner", source_fingerprint="e" * 64):
                assert kb.specify_triage_task(conn, task_id, title="Reviewed child")
        assert _state(conn, task_id)[0:2] == ("todo", "reviewed")
        assert kb.claim_task(conn, task_id) is None


def test_admission_policy_uses_the_connection_board_identity(intake_board):
    kb.create_board("alternate")
    kb.write_board_metadata("alternate", intake_profile="alternate-planner")
    with kbc.connect(board="alternate") as conn:
        task_id = kb.create_task(conn, title="alternate request", board="alternate")
        assert _state(conn, task_id)[1:] == ("pending", "alternate-planner")


def test_malformed_enrollment_policy_fails_closed(intake_board):
    kb.board_metadata_path("default").write_text("{broken", encoding="utf-8")
    with kbc.connect() as conn:
        with pytest.raises(IntakePolicyError, match="invalid intake policy metadata"):
            kb.create_task(conn, title="cannot enroll")
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
