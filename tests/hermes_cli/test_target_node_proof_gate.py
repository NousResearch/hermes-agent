"""Tests for target_node proof gate — t_728a7e43.

Verifies that infra/operator cards with target_node constraints cannot be
marked DONE unless completion metadata includes proof from the right host.
"""
import json
import socket
import sqlite3
import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def db():
    """Fresh in-memory kanban DB for each test."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        kb.init_db(db_path=db_path)
        conn = kb.connect(db_path=db_path)
        yield conn
        conn.close()


def test_target_node_missing_allows_completion(db):
    """Legacy cards with no target_node set should complete without proof."""
    tid = kb.create_task(
        db,
        title="legacy task",
        body="no target_node constraint",
        assignee="builder",
    )
    kb.claim_task(db, tid, claimer="conductor:99")
    ok = kb.complete_task(db, tid, summary="done", metadata={})
    assert ok, "legacy task with no target_node should complete without proof"


def test_target_node_any_allows_completion(db):
    """Cards with target_node=any should complete without proof."""
    tid = kb.create_task(
        db,
        title="any-node task",
        body="can run anywhere",
        assignee="builder",
        target_node="any",
    )
    kb.claim_task(db, tid, claimer="conductor:99")
    ok = kb.complete_task(db, tid, summary="done", metadata={})
    assert ok, "target_node=any should complete without proof"


def test_target_node_matching_proof_allows_completion(db):
    """Completion with matching verified_on_node should succeed."""
    tid = kb.create_task(
        db,
        title="conductor infra task",
        body="must run on conductor",
        assignee="operator",
        target_node="conductor",
    )
    kb.claim_task(db, tid, claimer="conductor:99")
    
    metadata = {
        "verified_on_node": "conductor",
        "hostname_proof": socket.gethostname(),
    }
    ok = kb.complete_task(db, tid, summary="done on conductor", metadata=metadata)
    assert ok, "matching target_node=conductor + verified_on_node=conductor should succeed"


def test_target_node_mismatched_proof_blocks_completion(db):
    """Completion with mismatched verified_on_node should fail."""
    tid = kb.create_task(
        db,
        title="conductor infra task",
        body="must run on conductor",
        assignee="operator",
        target_node="conductor",
    )
    kb.claim_task(db, tid, claimer="conductor:99")
    
    # Worker claims they verified on cfo-vm but target was conductor
    metadata = {
        "verified_on_node": "cfo-vm",
        "hostname_proof": "wrong-host",
    }
    ok = kb.complete_task(db, tid, summary="done on wrong node", metadata=metadata)
    assert not ok, "mismatched target_node should reject completion"
    
    # Task should still be running, not done
    row = db.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "running", "task should remain running after rejected completion"


def test_target_node_missing_proof_blocks_completion(db):
    """Completion without verified_on_node when target_node is set should fail."""
    tid = kb.create_task(
        db,
        title="conductor infra task",
        body="must run on conductor",
        assignee="operator",
        target_node="conductor",
    )
    kb.claim_task(db, tid, claimer="conductor:99")
    
    # Metadata has no verified_on_node at all
    metadata = {"some_other_key": "value"}
    ok = kb.complete_task(db, tid, summary="done without proof", metadata=metadata)
    assert not ok, "target_node set but no verified_on_node should reject completion"
    
    row = db.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "running"


def test_target_node_empty_metadata_blocks_completion(db):
    """Completion with empty metadata when target_node is set should fail."""
    tid = kb.create_task(
        db,
        title="conductor infra task",
        body="must run on conductor",
        assignee="operator",
        target_node="conductor",
    )
    kb.claim_task(db, tid, claimer="conductor:99")
    
    ok = kb.complete_task(db, tid, summary="done without metadata")
    assert not ok, "target_node set but metadata=None should reject completion"
    
    row = db.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "running"


def test_target_node_all_valid_nodes(db):
    """All acceptable node names (conductor, cfo-vm, mac, any) should work."""
    for node in ["conductor", "cfo-vm", "mac", "any"]:
        tid = kb.create_task(
            db,
            title=f"task for {node}",
            body=f"target={node}",
            assignee="operator",
            target_node=node,
        )
        
        # target_node=any doesn't require proof
        if node == "any":
            kb.claim_task(db, tid, claimer="conductor:99")
            ok = kb.complete_task(db, tid, summary=f"done on {node}")
            assert ok, f"target_node={node} should complete without proof"
        else:
            kb.claim_task(db, tid, claimer="conductor:99")
            metadata = {"verified_on_node": node, "hostname_proof": "test"}
            ok = kb.complete_task(db, tid, summary=f"done on {node}", metadata=metadata)
            assert ok, f"target_node={node} with matching proof should succeed"


def test_target_node_invalid_node_rejected_on_create(db):
    """Invalid target_node values should be rejected at task creation."""
    with pytest.raises(ValueError, match="target_node must be one of"):
        kb.create_task(
            db,
            title="invalid node",
            body="oops",
            assignee="operator",
            target_node="invalid-node",
        )


def test_completion_blocked_event_emitted_on_mismatch(db):
    """A completion_blocked_target_node_mismatch event should be emitted on rejection."""
    tid = kb.create_task(
        db,
        title="conductor infra task",
        body="must run on conductor",
        assignee="operator",
        target_node="conductor",
    )
    kb.claim_task(db, tid, claimer="conductor:99")
    
    metadata = {"verified_on_node": "cfo-vm"}
    ok = kb.complete_task(db, tid, summary="wrong node", metadata=metadata)
    assert not ok
    
    # Check that the event was logged
    events = db.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (tid,)
    ).fetchall()
    assert len(events) > 0
    evt = events[0]
    assert evt["kind"] == "completion_blocked_target_node_mismatch"
    payload = json.loads(evt["payload"])
    assert payload["expected"] == "conductor"
    assert payload["actual"] == "cfo-vm"


def test_target_node_persisted_and_readable(db):
    """target_node should be persisted and readable via list_tasks."""
    tid = kb.create_task(
        db,
        title="conductor task",
        body="body",
        assignee="operator",
        target_node="conductor",
    )
    
    rows = kb.list_tasks(db)
    task = next((t for t in rows if t.id == tid), None)
    assert task is not None
    assert task.target_node == "conductor"


def test_legacy_task_has_null_target_node(db):
    """Tasks created without target_node should have target_node=NULL."""
    tid = kb.create_task(
        db,
        title="legacy task",
        body="no target_node",
        assignee="builder",
    )
    
    row = db.execute("SELECT target_node FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["target_node"] is None


def test_target_node_proof_regression_t_e06f0001_class(db):
    """Regression: t_e06f0001-class false-DONE must be rejected.
    
    Context: t_e06f0001 was an infra card that claimed DONE with proof
    collected on cfo-vm when the artifact was actually on Conductor.
    This test simulates that scenario and verifies rejection.
    """
    tid = kb.create_task(
        db,
        title="[conductor-infra] Deploy artifact on conductor",
        body="Artifact must exist on conductor, not cfo-vm",
        assignee="operator",
        target_node="conductor",
    )
    kb.claim_task(db, tid, claimer="cfo-vm:99")
    
    # Worker on cfo-vm tries to complete with cfo-vm proof
    metadata = {
        "verified_on_node": "cfo-vm",
        "artifact_path": "/opt/artifact",
        "ls_proof": "-rw-r--r-- 1 ubuntu ubuntu 1234 artifact",
    }
    ok = kb.complete_task(db, tid, summary="artifact deployed", metadata=metadata)
    assert not ok, "t_e06f0001-class false-DONE should be rejected"
    
    row = db.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "running", "task should remain running after rejection"
