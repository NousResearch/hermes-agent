"""Packet-bound approval gates cannot be satisfied by generic task actions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_packet_approval as kpa


PACKET = b'{"project":"P1742-copy","operations":["switch"]}\n'
PACKET_SHA = hashlib.sha256(PACKET).hexdigest()
IDENTITY = {
    "host": "studio-mac",
    "project": "P1742-copy",
    "commit": "a" * 40,
    "operations": ["switch"],
    "rollback": "restore the previous camera-switch commit",
}
ACTOR = {
    "user_id": "user-123",
    "email": "david@example.com",
    "display_name": "David",
    "provider": "basic",
}


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _gate_and_child(conn):
    gate = kb.create_task(
        conn,
        title="Approve exact controlled-proof packet",
        initial_status="blocked",
    )
    child = kb.create_task(conn, title="Controlled proof", parents=[gate])
    kpa.configure_packet_approval(
        conn,
        gate,
        packet_sha256=PACKET_SHA,
        execution_identity=IDENTITY,
    )
    return gate, child


def test_generic_complete_unblock_comment_and_recompute_cannot_satisfy_gate(conn):
    gate, child = _gate_and_child(conn)

    kb.add_comment(conn, gate, author="operator", body="approved")
    with pytest.raises(kpa.PacketApprovalRequired):
        kb.complete_task(conn, gate, summary="approved")
    with pytest.raises(kpa.PacketApprovalRequired):
        kb.unblock_task(conn, gate)

    conn.execute("UPDATE tasks SET status='done' WHERE id=?", (gate,))
    conn.commit()
    assert kb.recompute_ready(conn) == 0
    assert kb.get_task(conn, child).status == "todo"


@pytest.mark.parametrize("action", ["unlink", "archive", "delete"])
def test_generic_graph_removal_cannot_bypass_or_orphan_gate(conn, action):
    gate, child = _gate_and_child(conn)

    with pytest.raises(kpa.PacketApprovalRequired):
        if action == "unlink":
            kb.unlink_tasks(conn, gate, child)
        elif action == "archive":
            kb.archive_task(conn, gate)
        else:
            kb.delete_task(conn, gate)

    assert kb.parent_ids(conn, child) == [gate]
    assert kb.get_task(conn, gate).status == "blocked"
    assert kpa.get_packet_approval(conn, gate) is not None
    assert kb.get_task(conn, child).status == "todo"


@pytest.mark.parametrize(
    "identity",
    [
        {"x": "y"},
        {"host": "studio", "project": "copy", "commit": "bad", "operations": ["switch"], "rollback": "restore"},
        {"host": "studio", "project": "copy", "commit": "a" * 40, "operations": [], "rollback": "restore"},
        {"host": "studio", "project": "copy", "commit": "a" * 40, "operations": ["switch"], "rollback": ""},
    ],
)
def test_gate_rejects_incomplete_execution_identity(conn, identity):
    gate = kb.create_task(conn, title="approval", initial_status="blocked")
    with pytest.raises(ValueError, match="execution_identity"):
        kpa.configure_packet_approval(
            conn,
            gate,
            packet_sha256=PACKET_SHA,
            execution_identity=identity,
        )
    assert kpa.get_packet_approval(conn, gate) is None


def test_exact_approval_records_actor_time_and_releases_child(conn):
    gate, child = _gate_and_child(conn)

    receipt = kpa.approve_packet(
        conn,
        gate,
        packet_sha256=PACKET_SHA,
        execution_identity=IDENTITY,
        actor=ACTOR,
    )

    assert receipt["packet_sha256"] == PACKET_SHA
    assert receipt["actor"] == ACTOR
    assert isinstance(receipt["approved_at"], int)
    assert kb.get_task(conn, gate).status == "done"
    assert kb.get_task(conn, child).status == "ready"
    assert kpa.require_packet_approval(
        conn,
        gate,
        packet_sha256=PACKET_SHA,
        execution_identity=IDENTITY,
    )["approved_at"] == receipt["approved_at"]


@pytest.mark.parametrize(
    "packet_sha256,identity,error",
    [
        ("b" * 64, IDENTITY, "packet hash"),
        (PACKET_SHA, {**IDENTITY, "project": "wrong-project"}, "execution identity"),
    ],
)
def test_approval_rejects_packet_or_identity_drift_without_state_change(
    conn, packet_sha256, identity, error,
):
    gate, child = _gate_and_child(conn)

    with pytest.raises(kpa.PacketApprovalMismatch, match=error):
        kpa.approve_packet(
            conn,
            gate,
            packet_sha256=packet_sha256,
            execution_identity=identity,
            actor=ACTOR,
        )

    assert kb.get_task(conn, gate).status == "blocked"
    assert kb.get_task(conn, child).status == "todo"
    stored = kpa.get_packet_approval(conn, gate)
    assert stored["approved_at"] is None


def _database_state(conn) -> tuple[str, ...]:
    return tuple(conn.iterdump())


@pytest.mark.parametrize(
    "packet_sha256,identity,error",
    [
        ("c" * 64, IDENTITY, "packet hash"),
        (PACKET_SHA, {**IDENTITY, "host": "wrong-host"}, "execution identity"),
    ],
)
def test_entrypoint_refusal_leaves_approved_state_unchanged(
    conn, packet_sha256, identity, error,
):
    gate, _child = _gate_and_child(conn)
    kpa.approve_packet(
        conn,
        gate,
        packet_sha256=PACKET_SHA,
        execution_identity=IDENTITY,
        actor=ACTOR,
    )
    before = _database_state(conn)

    with pytest.raises(kpa.PacketApprovalMismatch, match=error):
        kpa.require_packet_approval(
            conn,
            gate,
            packet_sha256=packet_sha256,
            execution_identity=identity,
        )

    assert _database_state(conn) == before


def test_entrypoint_rejects_unfinished_and_reblocked_gate_without_state_change(conn):
    gate, _child = _gate_and_child(conn)
    before = _database_state(conn)
    with pytest.raises(kpa.PacketApprovalRequired, match="absent"):
        kpa.require_packet_approval(
            conn,
            gate,
            packet_sha256=PACKET_SHA,
            execution_identity=IDENTITY,
        )
    assert _database_state(conn) == before

    kpa.approve_packet(
        conn,
        gate,
        packet_sha256=PACKET_SHA,
        execution_identity=IDENTITY,
        actor=ACTOR,
    )
    conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (gate,))
    conn.commit()
    before = _database_state(conn)
    with pytest.raises(kpa.PacketApprovalRequired, match="not complete"):
        kpa.require_packet_approval(
            conn,
            gate,
            packet_sha256=PACKET_SHA,
            execution_identity=IDENTITY,
        )
    assert _database_state(conn) == before


def test_replayed_approval_is_rejected_without_state_change(conn):
    gate, _child = _gate_and_child(conn)
    kpa.approve_packet(
        conn,
        gate,
        packet_sha256=PACKET_SHA,
        execution_identity=IDENTITY,
        actor=ACTOR,
    )
    before = _database_state(conn)

    with pytest.raises(kpa.PacketApprovalRequired, match="already approved"):
        kpa.approve_packet(
            conn,
            gate,
            packet_sha256=PACKET_SHA,
            execution_identity=IDENTITY,
            actor=ACTOR,
        )

    assert _database_state(conn) == before


def test_read_only_checker_runs_in_hermes_interpreter_and_preserves_state(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    with kbc.connect_closing(board="default") as seeded:
        gate, _child = _gate_and_child(seeded)
        kpa.approve_packet(
            seeded,
            gate,
            packet_sha256=PACKET_SHA,
            execution_identity=IDENTITY,
            actor=ACTOR,
        )
        before = _database_state(seeded)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "line-modules-must-not-load")
    checked = subprocess.run(
        [sys.executable, "-m", "hermes_cli.kanban_packet_check", "--task-id", gate, "--board", "default"],
        input=json.dumps({"packet_sha256": PACKET_SHA, "execution_identity": IDENTITY}),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        timeout=30,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["approved_at"] > 0
    with kbc.connect_closing(board="default") as verified:
        assert _database_state(verified) == before


def test_entrypoint_reads_approval_and_lifecycle_in_one_snapshot(conn):
    gate, _child = _gate_and_child(conn)
    kpa.approve_packet(conn, gate, packet_sha256=PACKET_SHA,
                       execution_identity=IDENTITY, actor=ACTOR)
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        approved = kpa.require_packet_approval(
            conn, gate, packet_sha256=PACKET_SHA, execution_identity=IDENTITY)
    finally:
        conn.set_trace_callback(None)
    reads = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(reads) == 1, "approval and lifecycle must share one SQLite snapshot"
    assert "JOIN tasks" in reads[0]
    assert approved["approved_packet_sha256"] == PACKET_SHA
    # A lifecycle withdrawal must refuse even while the exact approval row
    # remains intact: the joined read must inspect both values together.
    conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (gate,))
    conn.commit()
    with pytest.raises(kpa.PacketApprovalRequired, match="not complete"):
        kpa.require_packet_approval(
            conn, gate, packet_sha256=PACKET_SHA, execution_identity=IDENTITY)
