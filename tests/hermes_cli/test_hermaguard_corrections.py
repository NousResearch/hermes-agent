"""C1–C4 adversarial correction tests — Hermaguard event path.

RED witnesses ported from the KENSEI/Quan adversarial battery:
  C1 evidence must bind to the exact requirement + review cycle
  C2 gate-time tamper evidence (digest re-verified against report bytes)
  C3 exactly-once survives concurrent writers (two connections)
  C4 default-off enforced at every mutation boundary
"""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import hermaguard_events as he


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: True)
    conn = kb.connect()
    yield conn, home
    conn.close()


def _task(conn, tier="full", kind="task"):
    return kb.create_task(
        conn, title="t", assignee="octacon",
        body="## Problem\nx\n## Success Criteria\ny", triage=True,
        tier=tier, task_kind=kind,
    )


def _review(conn, tid):
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    conn.commit()
    kb.request_review(conn, tid, force=True)
    conn.commit()
    row = conn.execute(
        "SELECT id FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _report(home, tid, content=b"hermaguard report"):
    artifact = home / "feature-artifacts" / tid
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "hermaguard-report.md").write_bytes(content)
    return artifact


class TestC1Binding:
    def test_evidence_for_unrequired_review_rejected(self, env):
        """RED: evidence for cycle 2 accepted when only cycle 1 had a requirement."""
        conn, home = env
        tid = _task(conn)
        rid1 = _review(conn, tid)
        he.emit_requirement_on_review(conn, tid, rid1)
        conn.commit()
        # Simulate cycle 2: reject → re-review
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, 'review_rejected', '{}', strftime('%s','now'))", (tid,))
        conn.commit()
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        conn.commit()
        kb.request_review(conn, tid, force=True)
        conn.commit()
        rid2 = int(conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
            "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()["id"])
        artifact = _report(home, tid, b"cycle2 report")
        db = home / "kanban.db"
        # Cycle 2 has NO requirement — evidence must be refused
        assert he.record_evidence(db, tid, review_event_id=rid2,
                                  artifact_dir=artifact,
                                  report_relative="hermaguard-report.md",
                                  status="pass", version="1.0.0") is None
        # Gate for cycle 2 must fail (no requirement → no valid evidence)
        ok, reason = he.gate_review_transition(db, conn, tid, review_event_id=rid2,
                                               artifact_dir=artifact)
        assert ok is False

    def test_nonexistent_review_id_rejected(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        he.emit_requirement_on_review(conn, tid, rid)
        conn.commit()
        artifact = _report(home, tid)
        assert he.record_evidence(home / "kanban.db", tid, review_event_id=999999,
                                  artifact_dir=artifact,
                                  report_relative="hermaguard-report.md",
                                  status="pass", version="1.0.0") is None

    def test_cross_task_review_id_rejected(self, env):
        conn, home = env
        tid_a = _task(conn)
        tid_b = _task(conn)
        rid_b = _review(conn, tid_b)
        he.emit_requirement_on_review(conn, tid_b, rid_b)
        conn.commit()
        artifact = _report(home, tid_a)
        # Requirement exists for task B, evidence attempted for task A
        assert he.record_evidence(home / "kanban.db", tid_a, review_event_id=rid_b,
                                  artifact_dir=artifact,
                                  report_relative="hermaguard-report.md",
                                  status="pass", version="1.0.0") is None

    def test_stale_review_id_rejected_after_new_cycle(self, env):
        conn, home = env
        tid = _task(conn)
        rid1 = _review(conn, tid)
        he.emit_requirement_on_review(conn, tid, rid1)
        conn.commit()
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, 'review_rejected', '{}', strftime('%s','now'))", (tid,))
        conn.commit()
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        conn.commit()
        kb.request_review(conn, tid, force=True)
        conn.commit()
        rid2 = int(conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
            "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()["id"])
        assert rid2 != rid1
        artifact = _report(home, tid)
        # Stale cycle-1 evidence attempt must fail
        assert he.record_evidence(home / "kanban.db", tid, review_event_id=rid1,
                                  artifact_dir=artifact,
                                  report_relative="hermaguard-report.md",
                                  status="pass", version="1.0.0") is None


class TestC2Tamper:
    def test_modified_report_fails_gate(self, env):
        """RED: post-record tampering must not pass the gate."""
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        he.emit_requirement_on_review(conn, tid, rid)
        conn.commit()
        artifact = _report(home, tid, b"original bytes")
        db = home / "kanban.db"
        eid = he.record_evidence(db, tid, review_event_id=rid,
                                 artifact_dir=artifact,
                                 report_relative="hermaguard-report.md",
                                 status="pass", version="1.0.0")
        assert eid is not None
        # Tamper AFTER recording
        _report(home, tid, b"tampered bytes")
        ok, reason = he.gate_review_transition(db, conn, tid, review_event_id=rid,
                                               artifact_dir=artifact)
        assert ok is False
        assert "digest" in reason.lower() or "mismatch" in reason.lower() or "tamper" in reason.lower()

    def test_deleted_report_fails_gate(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        he.emit_requirement_on_review(conn, tid, rid)
        conn.commit()
        artifact = _report(home, tid)
        db = home / "kanban.db"
        assert he.record_evidence(db, tid, review_event_id=rid,
                                  artifact_dir=artifact,
                                  report_relative="hermaguard-report.md",
                                  status="pass", version="1.0.0") is not None
        (artifact / "hermaguard-report.md").unlink()
        ok, reason = he.gate_review_transition(db, conn, tid, review_event_id=rid,
                                               artifact_dir=artifact)
        assert ok is False

    def test_moved_report_fails_gate(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        he.emit_requirement_on_review(conn, tid, rid)
        conn.commit()
        artifact = _report(home, tid)
        db = home / "kanban.db"
        assert he.record_evidence(db, tid, review_event_id=rid,
                                  artifact_dir=artifact,
                                  report_relative="hermaguard-report.md",
                                  status="pass", version="1.0.0") is not None
        (artifact / "hermaguard-report.md").rename(artifact / "moved.md")
        ok, _ = he.gate_review_transition(db, conn, tid, review_event_id=rid,
                                          artifact_dir=artifact)
        assert ok is False

    def test_symlink_escape_fails_gate(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        he.emit_requirement_on_review(conn, tid, rid)
        conn.commit()
        artifact = _report(home, tid, b"clean")
        db = home / "kanban.db"
        assert he.record_evidence(db, tid, review_event_id=rid,
                                  artifact_dir=artifact,
                                  report_relative="hermaguard-report.md",
                                  status="pass", version="1.0.0") is not None
        # Replace report with symlink pointing outside the artifact dir
        outside = home / "outside-secret.md"
        outside.write_bytes(b"attacker bytes")
        (artifact / "hermaguard-report.md").unlink()
        (artifact / "hermaguard-report.md").symlink_to(outside)
        ok, _ = he.gate_review_transition(db, conn, tid, review_event_id=rid,
                                          artifact_dir=artifact)
        assert ok is False


class TestC3Atomicity:
    def test_concurrent_requirement_writes_exactly_one(self, env):
        """RED: two connections racing emit_requirement produce duplicates."""
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        conn.commit()
        db_path = str(home / "kanban.db")

        results = []
        barrier = threading.Barrier(2)

        def worker():
            con2 = sqlite3.connect(db_path, timeout=10)
            try:
                barrier.wait()
                he.emit_requirement_on_review(con2, tid, rid)
                con2.commit()
            finally:
                con2.close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        events = [e for e in kb.list_events(conn, tid) if e.kind == he.KIND_REQUIRED]
        assert len(events) == 1

    def test_concurrent_evidence_exactly_one(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        he.emit_requirement_on_review(conn, tid, rid)
        conn.commit()
        artifact = _report(home, tid)
        db_path = str(home / "kanban.db")
        barrier = threading.Barrier(2)

        def worker():
            con2 = sqlite3.connect(db_path, timeout=10)
            try:
                barrier.wait()
                he.record_evidence(db_path, tid, review_event_id=rid,
                                   artifact_dir=artifact,
                                   report_relative="hermaguard-report.md",
                                   status="pass", version="1.0.0")
            finally:
                con2.close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        events = [e for e in kb.list_events(conn, tid) if e.kind == he.KIND_EVIDENCE]
        assert len(events) == 1


class TestC4DefaultOff:
    def test_emit_requirement_noop_when_mode_off(self, env, monkeypatch):
        conn, home = env
        monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: False)
        tid = _task(conn)
        rid = _review(conn, tid)
        assert he.emit_requirement_on_review(conn, tid, rid) is None
        events = [e for e in kb.list_events(conn, tid) if e.kind == he.KIND_REQUIRED]
        assert events == []

    def test_reconcile_noop_when_mode_off(self, env, monkeypatch):
        conn, home = env
        monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: False)
        tid = _task(conn)
        _review(conn, tid)  # review with no requirement
        conn.commit()
        result = he.reconcile_missed_requirements(home / "kanban.db")
        assert result["repaired"] == 0

    def test_reconcile_writes_when_mode_on(self, env, monkeypatch):
        conn, home = env
        monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: True)
        tid = _task(conn)
        _review(conn, tid)
        conn.commit()
        result = he.reconcile_missed_requirements(home / "kanban.db")
        assert result["repaired"] == 1

    def test_invalid_config_produces_zero_writes(self, env, monkeypatch):
        conn, home = env
        monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: False)
        tid = _task(conn)
        rid = _review(conn, tid)
        assert he.emit_requirement_on_review(conn, tid, rid) is None

    def test_hook_noop_when_mode_off(self, env, monkeypatch):
        conn, home = env
        monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: False)
        tid = _task(conn)
        rid = _review(conn, tid)
        he.hook_request_review(conn, tid, rid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == he.KIND_REQUIRED]
        assert events == []