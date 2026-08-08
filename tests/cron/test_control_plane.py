from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_jobs(home: Path) -> Path:
    jobs_path = home / "cron" / "jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "job-1",
                        "name": "Job One",
                        "enabled": True,
                        "deliver": "telegram:123",
                        "state": "scheduled",
                        "schedule": {"kind": "interval", "minutes": 30, "display": "every 30m"},
                        "created_at": "2026-07-29T12:00:00+00:00",
                    }
                ],
                "updated_at": "2026-07-29T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return jobs_path


def _write_executions(home: Path) -> Path:
    db_path = home / "cron" / "executions.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE executions (
                   id TEXT PRIMARY KEY,
                   job_id TEXT NOT NULL,
                   source TEXT NOT NULL,
                   process_id TEXT NOT NULL,
                   pid INTEGER NOT NULL,
                   process_started_at INTEGER,
                   status TEXT NOT NULL,
                   claimed_at TEXT NOT NULL,
                   started_at TEXT,
                   finished_at TEXT,
                   error TEXT,
                   delivery_state TEXT,
                   receipt_path TEXT
               )"""
        )
        conn.execute(
            """INSERT INTO executions VALUES (
                   'exec-1', 'job-1', 'shadow-test', 'proc-1', 1000, 1,
                   'completed', '2026-07-29T12:30:00+00:00', '2026-07-29T12:30:05+00:00',
                   '2026-07-29T12:30:06+00:00', NULL, NULL, NULL
               )"""
        )
        conn.commit()
    return db_path


def _write_gateway_heartbeat(home: Path) -> Path:
    heartbeat_path = home / "state" / "gateway.heartbeat"
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path.write_text(
        json.dumps(
            {
                "pid": 4321,
                "start_time": 100.5,
                "updated_at": "2026-07-29T13:15:00+00:00",
                "monotonic": 1.25,
            }
        ),
        encoding="utf-8",
    )
    return heartbeat_path


def test_shadow_scan_collects_read_only_snapshot(hermes_home):
    from cron_control.shadow import collect_shadow_snapshot

    jobs_path = _write_jobs(hermes_home)
    executions_path = _write_executions(hermes_home)
    _write_gateway_heartbeat(hermes_home)

    snapshot = collect_shadow_snapshot(jobs_path=jobs_path, executions_path=executions_path)

    assert snapshot["jobs"][0]["id"] == "job-1"
    assert snapshot["executions"][0]["id"] == "exec-1"
    kinds = {item["kind"] for item in snapshot["evidence"]}
    assert {"job_metadata", "execution_state", "delivery_receipt", "scheduler_heartbeat", "dead_man_switch", "state_store"} <= kinds


def test_control_plane_db_is_append_only(hermes_home):
    from cron_control.store import append_audit_event, open_control_plane_db

    conn = open_control_plane_db()
    try:
        append_audit_event(
            conn,
            {
                "audit_id": "au_1",
                "timestamp": "2026-07-29T13:00:00+00:00",
                "incident_id": "inc_1",
                "job_id": "job-1",
                "event_type": "incident_opened",
                "actor": {"type": "system", "id": "shadow-scanner"},
                "evidence_refs": ["ev_1"],
                "result": "planned",
                "before_state": {},
                "after_state": {},
            },
        )
        assert conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE audit_events SET result='verified' WHERE audit_id='au_1'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM audit_events WHERE audit_id='au_1'")
    finally:
        conn.close()


def test_normalizer_builds_stable_evidence_id():
    from cron_control.normalizer import build_evidence

    base = {
        "incident_id": "inc_1",
        "job_id": "job-1",
        "execution_id": "exec-1",
        "kind": "execution_state",
        "source": "executions.db",
        "value": {"status": "running", "claimed_at": "2026-07-29T12:00:00+00:00"},
        "source_ref": "executions.db:id=exec-1",
    }
    first = build_evidence(**base)
    second = build_evidence(**base)
    assert first["evidence_id"] == second["evidence_id"]
    assert first["content_hash"] == second["content_hash"]


def test_shadow_main_emits_json(monkeypatch, capsys):
    from cron_control import shadow

    monkeypatch.setattr(
        shadow,
        "collect_shadow_snapshot",
        lambda **_kwargs: {"collected_at": "2026-07-29T13:30:00+00:00", "jobs": [], "executions": [], "evidence": []},
    )
    assert shadow.main([]) == 0
    out = capsys.readouterr().out
    assert '"collected_at": "2026-07-29T13:30:00+00:00"' in out


def test_shadow_snapshot_persists_control_plane_records(hermes_home):
    from cron_control.shadow import collect_shadow_snapshot, persist_shadow_snapshot

    jobs_path = _write_jobs(hermes_home)
    executions_path = _write_executions(hermes_home)
    _write_gateway_heartbeat(hermes_home)
    snapshot = collect_shadow_snapshot(jobs_path=jobs_path, executions_path=executions_path)
    control_plane_path = hermes_home / "cron" / "control-plane.db"

    persist_shadow_snapshot(snapshot, control_plane_path)

    with sqlite3.connect(control_plane_path) as conn:
        evidence_count = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
        incident_count = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        audit_count = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    assert evidence_count == len(snapshot["evidence"])
    assert incident_count >= len(snapshot["evidence"]) + 1
    assert audit_count == 1
