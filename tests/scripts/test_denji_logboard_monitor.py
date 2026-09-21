"""Behavioural tests for Denji logboard monitor freshness and live cron incidents.

All tests use a disposable HERMES_HOME. They never read or mutate live state.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "denji-logboard-monitor.py"


def _load_monitor(monkeypatch: pytest.MonkeyPatch, home: Path):
    monkeypatch.setenv("HERMES_HOME", str(home))
    name = f"denji_logboard_monitor_{id(home)}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_system_health(home: Path, *, timestamp: str) -> Path:
    path = home / "governance" / "logboard" / "system-health-test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "findings": [
                    {
                        "priority": "P1",
                        "slug": "disk-usage-high",
                        "title": "Disk usage high",
                        "body": "root disk exceeded threshold",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _create_executions_db(home: Path) -> sqlite3.Connection:
    path = home / "cron" / "executions.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE executions (
            id INTEGER PRIMARY KEY,
            job_id TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            error TEXT
        );
        CREATE TABLE cron_incidents (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            error_sig TEXT NOT NULL,
            state TEXT NOT NULL,
            failure_type TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            acked_at TEXT,
            closed_at TEXT,
            error TEXT NOT NULL,
            output_file TEXT
        );
        """
    )
    return conn


def _write_jobs(home: Path) -> None:
    path = home / "cron" / "jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"jobs": [{"id": "job-1", "name": "example-job"}]}),
        encoding="utf-8",
    )


def test_stale_system_health_is_not_reported(monkeypatch, tmp_path):
    old = datetime.now(timezone.utc) - timedelta(days=9)
    _write_system_health(tmp_path, timestamp=old.isoformat())
    monitor = _load_monitor(monkeypatch, tmp_path)

    assert monitor.parse_system_health() == []


def test_fresh_system_health_remains_supported(monkeypatch, tmp_path):
    fresh = datetime.now(timezone.utc) - timedelta(hours=1)
    source = _write_system_health(tmp_path, timestamp=fresh.isoformat())
    monitor = _load_monitor(monkeypatch, tmp_path)

    findings = monitor.parse_system_health()

    assert len(findings) == 1
    assert findings[0]["rule"] == "system_health_disk-usage-high"
    assert findings[0]["source_file"] == str(source)


def test_open_cron_incident_is_reported_when_latest_run_failed(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc).isoformat()
    conn = _create_executions_db(tmp_path)
    conn.execute(
        "INSERT INTO cron_incidents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("inc-1", "job-1", "sig", "detected", "script", now, now, None, None, "boom", None),
    )
    conn.execute(
        "INSERT INTO executions(job_id,status,started_at,finished_at,error) VALUES (?,?,?,?,?)",
        ("job-1", "failed", now, now, "boom"),
    )
    conn.commit()
    conn.close()
    _write_jobs(tmp_path)
    monitor = _load_monitor(monkeypatch, tmp_path)

    findings = monitor.parse_cron_incidents()

    assert len(findings) == 1
    assert findings[0]["rule"] == "cron_incident_inc-1"
    assert "example-job" in findings[0]["detail"]
    assert findings[0]["incident_id"] == "inc-1"


def test_recovered_cron_does_not_emit_stale_open_incident(monkeypatch, tmp_path):
    then = datetime.now(timezone.utc) - timedelta(hours=2)
    now = datetime.now(timezone.utc)
    conn = _create_executions_db(tmp_path)
    conn.execute(
        "INSERT INTO cron_incidents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "inc-1",
            "job-1",
            "sig",
            "alerted",
            "script",
            then.isoformat(),
            then.isoformat(),
            None,
            None,
            "old failure",
            None,
        ),
    )
    conn.execute(
        "INSERT INTO executions(job_id,status,started_at,finished_at,error) VALUES (?,?,?,?,?)",
        ("job-1", "failed", then.isoformat(), then.isoformat(), "old failure"),
    )
    conn.execute(
        "INSERT INTO executions(job_id,status,started_at,finished_at,error) VALUES (?,?,?,?,?)",
        ("job-1", "completed", now.isoformat(), now.isoformat(), None),
    )
    conn.commit()
    conn.close()
    _write_jobs(tmp_path)
    monitor = _load_monitor(monkeypatch, tmp_path)

    assert monitor.parse_cron_incidents() == []


def test_missing_cron_incident_store_is_silent(monkeypatch, tmp_path):
    monitor = _load_monitor(monkeypatch, tmp_path)

    assert monitor.parse_cron_incidents() == []
