"""Stuck-claim WARNING for #115692 — warn only, never reap live owners.

A claimed execution whose owner pid is still live but whose
heartbeat/staleness (claimed_at/started_at) is older than N x the cron
inactivity timeout must trigger a logger.warning. The claim itself
must be left untouched: live-owner reaping is owned by same-file PRs
#106627/#108484/#111454 and this path must not fight them.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import timedelta

from hermes_time import now as _hermes_now


def _point_ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def _backdate_claim(tmp_path, execution_id, age_seconds):
    old = (_hermes_now() - timedelta(seconds=age_seconds)).isoformat()
    with sqlite3.connect(tmp_path / "cron" / "executions.db") as conn:
        conn.execute(
            "UPDATE executions SET claimed_at=? WHERE id=?", (old, execution_id)
        )


def test_live_owner_stale_claim_warns_and_keeps_claim(monkeypatch, tmp_path, caplog):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("stuck-job", source="builtin")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: True)
    _backdate_claim(tmp_path, record["id"], age_seconds=3600)

    with caplog.at_level(logging.WARNING, logger="cron.executions"):
        warned = executions.warn_stuck_claims(inactivity_timeout=10.0)

    assert warned == 1
    assert any(
        "stuck" in r.message.lower() and record["id"] in r.message
        for r in caplog.records
    )
    assert executions.get_execution(record["id"])["status"] == "claimed"


def test_fresh_live_claim_stays_silent(monkeypatch, tmp_path, caplog):
    executions = _point_ledger(monkeypatch, tmp_path)
    executions.create_execution("fresh-job", source="builtin")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: True)

    with caplog.at_level(logging.WARNING, logger="cron.executions"):
        warned = executions.warn_stuck_claims(inactivity_timeout=600.0)

    assert warned == 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_recovery_path_warns_without_reaping_live_owner(monkeypatch, tmp_path, caplog):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("recovery-warn-job", source="builtin")
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: True)
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "10")
    _backdate_claim(tmp_path, record["id"], age_seconds=3600)

    with caplog.at_level(logging.WARNING, logger="cron.executions"):
        changed = executions.recover_interrupted_executions()

    assert changed == 0
    assert any("stuck" in r.message.lower() for r in caplog.records)
    assert executions.get_execution(record["id"])["status"] == "claimed"
