"""Desktop run-history must surface no_agent (script-only) cron runs.

``no_agent`` jobs short-circuit in ``run_job`` before any ``sessions`` row is
created (by design — they must not pay for SessionDB/AIAgent construction), so
``SessionDB.list_cron_job_runs`` never sees them. But ``run_one_job`` still
records every attempt in the durable execution ledger (cron/executions.db).

``hermes_cli.web_server._list_cron_job_runs_sync`` merges that ledger into the
session-backed run-history so no_agent jobs are visible in the Desktop UI
without changing the no_agent cost model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def _seed_execution(executions, job_id: str, *, success: bool, error=None):
    rec = executions.create_execution(job_id, source="builtin")
    executions.mark_execution_running(rec["id"])
    return executions.finish_execution(rec["id"], success=success, error=error)


def _make_session_run(db, job_id: str, started_at: float):
    """Mimic an LLM-path cron run row so the sessions source is also exercised."""
    sid = f"cron_{job_id}_{int(started_at)}"
    db.create_session(session_id=sid, source="cron")
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?", (started_at, sid)
    )
    db._conn.commit()
    # Checkpoint WAL before end_session() so a later read-only reopen of this
    # file (what the endpoint does) sees the row. list_cron_job_runs filters on
    # id prefix + source only, so ended_at visibility is irrelevant here.
    db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.end_session(sid, "completed")
    return sid


def test_execution_to_session_info_shapes_no_agent_run():
    import hermes_cli.web_server as ws

    row = {
        "id": "exec-1",
        "job_id": "watchdog",
        "source": "builtin",
        "status": "completed",
        "claimed_at": "2026-08-30T10:00:00",
        "started_at": "2026-08-30T10:00:01",
        "finished_at": "2026-08-30T10:00:05",
        "error": None,
    }
    info = ws._execution_to_session_info(row, "watchdog", "default")

    assert info["id"].startswith("exec_watchdog_")
    assert info["source"] == "cron"
    assert info["title"] == "cron watchdog"
    assert info["started_at"] > 0
    assert info["ended_at"] is not None
    assert info["is_active"] is False
    assert info["message_count"] == 0
    assert info["model"] is None
    assert info["profile"] == "default"


def test_execution_to_session_info_marks_active_and_error_preview():
    import hermes_cli.web_server as ws

    running = {
        "id": "exec-2", "job_id": "watchdog", "source": "builtin",
        "status": "running", "claimed_at": "2026-08-30T10:00:00",
        "started_at": "2026-08-30T10:00:01", "finished_at": None, "error": None,
    }
    info = ws._execution_to_session_info(running, "watchdog", None)
    assert info["is_active"] is True
    assert info["ended_at"] is None

    failed = {
        "id": "exec-3", "job_id": "watchdog", "source": "builtin",
        "status": "failed", "claimed_at": "2026-08-30T10:00:00",
        "started_at": "2026-08-30T10:00:01", "finished_at": "2026-08-30T10:00:09",
        "error": "script exited 1",
    }
    info_f = ws._execution_to_session_info(failed, "watchdog", None)
    assert info_f["is_active"] is False
    assert "script exited 1" in info_f["preview"]


def _patch_endpoint(monkeypatch, ws, home, job_profile=None):
    """Route the endpoint at an isolated temp store without the real profile layer.

    ``_open_session_db_for_profile`` is redirected to the REAL opener pointed at
    the seeded temp state.db (so ``_conn`` is initialized correctly); the
    execution ledger reads from the isolated file via EXECUTIONS_FILE.
    """
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(
        ws,
        "_open_session_db_for_profile",
        lambda profile, *, read_only=True: ws._open_session_db_at_path(
            home / "state.db", read_only=True
        ),
    )
    if job_profile is not None:
        monkeypatch.setattr(ws, "_find_cron_job_profile", lambda job_id: job_profile)


def test_run_history_merges_execution_ledger_for_no_agent(tmp_path, monkeypatch):
    """No session row exists (no_agent path) but the ledger does → surfaced."""
    import cron.executions as executions
    import hermes_cli.web_server as ws

    home = tmp_path / "default"
    (home / "cron").mkdir(parents=True)
    # Point the ledger at the temp store BEFORE seeding so seed + read agree.
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")

    # A no_agent job's attempts live ONLY in the ledger (no sessions row).
    _seed_execution(executions, "noagent-job", success=True)
    _seed_execution(executions, "noagent-job", success=False, error="boom")

    _patch_endpoint(monkeypatch, ws, home, job_profile=None)
    result = ws._list_cron_job_runs_sync("noagent-job", profile=None, limit=20)

    runs = result["runs"]
    assert len(runs) == 2, f"expected 2 ledger rows, got {len(runs)}"
    assert all(r["id"].startswith("exec_noagent-job_") for r in runs)
    # Newest-first ordering preserved.
    sts = [r["started_at"] for r in runs]
    assert sts == sorted(sts, reverse=True)
    # Failed run carries its error in the preview.
    assert any("boom" in r["preview"] for r in runs)


def test_run_history_keeps_session_runs_and_merges_ledger(tmp_path, monkeypatch):
    """LLM job (sessions) + no_agent sibling (ledger) both appear, newest-first."""
    import cron.executions as executions
    import hermes_cli.web_server as ws
    from hermes_state import SessionDB

    home = tmp_path / "default"
    (home / "cron").mkdir(parents=True)

    # Point the ledger at the temp store BEFORE seeding so seed + read agree.
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")

    # An LLM job run (sessions source) — seed the real temp state.db.
    seed_db = SessionDB(home / "state.db")
    _make_session_run(seed_db, "llm-job", 1_700_000_000.0)
    seed_db.close()
    # A no_agent job run (ledger only).
    _seed_execution(executions, "noagent-job", success=True)

    _patch_endpoint(monkeypatch, ws, home, job_profile=None)
    llm = ws._list_cron_job_runs_sync("llm-job", profile=None, limit=20)
    noagent = ws._list_cron_job_runs_sync("noagent-job", profile=None, limit=20)

    assert len(llm["runs"]) == 1
    assert llm["runs"][0]["id"].startswith("cron_llm-job_")
    assert len(noagent["runs"]) == 1
    assert noagent["runs"][0]["id"].startswith("exec_noagent-job_")


def test_run_history_ledger_merge_is_best_effort_on_failure(tmp_path, monkeypatch):
    """A broken ledger must not break the session-backed list."""
    import cron.executions as executions
    import hermes_cli.web_server as ws
    from hermes_state import SessionDB

    home = tmp_path / "default"
    (home / "cron").mkdir(parents=True)

    seed_db = SessionDB(home / "state.db")
    _make_session_run(seed_db, "llm-job", 1_700_000_000.0)
    seed_db.close()

    _patch_endpoint(monkeypatch, ws, home, job_profile=None)
    with patch.object(executions, "list_executions", side_effect=RuntimeError("ledger gone")):
        result = ws._list_cron_job_runs_sync("llm-job", profile=None, limit=20)

    assert len(result["runs"]) == 1
    assert result["runs"][0]["id"].startswith("cron_llm-job_")
