"""Selected PostgreSQL cron must refuse before every runnable side effect.

This is intentionally a safety boundary, not a partial PostgreSQL cron port:
cron still requires a durable SessionDB transcript lifecycle for session title,
lineage, finalization, retry, and lease semantics.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cron.scheduler import run_job, run_one_job
from state_store_runtime_readiness import trap_state_db_opens


_PG_CONFIG = (
    "state_store:\n"
    "  backend: postgresql\n"
    "  postgresql:\n"
    "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n"
)


def _selected_pg_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / ".hermes" / "profiles" / "selected-pg-cron"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    return home


def _assert_no_cron_or_transcript_artifacts(home: Path) -> None:
    assert not (home / "state.db").exists()
    assert not (home / "cron").exists()
    assert not (home / "sessions" / "sessions.json").exists()
    assert list((home / "sessions").glob("*.jsonl")) == [] if (home / "sessions").exists() else True


@pytest.mark.parametrize(
    "job",
    [
        {"id": "pg-agent", "name": "agent", "prompt": "must not run", "script": "pre.py"},
        {"id": "pg-script", "name": "script", "script": "only.py", "no_agent": True},
    ],
)
def test_selected_postgresql_run_job_refuses_before_agent_or_script(
    tmp_path, monkeypatch, caplog, job
):
    home = _selected_pg_home(tmp_path, monkeypatch)
    script_calls = []
    agent_calls = []

    with patch("cron.scheduler._hermes_home", home), \
         patch("cron.scheduler._run_job_script_with_claim_heartbeat", side_effect=lambda *a, **k: script_calls.append((a, k))), \
         patch("cron.scheduler._construct_cron_agent", side_effect=lambda *a, **k: agent_calls.append((a, k))):
        with trap_state_db_opens(home) as opens:
            success, output, final_response, error = run_job(job)

    assert success is False
    assert final_response == ""
    assert error and error.startswith("CRON_TRANSCRIPT_UNAVAILABLE:")
    assert "cron-session-transcript-lifecycle" in error
    assert "**Status:** CRON_TRANSCRIPT_UNAVAILABLE" in output
    assert script_calls == []
    assert agent_calls == []
    assert opens == []
    assert "CRON_TRANSCRIPT_UNAVAILABLE" in caplog.text
    _assert_no_cron_or_transcript_artifacts(home)


def test_selected_postgresql_run_one_job_refuses_before_handoff_delivery_or_finalization(
    tmp_path, monkeypatch, caplog
):
    home = _selected_pg_home(tmp_path, monkeypatch)
    calls = []
    job = {"id": "pg-control-plane", "name": "control", "script": "only.py", "no_agent": True}

    def forbidden(name):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            raise AssertionError(f"{name} must not run after selected-PG refusal")
        return call

    with patch("cron.scheduler._hermes_home", home), \
         patch("cron.scheduler.create_execution", side_effect=forbidden("create_execution")), \
         patch("cron.scheduler._launch_external_cron_worker", side_effect=forbidden("worker_handoff")), \
         patch("cron.scheduler._run_job_script_with_claim_heartbeat", side_effect=forbidden("script")), \
         patch("cron.scheduler.save_job_output", side_effect=forbidden("save_output")), \
         patch("cron.scheduler._deliver_result", side_effect=forbidden("delivery")), \
         patch("cron.scheduler.mark_job_run", side_effect=forbidden("mark_job_run")), \
         patch("cron.scheduler.finish_execution", side_effect=forbidden("finish_execution")):
        with trap_state_db_opens(home) as opens:
            assert run_one_job(job) is False

    assert calls == []
    assert opens == []
    assert "CRON_TRANSCRIPT_UNAVAILABLE" in caplog.text
    _assert_no_cron_or_transcript_artifacts(home)
