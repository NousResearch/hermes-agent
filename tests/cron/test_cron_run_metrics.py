"""hermes.cron.run: one row per terminal execution and per dropped occurrence."""

from datetime import timedelta

import pytest

from cron import executions, jobs
from hermes_cli.observability import relay_shared_metrics as rsm
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_gateway as smg


@pytest.fixture
def cron_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    got = []
    monkeypatch.setattr(rsm, "enabled", lambda: True)
    monkeypatch.setattr(rsm, "record_process_mark", lambda mark, data: got.append((mark, dict(data))))

    def read():
        smg.drain()
        found = [data for mark, data in got if mark == contract.CRON_RUN_MARK]
        assert all(contract.counter_dimensions_are_valid(contract.CRON_RUN_METRIC, d) for d in found)
        return found

    return read


def _execution(job_id, deliver, *, start=True):
    row = executions.create_execution(job_id, source="builtin")
    smg.note_cron_execution({"execution_id": row["id"], "deliver": deliver})
    if start:
        executions.mark_execution_running(row["id"])
    return row["id"]


def test_each_terminal_execution_is_one_row_classified_without_job_details(cron_rows):
    ok = _execution("job1", "telegram:-100555")
    executions.finish_execution(ok, success=True, delivery_outcome="delivered")
    executions.finish_execution(ok, success=False, error="late duplicate")  # write-once: no second row
    executions.finish_execution(_execution("job2", "local"), success=False, error="boom")
    refused = _execution("job3", "webhook", start=False)  # claim lost before any side effect
    executions.finish_execution(refused, success=False, error="Fire claim lost; execution was not started.")
    gated = _execution("job4", "origin")
    smg.note_cron_skipped({"execution_id": gated})  # pre-run script said wakeAgent=false
    executions.finish_execution(gated, success=True, delivery_outcome="suppressed")
    assert [(r["outcome"], r["delivery_kind"]) for r in cron_rows()] == [
        ("success", "platform"), ("failed", "local"), ("skipped", "webhook"), ("skipped", "platform"),
    ]


def test_an_occurrence_dropped_after_downtime_is_one_missed_row(tmp_path, cron_rows):
    (tmp_path / "config.yaml").write_text("cron:\n  catch_up_missed: false\n", encoding="utf-8")
    with jobs.use_cron_store(tmp_path / "cron"):
        jobs.create_job(prompt="secret prompt", schedule="every 1h", model="fixture", deliver="local")
        stored = jobs.load_jobs()
        stored[0]["next_run_at"] = (jobs._hermes_now() - timedelta(hours=4)).isoformat()
        jobs.save_jobs(stored)
        assert jobs.get_due_jobs() == []
        assert jobs.get_due_jobs() == []
    assert cron_rows() == [{"delivery_kind": "local", "duration_bucket": "lt_1s", "outcome": "missed"}]
