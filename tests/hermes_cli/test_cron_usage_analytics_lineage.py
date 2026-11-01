"""Run identity, continuation and auxiliary-accounting regressions for cron analytics."""
from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest

from cron import executions, jobs
from hermes_state import SessionDB
from tests.fakes.cron_owner import admit_fire, cron_owner


def _usage(db, sid, tokens, cost, *, actual=None):
    db.update_token_counts(sid, input_tokens=tokens, estimated_cost_usd=cost, actual_cost_usd=actual)


def _compress(db, parent, child):
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source="cron",
        messages=[{"role": "user", "content": "continue the scheduled run"}],
        require_compression_lease=False,
    )


def test_usage_includes_continuations_and_delegates_once_and_uses_completion_tip(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        root = "cron_daily_20261026_100000"
        db.create_session(root, source="cron", model="main")
        _usage(db, root, 120, 0.3, actual=0.25)
        _compress(db, root, "continuation-one")
        _usage(db, "continuation-one", 240, 0.5)
        _compress(db, "continuation-one", "continuation-two")
        _usage(db, "continuation-two", 60, 0.1)
        db.end_session("continuation-two", "cron_complete")
        db.create_session("delegate", source="subagent", parent_session_id="continuation-one",
                          model_config={"_delegate_from": "continuation-one"})
        _usage(db, "delegate", 35, 0.2)
        db.end_session("delegate", "completed")
        db.record_auxiliary_usage("continuation-one", task="compression", model="aux",
                                  input_tokens=7, output_tokens=3, estimated_cost_usd=0.05)
        # A user branch is not work performed by this cron execution.
        db.create_session("branch", source="cron", parent_session_id=root,
                          model_config={"_branched_from": root})
        _usage(db, "branch", 9999, 99)
        # A longer job id is not part of "daily", despite sharing the legacy prefix.
        db.create_session("cron_daily_digest_20261026_100000", source="cron")
        _usage(db, "cron_daily_digest_20261026_100000", 9999, 99)

        result = db.cron_job_usage_analytics("daily", since=time.time() - 86400)

    assert result["usage_runs"] == 1
    assert result["total_tokens"] == 120 + 240 + 60 + 35 + 7 + 3
    assert result["avg_tokens_per_run"] == result["total_tokens"]
    assert result["total_cost_usd"] == pytest.approx(0.25 + 0.5 + 0.1 + 0.2 + 0.05)
    assert result["avg_cost_usd_per_run"] == result["total_cost_usd"]
    assert result["actual_cost_runs"] == 0
    assert result["estimated_cost_runs"] == 1
    assert result["incomplete_runs"] == 0


def test_in_place_auxiliary_usage_is_added_without_counting_main_model_rows_twice(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        sid = "cron_daily_20261026_100000"
        db.create_session(sid, source="cron", model="main")
        _usage(db, sid, 120, 0.3)
        db.record_auxiliary_usage(sid, task="compression", model="aux",
                                  input_tokens=200, output_tokens=40, estimated_cost_usd=0.7)
        db.end_session(sid, "cron_complete")

        result = db.cron_job_usage_analytics("daily", since=time.time() - 86400)

    assert result["usage_runs"] == 1
    assert result["total_tokens"] == 360
    assert result["total_cost_usd"] == pytest.approx(1.0)
    assert result["incomplete_runs"] == 0


def test_cached_token_buckets_are_counted_once_for_main_and_auxiliary_usage(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        sid = "cron_daily_20261026_100000"
        db.create_session(sid, source="cron", model="main")
        db.update_token_counts(sid, input_tokens=100, output_tokens=20,
                               cache_read_tokens=900, cache_write_tokens=100, reasoning_tokens=10,
                               estimated_cost_usd=0.3)
        db.record_auxiliary_usage(sid, task="compression", model="aux",
                                  input_tokens=200, output_tokens=40,
                                  cache_read_tokens=600, cache_write_tokens=50, reasoning_tokens=30,
                                  estimated_cost_usd=0.7)

        result = db.cron_job_usage_analytics("daily", since=time.time() - 86400)

    # Reasoning is already part of output_tokens; canonical totals add only
    # input, output and the two disjoint cache buckets.
    assert result["total_tokens"] == 100 + 20 + 900 + 100 + 200 + 40 + 600 + 50
    assert result["avg_tokens_per_run"] == result["total_tokens"]
    assert result["total_cost_usd"] == pytest.approx(1.0)


def test_period_filters_root_executions_not_their_continuation_segments(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        root = "cron_daily_20261026_100000"
        db.create_session(root, source="cron")
        _usage(db, root, 120, 0.3)
        _compress(db, root, "recent-child")
        _usage(db, "recent-child", 240, 0.7)
        db.end_session("recent-child", "cron_complete")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (time.time() - 100 * 86400, root))
        db._conn.commit()

        result = db.cron_job_usage_analytics("daily", since=time.time() - 30 * 86400)

    assert result["usage_runs"] == result["total_tokens"] == 0
    assert result["avg_tokens_per_run"] is None
    assert result["avg_cost_usd_per_run"] is None


def test_usage_includes_unified_runtime_admissions_without_cross_job_leakage(tmp_path, monkeypatch):
    owner = cron_owner(tmp_path, monkeypatch)
    try:
        alpha = jobs.create_job(prompt="alpha", schedule="every 1h", deliver="local")["id"]
        beta = jobs.create_job(prompt="beta", schedule="every 1h", deliver="local")["id"]
        alpha_sid, _ = asyncio.run(admit_fire(owner, alpha))
        beta_sid, _ = asyncio.run(admit_fire(owner, beta))
        _usage(owner.db, alpha_sid, 120, 0.3)
        _usage(owner.db, beta_sid, 9999, 99)

        result = owner.db.cron_job_usage_analytics(alpha, since=time.time() - 86400)

        assert result["usage_runs"] == 1
        assert result["total_tokens"] == 120
        assert result["total_cost_usd"] == pytest.approx(0.3)
    finally:
        owner.db.close()


def test_execution_analytics_warns_when_another_jobs_runs_fill_retention(tmp_path, monkeypatch):
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 2)
    cutoff = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    first = executions.create_execution("quiet", source="direct")
    executions.mark_execution_running(first["id"])
    executions.finish_execution(first["id"], success=False)
    before = executions.execution_analytics("quiet", since=cutoff)
    assert before["failed"] == 1
    assert before["history_limited"] is False
    for _ in range(2):
        row = executions.create_execution("noisy", source="direct")
        executions.mark_execution_running(row["id"])
        executions.finish_execution(row["id"], success=True)

    result = executions.execution_analytics("quiet", since=cutoff)

    assert result["attempts"] == 0
    assert result["history_limited"] is True


def test_retry_at_period_boundary_is_not_relabelled_as_first_scheduled_run(tmp_path, monkeypatch):
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    old = (datetime.now(UTC) - timedelta(days=8)).isoformat()
    row = executions.create_execution("daily", source="builtin", scheduled_instant=old)
    executions.mark_execution_running(row["id"])
    executions.finish_execution(row["id"], success=False)
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET claimed_at=?, started_at=?, finished_at=? WHERE id=?",
                     (old, old, old, row["id"]))
    retry = executions.create_execution("daily", source="builtin", scheduled_instant=old)
    executions.mark_execution_running(retry["id"])
    executions.finish_execution(retry["id"], success=True)

    result = executions.execution_analytics("daily", since=(datetime.now(UTC) - timedelta(days=7)).isoformat())

    assert result["attempts"] == result["retry_attempts"] == 1
    assert result["scheduled_runs"] == 0
