"""Retention of the cron execution ledger: which rows survive a busy fleet.

The ledger is the only durable record of *what fired*; `hermes cron runs` and missed-occurrence
audits read it per job. A single global newest-N quota prunes by recency, so the rows it drops
first are always those of the LOWEST-frequency jobs — the weekly/monthly schedules whose history is
the only evidence their slot was accounted for — while a minute-level sibling keeps thousands of
rows it does not need.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta

from hermes_time import now as _hermes_now


def _ledger(monkeypatch, tmp_path):
    """Point the ledger at a temp store and return the module."""
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def _finish(executions, job_id: str, *, success: bool = True):
    row = executions.create_execution(job_id, source="builtin")
    assert executions.mark_execution_running(row["id"]) is not None
    finished = executions.finish_execution(row["id"], success=success, error=None if success else "boom")
    assert finished is not None
    return finished


def _age(executions, execution_id: str, *, hours: float) -> None:
    """Re-stamp one row into the past; every public writer stamps the current clock."""
    ts = (_hermes_now() - timedelta(hours=hours)).isoformat()
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        conn.execute(
            "UPDATE executions SET claimed_at=?, started_at=?, finished_at=? WHERE id=?",
            (ts, ts, ts, execution_id),
        )


def _surviving_ids(executions) -> set:
    return {row["id"] for row in executions.list_executions(limit=500)}


def test_a_quiet_jobs_history_outlives_its_chattiest_sibling(monkeypatch, tmp_path):
    """A chatty job's churn must not evict the only rows a low-frequency job has."""
    executions = _ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 500)
    quiet = _finish(executions, "weekly-report")["id"]
    _age(executions, quiet, hours=48)
    for index in range(6):
        _age(executions, _finish(executions, "minute-poller")["id"], hours=36 - index * 0.2)

    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 4)
    trigger = _finish(executions, "minute-poller")["id"]  # a terminal write applies retention

    surviving = _surviving_ids(executions)
    assert quiet in surviving, "the quiet job lost its history to the chatty job's churn"
    assert trigger in surviving
    assert len(surviving) == 4, "the cap must still bound the ledger"


def test_the_cap_evicts_completed_rows_before_failure_evidence(monkeypatch, tmp_path):
    """Failed rows are the highest-value audit rows: under cap pressure they go last."""
    executions = _ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 500)
    failed = _finish(executions, "nightly-audit", success=False)["id"]
    _age(executions, failed, hours=72)
    for index in range(3):
        _age(executions, _finish(executions, f"poller-{index}")["id"], hours=24 - index * 2)

    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 3)
    trigger = _finish(executions, "minute-poller")["id"]

    surviving = _surviving_ids(executions)
    assert failed in surviving, "cap pressure discarded failure evidence"
    assert trigger in surviving
    assert len(surviving) == 3


def test_the_cap_spends_each_jobs_excess_before_any_jobs_floor(monkeypatch, tmp_path):
    """Over the cap the per-job floor still binds: each job pays from its own excess first.

    The chatty job holds twice the floor, the quieter one barely above it, and every row is inside
    the success window — so a cap that simply trims the row count of the biggest job takes the
    chatty job below the floor it is supposed to keep.
    """
    executions = _ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "PRUNE_EVERY_N_FINISHES", 10**9)
    monkeypatch.setattr(executions, "PRUNE_MIN_INTERVAL_SECONDS", 10**9)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 10**9)
    monkeypatch.setattr(executions, "PER_JOB_TERMINAL_KEEP", 50)
    monkeypatch.setattr(executions, "SUCCESS_FLOOR_DAYS", 1.0 / 24)

    for _ in range(100):
        _finish(executions, "minute-poller")
    for _ in range(60):
        _finish(executions, "hourly-sync")

    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 100)  # 161 rows → 61 over the cap
    _finish(executions, "minute-poller")

    per_job = {}
    with sqlite3.connect(str(executions.EXECUTIONS_FILE)) as conn:
        for job_id, count in conn.execute("SELECT job_id, COUNT(*) FROM executions GROUP BY job_id"):
            per_job[job_id] = count
    assert len(_surviving_ids(executions)) == 100, "the cap must still bound the ledger"
    assert per_job.get("minute-poller", 0) >= 50, "the cap ate a job's floor while its sibling had excess"
    assert per_job.get("hourly-sync", 0) >= 50


def test_the_prune_budget_is_per_ledger_not_per_process(monkeypatch, tmp_path):
    """One process ticks every served profile: a shared budget defers one home's retention."""
    executions = _ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 10_000)
    monkeypatch.setattr(executions, "PRUNE_EVERY_N_FINISHES", 10**9)
    monkeypatch.setattr(executions, "PRUNE_MIN_INTERVAL_SECONDS", 3600.0)

    first = str(tmp_path / "cron" / "executions.db")
    second_path = tmp_path / "second" / "cron" / "executions.db"

    for _ in range(3):
        _finish(executions, "a-poller")  # this ledger has just pruned: its window is spent
    spent = dict(executions._prune_state[first])
    assert spent["finishes"] >= 1

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", second_path)
    _finish(executions, "b-poller")

    assert str(second_path) in executions._prune_state, "each ledger needs its own budget"
    assert executions._prune_state[str(second_path)]["last"] > 0.0, (
        "the second ledger's first terminal write was suppressed by the first ledger's budget"
    )
    assert executions._prune_state[first] == spent, "one ledger's churn spent another's budget"


def test_retention_is_amortized_under_the_cap_and_immediate_at_it(monkeypatch, tmp_path):
    """Aged rows are collected on a schedule, but the hard cap is never deferred."""
    executions = _ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 10_000)
    monkeypatch.setattr(executions, "SUCCESS_FLOOR_DAYS", 1.0 / 24)
    monkeypatch.setattr(executions, "PER_JOB_TERMINAL_KEEP", 0)

    _finish(executions, "poller")  # cold start: the first terminal write collects
    stale = _finish(executions, "poller")["id"]
    _age(executions, stale, hours=24 * 30)
    _finish(executions, "poller")  # under the cap, inside the interval: deferred
    assert stale in _surviving_ids(executions)

    monkeypatch.setattr(executions, "PRUNE_EVERY_N_FINISHES", 1)
    _finish(executions, "poller")  # the finish count is reached: collected
    assert stale not in _surviving_ids(executions)

    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 1)
    _finish(executions, "over-cap-1")
    over = _surviving_ids(executions)
    _finish(executions, "over-cap-2")
    assert len(_surviving_ids(executions)) <= len(over), "an over-cap table must be trimmed at once"
