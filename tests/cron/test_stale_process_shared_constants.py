"""A due scan survives a long-lived process whose cached cron.jobs predates a constant.

The stale process loads ``cron/occurrences.py`` fresh from disk while ``cron.jobs`` in
``sys.modules`` still comes from the boot-time tree. A sibling that reaches a shared
constant through that cached facade raises ImportError inside the per-job ``except``,
which turns a routine in-place update into every job being skipped as malformed."""

import sys

import pytest


@pytest.fixture
def stale_jobs_module(monkeypatch):
    """Mixed-version interpreter: jobs cached from a boot before the constants existed."""
    import cron.jobs as fresh_jobs

    stale_view = type(sys)("cron.jobs")
    stale_view.__dict__.update(fresh_jobs.__dict__)
    stale_view.__dict__.pop("FIRE_CLAIM_SKEW_SECONDS", None)
    stale_view.__dict__.pop("FIRE_CLAIM_TTL_SECONDS", None)
    monkeypatch.setitem(sys.modules, "cron.jobs", stale_view)
    monkeypatch.delitem(sys.modules, "cron.occurrences", raising=False)
    monkeypatch.delitem(sys.modules, "cron.constants", raising=False)
    return stale_view


def test_completed_occurrence_answers_without_the_jobs_facade(
    tmp_path, monkeypatch, stale_jobs_module
):
    from cron import executions
    from cron.occurrences import completed_occurrence

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    slot = "2026-01-05T00:00:00+00:00"
    run = executions.create_execution("job", source="builtin", scheduled_instant=slot)
    executions.finish_execution(run["id"], success=True)

    assert completed_occurrence({"id": "job"}, slot)
    assert not completed_occurrence({"id": "job"}, "2026-01-06T00:00:00+00:00")


def test_jobs_facade_still_reexports_the_shared_constants():
    import cron.jobs as jobs
    from cron import constants

    assert jobs.FIRE_CLAIM_SKEW_SECONDS == constants.FIRE_CLAIM_SKEW_SECONDS
    assert jobs.FIRE_CLAIM_TTL_SECONDS == constants.FIRE_CLAIM_TTL_SECONDS
