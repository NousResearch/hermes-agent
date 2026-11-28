"""A run that loses its fire claim keeps what it produced instead of discarding it.

The normal output save is fenced by the fire claim, so a run whose claim another fire took over
closed its ledger row with "stale result was discarded" and dropped the whole result: minutes of
agent work and the tokens it cost. The fence still holds (nothing is delivered, nothing is written
over the new owner); the output now lands in ``<stamp>-partial-<execution id>.md``.
"""
from unittest.mock import MagicMock

import pytest

from tests.cron.test_fire_claim_lost_after_delivery import _drive_heartbeat_thread

PLAIN = "Fire claim ownership lost; stale result was discarded."


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json/executions don't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_running_state():
    from cron import scheduler
    for registry in (scheduler._running_job_ids, scheduler._running_fire_owners, scheduler._interrupted_job_ids):
        registry.clear()
    yield
    for registry in (scheduler._running_job_ids, scheduler._running_fire_owners, scheduler._interrupted_job_ids):
        registry.clear()


def test_a_confirmed_claim_loss_mid_run_keeps_the_output(temp_home, monkeypatch):
    from cron.executions import get_execution
    from cron.jobs import get_job

    sched, job, _hb, delivered, run_cancel = _drive_heartbeat_thread(monkeypatch, misses=2, steal=True)

    assert sched.run_one_job(job) is True

    assert run_cancel == [True]
    assert delivered == [], "a lost claim still never delivers"
    record = get_job(job["id"])
    assert record["last_status"] is None and record["fire_claim"]["by"] == "replacement:deadbeef"
    error = get_execution(job["execution_id"])["error"]
    partials = list((temp_home / "cron" / "output" / job["id"]).glob("*-partial-*.md"))
    assert len(partials) == 1, partials
    assert partials[0].read_text(encoding="utf-8") == "output text"
    assert error.startswith("Fire claim ownership lost; stale result was not delivered.")
    assert str(partials[0]) in error


@pytest.fixture
def finished(monkeypatch):
    from cron import scheduler
    mock = MagicMock()
    monkeypatch.setattr(scheduler, "finish_execution", mock)
    return mock


def test_the_kept_file_is_named_by_execution_and_holds_the_output(tmp_path, finished):
    from cron import jobs
    from cron import scheduler

    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="work", schedule="every 5m", name="heavy", deliver="local")
        scheduler._record_fire_ownership_lost(job, None, "lost-execution", produced="the long report")
        files = sorted((tmp_path / "cron" / "output" / job["id"]).glob("*-partial-lost-execution.md"))

    assert len(files) == 1, files
    assert files[0].read_text(encoding="utf-8") == "the long report"
    _args, kwargs = finished.call_args
    assert _args == ("lost-execution",) and kwargs["success"] is False
    assert str(files[0]) in kwargs["error"]


def test_a_failed_keep_still_closes_the_ledger_row(tmp_path, finished, monkeypatch):
    """Best effort: a write that fails must not leave the row non-terminal (the dead-owner sweep
    would then report it ``unknown``); the failure is recorded exactly as before."""
    from cron import jobs
    from cron import scheduler
    import utils

    def refuse(*_args, **_kwargs):
        raise OSError("output dir is gone")
    monkeypatch.setattr(utils, "atomic_write_text", refuse)
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="work", schedule="every 5m", name="boom", deliver="local")
        scheduler._record_fire_ownership_lost(job, None, "boom-execution", produced="unwritable")

    finished.assert_called_once_with("boom-execution", success=False, error=PLAIN)


def test_nothing_produced_keeps_no_file_and_the_plain_error(tmp_path, finished):
    from cron import jobs
    from cron import scheduler

    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="work", schedule="every 5m", name="empty", deliver="local")
        scheduler._record_fire_ownership_lost(job, None, "empty-execution", produced="  ")
        assert list((tmp_path / "cron" / "output").glob("**/*-partial-*")) == []

    finished.assert_called_once_with("empty-execution", success=False, error=PLAIN)


def test_an_execution_id_cannot_steer_the_file_out_of_the_job_directory(tmp_path, finished):
    from cron import jobs
    from cron import scheduler

    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="work", schedule="every 5m", name="odd id", deliver="local")
        scheduler._record_fire_ownership_lost(job, None, "../../escape", produced="report")
        files = list((tmp_path / "cron" / "output" / job["id"]).glob("*-partial-*.md"))

    assert [f.name.split("-partial-")[1] for f in files] == ["escape.md"]
