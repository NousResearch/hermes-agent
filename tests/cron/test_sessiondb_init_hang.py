"""Canonical cron must bound store acquisition without unpersisted inference.

The owner lends its existing store; timeout/failure must not fall through to a
local writer or an agent without persistence. Late borrows are still released.
"""

import concurrent.futures
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cron import scheduler
from gateway import session_cron


@pytest.fixture
def owner_run(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = MagicMock(db_path=tmp_path / "state.db")
    db.get_compression_tip.return_value = None
    owner = SimpleNamespace(db=db)
    job = {"id": "owner-db", "name": "test", "prompt": "hello"}
    execution_id = "owner-fire"
    token = session_cron._execution.set((owner, "owner-session", job["id"], execution_id))
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(scheduler, "_reload_dotenv_and_publish_delivery_target", lambda *_: None)
    monkeypatch.setattr(scheduler, "_resolve_cron_agent_setup", lambda *_: scheduler._CronAgentSetup(
        model="test", runtime={"provider": "test"}))
    with patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value.run_conversation.return_value = {"final_response": "ok"}
        try:
            yield owner, job, execution_id, agent_cls
        finally:
            session_cron._execution.reset(token)


def test_sessiondb_init_preserves_multiplex_profile_context(owner_run, tmp_path):
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    owner, job, execution_id, agent_cls = owner_run
    profile_home = tmp_path / "profiles" / "jobsearch"
    owner.db.db_path = profile_home / "state.db"
    observed = []

    def acquire(path):
        observed.append((path, get_hermes_home()))
        return owner.db

    token = set_hermes_home_override(profile_home)
    try:
        with patch("hermes_state_registry.acquire", side_effect=acquire):
            success, _, response, error = scheduler.run_job(job, execution_id=execution_id)
    finally:
        reset_hermes_home_override(token)
    assert (success, response, error) == (True, "ok", None)
    assert observed == [(profile_home / "state.db", profile_home)]
    assert agent_cls.call_args.kwargs["session_db"] is owner.db


@pytest.mark.parametrize("setting", ["env", "config"])
def test_run_job_does_not_hang_when_sessiondb_init_wedges(owner_run, tmp_path, monkeypatch, setting):
    owner, job, execution_id, agent_cls = owner_run
    if setting == "env":
        monkeypatch.setenv("HERMES_CRON_SESSION_DB_TIMEOUT", "0.02")
    else:
        monkeypatch.delenv("HERMES_CRON_SESSION_DB_TIMEOUT", raising=False)
        (tmp_path / "config.yaml").write_text("cron:\n  session_db_timeout_seconds: 0.02\n")
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    owner.db.close.side_effect = closed.set

    def acquire(_path):
        entered.set()
        release.wait(10)
        return owner.db

    try:
        with patch("hermes_state_registry.acquire", side_effect=acquire):
            started = time.monotonic()
            success, _, response, error = scheduler.run_job(job, execution_id=execution_id)
            elapsed = time.monotonic() - started
            assert entered.is_set()
            assert elapsed < 5
            assert not success and response == "" and error and "TimeoutError" in error
            agent_cls.assert_not_called()
            owner.db.close.assert_not_called()
            release.set()
            assert closed.wait(5), "late borrowed reference was leaked"
            owner.db.close.assert_called_once()
    finally:
        release.set()
        assert closed.wait(5)


@pytest.mark.parametrize("failure", ["wrong-store", "acquire-error"])
def test_owner_store_failure_never_runs_unpersisted_agent(owner_run, failure):
    owner, job, execution_id, agent_cls = owner_run
    wrong_db = MagicMock()
    with patch("hermes_state_registry.acquire", return_value=wrong_db,
               side_effect=OSError("store unavailable") if failure == "acquire-error" else None):
        success, _, response, error = scheduler.run_job(job, execution_id=execution_id)
    assert not success and response == ""
    assert error and ("canonical owner store" if failure == "wrong-store" else "store unavailable") in error
    agent_cls.assert_not_called()
    owner.db.close.assert_not_called()
    assert wrong_db.close.call_count == (1 if failure == "wrong-store" else 0)


def test_invalid_timeout_env_falls_back_to_default(owner_run, monkeypatch, caplog):
    owner, job, _, _ = owner_run
    monkeypatch.setenv("HERMES_CRON_SESSION_DB_TIMEOUT", "not-a-number")
    future = concurrent.futures.Future()
    future.set_result(owner.db)
    with patch("cron.scheduler.concurrent.futures.ThreadPoolExecutor") as pool:
        pool.return_value.submit.return_value = future
        with patch.object(future, "result", wraps=future.result) as result:
            assert scheduler._open_cron_session_db(job) is owner.db
    assert result.call_args.kwargs["timeout"] == 10.0
    assert any("HERMES_CRON_SESSION_DB_TIMEOUT" in r.message and "not-a-number" in r.message
               for r in caplog.records)


def test_guard_is_released_and_job_refires_after_sessiondb_hang(owner_run, monkeypatch):
    owner, job, _, agent_cls = owner_run
    monkeypatch.setenv("HERMES_CRON_SESSION_DB_TIMEOUT", "0.02")
    from cron import jobs

    release = threading.Event()
    closed = threading.Event()
    owner.db.close.side_effect = closed.set
    calls = []

    def acquire(_path):
        calls.append(True)
        if len(calls) == 1:
            release.wait(10)
        return owner.db

    # The scheduler dispatches through the authority; emulate only transport and
    # execution admission, retaining the real guard and run_job lifecycle.
    def admitted(fired, **kwargs):
        token = session_cron._execution.set((owner, "owner-session", fired["id"], kwargs["execution_id"]))
        try:
            return scheduler.run_job(fired, **kwargs)
        finally:
            session_cron._execution.reset(token)

    with jobs.use_cron_store(owner.db.db_path.parent):
        saved = jobs.create_job(prompt="hello", schedule="every 5m", deliver="local")
        try:
            with patch("hermes_state_registry.acquire", side_effect=acquire), \
                 patch("cron.scheduler_authority.run_canonical_job", side_effect=admitted), \
                 patch("cron.scheduler_authority.reconcile_pending"), \
                 patch.object(scheduler, "get_due_jobs", return_value=[saved]):
                assert scheduler.tick() == 1
                assert saved["id"] not in scheduler.get_running_job_ids()
                agent_cls.assert_not_called()
                release.set()
                assert closed.wait(5)
                assert scheduler.tick() == 1
                assert len(calls) == 2
                assert agent_cls.call_count == 1
                final_job = jobs.get_job(saved["id"])
                assert final_job and final_job["last_status"] == "delivery_queued"
        finally:
            release.set()
            scheduler._shutdown_parallel_pool()


@pytest.mark.parametrize("result", ["db", None, RuntimeError("connect failed")])
def test_close_late_session_db_result(result):
    db = MagicMock()
    future = concurrent.futures.Future()
    if isinstance(result, Exception):
        future.set_exception(result)
    else:
        future.set_result(db if result == "db" else result)
    scheduler._close_late_session_db_result(future)
    assert db.close.call_count == (1 if result == "db" else 0)


def test_wake_gate_false_never_opens_session_db(owner_run):
    _, job, execution_id, agent_cls = owner_run
    job["script"] = "gate.py"
    with patch("hermes_state_registry.acquire") as acquire, \
         patch("cron.scheduler._run_job_script_with_claim_heartbeat", return_value=(True, '{"wakeAgent": false}')):
        success, _, _, error = scheduler.run_job(job, execution_id=execution_id)
    assert success and error is None
    acquire.assert_not_called()
    agent_cls.assert_not_called()
