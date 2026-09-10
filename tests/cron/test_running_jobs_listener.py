"""The running-jobs listener: a cron job's start and end notify the gateway so it can
persist ``active_agents`` the way a chat turn boundary does.

Without this, a cron job that outlives the last chat turn leaves ``gateway_state.json``
at the count it had when that turn ended (a fleet's polite restart probe then read
"mid-turn" for 26 hours on an idle box), and a cron job with no chat turn open is
invisible to the file.
"""

import gc

import pytest

import cron.scheduler as sched


@pytest.fixture(autouse=True)
def _clean_registry():
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._running_jobs_listeners.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._running_jobs_listeners.clear()


def test_register_and_release_notify_the_listener():
    calls = []
    sched.add_running_jobs_listener(lambda: calls.append(len(sched.get_running_job_ids())))
    assert sched.try_register_running_job("job-a") is True
    assert calls == [1]
    sched.release_running_job("job-a")
    assert calls == [1, 0]


def test_a_refused_duplicate_register_does_not_notify():
    calls = []
    sched.add_running_jobs_listener(lambda: calls.append("x"))
    assert sched.try_register_running_job("job-a") is True
    assert sched.try_register_running_job("job-a") is False
    assert calls == ["x"]


def test_a_raising_listener_never_breaks_the_scheduler():
    calls = []

    def boom():
        raise RuntimeError("listener bug")

    sched.add_running_jobs_listener(boom)
    sched.add_running_jobs_listener(lambda: calls.append("ok"))
    assert sched.try_register_running_job("job-a") is True
    sched.release_running_job("job-a")
    assert calls == ["ok", "ok"]
    assert "job-a" not in sched.get_running_job_ids()


def test_remove_listener_is_idempotent():
    calls = []
    fn = lambda: calls.append("x")  # noqa: E731
    sched.add_running_jobs_listener(fn)
    sched.add_running_jobs_listener(fn)  # a second add is a no-op
    sched.remove_running_jobs_listener(fn)
    sched.remove_running_jobs_listener(fn)
    sched.try_register_running_job("job-a")
    assert calls == []


def test_a_bound_method_is_held_weakly():
    calls = []

    class Runner:
        def persist(self):
            calls.append("persisted")

    runner = Runner()
    sched.add_running_jobs_listener(runner.persist)
    sched.try_register_running_job("job-a")
    assert calls == ["persisted"]
    del runner
    gc.collect()
    sched.release_running_job("job-a")
    assert calls == ["persisted"]  # the dead reference was dropped, not called
    assert sched._running_jobs_listeners == []
