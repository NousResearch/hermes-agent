"""Maintenance held for the first frame runs after it, once, and never gets lost."""

import threading
import types

import pytest

from hermes_cli import startup_idle


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(startup_idle, "_armed", False)
    monkeypatch.setattr(startup_idle, "_released", False)
    monkeypatch.setattr(startup_idle, "_pending", [])
    monkeypatch.setattr(startup_idle, "_timer", None)
    yield
    timer = startup_idle._timer
    if timer is not None:
        timer.cancel()


def test_unarmed_process_runs_work_at_once():
    ran = []
    startup_idle.defer_until_first_paint(lambda: ran.append("now"))
    assert ran == ["now"]


def test_armed_work_waits_for_the_first_frame_then_runs_once():
    ran = []
    startup_idle.arm()
    startup_idle.defer_until_first_paint(lambda: ran.append("a"))
    startup_idle.defer_until_first_paint(lambda: ran.append("b"))
    assert ran == []
    startup_idle.release(object())  # the after_render handler receives the app
    startup_idle.release(object())  # later frames do nothing
    assert ran == ["a", "b"]


def test_work_deferred_after_the_first_frame_runs_at_once():
    ran = []
    startup_idle.arm()
    startup_idle.release()
    startup_idle.defer_until_first_paint(lambda: ran.append("late"))
    assert ran == ["late"]


def test_a_process_that_never_paints_still_runs_the_work(monkeypatch):
    monkeypatch.setattr(startup_idle, "FALLBACK_SECONDS", 0.05)
    done = threading.Event()
    startup_idle.arm()
    startup_idle.defer_until_first_paint(done.set)
    assert done.wait(2.0)


def test_one_failing_job_does_not_stop_the_others():
    ran = []
    startup_idle.arm()

    def boom():
        raise RuntimeError("maintenance failed")

    startup_idle.defer_until_first_paint(boom)
    startup_idle.defer_until_first_paint(lambda: ran.append("after"))
    startup_idle.release()
    assert ran == ["after"]


@pytest.mark.parametrize("args, armed", [
    (types.SimpleNamespace(query=None, query_file=None), True),
    (types.SimpleNamespace(), True),
    (types.SimpleNamespace(query="what is 2+2", query_file=None), False),
    (types.SimpleNamespace(query=None, query_file="prompt.txt"), False),
])
def test_only_an_interactive_launch_arms_the_hold(args, armed):
    startup_idle.arm_for(args)
    assert startup_idle._armed is armed


def test_a_deferred_thread_starts_after_the_first_frame_with_the_callers_context():
    import contextvars

    scope = contextvars.ContextVar("profile_scope", default="unset")
    seen, done = [], threading.Event()
    startup_idle.arm()
    scope.set("profile-a")
    startup_idle.defer_thread(lambda: (seen.append(scope.get()), done.set()), name="deferred-test")
    assert seen == []
    startup_idle.release()
    assert done.wait(2.0)
    assert seen == ["profile-a"]


def test_the_fallback_timer_still_starts_work_in_the_callers_profile_scope(monkeypatch):
    """The timer thread has an empty context; the work must run in the one captured when it was deferred."""
    import contextvars

    scope = contextvars.ContextVar("profile_scope", default="unset")
    monkeypatch.setattr(startup_idle, "FALLBACK_SECONDS", 0.05)
    seen, done = [], threading.Event()
    startup_idle.arm()
    scope.set("profile-a")
    startup_idle.defer_thread(lambda: (seen.append(scope.get()), done.set()), name="fallback-scope-test")
    assert done.wait(2.0)  # released by the timer: nothing called release()
    assert seen == ["profile-a"]


def test_plain_deferred_work_also_runs_in_the_callers_context_when_the_timer_releases_it(monkeypatch):
    import contextvars

    scope = contextvars.ContextVar("profile_scope", default="unset")
    monkeypatch.setattr(startup_idle, "FALLBACK_SECONDS", 0.05)
    seen, done = [], threading.Event()
    startup_idle.arm()
    scope.set("profile-b")
    startup_idle.defer_until_first_paint(lambda: (seen.append(scope.get()), done.set()))
    assert done.wait(2.0)
    assert seen == ["profile-b"]
