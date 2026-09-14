"""Behavioral regressions for the residual multiplex ticker failure paths."""

import errno
import threading

from unittest.mock import patch


def _homes(tmp_path):
    home = tmp_path / "profile"
    (home / "cron").mkdir(parents=True)
    return home


def test_callable_enumerator_base_exception_does_not_kill_ticker(tmp_path):
    from cron.scheduler_provider import InProcessCronScheduler

    home = _homes(tmp_path)
    stop = threading.Event()
    enumerated = threading.Event()
    ticked = threading.Event()
    calls = 0

    def enumerate_homes():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SystemExit("enumerator failed")
        enumerated.set()
        return [("profile", home)]

    def tick(*args, **kwargs):
        ticked.set()
        stop.set()

    with patch("cron.scheduler.tick", side_effect=tick):
        thread = threading.Thread(
            target=InProcessCronScheduler().start,
            args=(stop,),
            kwargs={"interval": 0, "profile_homes": enumerate_homes},
            daemon=True,
        )
        thread.start()
        assert enumerated.wait(5)
        assert ticked.wait(5)
        stop.set()
        thread.join(5)

    assert not thread.is_alive()
    assert calls >= 2


def test_profile_gate_failure_fails_closed_and_retries_next_cycle(tmp_path):
    from cron.scheduler_provider import InProcessCronScheduler

    home = _homes(tmp_path)
    stop = threading.Event()
    second_gate_failure = threading.Event()
    retried = threading.Event()
    ticked = threading.Event()
    gate_calls = 0

    def gate(name, candidate):
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls <= 2:
            if gate_calls == 2:
                second_gate_failure.set()
            raise OSError(errno.EMFILE, "too many open files")
        retried.set()
        return True

    def tick(*args, **kwargs):
        ticked.set()
        stop.set()

    with (
        patch("cron.scheduler.tick", side_effect=tick) as tick_mock,
        patch(
            "cron.scheduler_provider._backoff_wait_seconds", return_value=0
        ) as backoff,
        patch(
            "cron.jobs.record_ticker_error",
            side_effect=OSError("status write failed"),
        ),
    ):
        thread = threading.Thread(
            target=InProcessCronScheduler().start,
            args=(stop,),
            kwargs={
                "interval": 0,
                "profile_homes": [("profile", home)],
                "profile_gate": gate,
            },
            daemon=True,
        )
        thread.start()
        assert second_gate_failure.wait(5)
        assert retried.wait(5)
        assert ticked.wait(5)
        stop.set()
        thread.join(5)

    assert not thread.is_alive()
    assert tick_mock.call_count == 1
    backoff_failures = [call.args[1] for call in backoff.call_args_list]
    assert backoff_failures[:3] == [1, 2, 0]


def test_status_projection_scope_failure_does_not_kill_ticker(tmp_path):
    from cron import scheduler_provider as provider

    first = _homes(tmp_path / "first")
    second = _homes(tmp_path / "second")
    stop = threading.Event()
    failed = threading.Event()
    later_cycle = threading.Event()
    scope_calls = 0

    class ScopeFailure:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def scope(home):
        nonlocal scope_calls
        scope_calls += 1
        if scope_calls == 5:
            failed.set()
            raise RuntimeError("status scope failed")
        if scope_calls >= 7:
            later_cycle.set()
        return ScopeFailure()

    with (
        patch.object(provider, "_profile_cron_scope", side_effect=scope),
        patch("cron.scheduler.tick", return_value=0),
        patch("cron.jobs.record_ticker_heartbeat"),
        patch("cron.jobs.clear_ticker_error"),
        patch("cron.jobs.record_ticker_error"),
    ):
        thread = threading.Thread(
            target=provider.InProcessCronScheduler().start,
            args=(stop,),
            kwargs={
                "interval": 0,
                "profile_homes": [("first", first), ("second", second)],
            },
            daemon=True,
        )
        thread.start()
        assert failed.wait(5)
        assert later_cycle.wait(5)
        stop.set()
        thread.join(5)

    assert not thread.is_alive()
    assert scope_calls >= 5
