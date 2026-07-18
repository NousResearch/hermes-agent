import concurrent.futures

import pytest

import gateway.run as gateway_run
import gateway.status as gateway_status


class _ExitCalled(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


class _NeverStops:
    def wait(self, timeout=None):
        return False


class _TimedOutFuture:
    def result(self, timeout=None):
        raise concurrent.futures.TimeoutError()

    def cancel(self):
        return True


def test_event_loop_watchdog_exits_after_consecutive_undispatched_probes(monkeypatch):
    def fake_schedule(coro, *args, **kwargs):
        coro.close()
        return _TimedOutFuture()

    monkeypatch.setattr(
        gateway_run,
        "safe_schedule_threadsafe",
        fake_schedule,
    )
    monkeypatch.setattr(gateway_status, "write_runtime_status", lambda **kwargs: None)

    def fake_exit(code):
        raise _ExitCalled(code)

    monkeypatch.setattr(gateway_run, "_exit_after_graceful_shutdown", fake_exit)

    with pytest.raises(_ExitCalled) as exc_info:
        gateway_run._start_event_loop_watchdog(
            _NeverStops(),
            loop=object(),
            interval=1,
            timeout=1,
            max_failures=2,
        )

    assert exc_info.value.code == gateway_run.GATEWAY_SERVICE_RESTART_EXIT_CODE
