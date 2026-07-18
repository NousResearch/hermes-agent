import concurrent.futures

import pytest

import gateway.run as gateway_run
import gateway.status as gateway_status
from gateway.config import GatewayConfig


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


def test_watchdog_config_parses_nested_gateway_values():
    config = GatewayConfig.from_dict(
        {
            "gateway": {
                "event_loop_watchdog": {
                    "enabled": True,
                    "interval_seconds": 12.5,
                    "timeout_seconds": 4,
                    "failure_threshold": 5,
                }
            }
        }
    )

    watchdog = config.event_loop_watchdog
    assert watchdog.enabled is True
    assert watchdog.interval_seconds == 12.5
    assert watchdog.timeout_seconds == 4
    assert watchdog.failure_threshold == 5


def test_watchdog_config_invalid_values_fall_back_to_safe_defaults():
    config = GatewayConfig.from_dict(
        {
            "event_loop_watchdog": {
                "enabled": "yes",
                "interval_seconds": 0,
                "timeout_seconds": "invalid",
                "failure_threshold": -1,
            }
        }
    )

    watchdog = config.event_loop_watchdog
    assert watchdog.enabled is True
    assert watchdog.interval_seconds == 60.0
    assert watchdog.timeout_seconds == 30.0
    assert watchdog.failure_threshold == 3


def test_watchdog_thread_is_built_only_when_config_enabled():
    disabled = GatewayConfig.from_dict({}).event_loop_watchdog
    assert gateway_run._build_event_loop_watchdog_thread(
        _NeverStops(), object(), disabled
    ) is None

    enabled = GatewayConfig.from_dict(
        {"event_loop_watchdog": {"enabled": True}}
    ).event_loop_watchdog
    thread = gateway_run._build_event_loop_watchdog_thread(
        _NeverStops(), object(), enabled
    )
    assert thread is not None
    assert thread.name == "gateway-loop-watchdog"
    assert thread.daemon is True
