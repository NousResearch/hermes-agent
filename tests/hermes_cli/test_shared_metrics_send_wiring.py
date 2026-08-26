"""Tests for wiring the sender into the shared-metrics export hook.

The properties that matter here are negative ones: the interactive path must
not block, and nothing must leave the machine unless the user opted in.
"""

from __future__ import annotations

import threading
import time

import pytest

from hermes_cli.observability import relay_shared_metrics as mod


class FakeStore:
    def __init__(self):
        self.exported = 0

    def create_and_export_package_if_due(self):
        self.exported += 1
        return []


class FakeSubscriber:
    def __init__(self):
        self.store = FakeStore()


class Runtime(mod._Runtime):
    """A _Runtime with the relay host stubbed out."""

    def __init__(self):
        self._sessions_lock = threading.RLock()
        self._sessions = {}
        self._task_creation_lock = threading.RLock()
        self._task_sessions_lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._send_thread = None
        self._task_sessions = {}
        self._turn_sessions = {}
        self.subscriber = FakeSubscriber()


@pytest.fixture
def runtime():
    return Runtime()


def _config(**shared):
    return {"telemetry": {"shared_metrics": shared}}


@pytest.fixture
def capture_sender(monkeypatch):
    """Replace the sender with a recorder and return the record."""
    record = {"passes": [], "endpoints": []}

    class FakeSender:
        def __init__(self, store, endpoint, **kwargs):
            record["endpoints"].append(endpoint)

        def send_pending(self):
            record["passes"].append(time.time())

    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_sender.SharedMetricsSender",
        FakeSender,
    )
    return record


def _set_config(monkeypatch, config):
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly", lambda: config, raising=False
    )


class TestOptIn:
    def test_no_send_when_nothing_is_configured(self, runtime, monkeypatch, capture_sender):
        _set_config(monkeypatch, {})
        runtime._export()
        runtime._join_send_thread(timeout=1)
        assert capture_sender["passes"] == []

    def test_no_send_when_only_collection_is_on(self, runtime, monkeypatch, capture_sender):
        _set_config(monkeypatch, _config(enabled=True))
        runtime._export()
        runtime._join_send_thread(timeout=1)
        assert capture_sender["passes"] == []

    def test_no_send_when_send_is_on_without_collection(
        self, runtime, monkeypatch, capture_sender
    ):
        _set_config(monkeypatch, _config(enabled=False, send=True))
        runtime._export()
        runtime._join_send_thread(timeout=1)
        assert capture_sender["passes"] == []

    def test_sends_when_both_are_on(self, runtime, monkeypatch, capture_sender):
        _set_config(monkeypatch, _config(enabled=True, send=True))
        runtime._export()
        runtime._join_send_thread(timeout=2)
        assert len(capture_sender["passes"]) == 1

    def test_uses_the_resolved_endpoint(self, runtime, monkeypatch, capture_sender):
        _set_config(
            monkeypatch,
            _config(enabled=True, send=True, endpoint="https://staging.test/v1"),
        )
        runtime._export()
        runtime._join_send_thread(timeout=2)
        assert capture_sender["endpoints"] == ["https://staging.test/v1"]

    def test_export_still_runs_when_sending_is_off(self, runtime, monkeypatch, capture_sender):
        _set_config(monkeypatch, _config(enabled=True))
        runtime._export()
        assert runtime.subscriber.store.exported == 1


class TestInteractivePathIsNotBlocked:
    def test_export_returns_before_the_send_finishes(
        self, runtime, monkeypatch
    ):
        started = threading.Event()
        release = threading.Event()

        class SlowSender:
            def __init__(self, store, endpoint, **kwargs):
                pass

            def send_pending(self):
                started.set()
                release.wait(5)

        monkeypatch.setattr(
            "hermes_cli.observability.shared_metrics_sender.SharedMetricsSender",
            SlowSender,
        )
        _set_config(monkeypatch, _config(enabled=True, send=True))

        began = time.monotonic()
        runtime._export()
        elapsed = time.monotonic() - began

        assert started.wait(2), "the send should have started"
        assert elapsed < 1.0, "finish_task must not wait on the network"
        release.set()
        runtime._join_send_thread(timeout=5)

    def test_the_send_thread_is_a_daemon(self, runtime, monkeypatch, capture_sender):
        _set_config(monkeypatch, _config(enabled=True, send=True))
        runtime._export()
        with runtime._send_lock:
            thread = runtime._send_thread
        assert thread is not None
        assert thread.daemon, "an unfinished send must not hold the process open"
        runtime._join_send_thread(timeout=2)

    def test_only_one_pass_runs_at_a_time(self, runtime, monkeypatch):
        release = threading.Event()
        starts = []

        class SlowSender:
            def __init__(self, store, endpoint, **kwargs):
                pass

            def send_pending(self):
                starts.append(1)
                release.wait(5)

        monkeypatch.setattr(
            "hermes_cli.observability.shared_metrics_sender.SharedMetricsSender",
            SlowSender,
        )
        _set_config(monkeypatch, _config(enabled=True, send=True))

        for _ in range(5):
            runtime._export()
        time.sleep(0.2)
        assert len(starts) == 1, "hook fires must not pile up send passes"
        release.set()
        runtime._join_send_thread(timeout=5)


class TestFailureIsolation:
    def test_a_sender_crash_does_not_propagate(self, runtime, monkeypatch):
        class Exploding:
            def __init__(self, store, endpoint, **kwargs):
                pass

            def send_pending(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            "hermes_cli.observability.shared_metrics_sender.SharedMetricsSender",
            Exploding,
        )
        _set_config(monkeypatch, _config(enabled=True, send=True))
        runtime._export()  # must not raise
        runtime._join_send_thread(timeout=2)

    def test_an_unreadable_config_does_not_break_export(self, runtime, monkeypatch, capture_sender):
        def explode():
            raise OSError("config unreadable")

        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config_readonly", explode, raising=False
        )
        runtime._export()
        assert runtime.subscriber.store.exported == 1
        assert capture_sender["passes"] == []

    def test_join_is_safe_with_no_thread(self, runtime):
        runtime._join_send_thread(timeout=0.1)

    def test_join_waits_for_an_in_flight_send(self, runtime, monkeypatch):
        """shutdown() must give a started send a chance to finish.

        A short-lived CLI exits straight after its final export; without the
        join the daemon thread is killed mid-request, and the hook path is the
        only delivery cadence this feature has.
        """
        finished = []
        release = threading.Event()

        class SlowSender:
            def __init__(self, store, endpoint, **kwargs):
                pass

            def send_pending(self):
                release.wait(3)
                finished.append(True)

        monkeypatch.setattr(
            "hermes_cli.observability.shared_metrics_sender.SharedMetricsSender",
            SlowSender,
        )
        _set_config(monkeypatch, _config(enabled=True, send=True))

        runtime._export()
        release.set()
        runtime._join_send_thread(timeout=3)
        assert finished == [True]

    def test_shutdown_joins_the_send_thread(self):
        """Regression: the join was wired into deactivate() but not shutdown()."""
        import inspect

        source = inspect.getsource(mod._Runtime.shutdown)
        assert "_join_send_thread" in source, (
            "shutdown() must join the sender, or a CLI exit kills it mid-send"
        )
