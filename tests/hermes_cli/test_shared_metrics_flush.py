from __future__ import annotations

import threading
import time

from hermes_cli.observability.relay_shared_metrics import _Runtime


def _runtime_for_flush_test() -> _Runtime:
    runtime = object.__new__(_Runtime)
    runtime._flush_lock = threading.RLock()
    runtime._flush_pending = False
    runtime._flush_thread = None
    return runtime


def test_scheduled_flush_does_not_wait_for_process_wide_barrier(monkeypatch):
    runtime = _runtime_for_flush_test()
    flush_started = threading.Event()
    release_flush = threading.Event()
    calls: list[str] = []

    def blocked_flush(self, failure_message):
        calls.append(failure_message)
        flush_started.set()
        assert release_flush.wait(2)

    monkeypatch.setattr(_Runtime, "_flush_and_export", blocked_flush)

    started = time.monotonic()
    runtime._schedule_flush_and_export("flush failed")
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert flush_started.wait(2)
    release_flush.set()
    thread = runtime._flush_thread
    assert thread is not None
    thread.join(2)
    assert calls == ["flush failed"]
