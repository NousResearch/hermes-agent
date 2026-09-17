"""Startup contract for the optional Skills Hub maintenance pass."""

import threading
import time

from gateway.run import _start_background_skills_sync


def test_skills_sync_runs_in_background_without_blocking_startup():
    started = threading.Event()
    release = threading.Event()

    def slow_sync() -> None:
        started.set()
        release.wait(timeout=5)

    begin = time.monotonic()
    worker = _start_background_skills_sync(slow_sync)
    elapsed = time.monotonic() - begin

    try:
        assert elapsed < 1.0
        assert started.wait(timeout=1.0)
        assert worker.daemon
        assert worker.name == "skills-hub-sync"
    finally:
        release.set()
        worker.join(timeout=2.0)
    assert not worker.is_alive()
