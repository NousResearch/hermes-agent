"""Reproducer + regression test for daemon_pool Python 3.14 compatibility.

Background
----------
``concurrent.futures.thread._worker`` changed signature in Python 3.14:

* 3.8-3.13: ``_worker(executor_reference, work_queue, initializer, initargs)``
* 3.14+:    ``_worker(executor_reference, ctx, work_queue)``

``tools/daemon_pool.py`` was written for the 3.8-3.13 signature and was never
updated when the stdlib changed. Calling ``pool.submit(...)`` on the patched
``DaemonThreadPoolExecutor`` triggers ``_adjust_thread_count`` which in turn
spawns a ``threading.Thread(target=_worker, args=(4-arg tuple))`` -- and the
stdlib's new 3-arg ``_worker`` raises::

    TypeError: _worker() takes 3 positional arguments but 4 were given

User-visible symptom: any tool chain that exercises concurrent
``DaemonThreadPoolExecutor`` (skills_hub batches, subagent timeout wrappers,
catalog fan-out) hits 502s on the first ``submit`` of a fresh pool. This
test fails on every Python 3.14 install before the fix.

Run from the repo root:

    python -m pytest tests/test_daemon_pool_py314_compat.py -v

Or smoke-test without pytest:

    python tests/test_daemon_pool_py314_compat.py
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def _ensure_importable() -> None:
    """Make ``tools`` importable when running this file directly via ``python tests/...``."""
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_importable()

# Imported lazily after sys.path tweak so the file also runs as a script.
from tools.daemon_pool import DaemonThreadPoolExecutor  # noqa: E402


def test_python_version_at_least_314():
    """This test only exercises the regression path on Python 3.14+."""
    assert sys.version_info >= (3, 14), (
        f"reproducer targets Python 3.14 stdlib API change; "
        f"current interpreter is {sys.version_info[:2]}"
    )


def test_basic_submit_does_not_raise():
    """A single submit() must not raise TypeError from _worker signature mismatch."""
    pool: DaemonThreadPoolExecutor = DaemonThreadPoolExecutor(max_workers=2)

    def task() -> str:
        return "ok"

    future = pool.submit(task)
    assert future.result(timeout=5) == "ok"

    pool.shutdown(wait=True)


def test_many_concurrent_submits():
    """20 parallel submits -- the call shape exercised by skills_hub / subagent batches."""
    pool: DaemonThreadPoolExecutor = DaemonThreadPoolExecutor(
        max_workers=min(len([1, 2, 3, 4, 5, 6]), 8)
    )

    def task(i: int) -> str:
        # tiny sleep to force _adjust_thread_count to actually create workers
        time.sleep(0.01)
        return f"task-{i}"

    futures = [pool.submit(task, i) for i in range(20)]
    results = [f.result(timeout=10) for f in futures]

    assert len(results) == 20
    assert results == [f"task-{i}" for i in range(20)]

    pool.shutdown(wait=True)


def test_initializer_and_initargs_propagate_to_worker():
    """The new ctx-based call shape must still run initializer with initargs."""
    initialized: list[tuple[int, int]] = []
    init_lock = threading.Lock()

    def initializer(x: int, y: int) -> None:
        with init_lock:
            initialized.append((x, y))

    pool: DaemonThreadPoolExecutor = DaemonThreadPoolExecutor(
        max_workers=1, initializer=initializer, initargs=(10, 20)
    )
    pool.submit(lambda: "task").result(timeout=5)
    pool.shutdown(wait=True)

    assert initialized == [(10, 20)], (
        f"initializer did not run with initargs on the worker; got {initialized!r}"
    )


def test_worker_threads_are_daemons():
    """The whole point of DaemonThreadPoolExecutor: workers must be daemon=True so
    interpreter exit is not blocked by the stdlib's _python_exit atexit hook."""
    pool: DaemonThreadPoolExecutor = DaemonThreadPoolExecutor(max_workers=2)

    # Force worker creation via submit
    pool.submit(lambda: None).result(timeout=5)

    threads = list(pool._threads)  # type: ignore[attr-defined]
    assert threads, "expected at least one worker thread to be created"

    for t in threads:
        assert t.daemon is True, (
            f"worker thread {t.name!r} is not daemon; "
            f"this defeats the purpose of DaemonThreadPoolExecutor"
        )

    pool.shutdown(wait=True)


def test_does_not_break_plain_threadingpoolexecutor():
    """The fix must not change stdlib ThreadPoolExecutor behavior -- sanity check
    that we are touching the right code path."""
    pool = ThreadPoolExecutor(max_workers=1)
    assert pool.submit(lambda: 42).result(timeout=5) == 42
    pool.shutdown(wait=True)


if __name__ == "__main__":
    # Allow `python tests/test_daemon_pool_py314_compat.py` smoke test
    # outside of pytest.
    failures = 0
    for fn in [
        test_python_version_at_least_314,
        test_basic_submit_does_not_raise,
        test_many_concurrent_submits,
        test_initializer_and_initargs_propagate_to_worker,
        test_worker_threads_are_daemons,
        test_does_not_break_plain_threadingpoolexecutor,
    ]:
        name = fn.__name__
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001 -- intentional aggregate
            failures += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")

    if failures:
        print(f"\n{failures} test(s) failed")
        sys.exit(1)
    print("\nAll tests passed")