"""Tests for the /model picker background cache prewarm.

``prewarm_picker_cache_async()`` warms the provider-models disk cache off the
user's critical path so the first ``/model`` open in a session is fast instead
of blocking ~1-2s on serial /v1/models fetches. These pin the two contracts
that matter: it runs the warm path exactly once per process (no thread leak),
and it delegates to ``list_authenticated_providers`` to do the warming.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import hermes_cli.model_switch as ms


def _reset_guard():
    ms._reset_picker_prewarm_for_tests()


def test_prewarm_runs_list_authenticated_providers_once():
    """First call spawns a thread that calls list_authenticated_providers;
    the warm side effect is delegated there (which disk-caches per provider)."""
    _reset_guard()
    with patch.object(ms, "list_authenticated_providers", return_value=[]) as mock_list:
        t = ms.prewarm_picker_cache_async()
        assert t is not None, "first call must spawn a prewarm thread"
        t.join(timeout=10)
        assert not t.is_alive(), "prewarm thread should finish promptly"
        mock_list.assert_called_once()
    _reset_guard()


def test_prewarm_guard_is_once_per_process():
    """The process-level Event guard must make repeat calls no-ops so a
    long-lived process never leaks one OS thread per call."""
    _reset_guard()
    with patch.object(ms, "list_authenticated_providers", return_value=[]):
        t1 = ms.prewarm_picker_cache_async()
        assert t1 is not None
        t1.join(timeout=10)
        # Subsequent calls return None (guard set) — no new thread.
        assert ms.prewarm_picker_cache_async() is None
        assert ms.prewarm_picker_cache_async() is None
    _reset_guard()


def test_prewarm_guard_is_atomic_under_concurrent_calls():
    """Concurrent session starts should still spawn only one prewarm thread."""
    _reset_guard()
    worker_count = 8
    start = threading.Barrier(worker_count)

    def invoke():
        start.wait(timeout=10)
        return ms.prewarm_picker_cache_async()

    def slow_list(*_args, **_kwargs):
        time.sleep(0.05)
        return []

    with patch.object(ms, "list_authenticated_providers", side_effect=slow_list) as mock_list:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            threads = [thread for thread in pool.map(lambda _: invoke(), range(worker_count)) if thread is not None]
        for thread in threads:
            thread.join(timeout=10)

    assert len(threads) == 1
    mock_list.assert_called_once()
    _reset_guard()


def test_prewarm_never_raises_on_failure():
    """A failing/offline provider path must be fully swallowed — the prewarm
    is best-effort and must never surface errors into the session."""
    _reset_guard()
    with patch.object(
        ms, "list_authenticated_providers", side_effect=RuntimeError("boom")
    ):
        t = ms.prewarm_picker_cache_async()
        assert t is not None
        # join must not raise; the worker swallows the exception internally.
        t.join(timeout=10)
        assert not t.is_alive()
    _reset_guard()
