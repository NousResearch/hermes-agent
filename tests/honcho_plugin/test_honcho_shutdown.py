"""Tests for Honcho memory provider shutdown — SIGABRT prevention.

Verifies that:
1. The prefetch-thread TOCTOU race is guarded by _prefetch_threads_lock.
2. _close_honcho_http_client warns when the Honcho SDK client can't be
   located (silent SDK attribute drift detection).
3. shutdown() joins all tracked prefetch threads.
"""

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.honcho import HonchoMemoryProvider


class TestPrefetchThreadLock:
    """Verify the _prefetch_threads_lock guards concurrent access."""

    def test_lock_exists(self):
        """HonchoSession must have a _prefetch_threads_lock attribute."""
        from plugins.memory.honcho.session import HonchoSessionManager

        manager = HonchoSessionManager.__new__(HonchoSessionManager)
        # Verify the attribute would exist on a real instance
        assert hasattr(HonchoSessionManager, "__init__")
        # Check the source includes the lock
        import inspect

        src = inspect.getsource(HonchoSessionManager.__init__)
        assert "_prefetch_threads_lock" in src, (
            "HonchoSessionManager.__init__ must initialize _prefetch_threads_lock"
        )

    def test_prefetch_context_uses_lock(self):
        """prefetch_context must acquire _prefetch_threads_lock."""
        import inspect

        from plugins.memory.honcho.session import HonchoSessionManager

        src = inspect.getsource(HonchoSessionManager.prefetch_context)
        assert "_prefetch_threads_lock" in src, (
            "prefetch_context must use _prefetch_threads_lock to guard the thread list"
        )

    def test_shutdown_uses_lock(self):
        """shutdown must acquire _prefetch_threads_lock for snapshot."""
        import inspect

        from plugins.memory.honcho.session import HonchoSessionManager

        src = inspect.getsource(HonchoSessionManager.shutdown)
        assert "_prefetch_threads_lock" in src, (
            "shutdown must use _prefetch_threads_lock to snapshot prefetch threads"
        )


class TestCloseHonchoHttpClientWarnings:
    """Verify SDK drift is observable via logger.warning."""

    def test_warns_when_manager_missing(self, caplog):
        """When _manager is None, a warning must be logged."""
        provider = HonchoMemoryProvider.__new__(HonchoMemoryProvider)
        provider._manager = None

        with caplog.at_level(logging.WARNING):
            provider._close_honcho_http_client()

        assert any("could not locate Honcho SDK client" in r.message for r in caplog.records), (
            "Must warn when Honcho SDK client can't be found"
        )

    def test_warns_when_honcho_attr_missing(self, caplog):
        """When _manager._honcho is missing, a warning must be logged."""
        provider = HonchoMemoryProvider.__new__(HonchoMemoryProvider)
        mock_manager = MagicMock()
        del mock_manager._honcho  # Make getattr return a spec'd mock, not None
        provider._manager = mock_manager

        # Actually we need _honcho to be None
        mock_manager._honcho = None
        provider._manager = mock_manager

        with caplog.at_level(logging.WARNING):
            provider._close_honcho_http_client()

        assert any("could not locate Honcho SDK client" in r.message for r in caplog.records), (
            "Must warn when _manager._honcho is None"
        )

    def test_no_warning_when_client_found(self, caplog):
        """When the client path is valid, no drift warning is logged."""
        provider = HonchoMemoryProvider.__new__(HonchoMemoryProvider)
        mock_http = MagicMock()
        mock_http.close = MagicMock()
        mock_honcho = MagicMock()
        mock_honcho._http = mock_http
        mock_honcho._async_http = None
        mock_manager = MagicMock()
        mock_manager._honcho = mock_honcho
        provider._manager = mock_manager

        with caplog.at_level(logging.WARNING):
            provider._close_honcho_http_client()

        assert not any("could not locate" in r.message for r in caplog.records), (
            "Should not warn when client is found"
        )
        mock_http.close.assert_called_once()


def _bare_manager():
    """A HonchoSessionManager with just the shutdown/prefetch plumbing."""
    from plugins.memory.honcho.session import HonchoSessionManager

    manager = HonchoSessionManager.__new__(HonchoSessionManager)
    manager._closed = False
    manager._prefetch_threads_lock = threading.Lock()
    manager._context_prefetch_threads = []
    manager._async_queue = None
    manager._async_thread = None
    return manager


class TestShutdownPrefetchRace:
    """shutdown() vs prefetch_context() — no thread may start after close."""

    def test_prefetch_after_shutdown_is_noop(self):
        """A prefetch attempted after shutdown must not start or track a thread."""
        manager = _bare_manager()
        manager.shutdown()

        with patch.object(threading.Thread, "start") as mock_start:
            manager.prefetch_context("key")

        mock_start.assert_not_called()
        assert manager._context_prefetch_threads == []

    def test_shutdown_interleaved_mid_prefetch_blocks_registration(self):
        """Deterministic TOCTOU: shutdown() lands between prefetch_context()'s
        fast closed-check and its registration. The lock-guarded re-check must
        prevent the thread from being started or tracked."""
        from plugins.memory.honcho import session as session_mod

        manager = _bare_manager()
        created: list[threading.Thread] = []
        real_thread = threading.Thread

        def racing_thread(*args, **kwargs):
            # Fires after prefetch_context() has already passed its fast
            # `if self._closed` check — the exact TOCTOU window.
            manager.shutdown()
            t = real_thread(*args, **kwargs)
            created.append(t)
            return t

        with patch.object(session_mod.threading, "Thread", side_effect=racing_thread):
            manager.prefetch_context("key")

        assert manager._closed is True
        assert created, "prefetch_context should have constructed its thread"
        assert created[0].ident is None, (
            "prefetch thread must never be started once shutdown() has run"
        )
        assert manager._context_prefetch_threads == [], (
            "prefetch thread must not be tracked after shutdown()'s snapshot"
        )

    def test_prefetch_before_shutdown_lands_in_snapshot(self):
        """A prefetch registered before shutdown() must be joined by it."""
        manager = _bare_manager()
        release = threading.Event()
        # Instance-level stubs: keep the worker alive until released so the
        # thread is still running when shutdown() takes its snapshot.
        manager.get_prefetch_context = lambda *a, **k: release.wait(timeout=5) and None
        manager.set_context_result = lambda *a, **k: None

        manager.prefetch_context("key")
        assert len(manager._context_prefetch_threads) == 1
        worker = manager._context_prefetch_threads[0]
        release.set()
        manager.shutdown()

        assert not worker.is_alive(), "shutdown() must join tracked prefetch threads"

    def test_shutdown_is_idempotent(self):
        """A second shutdown() must return early without re-joining."""
        manager = _bare_manager()
        manager.shutdown()
        assert manager._closed is True
        manager.shutdown()  # must not raise or deadlock
        assert manager._closed is True


class TestShutdownJoinPrefetchThreads:
    """Verify shutdown actually joins prefetch threads."""

    def test_shutdown_joins_tracked_thread(self):
        """A thread in _context_prefetch_threads must be joined during shutdown."""
        from plugins.memory.honcho.session import HonchoSessionManager

        manager = HonchoSessionManager.__new__(HonchoSessionManager)
        manager._closed = False
        manager._prefetch_threads_lock = threading.Lock()
        manager._context_prefetch_threads = []
        manager._async_queue = None
        manager._async_thread = None

        # Track a fake completed thread
        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = False
        manager._context_prefetch_threads = [fake_thread]

        manager.shutdown()

        # Thread was checked
        fake_thread.is_alive.assert_called()
