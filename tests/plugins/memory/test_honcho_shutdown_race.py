"""Tests for HonchoSessionManager.shutdown() race condition fix.

The shutdown() method sets _shutting_down before flush_all() so concurrent
save() calls drop new items instead of racing the shutdown sentinel.
"""
import queue
import threading
from unittest.mock import MagicMock, patch

import pytest


def _make_manager(write_frequency="async"):
    """Create a HonchoSessionManager with minimal init."""
    from plugins.memory.honcho.session import HonchoSessionManager
    config = MagicMock()
    config.write_frequency = write_frequency
    config.dialectic_reasoning_level = "low"
    config.dialectic_dynamic = True
    config.dialectic_max_chars = 600
    config.observation_mode = "directional"
    config.user_observe_me = True
    config.user_observe_others = True
    config.ai_observe_me = True
    config.ai_observe_others = True
    config.message_max_chars = 25000
    config.dialectic_max_input_chars = 10000
    manager = HonchoSessionManager(config=config)
    return manager


class TestShutdownRace:
    def test_save_dropped_after_shutdown_flag(self):
        """save() must be a no-op after _shutting_down is set."""
        manager = _make_manager()
        session = MagicMock()
        session.key = "test:1"
        session.messages = [{"role": "user", "content": "hi", "_synced": False}]
        session.user_peer_id = "user"
        session.assistant_peer_id = "ai"
        session.honcho_session_id = "test:1"

        manager._shutting_down = True
        with patch.object(manager, "_flush_session") as mock_flush:
            manager.save(session)
            mock_flush.assert_not_called()

    def test_shutdown_sets_flag_before_flush(self):
        """shutdown() must set _shutting_down before calling flush_all()."""
        manager = _make_manager()
        flag_was_set_during_flush = []

        original_flush_all = manager.flush_all

        def spy_flush_all():
            flag_was_set_during_flush.append(manager._shutting_down)
            original_flush_all()

        manager.flush_all = spy_flush_all
        manager._async_queue = queue.Queue()
        manager._async_thread = MagicMock()
        manager._async_thread.join = MagicMock()

        manager.shutdown()

        assert flag_was_set_during_flush == [True]
        assert manager._shutting_down is True

    def test_concurrent_save_and_shutdown_no_race(self):
        """Concurrent save() + shutdown() must not leave orphaned items."""
        manager = _make_manager()
        manager._async_queue = queue.Queue()

        session = MagicMock()
        session.key = "test:1"
        session.messages = []
        session.user_peer_id = "user"
        session.assistant_peer_id = "ai"
        session.honcho_session_id = "test:1"

        enqueued_after_shutdown = []
        original_put = manager._async_queue.put

        def counting_put(item, *args, **kwargs):
            if manager._shutting_down:
                enqueued_after_shutdown.append(item)
            original_put(item, *args, **kwargs)

        manager._async_queue.put = counting_put

        manager._async_thread = MagicMock()
        manager._async_thread.join = MagicMock()

        for _ in range(10):
            manager.save(session)
        manager.shutdown()

        assert len(enqueued_after_shutdown) == 0

    def test_shutdown_idempotent(self):
        """Calling shutdown() twice does not crash."""
        manager = _make_manager()
        manager._async_queue = queue.Queue()
        manager._async_thread = MagicMock()
        manager._async_thread.join = MagicMock()

        manager.shutdown()
        manager.shutdown()
        assert manager._shutting_down is True
