"""Provider shutdown joins every plugin thread within one budget (#37632, #33485, #60616)."""

import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho import session as session_module
from plugins.memory.honcho.client import (
    HonchoClientConfig,
    close_honcho_clients,
    join_plugin_threads,
    spawn_context_thread,
)
from plugins.memory.honcho.client_cache import _client_slots, _client_slots_lock
from plugins.memory.honcho.session import HonchoSessionManager
from plugins.plugin_utils import SingletonSlot


@pytest.fixture
def async_manager(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(session_module, "get_honcho_client", lambda *a, **k: fake)
    cfg = HonchoClientConfig(write_frequency="async", api_key="test-key", enabled=True)
    mgr = HonchoSessionManager(honcho=fake, config=cfg)
    mgr.fake_client = fake
    yield mgr
    mgr.shutdown()


class _Owner:
    pass


class TestThreadRegistry:
    def test_join_covers_only_the_given_owners(self):
        mine, theirs = _Owner(), _Owner()
        release = threading.Event()
        own = spawn_context_thread(lambda: release.wait(timeout=2), name="own", owner=mine)
        other = spawn_context_thread(lambda: release.wait(timeout=2), name="other", owner=theirs)
        own.start()
        other.start()
        try:
            started = time.monotonic()
            assert join_plugin_threads((theirs, None), timeout=0.05) == ["other"]
            release.set()
            assert join_plugin_threads((mine,), timeout=2) == []
            assert time.monotonic() - started < 1.5
        finally:
            release.set()
            own.join(timeout=1)
            other.join(timeout=1)


class TestProviderShutdown:
    def _provider(self, manager, cfg=None):
        provider = HonchoMemoryProvider()
        provider._manager = manager
        provider._config = cfg or manager._config
        provider._session_key = "test-session"
        provider._session_initialized = True
        return provider

    def _shutdown_blocks_until(self, provider, release):
        """shutdown() off-thread must still be running until ``release`` is set, then finish."""
        done = threading.Event()
        threading.Thread(target=lambda: (provider.shutdown(), done.set()), daemon=True).start()
        try:
            assert not done.wait(timeout=0.05)
            release.set()
            assert done.wait(timeout=2)
        finally:
            release.set()

    def test_shutdown_waits_for_an_in_flight_context_prefetch(self, async_manager):
        provider = self._provider(async_manager)
        started, release = threading.Event(), threading.Event()

        def slow_prefetch(session_key, user_message=None):
            started.set()
            release.wait(timeout=2)
            return {"representation": "ready"}

        async_manager.get_prefetch_context = slow_prefetch
        async_manager.prefetch_context("test-session", "query")
        assert started.wait(timeout=1)
        self._shutdown_blocks_until(provider, release)
        assert not any(t.name == "honcho-context-prefetch" and t.is_alive() for t in threading.enumerate())

    def test_shutdown_joins_a_stalled_init_thread_within_the_budget(self, monkeypatch, caplog):
        monkeypatch.setattr(HonchoMemoryProvider, "_SHUTDOWN_JOIN_FLOOR", 0.2)
        cfg = HonchoClientConfig(api_key="test-key", enabled=True, timeout=0.1)
        release = threading.Event()
        entered = threading.Event()

        class StalledManager:
            def __init__(self, *args, **kwargs):
                pass

            def get_or_create(self, session_key):
                entered.set()
                release.wait(timeout=5)
                return SimpleNamespace(messages=[])

            def migrate_memory_files(self, *a, **k):
                pass

        monkeypatch.setattr("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", lambda: cfg)
        monkeypatch.setattr("plugins.memory.honcho.client.get_honcho_client", lambda cfg: object())
        monkeypatch.setattr("plugins.memory.honcho.session.HonchoSessionManager", StalledManager)
        provider = HonchoMemoryProvider()
        provider.initialize("session-1", platform="cli")
        try:
            assert entered.wait(timeout=1)
            started = time.monotonic()
            with caplog.at_level(logging.WARNING, logger="plugins.memory.honcho"):
                provider.shutdown()
            assert 0.15 <= time.monotonic() - started < 1.0
            assert "honcho-session-init" in caplog.text
            assert "timed out after 0.2s" in caplog.text
        finally:
            release.set()
            provider._init_thread.join(timeout=2)

    def test_shutdown_joins_the_prewarm_dialectic_thread(self, async_manager):
        provider = self._provider(async_manager)
        release = threading.Event()
        async_manager.dialectic_query = lambda *a, **k: release.wait(timeout=2) and "answer"
        provider._spawn_dialectic("who?", thread_name="honcho-prewarm-dialectic", fired_at=0, log_label="prewarm")
        self._shutdown_blocks_until(provider, release)
        assert not provider._prefetch_thread.is_alive()

    def test_shutdown_does_not_close_the_shared_client(self, async_manager):
        provider = self._provider(async_manager)
        provider.shutdown()
        async_manager.fake_client._http.close.assert_not_called()


@pytest.mark.parametrize("timeout, budget", [(60.0, 60.0), (2.0, 5.0), (None, 5.0)])
def test_shutdown_join_budget_is_the_floor_or_the_longer_timeout(timeout, budget):
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(api_key="k", timeout=timeout) if timeout else SimpleNamespace()
    assert provider._shutdown_join_budget() == budget


class TestManagerAfterShutdown:
    def test_prefetch_is_skipped_after_shutdown(self, async_manager):
        calls = []
        async_manager.get_prefetch_context = lambda *a, **k: calls.append(a) or {}
        async_manager.shutdown()
        async_manager.prefetch_context("test-session", "query")
        time.sleep(0.05)
        assert calls == []


def test_close_honcho_clients_closes_every_pool_and_drops_the_slots():
    client = MagicMock()
    slot = SingletonSlot()
    slot.get(lambda: client)
    key = ("test", "close-me")
    with _client_slots_lock:
        _client_slots[key] = slot
    try:
        close_honcho_clients()
    finally:
        with _client_slots_lock:
            _client_slots.pop(key, None)
    client._http.close.assert_called_once_with()
    assert key not in _client_slots
