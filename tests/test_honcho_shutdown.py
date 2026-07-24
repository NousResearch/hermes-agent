"""Regression tests for Honcho provider shutdown."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.session import HonchoSessionManager


def _async_config() -> SimpleNamespace:
    return SimpleNamespace(
        write_frequency="async",
        dialectic_reasoning_level="low",
        dialectic_dynamic=True,
        dialectic_max_chars=600,
        observation_mode="directional",
        user_observe_me=True,
        user_observe_others=True,
        ai_observe_me=True,
        ai_observe_others=True,
        message_max_chars=25000,
        dialectic_max_input_chars=10000,
    )


def test_provider_shutdown_stops_honcho_async_writer() -> None:
    """Provider shutdown must not leave its session writer at interpreter exit."""
    manager = HonchoSessionManager(config=_async_config())
    provider = HonchoMemoryProvider()
    provider._manager = manager

    assert manager._async_thread is not None
    assert manager._async_thread.is_alive()

    try:
        provider.shutdown()
        assert not manager._async_thread.is_alive()
    finally:
        # Keep the regression test itself leak-free against the pre-fix code.
        manager.shutdown()


def test_provider_shutdown_waits_for_context_prefetch() -> None:
    """CLI cleanup must not leave Honcho HTTP work at interpreter finalization."""
    manager = HonchoSessionManager(config=_async_config())
    provider = HonchoMemoryProvider()
    provider._manager = manager
    started = threading.Event()
    release = threading.Event()
    shutdown_done = threading.Event()

    def slow_prefetch(
        session_key: str, user_message: str | None = None
    ) -> dict[str, str]:
        started.set()
        release.wait(timeout=2)
        return {"representation": "ready"}

    manager.get_prefetch_context = slow_prefetch  # type: ignore[method-assign]
    manager.prefetch_context("session", "query")
    assert started.wait(timeout=1)

    def shut_down_provider() -> None:
        try:
            provider.shutdown()
        finally:
            shutdown_done.set()

    shutdown_thread = threading.Thread(target=shut_down_provider)
    shutdown_thread.start()
    try:
        assert not shutdown_done.wait(timeout=0.05)
        release.set()
        shutdown_thread.join(timeout=1)
        assert shutdown_done.is_set()
        assert not manager._context_prefetch_threads
    finally:
        release.set()
        shutdown_thread.join(timeout=2)
        manager.shutdown()


def test_context_prefetch_is_rejected_after_shutdown() -> None:
    manager = HonchoSessionManager(config=_async_config())
    calls = 0

    def record_prefetch(
        session_key: str, user_message: str | None = None
    ) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {}

    manager.get_prefetch_context = record_prefetch  # type: ignore[method-assign]
    manager.shutdown()
    manager.prefetch_context("session", "query")

    assert calls == 0
