"""Tests for service-singleton lifecycle: atexit handler, idempotent shutdown.

These cover the exit-cleanup behavior added to plug the language-server
process leak — without the atexit hook, ``hermes chat`` exits while
pyright/gopls/etc. are still alive on the host.
"""
from __future__ import annotations

import atexit
import threading
from unittest.mock import MagicMock

import pytest

from agent import lsp as lsp_module


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Force a clean module state before each test.

    Tests in this file share process-global state (the lazy
    singleton + atexit registration flag); reset both before and
    after every test so order doesn't matter.
    """
    lsp_module._service = None
    lsp_module._atexit_registered = False
    yield
    lsp_module._service = None
    lsp_module._atexit_registered = False


def test_get_service_registers_atexit_handler_once(monkeypatch):
    """First call to ``get_service`` must register an atexit handler;
    subsequent calls must NOT register another one (Python's ``atexit``
    runs every registered callable, so a duplicate would shutdown
    twice — harmless but wasteful)."""
    fake_svc = MagicMock()
    fake_svc.is_active.return_value = True
    monkeypatch.setattr(
        lsp_module.LSPService, "create_from_config", classmethod(lambda cls: fake_svc)
    )

    registrations = []

    def fake_register(fn):
        registrations.append(fn)

    monkeypatch.setattr(atexit, "register", fake_register)

    a = lsp_module.get_service()
    b = lsp_module.get_service()
    c = lsp_module.get_service()

    assert a is fake_svc
    assert b is fake_svc
    assert c is fake_svc
    assert len(registrations) == 1
    # The registered callable must be our internal shutdown wrapper.
    assert registrations[0] is lsp_module._atexit_shutdown




def test_atexit_shutdown_swallows_exceptions(monkeypatch):
    def boom():
        raise RuntimeError("server already dead")

    monkeypatch.setattr(lsp_module, "shutdown_service", boom)
    # Must not raise.
    lsp_module._atexit_shutdown()


def test_shutdown_service_idempotent(monkeypatch):
    """Calling shutdown twice must be safe — first call cleans up,
    second call no-ops (nothing to shut down)."""
    fake_svc = MagicMock()
    fake_svc.is_active.return_value = True
    fake_svc.shutdown = MagicMock(return_value=True)
    monkeypatch.setattr(
        lsp_module.LSPService, "create_from_config", classmethod(lambda cls: fake_svc)
    )
    monkeypatch.setattr(atexit, "register", lambda fn: None)

    lsp_module.get_service()
    lsp_module.shutdown_service()
    lsp_module.shutdown_service()  # must not raise

    assert fake_svc.shutdown.call_count == 1


def test_shutdown_fences_concurrent_get_service_until_teardown_finishes(
    monkeypatch,
):
    shutdown_started = threading.Event()
    cleanup_complete = threading.Event()
    replacement_created = threading.Event()

    class TrackingLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.waiter_attempted = threading.Event()

        def __enter__(self):
            if self._lock.locked():
                self.waiter_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            self._lock.release()

    service_lock = TrackingLock()
    monkeypatch.setattr(lsp_module, "_service_lock", service_lock)

    old_service = MagicMock()
    old_service.is_active.return_value = True

    def shutdown():
        shutdown_started.set()
        assert service_lock.waiter_attempted.wait(timeout=2.0)
        assert not replacement_created.is_set()
        cleanup_complete.set()
        return True

    old_service.shutdown.side_effect = shutdown
    replacement = MagicMock()
    replacement.is_active.return_value = True

    def create_replacement(cls):
        assert cleanup_complete.is_set()
        replacement_created.set()
        return replacement

    monkeypatch.setattr(
        lsp_module.LSPService,
        "create_from_config",
        classmethod(create_replacement),
    )
    monkeypatch.setattr(atexit, "register", lambda fn: None)
    lsp_module._service = old_service

    shutdown_result = []
    get_result = []
    shutdown_thread = threading.Thread(
        target=lambda: shutdown_result.append(lsp_module.shutdown_service())
    )

    def get_concurrently():
        get_result.append(lsp_module.get_service())

    get_thread = threading.Thread(target=get_concurrently)
    shutdown_thread.start()
    assert shutdown_started.wait(timeout=2.0)
    get_thread.start()
    shutdown_thread.join(timeout=2.0)
    get_thread.join(timeout=2.0)

    assert not shutdown_thread.is_alive()
    assert not get_thread.is_alive()
    assert shutdown_result == [True]
    assert get_result == [replacement]
    assert replacement_created.is_set()


def test_failed_shutdown_leaves_tombstone_and_refuses_replacement(monkeypatch):
    failed_service = MagicMock()
    failed_service.is_active.return_value = True
    failed_service.shutdown.return_value = False
    failed_service._get_shutdown_error.return_value = "cleanup blocked"
    lsp_module._service = failed_service

    replacement = MagicMock()
    replacement.is_active.return_value = True
    create = MagicMock(return_value=replacement)
    monkeypatch.setattr(
        lsp_module.LSPService,
        "create_from_config",
        classmethod(lambda cls: create()),
    )

    assert lsp_module.shutdown_service() is False
    assert isinstance(lsp_module._service, lsp_module._ServiceTombstone)
    assert lsp_module.get_service() is None
    assert create.call_count == 0

    failed_service.shutdown.return_value = True
    assert lsp_module.shutdown_service() is True
    assert lsp_module._service is None
    assert lsp_module.get_service() is replacement
    assert create.call_count == 1


def test_singleton_tombstone_survives_loop_stop_failure():
    svc = lsp_module.LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=0,
    )
    real_stop = svc._loop.stop
    stop_calls = 0

    def flaky_stop():
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls == 1:
            return False
        return real_stop()

    svc._loop.stop = flaky_stop  # type: ignore[method-assign]
    lsp_module._service = svc
    try:
        assert lsp_module.shutdown_service() is False
        assert isinstance(lsp_module._service, lsp_module._ServiceTombstone)
        assert svc._loop._thread is not None
        assert svc._loop._thread.is_alive()
        assert lsp_module.get_service() is None

        assert lsp_module.shutdown_service() is True
        assert lsp_module._service is None
        assert stop_calls == 2
        assert svc._loop_stopped is True
    finally:
        real_stop()



