"""Tests for service-singleton lifecycle: atexit handler, idempotent shutdown.

These cover the exit-cleanup behavior added to plug the language-server
process leak — without the atexit hook, ``hermes chat`` exits while
pyright/gopls/etc. are still alive on the host.
"""
from __future__ import annotations

import atexit
from unittest.mock import MagicMock, patch

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


def test_atexit_shutdown_calls_shutdown_service(monkeypatch):
    """The atexit-registered wrapper invokes ``shutdown_service`` and
    swallows any exception — by the time atexit fires, the user has
    already seen the response and a noisy traceback would be clutter."""
    called = []
    monkeypatch.setattr(
        lsp_module, "shutdown_service", lambda: called.append("shutdown")
    )
    lsp_module._atexit_shutdown()
    assert called == ["shutdown"]


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
    fake_svc.shutdown = MagicMock()
    monkeypatch.setattr(
        lsp_module.LSPService, "create_from_config", classmethod(lambda cls: fake_svc)
    )
    monkeypatch.setattr(atexit, "register", lambda fn: None)

    lsp_module.get_service()
    lsp_module.shutdown_service()
    lsp_module.shutdown_service()  # must not raise

    assert fake_svc.shutdown.call_count == 1


def test_shutdown_service_no_op_when_never_started():
    """Calling shutdown without ever creating the service is safe."""
    lsp_module.shutdown_service()  # must not raise


def test_shutdown_service_swallows_exception(monkeypatch):
    """An exception during ``svc.shutdown()`` must not propagate —
    the caller (often atexit) has nothing useful to do with it."""
    fake_svc = MagicMock()
    fake_svc.is_active.return_value = True
    fake_svc.shutdown = MagicMock(side_effect=RuntimeError("kill -9 already"))
    monkeypatch.setattr(
        lsp_module.LSPService, "create_from_config", classmethod(lambda cls: fake_svc)
    )
    monkeypatch.setattr(atexit, "register", lambda fn: None)

    lsp_module.get_service()
    lsp_module.shutdown_service()  # must not raise


def test_get_service_returns_none_for_inactive_service(monkeypatch):
    """A service whose ``is_active()`` returns False is treated as
    not running — callers see ``None`` and fall back."""
    fake_svc = MagicMock()
    fake_svc.is_active.return_value = False
    monkeypatch.setattr(
        lsp_module.LSPService, "create_from_config", classmethod(lambda cls: fake_svc)
    )
    monkeypatch.setattr(atexit, "register", lambda fn: None)

    assert lsp_module.get_service() is None
    # Subsequent call returns None too — but the inactive instance is
    # cached so we don't re-build it on every check.
    assert lsp_module.get_service() is None


def test_get_service_returns_none_when_create_fails(monkeypatch):
    """Service factory returning ``None`` (no config, etc.) propagates."""
    monkeypatch.setattr(
        lsp_module.LSPService, "create_from_config", classmethod(lambda cls: None)
    )
    monkeypatch.setattr(atexit, "register", lambda fn: None)

    assert lsp_module.get_service() is None
