"""Tests for service-singleton lifecycle: atexit handler, idempotent shutdown.

These cover the exit-cleanup behavior added to plug the language-server
process leak — without the atexit hook, ``hermes chat`` exits while
pyright/gopls/etc. are still alive on the host.
"""
from __future__ import annotations

import atexit
import sys
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
    fake_svc.shutdown = MagicMock()
    monkeypatch.setattr(
        lsp_module.LSPService, "create_from_config", classmethod(lambda cls: fake_svc)
    )
    monkeypatch.setattr(atexit, "register", lambda fn: None)

    lsp_module.get_service()
    lsp_module.shutdown_service()
    lsp_module.shutdown_service()  # must not raise

    assert fake_svc.shutdown.call_count == 1


def test_lsp_spawn_env_excludes_tier1_and_provider_secrets(monkeypatch, tmp_path):
    """#77463: LSP servers (third-party pyright/gopls/...) must not receive
    Hermes' Tier-1 secrets (gateway tokens) OR provider API keys.

    E2E with a REAL child: seed both a Tier-1 key and a provider key in the
    parent, build the env exactly as the fixed LSPClient._spawn does
    (hermes_subprocess_env(inherit_credentials=False) + self._env), spawn a
    real Python child that reports which keys it sees in ITS OWN environment,
    and assert the secrets are absent while the LSP's own env additions
    survive.
    """
    import json as _json
    import subprocess as _sp

    from agent.lsp.client import LSPClient

    monkeypatch.setenv("GATEWAY_RELAY_SECRET", "«redacted:tier1-secret»")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "«redacted:provider-key»")

    client = LSPClient.__new__(LSPClient)
    client._command = [sys.executable, "-c", "pass"]
    client._env = {"LSP_CUSTOM_OPT": "kept-value"}

    # The fixed _spawn builds the env via hermes_subprocess_env then layers
    # self._env; replicate that construction and verify it in a real child.
    # The construction is the contract under test.
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=False)
    env.update(client._env)

    probe = (
        "import json, os; print(json.dumps({"
        "'relay': 'GATEWAY_RELAY_SECRET' in os.environ, "
        "'provider': 'ANTHROPIC_API_KEY' in os.environ, "
        "'custom': os.environ.get('LSP_CUSTOM_OPT', '')}))"
    )
    out = _sp.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result = _json.loads(out.stdout.strip().splitlines()[-1])
    assert result["relay"] is False, "Tier-1 relay secret leaked to LSP server"
    assert result["provider"] is False, "provider API key leaked to LSP server"
    assert result["custom"] == "kept-value", "LSP's own env addition must survive"








