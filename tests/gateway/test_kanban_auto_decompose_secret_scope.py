"""Regression tests: the gateway auto-decomposer binds its own profile secrets.

``_auto_decompose_tick`` calls the auxiliary LLM from a watcher thread, outside
any per-turn ``_profile_runtime_scope``. With ``gateway.multiplex_profiles``
on, ``get_secret`` fails closed when no scope is installed, so every tick
logged ``no Anthropic credentials found`` and ended in ``UnscopedSecretError``
on ``OPENAI_API_KEY``. ``_gateway_aux_secret_scope`` installs the gateway
home's secret mapping for the duration of the call.
"""

from __future__ import annotations

import pytest

from agent import secret_scope
from gateway import kanban_watchers as kw


@pytest.fixture
def _clean_scope(monkeypatch):
    """Start every test unscoped and with multiplexing off; restore after."""
    prev_mux = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    yield
    secret_scope.reset_secret_scope(token)
    secret_scope.set_multiplex_active(prev_mux)


def _fake_home(monkeypatch, tmp_path, env_text: str):
    (tmp_path / ".env").write_text(env_text, encoding="utf-8")
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def test_scope_installs_gateway_home_secrets_when_multiplexing(
    _clean_scope, monkeypatch, tmp_path
):
    _fake_home(monkeypatch, tmp_path, "OPENAI_API_KEY=sk-test-gateway\n")
    secret_scope.set_multiplex_active(True)

    with pytest.raises(secret_scope.UnscopedSecretError):
        secret_scope.get_secret("OPENAI_API_KEY")

    with kw._gateway_aux_secret_scope():
        assert secret_scope.get_secret("OPENAI_API_KEY") == "sk-test-gateway"
        assert secret_scope.current_secret_scope() is not None

    # Previous (unscoped) state restored on exit.
    assert secret_scope.current_secret_scope() is None
    with pytest.raises(secret_scope.UnscopedSecretError):
        secret_scope.get_secret("OPENAI_API_KEY")


def test_scope_resets_on_exception(_clean_scope, monkeypatch, tmp_path):
    _fake_home(monkeypatch, tmp_path, "OPENAI_API_KEY=x\n")
    secret_scope.set_multiplex_active(True)

    with pytest.raises(RuntimeError, match="boom"):
        with kw._gateway_aux_secret_scope():
            assert secret_scope.current_secret_scope() is not None
            raise RuntimeError("boom")
    assert secret_scope.current_secret_scope() is None


def test_scope_restores_enclosing_scope_when_nested(_clean_scope, monkeypatch, tmp_path):
    _fake_home(monkeypatch, tmp_path, "OPENAI_API_KEY=inner\n")
    secret_scope.set_multiplex_active(True)
    outer = {"OPENAI_API_KEY": "outer"}
    tok = secret_scope.set_secret_scope(outer)
    try:
        with kw._gateway_aux_secret_scope():
            assert secret_scope.get_secret("OPENAI_API_KEY") == "inner"
        assert secret_scope.current_secret_scope() is outer
        assert secret_scope.get_secret("OPENAI_API_KEY") == "outer"
    finally:
        secret_scope.reset_secret_scope(tok)


def test_scope_is_noop_when_multiplexing_off(_clean_scope, monkeypatch, tmp_path):
    _fake_home(monkeypatch, tmp_path, "OPENAI_API_KEY=should-not-load\n")
    secret_scope.set_multiplex_active(False)
    calls = []
    monkeypatch.setattr(
        secret_scope, "build_profile_secret_scope",
        lambda home: calls.append(home) or {},
    )
    monkeypatch.setattr(
        secret_scope, "set_secret_scope",
        lambda s: calls.append(("set", s)) or secret_scope._SECRET_SCOPE.set(s),
    )
    with kw._gateway_aux_secret_scope():
        assert secret_scope.current_secret_scope() is None
    assert calls == []


def test_scope_survives_builder_failure(_clean_scope, monkeypatch, tmp_path):
    """A broken .env must not take the dispatcher down — degrade to unscoped."""
    _fake_home(monkeypatch, tmp_path, "")
    secret_scope.set_multiplex_active(True)

    def _boom(home):
        raise OSError("unreadable .env")

    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", _boom)
    with kw._gateway_aux_secret_scope():
        assert secret_scope.current_secret_scope() is None
    assert secret_scope.current_secret_scope() is None
