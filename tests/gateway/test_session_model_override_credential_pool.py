"""Session /model overrides must attach credential_pool for 402 rotation."""

from __future__ import annotations

from unittest.mock import MagicMock

from gateway.run import GatewayRunner, _credential_pool_for_provider


def test_fast_session_override_includes_credential_pool(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "kimi-k2.7",
            "provider": "custom:hyper",
            "api_key": "sk-test",
            "base_url": "https://hyper.charm.land/v1",
            "api_mode": "chat_completions",
        },
    }
    fake_pool = object()

    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda _uc=None: "default-model",
    )
    monkeypatch.setattr(
        "gateway.run._credential_pool_for_provider",
        lambda provider: fake_pool if provider == "custom:hyper" else None,
    )

    model, runtime = runner._resolve_session_agent_runtime(session_key="sess-1")

    assert model == "kimi-k2.7"
    assert runtime.get("credential_pool") is fake_pool


def test_apply_session_override_backfills_credential_pool(monkeypatch):
    runner = object.__new__(GatewayRunner)
    fake_pool = MagicMock(name="pool")
    runner._session_model_overrides = {
        "sess-2": {
            "model": "kimi-k2.7",
            "provider": "custom:hyper",
            "api_key": "sk-test",
        },
    }
    monkeypatch.setattr(
        "gateway.run._credential_pool_for_provider",
        lambda provider: fake_pool,
    )

    model, runtime = runner._apply_session_model_override(
        "sess-2",
        "default-model",
        {"api_key": "old", "provider": "x"},
    )

    assert model == "kimi-k2.7"
    assert runtime["credential_pool"] is fake_pool


def test_credential_pool_for_provider_delegates(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda p: {"credential_pool": sentinel, "provider": p},
    )
    assert _credential_pool_for_provider("custom:hyper") is sentinel