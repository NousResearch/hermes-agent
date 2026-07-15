"""Session /model overrides must attach credential_pool for 402 rotation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gateway.run import GatewayRunner, _credential_pool_for_provider


def test_session_override_re_resolves_live_credentials_and_pool(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "kimi-k2.7",
            "provider": "custom:hyper",
            "api_key": "sk-stale",
            "base_url": "https://hyper.charm.land/v1",
            "api_mode": "chat_completions",
            "max_tokens": 8192,
        },
    }
    fake_pool = object()
    calls = []

    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda _uc=None: "default-model",
    )
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda provider, *, target_model=None: (
            calls.append((provider, target_model))
            or {
                "provider": provider,
                "api_key": "sk-fresh",
                "base_url": "https://live.example/v1",
                "api_mode": "responses",
                "credential_pool": fake_pool,
            }
        ),
    )

    model, runtime = runner._resolve_session_agent_runtime(session_key="sess-1")

    assert model == "kimi-k2.7"
    assert calls == [("custom:hyper", "kimi-k2.7")]
    assert runtime["api_key"] == "sk-fresh"
    assert runtime["base_url"] == "https://live.example/v1"
    assert runtime["api_mode"] == "responses"
    assert runtime["credential_pool"] is fake_pool
    assert runtime["max_tokens"] == 8192


def test_session_override_token_change_busts_signature_but_stable_token_does_not(
    monkeypatch,
):
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "gpt-5.6-sol",
            "provider": "openai-codex",
            "api_key": "cached-token-must-not-win",
        },
    }
    live = {"token": "live-token-a"}

    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda _uc=None: "default-model",
    )

    def resolve_live(provider, *, target_model=None):
        assert provider == "openai-codex"
        assert target_model == "gpt-5.6-sol"
        return {
            "provider": provider,
            "api_key": live["token"],
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "credential_pool": None,
        }

    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        resolve_live,
    )

    model_a, runtime_a = runner._resolve_session_agent_runtime(session_key="sess-1")
    model_b, runtime_b = runner._resolve_session_agent_runtime(session_key="sess-1")
    sig_a = runner._agent_config_signature(model_a, runtime_a, [], "")
    sig_b = runner._agent_config_signature(model_b, runtime_b, [], "")
    assert sig_a == sig_b

    live["token"] = "live-token-b"
    model_c, runtime_c = runner._resolve_session_agent_runtime(session_key="sess-1")
    sig_c = runner._agent_config_signature(model_c, runtime_c, [], "")
    assert runtime_c["api_key"] == "live-token-b"
    assert sig_c != sig_b


def test_session_override_resolution_failure_never_uses_cached_token(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "gpt-5.6-sol",
            "provider": "openai-codex",
            "api_key": "stale-token",
        },
    }
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda _uc=None: "default-model",
    )

    def fail_live_resolution(provider, *, target_model=None):
        raise RuntimeError("live credentials unavailable")

    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        fail_live_resolution,
    )

    with pytest.raises(RuntimeError, match="live credentials unavailable"):
        runner._resolve_session_agent_runtime(session_key="sess-1")


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
