"""Secret-safe runtime provider authority probes.

These tests use synthetic values and isolated homes only.  They never read the
operator's .env, auth.json, or credential store.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from agent import secret_scope
from hermes_cli import auth
from hermes_cli.runtime_provider import resolve_runtime_provider


@contextmanager
def _multiplex_scope(values):
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope(values)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_azure_deepseek_profile_route_uses_scoped_configured_credential(monkeypatch):
    provider_id = "azure-deepseek"
    monkeypatch.setitem(
        auth.PROVIDER_REGISTRY,
        provider_id,
        auth.ProviderConfig(
            id=provider_id,
            name="Azure DeepSeek fixture",
            auth_type="api_key",
            inference_base_url="https://fixture.openai.azure.com/openai/v1",
            api_key_env_vars=("AZURE_DEEPSEEK_API_KEY",),
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {
            "provider": provider_id,
            "default": "DeepSeek-V4-Flash-0731",
            "base_url": "https://fixture.openai.azure.com/openai/v1",
        },
    )
    monkeypatch.setattr(auth, "_model_level_key_env", lambda _provider: "")
    monkeypatch.setattr(auth, "_load_auth_store", lambda: {})

    with _multiplex_scope({"AZURE_DEEPSEEK_API_KEY": "fixture-azure-key"}):
        runtime = resolve_runtime_provider(
            requested=provider_id, target_model="DeepSeek-V4-Flash-0731")

    assert runtime["provider"] == provider_id
    assert runtime["base_url"] == "https://fixture.openai.azure.com/openai/v1"
    assert runtime["api_key"] == "fixture-azure-key"
    assert runtime["source"] in {"AZURE_DEEPSEEK_API_KEY", "env:AZURE_DEEPSEEK_API_KEY"}


def test_openrouter_route_uses_scoped_key_without_cross_profile_fallback(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"provider": "openrouter", "default": "fixture/model"},
    )
    monkeypatch.setattr(auth, "_load_auth_store", lambda: {})
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _provider: None)

    with _multiplex_scope({"OPENROUTER_API_KEY": "sk-or-v1-fixture"}):
        runtime = resolve_runtime_provider(requested="openrouter", target_model="fixture/model")

    assert runtime["provider"] == "openrouter"
    assert runtime["base_url"] == "https://openrouter.ai/api/v1"
    assert runtime["api_key"] == "sk-or-v1-fixture"
    assert runtime["source"] == "env:OPENROUTER_API_KEY"


def test_unscoped_secret_read_fails_closed_in_multiplex_mode():
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            secret_scope.get_secret_str("AZURE_DEEPSEEK_API_KEY")
    finally:
        secret_scope.set_multiplex_active(False)
