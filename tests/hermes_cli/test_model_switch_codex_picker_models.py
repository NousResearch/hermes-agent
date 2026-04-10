"""Tests for Codex model discovery in shared /model provider listings."""

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.models import _PROVIDER_MODELS


def test_list_authenticated_providers_uses_dynamic_codex_models(monkeypatch):
    """Gateway/Telegram /model should reuse the richer Codex discovery path."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        providers_mod,
        "HERMES_OVERLAYS",
        {"openai-codex": providers_mod.HERMES_OVERLAYS["openai-codex"]},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._load_auth_store",
        lambda: {"providers": {"openai-codex": {"tokens": {"access_token": "abc"}}}, "credential_pool": {}},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"logged_in": True, "api_key": "codex-live-token"},
    )

    captured = {}

    def _fake_get_codex_model_ids(access_token=None):
        captured["access_token"] = access_token
        return ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.2", "gpt-5.3-codex-spark"]

    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids",
        _fake_get_codex_model_ids,
    )

    providers = list_authenticated_providers(current_provider="openai-codex", max_models=50)
    codex = next(p for p in providers if p["slug"] == "openai-codex")

    assert captured["access_token"] == "codex-live-token"
    assert codex["models"] == ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.2", "gpt-5.3-codex-spark"]
    assert codex["total_models"] == 5


def test_list_authenticated_providers_falls_back_to_curated_codex_models(monkeypatch):
    """If dynamic Codex discovery fails, keep the old curated picker list."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        providers_mod,
        "HERMES_OVERLAYS",
        {"openai-codex": providers_mod.HERMES_OVERLAYS["openai-codex"]},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._load_auth_store",
        lambda: {"providers": {"openai-codex": {"tokens": {"access_token": "abc"}}}, "credential_pool": {}},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"logged_in": True, "api_key": "codex-live-token"},
    )
    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids",
        lambda access_token=None: [],
    )

    providers = list_authenticated_providers(current_provider="openai-codex", max_models=50)
    codex = next(p for p in providers if p["slug"] == "openai-codex")

    assert codex["models"] == _PROVIDER_MODELS["openai-codex"]
    assert codex["total_models"] == len(_PROVIDER_MODELS["openai-codex"])
