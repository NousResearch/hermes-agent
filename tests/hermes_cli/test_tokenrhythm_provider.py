"""TokenRhythm provider wiring: identity, credentials, picker catalog and model capabilities.

TokenRhythm is an OpenAI-compatible Chat Completions gateway (no Responses API) with bare model
ids. It is absent from models.dev, so the profile's ``model_capabilities`` are what context
budgeting, image routing and the picker badges read.
"""

from __future__ import annotations

import pytest

from hermes_cli.auth import resolve_api_key_provider_credentials, resolve_provider
from hermes_cli.model_normalize import normalize_model_for_provider
from hermes_cli.models import normalize_provider, provider_model_ids

BASE_URL = "https://tokenrhythm.studio/v1"


@pytest.fixture
def profile():
    from providers import get_provider_profile

    prof = get_provider_profile("tokenrhythm")
    assert prof is not None, "tokenrhythm provider profile must be registered"
    return prof


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {"provider": provider_id, "api_key": "sk_live", "base_url": BASE_URL,
                             "source": "TOKENRHYTHM_API_KEY"},
    )


def test_aliases_resolve_to_one_identity(monkeypatch):
    from hermes_cli.providers import normalize_provider as normalize_in_providers

    monkeypatch.setenv("TOKENRHYTHM_API_KEY", "sk_test")
    for name in ("tokenrhythm", "token-rhythm", "TokenRhythm"):
        assert resolve_provider(name) == "tokenrhythm"
        assert normalize_provider(name) == "tokenrhythm"
        assert normalize_in_providers(name) == "tokenrhythm"


def test_env_key_and_default_endpoint(monkeypatch):
    monkeypatch.setenv("TOKENRHYTHM_API_KEY", "sk_secret")
    monkeypatch.delenv("TOKENRHYTHM_BASE_URL", raising=False)

    creds = resolve_api_key_provider_credentials("tokenrhythm")
    assert creds["provider"] == "tokenrhythm"
    assert creds["api_key"] == "sk_secret"
    assert creds["base_url"].rstrip("/") == BASE_URL


def test_base_url_env_override(monkeypatch):
    monkeypatch.setenv("TOKENRHYTHM_API_KEY", "sk_secret")
    monkeypatch.setenv("TOKENRHYTHM_BASE_URL", "https://relay.example/v1")

    assert resolve_api_key_provider_credentials("tokenrhythm")["base_url"] == "https://relay.example/v1"


def test_routes_over_chat_completions():
    from hermes_cli.providers import determine_api_mode, get_label, get_provider

    provider = get_provider("token-rhythm")
    assert provider is not None
    assert provider.id == "tokenrhythm"
    assert provider.base_url == BASE_URL
    assert "TOKENRHYTHM_API_KEY" in provider.api_key_env_vars
    assert get_label("tokenrhythm") == "TokenRhythm"
    assert determine_api_mode("tokenrhythm", BASE_URL) == "chat_completions"


def test_picker_leads_with_default_model_and_appends_live_only(monkeypatch, profile, keyed):
    monkeypatch.setattr(profile, "fetch_models",
                        lambda *, api_key=None, base_url=None, timeout=8.0: ["glm-5.1", "brand-new-model"])

    result = provider_model_ids("tokenrhythm")
    assert result[0] == "deepseek-v4-pro-0813"
    assert result[: len(profile.fallback_models)] == list(profile.fallback_models)
    assert result[len(profile.fallback_models):] == ["brand-new-model"]


def test_picker_falls_back_to_static_catalog(monkeypatch, profile, keyed):
    monkeypatch.setattr(profile, "fetch_models", lambda *, api_key=None, base_url=None, timeout=8.0: None)

    assert provider_model_ids("token-rhythm") == list(profile.fallback_models)


def test_aux_model_is_in_catalog(profile):
    assert profile.default_aux_model == "deepseek-v4-flash-0731"
    assert profile.default_aux_model in profile.fallback_models


@pytest.mark.parametrize(
    ("model", "context", "vision"),
    [("deepseek-v4-pro-0813", 1_000_000, False), ("kimi-k2.6", 256_000, True),
     ("glm-5.1", 200_000, False), ("glm-5.3-flash", 1_048_576, True)],
)
def test_model_capabilities_reach_consumers(model, context, vision):
    from agent.model_metadata import get_model_context_length
    from agent.models_dev import get_model_capabilities

    caps = get_model_capabilities("tokenrhythm", model)
    assert caps is not None
    assert caps.supports_tools is True
    assert caps.supports_reasoning is True
    assert caps.supports_vision is vision
    assert caps.context_window == context
    assert get_model_context_length(model, base_url=BASE_URL, api_key="sk", provider="tokenrhythm") == context


def test_wire_has_no_reasoning_knobs(profile):
    """No documented effort vocabulary: nothing reasoning-shaped goes on the wire."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="deepseek-v4-pro-0813", messages=[{"role": "user", "content": "ping"}], tools=None,
        provider_profile=profile, reasoning_config={"enabled": True, "effort": "high"},
        base_url=BASE_URL, provider_name="tokenrhythm",
    )
    assert kwargs["model"] == "deepseek-v4-pro-0813"
    assert "reasoning_effort" not in kwargs
    assert "reasoning" not in kwargs.get("extra_body", {})


def test_copied_provider_prefix_is_stripped():
    assert normalize_model_for_provider("tokenrhythm/kimi-k2.6", "tokenrhythm") == "kimi-k2.6"
    assert normalize_model_for_provider("kimi-k2.6", "tokenrhythm") == "kimi-k2.6"
