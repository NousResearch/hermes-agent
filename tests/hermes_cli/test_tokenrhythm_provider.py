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


def _wire(profile, model, reasoning_config):
    from agent.transports.chat_completions import ChatCompletionsTransport

    return ChatCompletionsTransport().build_kwargs(
        model=model, messages=[{"role": "user", "content": "ping"}], tools=None, provider_profile=profile,
        reasoning_config=reasoning_config, base_url=BASE_URL, provider_name="tokenrhythm",
    )


_OFF = {"enabled": False}
_EFFORT_OFF = {"enabled": True, "effort": "none"}


@pytest.mark.parametrize(
    ("reasoning_config", "thinking", "effort"),
    [({"enabled": True, "effort": "low"}, "enabled", "low"),
     ({"enabled": True, "effort": "medium"}, "enabled", "high"),
     ({"enabled": True, "effort": "xhigh"}, "enabled", "max"),
     ({"enabled": True, "effort": "ultra"}, "enabled", "max"),
     (_OFF, "disabled", None), (_EFFORT_OFF, "disabled", None)],
)
def test_deepseek_toggles_thinking_and_grades_effort(profile, reasoning_config, thinking, effort):
    kwargs = _wire(profile, "deepseek-v4-pro-0813", reasoning_config)
    assert kwargs["extra_body"] == {"thinking": {"type": thinking}}
    assert kwargs.get("reasoning_effort") == effort


def test_deepseek_main_loop_default_turns_thinking_on(profile):
    """deepseek-v4-* do not think when the toggle is omitted; Hermes' unset effort means medium."""
    default = profile.default_reasoning_config("deepseek-v4-flash-0731")
    kwargs = _wire(profile, "deepseek-v4-flash-0731", default)
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert kwargs["reasoning_effort"] == "high"


def test_unset_aux_call_sends_no_reasoning_fields(profile):
    kwargs = _wire(profile, "deepseek-v4-flash-0731", None)
    assert "extra_body" not in kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("model", ["qwen3.8-max", "seed-2.1-pro"])
def test_qwen_seed_toggle_only_never_send_effort(profile, model):
    assert profile.default_reasoning_config(model) is None
    for config in (_OFF, _EFFORT_OFF):
        kwargs = _wire(profile, model, config)
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert "reasoning_effort" not in kwargs
    kwargs = _wire(profile, model, {"enabled": True, "effort": "high"})
    assert "extra_body" not in kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("model", ["glm-5.3", "kimi-k2.7-code", "glm-5.1", "mimo-v2.5-pro", "some-new-model"])
@pytest.mark.parametrize("reasoning_config", [_OFF, _EFFORT_OFF, {"enabled": True, "effort": "max"}, None])
def test_always_on_models_never_receive_a_disable_or_effort(profile, model, reasoning_config):
    """GLM-5.3 / kimi-k2.7-code / longcat / minimax 400 on any disable ("当前模型必须开启深度思考");
    unknown live-only ids are treated the same way."""
    assert profile.default_reasoning_config(model) is None
    kwargs = _wire(profile, model, reasoning_config)
    assert "extra_body" not in kwargs
    assert "reasoning_effort" not in kwargs
    assert "thinking" not in kwargs and "enable_thinking" not in kwargs


def test_reasoning_content_is_read_from_the_response():
    from types import SimpleNamespace

    from agent.transports.chat_completions import ChatCompletionsTransport

    msg = SimpleNamespace(content="hi", tool_calls=None, reasoning_content="thinking...", reasoning=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)
    normalized = ChatCompletionsTransport().normalize_response(response)
    assert normalized.provider_data["reasoning_content"] == "thinking..."


def test_copied_provider_prefix_is_stripped():
    assert normalize_model_for_provider("tokenrhythm/kimi-k2.6", "tokenrhythm") == "kimi-k2.6"
    assert normalize_model_for_provider("kimi-k2.6", "tokenrhythm") == "kimi-k2.6"
