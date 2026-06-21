"""Provider plugin and curated catalog regressions for smaller inference providers."""

from unittest.mock import patch

from agent.models_dev import PROVIDER_TO_MODELS_DEV
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.models import _MODELS_DEV_PREFERRED, _PROVIDER_MODELS, provider_model_ids
from providers import get_provider_profile


def test_groq_plugin_registers_profile_and_auth_registry_entry():
    profile = get_provider_profile("groq")
    assert profile is not None
    assert profile.base_url == "https://api.groq.com/openai/v1"

    registry_entry = PROVIDER_REGISTRY["groq"]
    assert registry_entry.api_key_env_vars == ("GROQ_API_KEY",)
    assert registry_entry.base_url_env_var == "GROQ_BASE_URL"


def test_cerebras_plugin_registers_profile_auth_registry_and_models_dev_mapping():
    profile = get_provider_profile("cerebras")
    assert profile is not None
    assert profile.base_url == "https://api.cerebras.ai/v1"

    registry_entry = PROVIDER_REGISTRY["cerebras"]
    assert registry_entry.api_key_env_vars == ("CEREBRAS_API_KEY",)
    assert registry_entry.base_url_env_var == "CEREBRAS_BASE_URL"
    assert PROVIDER_TO_MODELS_DEV["cerebras"] == "cerebras"
    assert "cerebras" in _MODELS_DEV_PREFERRED


def test_minimax_fetch_models_uses_anthropic_messages_catalog_auth():
    profile = get_provider_profile("minimax")
    assert profile is not None
    credential = "dummy-" + "key"

    with patch("hermes_cli.models.fetch_api_models", return_value=["MiniMax-M3"]) as mocked_fetch:
        result = profile.fetch_models(
            api_key=credential,
            base_url="https://api.minimax.io/anthropic",
            timeout=3.0,
        )

    assert result == ["MiniMax-M3"]
    assert mocked_fetch.call_args.kwargs["api_mode"] == "anthropic_messages"
    assert mocked_fetch.call_args.kwargs["timeout"] == 3.0


def test_groq_picker_uses_curated_chat_catalog_not_live_or_models_dev_noise():
    assert "groq" not in _MODELS_DEV_PREFERRED
    profile = get_provider_profile("groq")
    assert profile is not None
    credential = "dummy-" + "key"
    assert profile.fetch_models(api_key=credential) is None

    with (
        patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": credential, "base_url": "https://api.groq.com/openai/v1"},
        ),
        patch(
            "providers.base.ProviderProfile.fetch_models",
            side_effect=AssertionError("groq should not use broad live /models"),
        ),
        patch(
            "agent.models_dev.list_agentic_models",
            side_effect=AssertionError("groq should not merge models.dev"),
        ),
    ):
        models = provider_model_ids("groq", force_refresh=True)

    assert models == _PROVIDER_MODELS["groq"]
    joined = "\n".join(models).lower()
    assert "whisper" not in joined
    assert "safeguard" not in joined
    assert "prompt-guard" not in joined


def test_xiaomi_catalog_excludes_token_plan_rejected_ultraspeed_model():
    assert "mimo-v2.5-pro-ultraspeed" not in _PROVIDER_MODELS["xiaomi"]

    with patch(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        return_value={"api_key": "", "base_url": ""},
    ):
        models = provider_model_ids("xiaomi", force_refresh=True)

    assert "mimo-v2.5-pro" in models
    assert "mimo-v2.5-pro-ultraspeed" not in models
