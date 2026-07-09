from __future__ import annotations

from hermes_cli.fallback_config import get_fallback_chain
from hermes_cli.model_aliases import apply_model_alias_to_model_config, get_model_alias_entry


def _config() -> dict:
    return {
        "model": {"default": "smart-cheap", "provider": "stale-provider"},
        "model_aliases": {
            "smart-cheap": {
                "provider": "zai",
                "model": "glm-5.2",
                "api_format": "anthropic",
                "endpoint": "https://api.z.ai/api/anthropic/v1/messages",
                "fallback_chain": [
                    {
                        "provider": "openai-codex",
                        "model": "gpt-5.4",
                        "api_mode": "codex_responses",
                    }
                ],
            }
        },
    }


def test_model_alias_entry_resolves_endpoint_to_transport_base_url():
    alias = get_model_alias_entry(_config(), "smart-cheap")

    assert alias is not None
    assert alias["provider"] == "zai"
    assert alias["model"] == "glm-5.2"
    assert alias["api_mode"] == "anthropic_messages"
    assert alias["base_url"] == "https://api.z.ai/api/anthropic"


def test_apply_model_alias_to_model_config_expands_main_model():
    model_cfg = apply_model_alias_to_model_config(_config())

    assert model_cfg["default"] == "glm-5.2"
    assert model_cfg["provider"] == "zai"
    assert model_cfg["api_mode"] == "anthropic_messages"
    assert model_cfg["base_url"] == "https://api.z.ai/api/anthropic"
    assert model_cfg["configured_alias_intent"] == "smart-cheap"


def test_fallback_chain_inherits_active_model_alias_fallback_chain():
    chain = get_fallback_chain(_config())

    assert chain == [
        {
            "provider": "openai-codex",
            "model": "gpt-5.4",
            "api_mode": "codex_responses",
        }
    ]


def test_explicit_fallback_providers_precede_alias_fallback_and_dedupe():
    cfg = _config()
    cfg["fallback_providers"] = [
        {"provider": "nous", "model": "Hermes-4"},
        {"provider": "openai-codex", "model": "gpt-5.4"},
    ]

    chain = get_fallback_chain(cfg)

    assert chain == [
        {"provider": "nous", "model": "Hermes-4"},
        {"provider": "openai-codex", "model": "gpt-5.4"},
    ]
