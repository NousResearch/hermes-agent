import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hermes_cli.config import build_model_picker_user_providers
from hermes_cli.model_switch import list_authenticated_providers


def test_build_model_picker_user_providers_merges_custom_providers():
    cfg = {
        "providers": {
            "lmstudio": {
                "name": "LM Studio",
                "api": "http://localhost:1234/v1",
                "default_model": "llama-3.1-8b-instruct",
            }
        },
        "custom_providers": [
            {
                "name": "My OpenAI Proxy",
                "base_url": "http://localhost:8080/v1",
                "api_key": "sk-test",
                "default_model": "gpt-4o-mini",
            }
        ],
    }

    merged = build_model_picker_user_providers(cfg)

    assert merged is not None
    assert "lmstudio" in merged
    assert merged["lmstudio"]["name"] == "LM Studio"
    assert "custom:my-openai-proxy" in merged
    assert merged["custom:my-openai-proxy"]["name"] == "My OpenAI Proxy"
    assert merged["custom:my-openai-proxy"]["api"] == "http://localhost:8080/v1"
    assert merged["custom:my-openai-proxy"]["api_key"] == "sk-test"
    assert merged["custom:my-openai-proxy"]["default_model"] == "gpt-4o-mini"


def test_list_authenticated_providers_probes_custom_provider_models(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        lambda api_key, base_url, timeout=5.0: ["gpt-4o-mini", "gpt-4o"],
    )

    merged = build_model_picker_user_providers(
        {
            "custom_providers": [
                {
                    "name": "My OpenAI Proxy",
                    "base_url": "http://localhost:8080/v1",
                    "api_key": "sk-test",
                }
            ]
        }
    )

    providers = list_authenticated_providers(user_providers=merged)
    custom = next(p for p in providers if p["slug"] == "custom:my-openai-proxy")

    assert custom["name"] == "My OpenAI Proxy"
    assert custom["models"] == ["gpt-4o-mini", "gpt-4o"]
    assert custom["total_models"] == 2
