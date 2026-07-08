"""Focused tests for Pioneer provider wiring."""

from __future__ import annotations


def test_model_flow_api_key_provider_prefers_live_pioneer_models(monkeypatch):
    monkeypatch.setenv("PIONEER_API_KEY", "pioneer-test-key")

    fetch_calls = []

    def fake_fetch_api_models(api_key, base_url, *args, **kwargs):
        fetch_calls.append((api_key, base_url))
        return [
            "anthropic/pioneer/gpt-5.5",
            "live-model-a",
            "gpt-5.5",
            "live-model-b",
        ]

    models_dev_calls = []

    def fake_models_dev(provider_id):
        models_dev_calls.append(provider_id)
        return ["models-dev-model"]

    prompt_calls = []

    def fake_prompt_model_selection(model_list, current_model="", **kwargs):
        prompt_calls.append((model_list, current_model))
        return "live-model-b"

    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)
    monkeypatch.setattr("agent.models_dev.list_agentic_models", fake_models_dev)
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", fake_prompt_model_selection)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)

    from hermes_cli.config import load_config
    from hermes_cli.main import _model_flow_api_key_provider

    _model_flow_api_key_provider(load_config(), "pioneer", "old-model")

    assert fetch_calls == [("pioneer-test-key", "https://api.pioneer.ai/v1")]
    assert models_dev_calls == []
    assert prompt_calls == [
        (["gpt-5.5", "live-model-a", "live-model-b"], "old-model")
    ]

    config = load_config()
    assert config["model"]["provider"] == "pioneer"
    assert config["model"]["default"] == "live-model-b"
    assert config["model"]["base_url"] == "https://api.pioneer.ai/v1"


def test_provider_model_ids_cleans_pioneer_routed_aliases(monkeypatch):
    class FakeProfile:
        auth_type = "api_key"
        base_url = "https://api.pioneer.ai/v1"
        fallback_models = ()

        def fetch_models(self, *, api_key, **kwargs):
            assert api_key == "pioneer-test-key"
            return [
                "anthropic/pioneer/gpt-5.5",
                "mistralai/Mistral-Small-4-119B-2603",
                "gpt-5.5",
                "openai/pioneer/gpt-4.1-mini",
                "gpt-4.1-mini",
                "anthropic/pioneer/gpt-5.4",
            ]

    monkeypatch.setattr("providers.get_provider_profile", lambda provider: FakeProfile())
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider: {
            "api_key": "pioneer-test-key",
            "base_url": "https://api.pioneer.ai/v1",
        },
    )

    from hermes_cli.models import provider_model_ids

    models = provider_model_ids("pioneer")
    assert models[:4] == [
        "gpt-5.5",
        "mistralai/Mistral-Small-4-119B-2603",
        "gpt-4.1-mini",
        "gpt-5.4",
    ]
    assert "anthropic/pioneer/gpt-5.5" not in models
    assert "openai/pioneer/gpt-4.1-mini" not in models
    assert len(models) == len({m.lower() for m in models})


def test_cached_provider_model_ids_cleans_pioneer_cache(monkeypatch):
    monkeypatch.setattr("hermes_cli.models.time.time", lambda: 100.0)
    monkeypatch.setattr(
        "hermes_cli.models._credential_fingerprint",
        lambda provider: "cache-fingerprint",
    )
    monkeypatch.setattr(
        "hermes_cli.models._load_provider_models_cache",
        lambda: {
            "pioneer": {
                "fp": "cache-fingerprint",
                "at": 90.0,
                "models": [
                    "anthropic/pioneer/gpt-5.5",
                    "gpt-5.5",
                    "openai/pioneer/gpt-4.1-mini",
                    "gpt-4.1-mini",
                ],
            }
        },
    )

    live_calls = []

    def fake_provider_model_ids(*args, **kwargs):
        live_calls.append((args, kwargs))
        return []

    monkeypatch.setattr("hermes_cli.models.provider_model_ids", fake_provider_model_ids)

    from hermes_cli.models import cached_provider_model_ids

    assert cached_provider_model_ids("pioneer") == ["gpt-5.5", "gpt-4.1-mini"]
    assert live_calls == []
