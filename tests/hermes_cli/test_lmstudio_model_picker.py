"""Regression tests for LM Studio visibility in the /model picker."""

from hermes_cli.model_switch import list_picker_providers


def test_picker_includes_lmstudio_when_base_url_configured_without_api_key(monkeypatch):
    """LM Studio often runs locally without auth; LM_BASE_URL should be enough.

    The Telegram/Discord model picker is credential-gated so it does not show
    unusable cloud providers. LM Studio is different: a configured local base
    URL is the user's opt-in signal even when no LM_API_KEY exists.
    """
    monkeypatch.setenv("LM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.delenv("LM_API_KEY", raising=False)

    # Keep the provider listing deterministic and offline. The LM Studio row is
    # populated from the live native probe, then falls back to this curated list
    # when cached_provider_model_ids() is empty.
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda _slug: [])
    monkeypatch.setattr(
        "hermes_cli.models.fetch_lmstudio_models",
        lambda **_kwargs: ["google/gemma-4-12b-qat", "openai/gpt-oss-20b"],
    )

    providers = list_picker_providers(
        current_provider="openai-codex",
        current_base_url="https://chatgpt.com/backend-api/codex",
        current_model="gpt-5.5",
        max_models=50,
    )

    lmstudio = next((p for p in providers if p["slug"] == "lmstudio"), None)
    assert lmstudio is not None
    assert lmstudio["name"] == "LM Studio"
    assert lmstudio["models"] == ["google/gemma-4-12b-qat", "openai/gpt-oss-20b"]
    assert lmstudio["total_models"] == 2
