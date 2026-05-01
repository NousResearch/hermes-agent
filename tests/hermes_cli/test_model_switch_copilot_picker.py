import agent.models_dev as models_dev_mod
import hermes_cli.auth as auth_mod
import hermes_cli.models as models_mod
import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers


class _ApiKeyProvider:
    auth_type = "api_key"
    api_key_env_vars = ("COPILOT_GITHUB_TOKEN",)
    base_url_env_var = None
    inference_base_url = "https://models.inference.ai.azure.com"


def test_list_authenticated_providers_uses_live_copilot_catalog_without_truncation(monkeypatch):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_test_token")

    monkeypatch.setattr(
        models_dev_mod,
        "fetch_models_dev",
        lambda: {
            "github-copilot": {
                "env": ["COPILOT_GITHUB_TOKEN"],
                "name": "GitHub Copilot",
            }
        },
    )
    monkeypatch.setattr(
        models_dev_mod,
        "PROVIDER_TO_MODELS_DEV",
        {"copilot": "github-copilot"},
    )
    monkeypatch.setattr(
        auth_mod,
        "PROVIDER_REGISTRY",
        {"copilot": _ApiKeyProvider()},
    )
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        models_mod,
        "provider_model_ids",
        lambda provider: ["live-a", "live-b", "live-c"] if provider == "copilot" else [],
    )

    providers = list_authenticated_providers(
        current_provider="copilot",
        current_model="live-a",
        user_providers={},
        custom_providers=[],
        max_models=1,
    )

    copilot = next(p for p in providers if p["slug"] == "copilot")
    assert copilot["models"] == ["live-a", "live-b", "live-c"]
    assert copilot["total_models"] == 3
    assert copilot["is_current"] is True
