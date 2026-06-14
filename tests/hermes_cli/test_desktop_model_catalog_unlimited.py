"""Regression tests for Desktop model picker catalog truncation."""

from hermes_cli.inventory import ConfigContext, build_models_payload
from hermes_cli.model_switch import list_authenticated_providers


def test_build_models_payload_allows_unlimited_catalogs(monkeypatch):
    """max_models=None means Desktop receives every model, not just 50."""
    seen = {}

    def fake_list_authenticated_providers(**kwargs):
        seen.update(kwargs)
        models = [f"model-{i}" for i in range(75)]
        max_models = kwargs.get("max_models")
        return [
            {
                "slug": "many",
                "name": "Many",
                "models": models if max_models is None else models[:max_models],
                "total_models": len(models),
                "is_current": False,
                "is_user_defined": False,
                "source": "built-in",
            }
        ]

    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        fake_list_authenticated_providers,
    )

    payload = build_models_payload(
        ConfigContext(
            current_provider="",
            current_model="",
            current_base_url="",
            user_providers={},
            custom_providers=[],
        ),
        max_models=None,
    )

    assert seen["max_models"] is None
    assert len(payload["providers"][0]["models"]) == 75


def test_list_authenticated_providers_none_max_keeps_more_than_fifty_models(monkeypatch):
    """The lower-level provider lister must treat None as unbounded."""
    many = [f"nvidia/model-{i}" for i in range(75)]

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda slug: many if slug == "nvidia" else [])

    providers = list_authenticated_providers(max_models=None)
    nvidia = next(provider for provider in providers if provider["slug"] == "nvidia")

    assert nvidia["total_models"] == 75
    assert len(nvidia["models"]) == 75
