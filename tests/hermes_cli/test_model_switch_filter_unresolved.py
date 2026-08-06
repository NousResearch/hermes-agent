"""Picker rows must resolve to a runtime provider (#57503)."""

from unittest.mock import patch

from hermes_cli.auth import is_runtime_provider_routable
from hermes_cli.model_switch import list_authenticated_providers


def _rows_with_env(monkeypatch, env_name: str, provider: str) -> list[dict]:
    monkeypatch.setenv(env_name, "test-key")
    with (
        patch(
            "agent.models_dev.fetch_models_dev",
            return_value={provider: {"env": [env_name], "name": provider.title()}},
        ),
        patch(
            "agent.models_dev.PROVIDER_TO_MODELS_DEV",
            {provider: provider},
        ),
        patch("hermes_cli.models.cached_provider_model_ids", return_value=["model-a"]),
        patch("hermes_cli.providers.HERMES_OVERLAYS", {}),
    ):
        return list_authenticated_providers(max_models=5)


def test_models_dev_only_provider_is_not_selectable(monkeypatch):
    rows = _rows_with_env(monkeypatch, "MISTRAL_API_KEY", "mistral")

    assert all(row["slug"] != "mistral" for row in rows)
    assert not is_runtime_provider_routable("mistral")


def test_registered_provider_remains_selectable(monkeypatch):
    rows = _rows_with_env(monkeypatch, "DEEPSEEK_API_KEY", "deepseek")

    row = next(row for row in rows if row["slug"] == "deepseek")
    assert row["models"] == ["model-a"]
    assert row["total_models"] == 1
    assert is_runtime_provider_routable("deepseek")


def test_special_runtime_provider_does_not_require_registry_membership():
    assert is_runtime_provider_routable("openrouter")
    assert is_runtime_provider_routable("custom:local-lab")
