"""Regression coverage for native model/provider pair validation (#101091)."""

from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli.models import model_provider_compatibility_error


def test_rejects_model_claimed_by_another_native_provider():
    error = model_provider_compatibility_error("deepseek-v4-pro", "xiaomi")

    assert error is not None
    assert "deepseek" in error
    assert "xiaomi" in error


def test_accepts_model_from_requested_native_provider():
    assert model_provider_compatibility_error("mimo-v2.5-pro", "xiaomi") is None


def test_accepts_unknown_future_native_model():
    assert model_provider_compatibility_error("mimo-v3-future", "xiaomi") is None


def test_accepts_aggregator_model():
    assert model_provider_compatibility_error("deepseek-v4-pro", "openrouter") is None


def test_accepts_user_defined_endpoint():
    assert (
        model_provider_compatibility_error(
            "deepseek-v4-pro",
            "lab",
            user_providers={"lab": {"base_url": "http://127.0.0.1:8000/v1"}},
        )
        is None
    )


def test_accepts_explicitly_declared_model_for_builtin_provider():
    assert (
        model_provider_compatibility_error(
            "deepseek-v4-pro",
            "xiaomi",
            user_providers={"xiaomi": {"models": ["deepseek-v4-pro"]}},
        )
        is None
    )


def test_inventory_does_not_inject_foreign_current_model_into_native_provider(monkeypatch):
    monkeypatch.setenv("XIAOMI_API_KEY", "sk-test-xiaomi")

    with (
        patch(
            "agent.models_dev.fetch_models_dev",
            return_value={"xiaomi": {"env": ["XIAOMI_API_KEY"], "name": "Xiaomi"}},
        ),
        patch("agent.models_dev.PROVIDER_TO_MODELS_DEV", {"xiaomi": "xiaomi"}),
        patch(
            "hermes_cli.models.cached_provider_model_ids",
            return_value=["mimo-v2.5-pro"],
        ),
    ):
        rows = list_authenticated_providers(
            current_provider="xiaomi",
            current_model="deepseek-v4-pro",
            max_models=10,
        )

    xiaomi = next(row for row in rows if row["slug"] == "xiaomi")
    assert "mimo-v2.5-pro" in xiaomi["models"]
    assert "deepseek-v4-pro" not in xiaomi["models"]


def test_inventory_keeps_current_model_for_custom_provider():
    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        rows = list_authenticated_providers(
            current_provider="custom:lab",
            current_base_url="https://lab.example/v1",
            current_model="private-current-model",
            custom_providers=[
                {
                    "name": "Lab",
                    "base_url": "https://lab.example/v1",
                    "models": ["catalog-model"],
                }
            ],
            max_models=10,
            probe_custom_providers=False,
        )

    lab = next(row for row in rows if row["is_current"])
    assert lab["slug"] == "custom:lab"
    assert lab["models"][0] == "private-current-model"
    assert "catalog-model" in lab["models"]


def test_model_switch_rejects_clear_native_conflict_before_remote_validation(monkeypatch):
    monkeypatch.setenv("XIAOMI_API_KEY", "sk-test-xiaomi")

    result = switch_model(
        raw_input="deepseek-v4-pro",
        current_provider="xiaomi",
        current_model="mimo-v2.5-pro",
        explicit_provider="xiaomi",
    )

    assert result.success is False
    assert "belongs to provider 'deepseek'" in result.error_message
