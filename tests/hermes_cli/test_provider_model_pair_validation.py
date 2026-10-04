"""Provider/model validation must reject foreign native-model families."""

from hermes_cli.model_normalize import detect_vendor
from hermes_cli.models import _AGGREGATOR_PROVIDERS, _PROVIDER_MODELS
from hermes_cli.models_validate import validate_requested_model


def test_native_provider_rejects_model_family_owned_by_another_catalog(monkeypatch):
    provider = "anthropic"
    own_vendors = {
        vendor
        for model in _PROVIDER_MODELS[provider]
        if (vendor := detect_vendor(model)) is not None
    }
    foreign_model = next(
        model
        for candidate, models in _PROVIDER_MODELS.items()
        if candidate != provider and candidate not in _AGGREGATOR_PROVIDERS
        for model in models
        if (vendor := detect_vendor(model)) is not None and vendor not in own_vendors
    )
    monkeypatch.setattr(
        "hermes_cli.models._fetch_anthropic_models",
        lambda **_kwargs: list(_PROVIDER_MODELS[provider]),
    )

    result = validate_requested_model(foreign_model, provider, api_key="test-key")

    assert result["accepted"] is False
    assert "catalog family served by" in result["message"]
