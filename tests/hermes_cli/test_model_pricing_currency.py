"""Currency handling for model-picker pricing catalogs."""

import json
from unittest.mock import MagicMock

from hermes_cli import models


def _catalog_response(payload):
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_generic_picker_keeps_only_usd_compatible_pricing(monkeypatch):
    payload = {
        "data": [
            {
                "id": "usd",
                "pricing": {"currency": "USD", "prompt": "1e-6", "completion": "2e-6"},
            },
            {
                "id": "inherited-idr",
                "currency": "IDR",
                "pricing": {"prompt": "1", "completion": "2"},
            },
            {
                "id": "overridden-usd",
                "currency": "IDR",
                "pricing": {"currency": "$", "prompt": "3e-6", "completion": "4e-6"},
            },
            {
                "id": "direct-idr",
                "pricing": {"currency": "IDR", "prompt": "5", "completion": "6"},
            },
        ]
    }
    monkeypatch.setattr(
        models,
        "_urlopen_model_catalog_request",
        lambda *_args, **_kwargs: _catalog_response(payload),
    )

    result = models.fetch_models_with_pricing(
        base_url="https://currency-picker.example/api",
        force_refresh=True,
    )

    assert set(result) == {"usd", "overridden-usd"}


def test_novita_picker_keeps_only_usd_compatible_pricing(monkeypatch):
    payload = {
        "data": [
            {
                "id": "currencyless",
                "input_token_price_per_m": 10_000,
                "output_token_price_per_m": 20_000,
            },
            {
                "id": "usd",
                "currency": "USD",
                "input_token_price_per_m": 10_000,
                "output_token_price_per_m": 20_000,
            },
            {
                "id": "idr",
                "currency": "IDR",
                "input_token_price_per_m": 10_000,
                "output_token_price_per_m": 20_000,
            },
        ]
    }
    monkeypatch.setenv("NOVITA_API_KEY", "test-key")
    monkeypatch.setenv("NOVITA_BASE_URL", "https://currency-novita.example/v1")
    monkeypatch.setattr(
        models,
        "_urlopen_model_catalog_request",
        lambda *_args, **_kwargs: _catalog_response(payload),
    )

    result = models._fetch_novita_pricing(force_refresh=True)

    assert set(result) == {"currencyless", "usd"}


def test_deepinfra_picker_uses_nearest_currency_declaration(monkeypatch):
    items = [
        {
            "id": "metadata-idr",
            "metadata": {
                "currency": "IDR",
                "pricing": {"input_tokens": 1, "output_tokens": 2},
            },
        },
        {
            "id": "child-usd",
            "metadata": {
                "currency": "IDR",
                "pricing": {
                    "currency": "USD",
                    "input_tokens": 1,
                    "output_tokens": 2,
                },
            },
        },
        {
            "id": "child-empty-inherits-idr",
            "metadata": {
                "currency": "IDR",
                "pricing": {
                    "currency": "   ",
                    "input_tokens": 1,
                    "output_tokens": 2,
                },
            },
        },
        {
            "id": "currencyless",
            "metadata": {"pricing": {"input_tokens": 1, "output_tokens": 2}},
        },
    ]
    monkeypatch.setattr(
        models,
        "_fetch_deepinfra_models_by_tag",
        lambda *_args, **_kwargs: items,
    )

    result = models._fetch_deepinfra_pricing(force_refresh=True)

    assert set(result) == {"child-usd", "currencyless"}
