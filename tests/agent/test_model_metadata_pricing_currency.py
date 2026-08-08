"""Currency handling for model-catalog pricing metadata."""

from unittest.mock import MagicMock, patch

import pytest

import agent.model_metadata as model_metadata
from agent.model_metadata import (
    _extract_pricing,
    fetch_endpoint_model_metadata,
    fetch_model_metadata,
)


@pytest.mark.parametrize(
    ("has_currency_field", "currency", "expected"),
    (
        (False, None, {"prompt": "0.000001", "completion": "0.000002"}),
        (True, None, {"prompt": "0.000001", "completion": "0.000002"}),
        (True, "", {"prompt": "0.000001", "completion": "0.000002"}),
        (True, "   ", {"prompt": "0.000001", "completion": "0.000002"}),
        (True, "USD", {"prompt": "0.000001", "completion": "0.000002"}),
        (True, "usd", {"prompt": "0.000001", "completion": "0.000002"}),
        (True, " USD ", {"prompt": "0.000001", "completion": "0.000002"}),
        (True, "$", {"prompt": "0.000001", "completion": "0.000002"}),
        (True, "IDR", {}),
        (True, "EUR", {}),
    ),
)
def test_extract_pricing_currency_matrix(has_currency_field, currency, expected):
    pricing = {"prompt": "0.000001", "completion": "0.000002"}
    if has_currency_field:
        pricing["currency"] = currency

    assert _extract_pricing({"pricing": pricing}) == expected


@pytest.mark.parametrize(
    "payload",
    (
        {
            "currency": "IDR",
            "input_token_price_per_m": 1000,
            "output_token_price_per_m": 2000,
        },
        {
            "metadata": {
                "pricing": {
                    "currency": "EUR",
                    "input_tokens": 1,
                    "output_tokens": 2,
                }
            }
        },
    ),
)
def test_non_usd_currency_rejected_for_special_pricing_shapes(payload):
    assert _extract_pricing(payload) == {}


def test_unrelated_non_usd_currency_does_not_suppress_usd_pricing():
    payload = {
        "regional_metadata": {"currency": "EUR"},
        "pricing": {
            "currency": "USD",
            "prompt": "0.000001",
            "completion": "0.000002",
        },
    }

    assert _extract_pricing(payload) == {
        "prompt": "0.000001",
        "completion": "0.000002",
    }


@pytest.mark.parametrize("currency", (None, "", "   "))
def test_empty_child_currency_inherits_non_usd_parent(currency):
    payload = {
        "currency": "IDR",
        "pricing": {
            "currency": currency,
            "prompt": "1",
            "completion": "2",
        },
    }

    assert _extract_pricing(payload) == {}


def test_generic_pricing_inherits_non_usd_parent_currency():
    payload = {
        "currency": "IDR",
        "pricing": {
            "prompt": "1",
            "completion": "2",
        },
    }

    assert _extract_pricing(payload) == {}


def test_deepinfra_pricing_inherits_metadata_currency():
    payload = {
        "metadata": {
            "currency": "IDR",
            "pricing": {
                "input_tokens": 1,
                "output_tokens": 2,
            },
        }
    }

    assert _extract_pricing(payload) == {}


def test_child_usd_currency_overrides_non_usd_parent():
    payload = {
        "currency": "IDR",
        "pricing": {
            "currency": "USD",
            "prompt": "0.000001",
            "completion": "0.000002",
        },
    }

    assert _extract_pricing(payload) == {
        "prompt": "0.000001",
        "completion": "0.000002",
    }


def test_endpoint_catalog_keeps_usd_model_when_sibling_is_non_usd():
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {
                "id": "usd-model",
                "pricing": {
                    "currency": "USD",
                    "prompt": "0.000001",
                    "completion": "0.000002",
                },
            },
            {
                "id": "non-usd-model",
                "pricing": {
                    "currency": "IDR",
                    "prompt": "1",
                    "completion": "2",
                },
            },
        ]
    }
    response.raise_for_status.return_value = None

    with patch("agent.model_metadata.requests.get", return_value=response):
        metadata = fetch_endpoint_model_metadata(
            "https://currency-matrix.example/v1",
            force_refresh=True,
        )

    assert metadata["usd-model"]["pricing"] == {
        "prompt": "0.000001",
        "completion": "0.000002",
    }
    assert "pricing" not in metadata["non-usd-model"]


def test_openrouter_cache_filters_non_usd_pricing(monkeypatch):
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {
                "id": "usd-model",
                "pricing": {
                    "currency": "USD",
                    "prompt": "0.000001",
                    "completion": "0.000002",
                },
            },
            {
                "id": "non-usd-model",
                "pricing": {
                    "currency": "IDR",
                    "prompt": "1",
                    "completion": "2",
                },
            },
        ]
    }
    response.raise_for_status.return_value = None
    monkeypatch.setattr(model_metadata, "_model_metadata_cache", {})
    monkeypatch.setattr(model_metadata, "_model_metadata_cache_time", 0)
    monkeypatch.setattr(
        model_metadata, "_save_model_metadata_disk_cache", lambda _cache: None
    )

    with patch("agent.model_metadata.requests.get", return_value=response):
        metadata = fetch_model_metadata(force_refresh=True)

    assert metadata["usd-model"]["pricing"] == {
        "prompt": "0.000001",
        "completion": "0.000002",
    }
    assert metadata["non-usd-model"]["pricing"] == {}
