"""Sale UI pricing helpers: gateway pricing.original → discount chrome."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import hermes_cli.models as models_mod
from hermes_cli.models import (
    compute_sale_discount,
    fetch_models_with_pricing,
)


def test_compute_sale_discount_from_prompt():
    sale = compute_sale_discount(
        "0.0000016000",
        "0.0000080000",
        {"prompt": "0.0000020000", "completion": "0.0000100000"},
    )
    assert sale is not None
    pct, was_prompt, was_completion = sale
    assert pct == 20
    assert was_prompt == "0.0000020000"
    assert was_completion == "0.0000100000"


def test_compute_sale_discount_omits_when_no_original():
    assert compute_sale_discount("0.0000016", "0.000008", None) is None
    assert compute_sale_discount("0.0000016", "0.000008", "not-a-dict") is None


def test_compute_sale_discount_omits_for_free_models_even_with_higher_original():
    """Free / $0 models must never show sale chrome against a list price."""
    assert (
        compute_sale_discount(
            "0",
            "0",
            {"prompt": "0.000002", "completion": "0.00001"},
        )
        is None
    )


def test_compute_sale_discount_omits_sub_one_percent_discounts():
    """A discount that rounds below 1% must not render as '-0%'."""
    assert (
        compute_sale_discount(
            "0.0000099999",
            "0.00005",
            {"prompt": "0.00001", "completion": "0.00005"},
        )
        is None
    )


def test_compute_sale_discount_omits_when_not_cheaper():
    assert (
        compute_sale_discount(
            "0.000002",
            "0.00001",
            {"prompt": "0.000002", "completion": "0.00001"},
        )
        is None
    )


def test_compute_sale_discount_falls_back_to_completion():
    sale = compute_sale_discount(
        "",
        "0.000008",
        {"prompt": "", "completion": "0.00001"},
    )
    assert sale is not None
    assert sale[0] == 20


def test_fetch_models_with_pricing_copies_nested_original(monkeypatch):
    models_mod._pricing_cache.clear()
    payload = {
        "data": [
            {
                "id": "anthropic/claude-sonnet-5",
                "pricing": {
                    "prompt": "0.0000016",
                    "completion": "0.000008",
                    "input_cache_read": "0.00000016",
                    "original": {
                        "prompt": "0.000002",
                        "completion": "0.00001",
                        "input_cache_read": "0.0000002",
                    },
                },
            },
            {
                "id": "free/model",
                "pricing": {"prompt": "0", "completion": "0"},
            },
        ]
    }
    body = json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda *a: False

    monkeypatch.setattr(
        models_mod,
        "_urlopen_model_catalog_request",
        lambda req, timeout=8.0: resp,
    )

    # Nous Portal opts in via include_sale_original=True.
    result = fetch_models_with_pricing(
        api_key="sk-test",
        base_url="https://example.test",
        force_refresh=True,
        include_sale_original=True,
    )
    paid = result["anthropic/claude-sonnet-5"]
    assert paid["prompt"] == "0.0000016"
    assert paid["completion"] == "0.000008"
    assert paid["original"] == {
        "prompt": "0.000002",
        "completion": "0.00001",
        "input_cache_read": "0.0000002",
    }
    assert "original" not in result["free/model"]


def test_fetch_models_with_pricing_omits_original_when_absent(monkeypatch):
    models_mod._pricing_cache.clear()
    payload = {
        "data": [
            {
                "id": "x/y",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            }
        ]
    }
    body = json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda *a: False
    monkeypatch.setattr(
        models_mod,
        "_urlopen_model_catalog_request",
        lambda req, timeout=8.0: resp,
    )

    result = fetch_models_with_pricing(
        base_url="https://example.test/no-sale",
        force_refresh=True,
        include_sale_original=True,
    )
    assert "original" not in result["x/y"]


def test_fetch_models_with_pricing_ignores_original_unless_opted_in(monkeypatch):
    """OpenRouter / default path must never surface pricing.original."""
    models_mod._pricing_cache.clear()
    payload = {
        "data": [
            {
                "id": "anthropic/claude-sonnet-5",
                "pricing": {
                    "prompt": "0.0000016",
                    "completion": "0.000008",
                    "original": {
                        "prompt": "0.000002",
                        "completion": "0.00001",
                    },
                },
            }
        ]
    }
    body = json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda *a: False
    monkeypatch.setattr(
        models_mod,
        "_urlopen_model_catalog_request",
        lambda req, timeout=8.0: resp,
    )

    # Default (OpenRouter path): strip original even when the payload has it.
    result = fetch_models_with_pricing(
        base_url="https://openrouter.ai/api",
        force_refresh=True,
    )
    assert "original" not in result["anthropic/claude-sonnet-5"]
    assert result["anthropic/claude-sonnet-5"]["prompt"] == "0.0000016"


def test_resolve_nous_pricing_credentials_honors_inference_env_override(monkeypatch):
    """Staging profiles set NOUS_INFERENCE_BASE_URL — pricing must follow it.

    Without this, anonymous/failed-auth fallback hits prod and sale
    ``pricing.original`` never reaches Desktop/CLI pickers.
    """
    monkeypatch.setenv(
        "NOUS_INFERENCE_BASE_URL",
        "https://stg-inference-api.nousresearch.com/v1",
    )
    # Auth resolution fails / returns nothing — the env override must still win.
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials",
        lambda: None,
    )
    api_key, base_url = models_mod._resolve_nous_pricing_credentials()
    assert api_key == ""
    assert base_url == "https://stg-inference-api.nousresearch.com/v1"


def test_resolve_nous_pricing_credentials_env_wins_over_stored_prod(monkeypatch):
    monkeypatch.setenv(
        "NOUS_INFERENCE_BASE_URL",
        "https://stg-inference-api.nousresearch.com/v1",
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials",
        lambda: {
            "api_key": "ak-test",
            "base_url": "https://inference-api.nousresearch.com/v1",
        },
    )
    api_key, base_url = models_mod._resolve_nous_pricing_credentials()
    assert api_key == "ak-test"
    assert base_url == "https://stg-inference-api.nousresearch.com/v1"
