"""Together AI catalog, pricing, and usage-accounting tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.model_metadata import fetch_endpoint_model_metadata
from agent.usage_pricing import (
    get_pricing_entry,
    normalize_usage,
    resolve_billing_route,
)


def _together_catalog():
    return [
        {
            "id": "thinkingmachines/Inkling",
            "type": "chat",
            "display_name": "Inkling",
            "context_length": 524288,
            "pricing": {
                "base": 0,
                "finetune": 0,
                "hourly": 0,
                "input": 1.0,
                "cached_input": 0.17,
                "output": 4.05,
            },
        }
    ]


def test_bare_list_catalog_parses_context_and_per_million_pricing():
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=_together_catalog,
    )
    with patch("agent.model_metadata.requests.get", return_value=response):
        metadata = fetch_endpoint_model_metadata(
            "https://api.together.test/v1",
            api_key="test-key",
            force_refresh=True,
        )

    entry = metadata["thinkingmachines/Inkling"]
    assert entry["context_length"] == 524288
    assert float(entry["pricing"]["prompt"]) == pytest.approx(1.0 / 1_000_000)
    assert float(entry["pricing"]["completion"]) == pytest.approx(4.05 / 1_000_000)
    assert float(entry["pricing"]["cache_read"]) == pytest.approx(0.17 / 1_000_000)


@pytest.mark.parametrize(
    "provider,base_url",
    [
        ("together", "https://api.together.ai/v1"),
        ("together-ai", "https://api.together.ai/v1"),
        ("", "https://api.together.xyz/v1"),
    ],
)
def test_billing_route_normalizes_provider_and_legacy_host(provider, base_url):
    route = resolve_billing_route(
        "thinkingmachines/Inkling",
        provider=provider,
        base_url=base_url,
    )
    assert route.provider == "together"
    assert route.model == "thinkingmachines/Inkling"
    assert route.billing_mode == "official_models_api"


def test_live_models_metadata_drives_usage_pricing():
    metadata = {
        "thinkingmachines/Inkling": {
            "pricing": {
                "prompt": str(1.0 / 1_000_000),
                "completion": str(4.05 / 1_000_000),
                "cache_read": str(0.17 / 1_000_000),
            }
        }
    }
    with patch(
        "agent.usage_pricing.fetch_endpoint_model_metadata",
        return_value=metadata,
    ):
        entry = get_pricing_entry(
            "thinkingmachines/Inkling",
            provider="together",
            base_url="https://api.together.ai/v1",
            api_key="test-key",
        )

    assert entry is not None
    assert float(entry.input_cost_per_million) == 1.0
    assert float(entry.output_cost_per_million) == 4.05
    assert float(entry.cache_read_cost_per_million) == 0.17
    assert entry.source == "provider_models_api"


def test_models_dev_is_offline_pricing_fallback():
    model_info = SimpleNamespace(
        cost_input=0.3,
        cost_output=1.2,
        cost_cache_read=0.06,
        cost_cache_write=None,
        has_cost_data=lambda: True,
    )
    with (
        patch(
            "agent.usage_pricing.fetch_endpoint_model_metadata",
            return_value={},
        ),
        patch("agent.models_dev.get_model_info", return_value=model_info),
    ):
        entry = get_pricing_entry(
            "MiniMaxAI/MiniMax-M3",
            provider="together",
            base_url="https://api.together.ai/v1",
        )

    assert entry is not None
    assert float(entry.input_cost_per_million) == 0.3
    assert float(entry.output_cost_per_million) == 1.2
    assert float(entry.cache_read_cost_per_million) == 0.06


def test_flat_cached_tokens_are_normalized():
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=50,
        prompt_tokens_details=None,
        cached_tokens=400,
        completion_tokens_details=None,
        output_tokens_details=None,
    )

    normalized = normalize_usage(usage, provider="together")

    assert normalized.input_tokens == 600
    assert normalized.cache_read_tokens == 400
    assert normalized.output_tokens == 50
