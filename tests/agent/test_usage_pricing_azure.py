"""Tests for Azure Foundry pricing & route resolution in agent.usage_pricing.

Covers the 7 catalog models the setup wizard prefills (gpt-5, gpt-5-mini,
gpt-5-codex, gpt-4.1, gpt-4.1-mini, o3, o4-mini) for both:
  - canonical-model lookup path
  - deployment-name canonicalisation path (e.g. "prod-gpt5-eus" → "gpt-5")
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from agent.usage_pricing import (
    _canonicalize_azure_model,
    get_pricing_entry,
    resolve_billing_route,
)


AZURE_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-codex",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o3",
    "o4-mini",
]


@pytest.mark.parametrize("model", AZURE_MODELS)
def test_canonical_model_route_resolves(model):
    """resolve_billing_route on canonical model name → azure-foundry."""
    route = resolve_billing_route(model, provider="azure-foundry")
    assert route.provider == "azure-foundry"
    assert route.model == model
    assert route.billing_mode == "azure-foundry-list-fallback"


@pytest.mark.parametrize("model", AZURE_MODELS)
def test_canonical_model_pricing_lookup(model):
    """Each canonical model has a published pricing entry."""
    entry = get_pricing_entry(model, provider="azure-foundry")
    assert entry is not None, f"no pricing entry for {model}"
    assert entry.input_cost_per_million > Decimal("0")
    assert entry.output_cost_per_million > Decimal("0")
    assert entry.source == "azure-foundry-list-fallback"


@pytest.mark.parametrize(
    "deployment,expected",
    [
        ("prod-gpt-5-eus", "gpt-5"),
        ("prod_gpt-5_eus", "gpt-5"),
        ("staging-gpt-5-mini", "gpt-5-mini"),
        ("gpt-5-codex-westus", "gpt-5-codex"),
        ("my-gpt-4.1-deployment", "gpt-4.1"),
        ("eu-gpt-4.1-mini", "gpt-4.1-mini"),
        ("o3-prod", "o3"),
        ("o4-mini-eastus2", "o4-mini"),
    ],
)
def test_deployment_name_canonicalises(deployment, expected):
    """User-named deployments map back to canonical catalog model."""
    assert _canonicalize_azure_model(deployment) == expected


@pytest.mark.parametrize(
    "deployment,expected",
    [
        ("prod-gpt-5-eus", "gpt-5"),
        ("staging-gpt-5-mini", "gpt-5-mini"),
        ("gpt-5-codex-westus", "gpt-5-codex"),
    ],
)
def test_deployment_name_pricing_lookup(deployment, expected):
    """Deployment name resolves via canonicalisation for pricing."""
    entry = get_pricing_entry(deployment, provider="azure-foundry")
    assert entry is not None
    # Sanity: same entry as canonical
    canonical_entry = get_pricing_entry(expected, provider="azure-foundry")
    assert entry.input_cost_per_million == canonical_entry.input_cost_per_million
    assert entry.output_cost_per_million == canonical_entry.output_cost_per_million


def test_unknown_azure_model_returns_none_gracefully():
    """A model with no pricing entry returns None instead of raising."""
    entry = get_pricing_entry("gpt-99-future-vapourware", provider="azure-foundry")
    assert entry is None


def test_canonicalize_passthrough_when_no_match():
    """Canonicaliser returns input unchanged when no rule matches."""
    assert _canonicalize_azure_model("brand-new-model") == "brand-new-model"
    assert _canonicalize_azure_model("") == ""


def test_route_via_base_url_host_matches():
    """Azure-detected via base_url host even without explicit provider name."""
    route = resolve_billing_route(
        "prod-gpt-5",
        provider="",
        base_url="https://my-resource.openai.azure.com/openai/v1",
    )
    assert route.provider == "azure-foundry"
    assert route.model == "gpt-5"
