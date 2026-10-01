"""Tests for Azure Foundry's model-dependent wire policy."""

from __future__ import annotations

import pytest


@pytest.fixture
def azure_foundry_profile():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("azure-foundry")
    assert profile is not None
    return profile


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "codex-mini", "o3-mini"])
def test_responses_only_families_use_the_responses_wire(azure_foundry_profile, model):
    assert azure_foundry_profile.resolve_route_policy(model) == "codex_responses"


def test_other_azure_models_leave_the_profile_default_in_charge(azure_foundry_profile):
    assert azure_foundry_profile.resolve_route_policy("gpt-4o") is None


def test_vendor_qualified_model_uses_the_bare_model_family(azure_foundry_profile):
    assert azure_foundry_profile.resolve_route_policy("openai/gpt-5.3") == "codex_responses"
