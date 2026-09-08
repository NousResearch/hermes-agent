"""Bundled discovery and offline native metadata contracts."""
from pathlib import Path
import sys

import pytest

import providers


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    # Rediscover the actual bundled package, not a test-only importlib name.
    for name in tuple(sys.modules):
        if name.startswith("plugins.model_providers."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_discovered", False)
    providers._discover_providers()
    result = providers.get_provider_profile("claude-oauth-directsdk")
    assert result is not None
    return result


def test_discovered_profile_constructs_bundled_client_without_native_process(profile):
    client = profile.create_client(command="unused-offline-native", env={})
    try:
        module = sys.modules[type(client).__module__]
        expected = Path(providers.__file__).resolve().parent.parent / "plugins" / "model-providers" / profile.name
        assert Path(module.__file__).resolve().parent == expected
        assert callable(client.chat.completions.create)
        assert profile.fetch_models() is None  # No invented live /models endpoint.
        assert profile.supports_health_check is False
    finally:
        client.close()


def test_native_alias_metadata_is_bounded_and_never_claims_subscription_invoice(profile):
    from decimal import Decimal
    from agent.model_metadata import get_model_context_length
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, normalize_usage

    for alias in profile.fallback_models:
        # Qualified catalog capacities are distinct from the conservative native
        # preflight bound: entitlement/alias overrides are owned by Claude Code.
        metadata = profile.model_metadata[alias]
        assert metadata["canonical_model"].startswith("claude-")
        assert 0 < profile.get_model_context_length(alias) <= metadata["context_window"]
        assert profile.get_model_context_length(metadata["canonical_model"]) == profile.get_model_context_length(alias)
        assert get_model_context_length(alias, provider=profile.name, base_url=profile.base_url) == profile.get_model_context_length(alias)
        assert get_model_context_length(alias, provider=profile.name, config_context_length=123456) == 123456
        cost = estimate_usage_cost(alias, CanonicalUsage(input_tokens=1000, output_tokens=100),
                             provider=profile.name, base_url=profile.base_url)
        assert cost.status == "unknown"
        assert cost.amount_usd is None
    assert profile.get_model_context_length("unqualified-future-model") is None
    reported = {"prompt_tokens": 50, "completion_tokens": 10, "native_cost": {"total_cost_usd": .012345, "modelUsage": {"claude-sonnet-5": {"costBasis": "list"}}}}
    usage = normalize_usage(reported, provider=profile.name)
    cost = estimate_usage_cost("sonnet", usage, provider=profile.name, base_url=profile.base_url)
    assert cost.amount_usd == Decimal("0.012345")
    assert cost.status == "estimated"
    assert any("not subscription invoice" in note for note in cost.notes)
    for invalid in (float('nan'), float('inf'), -1, True, None):
        reported['native_cost']['total_cost_usd'] = invalid
        cost = estimate_usage_cost('sonnet', normalize_usage(reported), provider=profile.name)
        assert cost.status == 'unknown' and cost.amount_usd is None
