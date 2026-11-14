"""Fast-mode eligibility shares OpenAI's regional-host policy, not a second allowlist."""

import pytest

from hermes_cli.models import resolve_fast_mode_overrides


@pytest.mark.parametrize("provider", ["openai", "openai-api", None])
@pytest.mark.parametrize("base_url", [
    "https://api.openai.com/v1", "https://us.api.openai.com/v1",
    "https://eu.api.openai.com/v1", "https://US.API.OPENAI.COM.:443/v1",
])
def test_official_openai_routes_keep_the_requested_tier(provider, base_url):
    for tier in ("priority", "ultrafast"):
        assert resolve_fast_mode_overrides(
            "gpt-6.1-sol", provider=provider, base_url=base_url, tier=tier,
        ) == {"service_tier": tier}
    # The endpoint cannot grant a model a tier it does not support.
    assert resolve_fast_mode_overrides(
        "gpt-5.4", provider=provider, base_url=base_url, tier="ultrafast",
    ) is None


@pytest.mark.parametrize("model,provider,base_url,expected", [
    ("gpt-6.1-sol", "openai-api", "https://api.openai.com.attacker.test/v1", None),
    ("gpt-6.1-sol", "openai-api", "https://us.api.openai.com.attacker.test/v1", None),
    ("gpt-6.1-sol", "openai-api", "https://notapi.openai.com/v1", None),
    ("gpt-6.1-sol", "openai-api", "https://proxy.test/api.openai.com/v1", None),
    ("gpt-6.1-sol", "openai-api", "https://us.api.openai.com@proxy.test/v1", None),
    ("gpt-6.1-sol", "openai-api", "https://proxy.test/v1", None),
    ("gpt-6.1-sol", "custom", "https://us.api.openai.com/v1", None),
    ("gpt-6.1-sol", "openrouter", "https://us.api.openai.com/v1", None),
    ("gpt-6.1-sol", "bedrock", "https://us.api.openai.com/v1", None),
    ("gpt-5-codex", "openai-api", "https://us.api.openai.com/v1", None),
    ("claude-opus-5.5", "anthropic", "https://us.api.openai.com/v1", None),
    ("grok-4.6", "xai", "https://us.api.openai.com/v1", None),
    ("claude-opus-5.5", "anthropic", "https://api.anthropic.com", {"speed": "fast"}),
    ("claude-opus-5.5", "anthropic", "https://us.api.anthropic.com", None),
    ("grok-4.6", "xai", "https://api.x.ai/v1", {"service_tier": "priority"}),
    ("grok-4.6", "xai", "https://us.api.x.ai/v1", None),
    ("gpt-6.1-sol", "openai-codex", "https://chatgpt.com/backend-api/codex", {"service_tier": "priority"}),
    ("gpt-6.1-sol", "openai-codex", "https://proxy.test/backend-api/codex", None),
    ("gpt-6.1-sol", None, None, {"service_tier": "priority"}),
    ("gpt-6.1-sol", "openai-api", None, {"service_tier": "priority"}),
])
def test_fast_route_boundaries_are_preserved(model, provider, base_url, expected):
    assert resolve_fast_mode_overrides(model, provider=provider, base_url=base_url) == expected


@pytest.mark.parametrize("model,host,tier", [
    ("gpt-6-luna", "jp.api.openai.com", None),  # au/jp/in: "Standard pay-as-you-go only"
    ("gpt-6.1-sol", "au.api.openai.com", None),
    ("gpt-6.1-sol", "ca.api.openai.com", None),  # storage-only region, no documented Fast
    ("gpt-6-astra", "eu.api.openai.com", None),  # EU Fast excludes GPT-6 Astra / GPT-6 Sol
    ("gpt-6-sol", "eu.api.openai.com", None),
    ("gpt-6-astra", "eu.api.openai.com", "ultrafast"),  # Astra Ultrafast: US + global only
])
def test_regional_hosts_only_get_documented_fast_tiers(model, host, tier):
    assert resolve_fast_mode_overrides(model, provider="openai-api", base_url=f"https://{host}/v1", tier=tier) is None
