"""Contract tests for the Z.ai Indirect provider.

The indirect route must use Z.ai's Claude Code-compatible Anthropic endpoint;
it must never fall back to the OpenAI-compatible PaaS route.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def profile():
    import model_tools  # noqa: F401 -- triggers bundled plugin discovery
    import providers

    value = providers.get_provider_profile("zai-indirect")
    assert value is not None, "zai-indirect provider profile must be registered"
    return value


def test_provider_is_a_distinct_claude_code_compatible_route(profile):
    assert profile.name == "zai-indirect"
    assert profile.display_name == "Z.ai Indirect"
    assert profile.api_mode == "anthropic_messages"
    assert profile.base_url == "https://api.z.ai/api/anthropic"


def test_provider_exposes_only_glm_5_2_without_openai_model_probe(profile):
    assert profile.fallback_models == ("glm-5.2",)
    assert profile.supports_health_check is False
    assert profile.models_url == ""


def test_provider_uses_a_dedicated_credential_namespace(profile):
    assert profile.env_vars[0] == "ZAI_INDIRECT_API_KEY"


def test_provider_id_alone_determines_anthropic_api_mode():
    from hermes_cli.providers import determine_api_mode

    assert determine_api_mode("zai-indirect") == "anthropic_messages"


def test_manifest_declares_model_provider_kind():
    manifest = Path("plugins/model-providers/zai-indirect/plugin.yaml").read_text(encoding="utf-8")
    assert "kind: model-provider" in manifest


def test_direct_agent_construction_uses_anthropic_transport():
    """Provider-only callers must not silently fall back to chat completions."""
    from run_agent import AIAgent

    agent = AIAgent(
        provider="zai-indirect",
        model="glm-5.2",
        api_key="test-token-not-a-real-secret",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
    )

    assert agent.base_url == "https://api.z.ai/api/anthropic"
    assert agent.api_mode == "anthropic_messages"
    assert agent._anthropic_client is not None


def test_desktop_picker_catalog_contains_distinct_indirect_row(monkeypatch):
    from hermes_cli.model_switch import list_authenticated_providers

    monkeypatch.setenv("ZAI_INDIRECT_API_KEY", "test-token-not-a-real-secret")
    rows = list_authenticated_providers(
        current_provider="zai-indirect",
        current_base_url="https://api.z.ai/api/anthropic",
        max_models=50,
        probe_custom_providers=False,
        for_picker=True,
    )

    indirect = [row for row in rows if row["slug"] == "zai-indirect"]
    assert indirect == [
        {
            "slug": "zai-indirect",
            "name": "Z.ai Indirect",
            "is_current": True,
            "is_user_defined": False,
            "models": ["glm-5.2"],
            "total_models": 1,
            "source": "canonical",
        }
    ]
