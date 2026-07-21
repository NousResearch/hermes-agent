"""Regression test: custom providers preserve dots in model names.

When provider is "custom", _anthropic_preserve_dots() must return True
so that model names like "claude-sonnet-4.6" or "glm-4.7" are sent to
the endpoint unchanged.  Dot-to-hyphen conversion is Anthropic-specific
and must not be applied to user-configured custom endpoints.

Refs #16417, #13061.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from run_agent import AIAgent


def _make_agent(provider: str = "", base_url: str = "") -> AIAgent:
    agent = MagicMock(spec=AIAgent)
    agent.provider = provider
    agent.base_url = base_url
    # Bind the real method
    agent._anthropic_preserve_dots = AIAgent._anthropic_preserve_dots.__get__(agent)
    return agent


class TestCustomProviderPreserveDots:
    """Custom providers must preserve dots in model names."""

    def test_custom_provider_preserves_dots(self) -> None:
        agent = _make_agent(provider="custom")
        assert agent._anthropic_preserve_dots() is True

    def test_custom_with_base_url_preserves_dots(self) -> None:
        agent = _make_agent(
            provider="custom",
            base_url="https://my-relay.example.com/v1",
        )
        assert agent._anthropic_preserve_dots() is True

    def test_anthropic_provider_still_converts_dots(self) -> None:
        """Anthropic provider should NOT preserve dots (default behavior)."""
        agent = _make_agent(provider="anthropic")
        assert agent._anthropic_preserve_dots() is False

    def test_openrouter_provider_still_converts_dots(self) -> None:
        agent = _make_agent(provider="openrouter")
        assert agent._anthropic_preserve_dots() is False

    def test_known_providers_still_preserve_dots(self) -> None:
        """Existing allowlist providers still work."""
        for prov in ("alibaba", "minimax", "zai", "bedrock"):
            agent = _make_agent(provider=prov)
            assert agent._anthropic_preserve_dots() is True, f"{prov} should preserve dots"
