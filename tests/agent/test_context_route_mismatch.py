"""Tests for agent_init._context_route_mismatch context-pin scoping."""

from agent.agent_init import _context_route_mismatch


class TestContextRouteMismatchNamedCustomProvider:
    """Named custom providers store the URL under custom_providers, not model.base_url.

    Gateway session-reset banners used to treat empty model.base_url + a runtime
    custom URL as a route mismatch, drop model.context_length, and fall back to
    the Qwen family default (131072) even though /status still showed the pin.
    """

    def test_same_named_custom_provider_keeps_pin_without_configured_url(self):
        assert (
            _context_route_mismatch(
                None,
                "http://127.0.0.1:8080/v1",
                "custom-local-agentw",
                "custom-local-agentw",
            )
            is False
        )

    def test_different_provider_still_clears_pin(self):
        assert (
            _context_route_mismatch(
                None,
                "http://127.0.0.1:8080/v1",
                "custom-local-agentw",
                "openrouter",
            )
            is True
        )

    def test_explicit_base_url_mismatch_still_clears_pin(self):
        assert (
            _context_route_mismatch(
                "http://127.0.0.1:8080/v1",
                "http://10.0.0.2:8080/v1",
                "custom-local-agentw",
                "custom-local-agentw",
            )
            is True
        )

    def test_catalog_provider_rejects_non_default_runtime_url(self):
        assert (
            _context_route_mismatch(
                None,
                "http://127.0.0.1:8080/v1",
                "openrouter",
                "openrouter",
            )
            is True
        )


class TestContextRouteMismatchCopilotHosts:
    """The Copilot token exchange swaps api.githubcopilot.com for the account's host."""

    def test_exchanged_copilot_host_keeps_pin(self):
        for host in ("individual", "business", "enterprise"):
            assert (
                _context_route_mismatch(
                    "https://api.githubcopilot.com",
                    f"https://api.{host}.githubcopilot.com",
                    "copilot",
                    "copilot",
                )
                is False
            )

    def test_copilot_to_other_host_still_clears_pin(self):
        assert (
            _context_route_mismatch(
                "https://api.githubcopilot.com",
                "https://api.openai.com/v1",
                "copilot",
                "openai",
            )
            is True
        )
