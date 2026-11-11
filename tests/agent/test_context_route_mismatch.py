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
