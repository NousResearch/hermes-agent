"""Tests for xAI OAuth auth refresh route mapping (#60264)."""

import pytest


class TestAuthRefreshXaiRoute:
    def test_xai_oauth_route_mapped(self):
        """api.x.ai and x.ai should map to xai-oauth."""
        from agent.auxiliary_client import _auth_refresh_provider_for_route

        assert _auth_refresh_provider_for_route("auto", "https://api.x.ai/v1") == "xai-oauth"
        assert _auth_refresh_provider_for_route("auto", "https://x.ai/v1") == "xai-oauth"

    def test_existing_routes_unchanged(self):
        """Existing provider mappings should still work."""
        from agent.auxiliary_client import _auth_refresh_provider_for_route

        assert _auth_refresh_provider_for_route("auto", "https://api.githubcopilot.com") == "copilot"
        assert _auth_refresh_provider_for_route("auto", "https://chatgpt.com/backend-api/codex") == "openai-codex"
        assert _auth_refresh_provider_for_route("auto", "https://api.anthropic.com") == "anthropic"

    def test_explicit_provider_passthrough(self):
        """Explicit (non-auto) providers should pass through unchanged."""
        from agent.auxiliary_client import _auth_refresh_provider_for_route

        assert _auth_refresh_provider_for_route("openai", "https://api.openai.com") == "openai"
        assert _auth_refresh_provider_for_route("nous", "https://any.url") == "nous"
