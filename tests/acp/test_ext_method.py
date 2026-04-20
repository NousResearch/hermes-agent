"""Tests for HermesACPAgent.ext_method extension method handling."""

import pytest
from acp_adapter.server import HermesACPAgent


class TestExtMethod:
    """Test suite for ACP extension method handling."""

    def test_hermes_acp_agent_has_ext_method(self):
        """Test that HermesACPAgent has an ext_method method."""
        agent = HermesACPAgent()
        assert hasattr(agent, "ext_method"), "HermesACPAgent should have ext_method method"
        assert callable(agent.ext_method), "ext_method should be callable"

    @pytest.mark.asyncio
    async def test_ext_method_ping_returns_empty_dict(self):
        """Test that calling ext_method with 'ping' returns empty dict without error."""
        agent = HermesACPAgent()
        result = await agent.ext_method("ping", {})
        assert result == {}, "ext_method('ping', {}) should return {}"

    @pytest.mark.asyncio
    async def test_ext_method_unknown_method_returns_empty_dict(self):
        """Test that calling ext_method with any unknown method returns empty dict."""
        agent = HermesACPAgent()
        result = await agent.ext_method("some_unknown_method", {"key": "value"})
        assert result == {}, "ext_method with unknown method should return {}"

    @pytest.mark.asyncio
    async def test_ext_method_with_various_unknown_methods(self):
        """Test that ext_method gracefully handles various unknown extension methods."""
        agent = HermesACPAgent()
        unknown_methods = ["ping", "health", "status", "custom_extension", "_internal"]
        for method in unknown_methods:
            result = await agent.ext_method(method, {})
            assert result == {}, f"ext_method('{method}', {{}}) should return {{}}"
