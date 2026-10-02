"""Tests for hermes-api-server toolset and API server tool availability."""

class TestApiServerPlatformConfig:

    def test_default_api_server_includes_terminal_toolset(self):
        """Regression #49622: desktop-only read_terminal is registered into the
        'terminal' toolset (ships in-repo), so resolve_toolset('terminal') grows
        to include it after discovery. read_terminal is NOT in the
        hermes-api-server composite, so the old all-tools subset test dropped
        'terminal' entirely. Its static membership (terminal, process) IS in the
        composite, so it must stay enabled."""
        from hermes_cli.config import has_xai_tool_credentials

        from tools.registry import discover_builtin_tools
        from tools.platform_policy import get_platform_tools
        discover_builtin_tools()
        assert "terminal" in get_platform_tools({}, "api_server", xai_credentials_present=has_xai_tool_credentials)
