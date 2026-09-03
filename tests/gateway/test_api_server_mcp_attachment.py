"""Regression coverage for MCP snapshots in API Server-created agents."""

import unittest
from unittest.mock import patch


def _tool(name):
    return {
        "type": "function",
        "function": {"name": name, "description": "", "parameters": {}},
    }


class _FakeAgent:
    """Models the post-construction snapshot before an MCP refresh."""

    def __init__(self, **kwargs):
        self.enabled_toolsets = kwargs["enabled_toolsets"]
        self.disabled_toolsets = None
        self.tools = [_tool("read_file")]
        self.valid_tool_names = {"read_file"}


class TestApiServerMcpAttachment(unittest.TestCase):
    def _create_agent(self, toolsets):
        from gateway.config import PlatformConfig
        from gateway.platforms.api_server import APIServerAdapter

        def _definitions(*, enabled_toolsets, **_kwargs):
            tools = [_tool("read_file")]
            if "mcp-profile-a" in enabled_toolsets:
                tools.append(_tool("mcp_profile_a_expected"))
            if "mcp-profile-b" in enabled_toolsets:
                tools.append(_tool("mcp_profile_b_unexpected"))
            return tools

        with patch("run_agent.AIAgent", _FakeAgent), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "api_key": "test-key", "base_url": None, "provider": None,
                 "api_mode": None, "command": None, "args": [],
             }), \
             patch("gateway.run._resolve_gateway_model", return_value="test/model"), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None)), \
             patch("gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda: {})), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value=toolsets), \
             patch("model_tools.get_tool_definitions", side_effect=_definitions):
            adapter = APIServerAdapter(PlatformConfig(enabled=True))
            adapter._ensure_session_db = lambda: None
            return adapter._create_agent(session_id="api-session")

    def test_allowed_mcp_tool_is_attached_without_cross_toolset_leakage(self):
        agent = self._create_agent({"mcp-profile-a"})

        self.assertIn("mcp_profile_a_expected", agent.valid_tool_names)
        self.assertNotIn("mcp_profile_b_unexpected", agent.valid_tool_names)

    def test_profile_without_an_mcp_toolset_receives_no_mcp_tools(self):
        agent = self._create_agent(set())

        self.assertEqual(agent.valid_tool_names, {"read_file"})


if __name__ == "__main__":
    unittest.main()
