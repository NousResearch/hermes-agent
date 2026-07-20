"""Regression: one-shot (`hermes -z`) must wait for background MCP discovery
before building the agent, so slow stdio MCP servers aren't silently dropped.

See the issue: CLI startup kicks MCP discovery onto the ``cli-mcp-discovery``
background thread for the one-shot command, but ``_run_agent`` used to build
``AIAgent`` (which snapshots the tool registry) without joining that thread —
so tools from slow servers never reached the model. The fix mirrors the
interactive path by calling ``wait_for_mcp_discovery()`` first.
"""

from unittest.mock import MagicMock, patch

from hermes_cli import oneshot


def _run_agent_with_stubs(calls):
    """Invoke oneshot._run_agent with every collaborator stubbed, recording the
    order in which discovery-wait and agent construction happen in ``calls``."""

    def _record_wait():
        calls.append("wait")

    def _make_agent(*_args, **_kwargs):
        calls.append("agent")
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "ok"}
        return agent

    runtime = {
        "api_key": "k",
        "base_url": "https://example.test",
        "provider": "openrouter",
        "api_mode": "openai",
        "credential_pool": None,
    }

    with patch("hermes_cli.config.load_config", return_value={}), \
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime), \
        patch("hermes_cli.tools_config._get_platform_tools", return_value=set()), \
        patch("hermes_cli.oneshot.get_fallback_chain", return_value=[]), \
        patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None), \
        patch("hermes_cli.mcp_startup.wait_for_mcp_discovery", side_effect=_record_wait), \
        patch("run_agent.AIAgent", side_effect=_make_agent):
        return oneshot._run_agent("say hi", model="openrouter/some-model")


class TestOneshotWaitsForMcpDiscovery:
    def test_waits_before_building_agent(self):
        calls: list[str] = []
        response, result = _run_agent_with_stubs(calls)

        assert response == "ok"
        assert result == {"final_response": "ok"}
        # The wait must happen, and it must happen BEFORE the agent snapshots
        # the tool registry — otherwise slow MCP servers are dropped.
        assert "wait" in calls, "wait_for_mcp_discovery was never called"
        assert calls.index("wait") < calls.index("agent")

    def test_wait_failure_does_not_break_oneshot(self):
        """A failure while waiting must not abort the one-shot run."""
        def _boom():
            raise RuntimeError("discovery join blew up")

        runtime = {
            "api_key": "k",
            "base_url": "https://example.test",
            "provider": "openrouter",
            "api_mode": "openai",
            "credential_pool": None,
        }
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "ok"}

        with patch("hermes_cli.config.load_config", return_value={}), \
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime), \
            patch("hermes_cli.tools_config._get_platform_tools", return_value=set()), \
            patch("hermes_cli.oneshot.get_fallback_chain", return_value=[]), \
            patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None), \
            patch("hermes_cli.mcp_startup.wait_for_mcp_discovery", side_effect=_boom), \
            patch("run_agent.AIAgent", return_value=agent):
            response, _ = oneshot._run_agent("say hi", model="openrouter/some-model")

        assert response == "ok"
