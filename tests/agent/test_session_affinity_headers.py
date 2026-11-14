"""Test propagation of session affinity headers in AIAgent initialization.
"""

from __future__ import annotations

from unittest.mock import patch

from run_agent import AIAgent


def _mock_build_client(agent, *args, **kwargs):
    agent._client_kwargs = {}
    agent.client = None


def test_session_affinity_headers_injected_with_session_id():
    with patch("hermes_cli.config.load_config", return_value={}), \
         patch("agent.agent_init._build_client", side_effect=_mock_build_client), \
         patch("model_tools.get_tool_definitions", return_value=[]):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="test-model",
            session_id="session_12345",
            skip_context_files=True,
            skip_memory=True,
        )
        extra_headers = agent.request_overrides.get("extra_headers", {})
        assert extra_headers.get("X-Session-ID") == "session_12345"
        assert extra_headers.get("X-OmniRoute-Session-Key") == "session_12345"


def test_session_affinity_headers_parent_session_id_precedence():
    with patch("hermes_cli.config.load_config", return_value={}), \
         patch("agent.agent_init._build_client", side_effect=_mock_build_client), \
         patch("model_tools.get_tool_definitions", return_value=[]):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="test-model",
            session_id="child_subagent_678",
            parent_session_id="parent_root_123",
            skip_context_files=True,
            skip_memory=True,
        )
        extra_headers = agent.request_overrides.get("extra_headers", {})
        # Parent session ID is preferred for prompt cache affinity across subagents
        assert extra_headers.get("X-Session-ID") == "parent_root_123"
        assert extra_headers.get("X-OmniRoute-Session-Key") == "parent_root_123"


def test_session_affinity_headers_preserves_custom_headers():
    with patch("hermes_cli.config.load_config", return_value={}), \
         patch("agent.agent_init._build_client", side_effect=_mock_build_client), \
         patch("model_tools.get_tool_definitions", return_value=[]):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="test-model",
            session_id="session_custom",
            request_overrides={"extra_headers": {"X-Custom-Auth": "secret", "X-Session-ID": "existing_sid"}},
            skip_context_files=True,
            skip_memory=True,
        )
        extra_headers = agent.request_overrides.get("extra_headers", {})
        assert extra_headers.get("X-Custom-Auth") == "secret"
        # setdefault does not overwrite existing value
        assert extra_headers.get("X-Session-ID") == "existing_sid"
        assert extra_headers.get("X-OmniRoute-Session-Key") == "session_custom"
