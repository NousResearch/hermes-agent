"""Unit tests for the generic ACP client (issue #5257)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.acp_client import (
    ACPClient,
    create_acp_client,
    extract_agent_from_url,
    marker_base_url,
)
from agent.copilot_acp_client import ACP_MARKER_BASE_URL, CopilotACPClient


def test_extract_agent_from_url():
    assert extract_agent_from_url("acp://copilot") == "copilot"
    assert extract_agent_from_url("acp://claude") == "claude"
    assert extract_agent_from_url("ACP://Claude") == "claude"
    assert extract_agent_from_url("acp://gemini/extra") == "gemini"
    assert extract_agent_from_url("https://api.openai.com/v1") is None
    assert extract_agent_from_url("") is None
    assert extract_agent_from_url(None) is None
    assert extract_agent_from_url("acp://") is None


def test_marker_base_url():
    assert marker_base_url("claude") == "acp://claude"
    assert marker_base_url(" Codex ") == "acp://codex"


def test_agent_name_derived_from_base_url():
    # Nothing installed on PATH -> registry resolves to the preferred bin.
    with patch("agent.acp_agent_registry.shutil.which", return_value=None):
        client = ACPClient(base_url="acp://claude", acp_cwd="/tmp")
    assert client.agent_name == "claude"
    assert client.agent_display_name == "Claude Code"
    assert client._acp_command == "claude-agent-acp"
    assert client.api_key == "claude-acp"


def test_agent_name_param_used_when_no_base_url():
    client = ACPClient(agent_name="gemini", acp_cwd="/tmp")
    assert client.base_url == "acp://gemini"
    assert client._acp_command == "gemini"
    assert client._acp_args == ["--experimental-acp"]


def test_explicit_command_keeps_registry_default_args():
    client = ACPClient(agent_name="copilot", command="/opt/bin/copilot", acp_cwd="/tmp")
    assert client._acp_command == "/opt/bin/copilot"
    assert client._acp_args == ["--acp", "--stdio"]


def test_explicit_args_override_registry_defaults():
    client = ACPClient(agent_name="copilot", acp_args=["--acp"], acp_cwd="/tmp")
    assert client._acp_args == ["--acp"]


def test_unknown_agent_without_command_raises():
    with pytest.raises(ValueError):
        ACPClient(agent_name="mystery-agent", acp_cwd="/tmp")


def test_unknown_agent_with_explicit_command_is_allowed():
    client = ACPClient(agent_name="mystery-agent", command="/opt/bin/mystery", acp_cwd="/tmp")
    assert client._acp_command == "/opt/bin/mystery"
    assert client._acp_args == []


def test_factory_returns_copilot_subclass_for_copilot():
    client = create_acp_client(agent_name="copilot", acp_cwd="/tmp")
    assert isinstance(client, CopilotACPClient)
    assert client.base_url == ACP_MARKER_BASE_URL


def test_factory_returns_generic_client_for_other_agents():
    client = create_acp_client(agent_name="claude", acp_cwd="/tmp")
    assert isinstance(client, ACPClient)
    assert not isinstance(client, CopilotACPClient)


def test_factory_infers_agent_from_base_url():
    client = create_acp_client(base_url="acp://codex", acp_cwd="/tmp")
    assert client.agent_name == "codex"
    assert not isinstance(client, CopilotACPClient)


def test_copilot_subclass_passes_generic_isinstance_check():
    # auxiliary_client routes on isinstance(x, ACPClient)
    assert isinstance(CopilotACPClient(acp_cwd="/tmp"), ACPClient)


def test_missing_command_error_includes_install_hint():
    client = ACPClient(agent_name="claude", acp_cwd="/tmp")
    with patch("agent.acp_client.subprocess.Popen", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError) as excinfo:
            client._run_prompt("hi", timeout_seconds=1.0)
    message = str(excinfo.value)
    assert "Claude Code" in message
    assert "@agentclientprotocol/claude-agent-acp" in message


def test_claude_bridge_prefers_official_but_falls_back_to_zed():
    # Only the older Zed bin is installed -> registry resolves to it.
    def _only_zed(name):
        return "/usr/bin/claude-code-acp" if name == "claude-code-acp" else None

    with patch("agent.acp_agent_registry.shutil.which", side_effect=_only_zed):
        client = ACPClient(agent_name="claude", acp_cwd="/tmp")
    assert client._acp_command == "claude-code-acp"


def test_claude_env_unset_strips_session_markers():
    # The claude agent declares the Claude Code session markers to strip so
    # the bridge launches even inside a parent Claude Code session.
    from agent.acp_agent_registry import agent_env_unset
    from agent.acp_client import _build_subprocess_env

    markers = agent_env_unset("claude")
    assert "CLAUDECODE" in markers

    with patch.dict(
        "os.environ",
        {"CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "cli", "CLAUDE_CODE_SSE_PORT": "9"},
    ):
        env = _build_subprocess_env(env_unset=markers)
    for marker in markers:
        assert marker not in env
    # Non-claude agents strip nothing.
    assert agent_env_unset("gemini") == ()


def test_early_exit_hook_default_is_generic(monkeypatch):
    client = ACPClient(agent_name="claude", acp_cwd="/tmp")
    assert client._early_exit_error("some stderr") is None


def test_copilot_early_exit_hook_detects_deprecated_extension():
    client = CopilotACPClient(acp_cwd="/tmp")
    stderr = "gh-copilot has been deprecated, no commands will be executed"
    message = client._early_exit_error(stderr)
    assert message and "deprecated" in message
    assert client._early_exit_error("unrelated stderr noise") is None
