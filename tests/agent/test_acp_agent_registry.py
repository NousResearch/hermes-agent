"""Unit tests for the ACP agent registry (issue #5257)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agent.acp_agent_registry import (
    ACP_AGENT_REGISTRY,
    agent_display_name,
    agent_install_hint,
    is_acp_agent_available,
    known_agents,
    normalize_agent_name,
    resolve_agent_launch,
)


def test_known_agents_include_core_set():
    agents = known_agents()
    for name in ("copilot", "claude", "codex", "gemini", "qwen"):
        assert name in agents


def test_builtin_resolution():
    # Pin PATH resolution off so the preferred bin is chosen deterministically.
    with patch("agent.acp_agent_registry.shutil.which", return_value=None):
        assert resolve_agent_launch("claude") == ("claude-agent-acp", [])
        assert resolve_agent_launch("codex") == ("codex-acp", [])
        assert resolve_agent_launch("gemini") == ("gemini", ["--experimental-acp"])
        assert resolve_agent_launch("copilot") == ("copilot", ["--acp", "--stdio"])


def test_claude_falls_back_to_zed_bin_when_only_it_is_installed():
    def _only_zed(name):
        return "/usr/bin/claude-code-acp" if name == "claude-code-acp" else None

    with patch("agent.acp_agent_registry.shutil.which", side_effect=_only_zed):
        assert resolve_agent_launch("claude") == ("claude-code-acp", [])


def test_agent_env_unset_declares_claude_session_markers():
    from agent.acp_agent_registry import agent_env_unset

    markers = agent_env_unset("claude")
    assert {"CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT"} <= set(markers)
    assert agent_env_unset("copilot") == ()
    assert agent_env_unset("unknown-agent") == ()


def test_normalization_is_case_and_whitespace_insensitive():
    assert normalize_agent_name("  Claude ") == "claude"
    with patch("agent.acp_agent_registry.shutil.which", return_value=None):
        assert resolve_agent_launch("CLAUDE") == ("claude-agent-acp", [])


def test_generic_env_override_is_shlex_split_full_command():
    with patch.dict(os.environ, {"HERMES_ACP_CLINE_COMMAND": "npx cline-acp --stdio"}, clear=False):
        assert resolve_agent_launch("cline") == ("npx", ["cline-acp", "--stdio"])
        assert is_acp_agent_available("cline")


def test_generic_env_override_beats_registry():
    with patch.dict(os.environ, {"HERMES_ACP_CLAUDE_COMMAND": "/opt/custom/claude-acp --flag"}, clear=False):
        assert resolve_agent_launch("claude") == ("/opt/custom/claude-acp", ["--flag"])


def test_legacy_copilot_env_vars_are_command_path_only():
    with patch.dict(
        os.environ,
        {"HERMES_COPILOT_ACP_COMMAND": "/usr/local/bin/copilot"},
        clear=False,
    ):
        command, args = resolve_agent_launch("copilot")
        assert command == "/usr/local/bin/copilot"
        # default args preserved — the legacy var never carried args
        assert args == ["--acp", "--stdio"]

    with patch.dict(os.environ, {"COPILOT_CLI_PATH": "/opt/bin/copilot"}, clear=False):
        command, _ = resolve_agent_launch("copilot")
        assert command == "/opt/bin/copilot"


def test_legacy_copilot_args_env_var_replaces_args():
    with patch.dict(os.environ, {"HERMES_COPILOT_ACP_ARGS": "--acp"}, clear=False):
        assert resolve_agent_launch("copilot") == ("copilot", ["--acp"])


def test_generic_args_env_var_replaces_args():
    with patch.dict(os.environ, {"HERMES_ACP_GEMINI_ARGS": "--acp --sandbox"}, clear=False):
        assert resolve_agent_launch("gemini") == ("gemini", ["--acp", "--sandbox"])


def test_unknown_agent_without_override_raises_with_hint():
    with pytest.raises(ValueError) as excinfo:
        resolve_agent_launch("definitely-not-an-agent")
    message = str(excinfo.value)
    assert "HERMES_ACP_DEFINITELY_NOT_AN_AGENT_COMMAND" in message
    assert not is_acp_agent_available("definitely-not-an-agent")


def test_display_name_and_install_hint():
    assert agent_display_name("claude") == "Claude Code"
    assert "claude-code-acp" in agent_install_hint("claude")
    # Unknown agents get a generic-but-actionable hint
    assert "HERMES_ACP_MYSTERY_COMMAND" in agent_install_hint("mystery")


def test_registry_entries_have_display_names_and_hints():
    for name, entry in ACP_AGENT_REGISTRY.items():
        assert entry.command, name
        assert entry.display_name, name
        assert entry.install_hint, name
