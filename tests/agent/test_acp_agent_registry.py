"""Tests for the ACP agent command registry."""

import os
from unittest import mock

import pytest

from agent.acp_agent_registry import (
    ACP_AGENT_REGISTRY,
    list_agents,
    resolve_agent_command,
    resolve_agent_env,
    split_agent_command,
)


class TestResolveAgentCommand:
    def test_builtin_claude(self):
        cmd = resolve_agent_command("claude")
        assert cmd is not None
        assert "claude-agent-acp" in cmd

    def test_builtin_codex(self):
        cmd = resolve_agent_command("codex")
        assert cmd is not None
        assert "codex-acp" in cmd

    def test_builtin_gemini(self):
        cmd = resolve_agent_command("gemini")
        assert cmd == "gemini --acp"

    def test_builtin_copilot(self):
        cmd = resolve_agent_command("copilot")
        assert cmd is not None
        assert "--acp" in cmd

    def test_unknown_agent_returns_none(self):
        assert resolve_agent_command("nonexistent-agent-xyz") is None

    def test_case_insensitive(self):
        assert resolve_agent_command("Claude") == resolve_agent_command("claude")
        assert resolve_agent_command("GEMINI") == resolve_agent_command("gemini")

    def test_env_override(self):
        with mock.patch.dict(os.environ, {"HERMES_ACP_CLAUDE_COMMAND": "/custom/claude --acp"}):
            cmd = resolve_agent_command("claude")
            assert cmd == "/custom/claude --acp"

    def test_env_override_with_hyphen(self):
        """Agent name hyphens become underscores in env var name."""
        with mock.patch.dict(os.environ, {"HERMES_ACP_MY_AGENT_COMMAND": "my-agent --acp"}):
            cmd = resolve_agent_command("my-agent")
            assert cmd == "my-agent --acp"

    def test_empty_env_falls_through(self):
        with mock.patch.dict(os.environ, {"HERMES_ACP_CLAUDE_COMMAND": "  "}):
            cmd = resolve_agent_command("claude")
            # Should fall through to built-in
            assert "claude-agent-acp" in cmd


class TestSplitAgentCommand:
    def test_simple(self):
        assert split_agent_command("gemini --acp") == ["gemini", "--acp"]

    def test_npx(self):
        parts = split_agent_command("npx -y @zed-industries/codex-acp@0.9.5")
        assert parts == ["npx", "-y", "@zed-industries/codex-acp@0.9.5"]

    def test_multiple_args(self):
        parts = split_agent_command("copilot --acp --stdio")
        assert parts == ["copilot", "--acp", "--stdio"]


class TestListAgents:
    def test_returns_sorted_list(self):
        agents = list_agents()
        assert agents == sorted(agents)
        assert "claude" in agents
        assert "codex" in agents
        assert "gemini" in agents

    def test_all_registry_entries_present(self):
        agents = list_agents()
        for name in ACP_AGENT_REGISTRY:
            assert name in agents


class TestResolveAgentEnv:
    def test_claude_bridges_anthropic_token(self):
        """ANTHROPIC_TOKEN is bridged to ANTHROPIC_API_KEY for Claude."""
        with mock.patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_TOKEN": "sk-ant-test-token",
        }):
            env = resolve_agent_env("claude")
            assert env.get("ANTHROPIC_API_KEY") == "sk-ant-test-token"

    def test_claude_skips_if_api_key_set(self):
        """If ANTHROPIC_API_KEY is already set, don't override it."""
        with mock.patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-existing",
            "ANTHROPIC_TOKEN": "sk-fallback",
        }):
            env = resolve_agent_env("claude")
            assert "ANTHROPIC_API_KEY" not in env  # not overridden

    def test_claude_bridges_oauth_token(self):
        """CLAUDE_CODE_OAUTH_TOKEN is used as last resort."""
        with mock.patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_TOKEN": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token-123",
        }):
            env = resolve_agent_env("claude")
            assert env.get("ANTHROPIC_API_KEY") == "oauth-token-123"

    def test_codex_bridges_openai_key(self):
        """Codex auth env is resolved."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai-test"}):
            env = resolve_agent_env("codex")
            assert "OPENAI_API_KEY" not in env  # already set, no override

    def test_unknown_agent_returns_empty(self):
        env = resolve_agent_env("nonexistent-xyz")
        assert env == {}

    def test_agent_with_no_auth_env(self):
        """Gemini has no auth_env mapping — returns empty."""
        env = resolve_agent_env("gemini")
        assert env == {}

    def test_no_source_vars_set(self):
        """When no source vars have values, nothing is injected."""
        with mock.patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_TOKEN": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "",
        }):
            env = resolve_agent_env("claude")
            assert env == {}
