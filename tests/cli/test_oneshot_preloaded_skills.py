"""Tests for oneshot (-z) mode's --skills / --ignore-rules / --ignore-user-config.

These verify the three parameters that were previously accepted by argparse
but silently dropped by run_oneshot() — the fix connects them to the actual
agent construction, matching chat mode behavior.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Tests for _parse_skills_list
# ---------------------------------------------------------------------------

class TestParseSkillsList:
    """Test the _parse_skills_list helper."""

    def test_none_returns_none(self):
        from hermes_cli.oneshot import _parse_skills_list
        assert _parse_skills_list(None) is None

    def test_empty_string_returns_none(self):
        from hermes_cli.oneshot import _parse_skills_list
        assert _parse_skills_list("") is None

    def test_comma_separated(self):
        from hermes_cli.oneshot import _parse_skills_list
        assert _parse_skills_list("a,b,c") == ["a", "b", "c"]

    def test_list_passthrough(self):
        from hermes_cli.oneshot import _parse_skills_list
        assert _parse_skills_list(["x", "y"]) == ["x", "y"]

    def test_whitespace_stripped(self):
        from hermes_cli.oneshot import _parse_skills_list
        assert _parse_skills_list(" a , b ") == ["a", "b"]


# ---------------------------------------------------------------------------
# Tests for run_oneshot env var setting
# ---------------------------------------------------------------------------

class TestOneshotEnvVars:
    """Test that --ignore-rules and --ignore-user-config set env vars."""

    def test_oneshot_ignore_rules_sets_env_var(self, monkeypatch):
        """run_oneshot(prompt, ignore_rules=True) should set HERMES_IGNORE_RULES=1."""
        from hermes_cli.oneshot import run_oneshot

        monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)

        with patch("hermes_cli.oneshot._run_agent", return_value="mock response"):
            run_oneshot("test", ignore_rules=True)

        assert os.environ.get("HERMES_IGNORE_RULES") == "1"

    def test_oneshot_ignore_user_config_sets_env_var(self, monkeypatch):
        """run_oneshot(prompt, ignore_user_config=True) should set HERMES_IGNORE_USER_CONFIG=1."""
        from hermes_cli.oneshot import run_oneshot

        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)

        with patch("hermes_cli.oneshot._run_agent", return_value="mock response"):
            run_oneshot("test", ignore_user_config=True)

        assert os.environ.get("HERMES_IGNORE_USER_CONFIG") == "1"


# ---------------------------------------------------------------------------
# Tests for skills injection in _run_agent
# ---------------------------------------------------------------------------

class TestOneshotSkillsInjection:
    """Test that --skills is honored in oneshot _run_agent."""

    def test_oneshot_applies_skills_to_system_prompt(self, monkeypatch):
        """_run_agent(prompt, skills='x') should inject skill body into ephemeral_system_prompt."""
        from hermes_cli.oneshot import _run_agent

        mock_agent = MagicMock()
        mock_agent.ephemeral_system_prompt = ""
        mock_agent.session_id = "test-session-001"
        mock_agent.chat.return_value = "mock response"

        with patch("run_agent.AIAgent", return_value=mock_agent), \
             patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None), \
             patch("hermes_cli.oneshot.get_fallback_chain", return_value=None), \
             patch("hermes_cli.config.load_config", return_value={"model": {}}), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "api_key": "k", "base_url": None, "provider": "test", "api_mode": None,
                 "credential_pool": None,
             }), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value=[]), \
             patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
             patch("agent.skill_commands.build_preloaded_skills_prompt",
                   return_value=("SKILL_BODY libero-decompose", ["libero-decompose"], [])):
            _run_agent("test prompt", skills="libero-decompose")

        assert "SKILL_BODY libero-decompose" in mock_agent.ephemeral_system_prompt
        assert mock_agent.chat.called

    def test_oneshot_skills_none_leaves_system_prompt_unchanged(self, monkeypatch):
        """When skills=None, ephemeral_system_prompt should not be modified."""
        from hermes_cli.oneshot import _run_agent

        mock_agent = MagicMock()
        mock_agent.ephemeral_system_prompt = ""
        mock_agent.session_id = "test-session-001"
        mock_agent.chat.return_value = "mock response"

        with patch("run_agent.AIAgent", return_value=mock_agent), \
             patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None), \
             patch("hermes_cli.oneshot.get_fallback_chain", return_value=None), \
             patch("hermes_cli.config.load_config", return_value={"model": {}}), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "api_key": "k", "base_url": None, "provider": "test", "api_mode": None,
                 "credential_pool": None,
             }), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value=[]), \
             patch("hermes_cli.models.detect_provider_for_model", return_value=None):
            _run_agent("test prompt", skills=None)

        assert mock_agent.ephemeral_system_prompt == ""

    def test_oneshot_unknown_skill_warns_not_fatal(self, monkeypatch, capsys):
        """An unknown skill name should print a warning to stderr, not raise."""
        from hermes_cli.oneshot import _run_agent

        mock_agent = MagicMock()
        mock_agent.ephemeral_system_prompt = ""
        mock_agent.session_id = "test-session-001"
        mock_agent.chat.return_value = "mock response"

        with patch("run_agent.AIAgent", return_value=mock_agent), \
             patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None), \
             patch("hermes_cli.oneshot.get_fallback_chain", return_value=None), \
             patch("hermes_cli.config.load_config", return_value={"model": {}}), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "api_key": "k", "base_url": None, "provider": "test", "api_mode": None,
                 "credential_pool": None,
             }), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value=[]), \
             patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
             patch("agent.skill_commands.build_preloaded_skills_prompt",
                   return_value=("", [], ["nonexistent-skill"])):
            # Should NOT raise — just warn
            result = _run_agent("test prompt", skills="nonexistent-skill")

        assert result == "mock response"
