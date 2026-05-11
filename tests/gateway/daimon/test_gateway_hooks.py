"""Tests for gateway.daimon.gateway_hooks module."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from gateway.daimon.gateway_hooks import (
    apply_overrides,
    gate_tool_call,
    get_agent_overrides,
    load_system_prompt,
    redact_output,
    setup_tool_gate,
    teardown_tool_gate,
)
from gateway.daimon.agent_overrides import AgentOverrides
from gateway.daimon.tier import Tier
from gateway.daimon.tool_gate import get_limiter


def _make_config(admin_users=None, user_model=None, admin_model=None, tool_limits=None, **kwargs):
    """Build a raw config dict matching gateway.discord.daimon namespace."""
    daimon = {}
    if admin_users is not None:
        daimon["admin_users"] = admin_users
    if user_model is not None:
        daimon["user_model"] = user_model
    if admin_model is not None:
        daimon["admin_model"] = admin_model
    if tool_limits is not None:
        daimon["tool_limits"] = tool_limits
    daimon.update(kwargs)
    return {"gateway": {"discord": {"daimon": daimon}}}


class TestGetAgentOverridesNonDiscord:
    """Returns None for non-discord platforms."""

    def test_telegram(self):
        cfg = _make_config(admin_users=["admin1"])
        result = get_agent_overrides(cfg, "admin1", "telegram")
        assert result is None

    def test_slack(self):
        cfg = _make_config(admin_users=["admin1"])
        result = get_agent_overrides(cfg, "admin1", "slack")
        assert result is None

    def test_cli(self):
        cfg = _make_config(admin_users=["admin1"])
        result = get_agent_overrides(cfg, "admin1", "cli")
        assert result is None


class TestGetAgentOverridesAdmin:
    """Admin user gets admin model."""

    def test_returns_admin_model(self):
        cfg = _make_config(admin_users=["admin1"], admin_model="openai/gpt-4o")
        result = get_agent_overrides(cfg, "admin1", "discord")
        assert result is not None
        assert result.model == "openai/gpt-4o"
        assert result.tier == Tier.ADMIN
        assert result.max_iterations is None
        assert result.disabled_toolsets is None

    def test_default_admin_model(self):
        cfg = _make_config(admin_users=["admin1"])
        result = get_agent_overrides(cfg, "admin1", "discord")
        assert result.model == "anthropic/claude-sonnet-4.6"


class TestGetAgentOverridesUser:
    """Regular user gets user model + caps."""

    def test_returns_user_model_and_caps(self):
        cfg = _make_config(admin_users=["admin1"], user_model="openai/gpt-4o-mini")
        result = get_agent_overrides(cfg, "regular_user", "discord")
        assert result is not None
        assert result.model == "openai/gpt-4o-mini"
        assert result.tier == Tier.USER
        assert result.max_iterations == 30

    def test_disabled_toolsets_for_zero_limit(self):
        cfg = _make_config(admin_users=["admin1"])
        result = get_agent_overrides(cfg, "regular_user", "discord")
        # Default config has text_to_speech, cronjob, send_message at limit=0
        assert "text_to_speech" in result.disabled_toolsets
        assert "cronjob" in result.disabled_toolsets
        assert "send_message" in result.disabled_toolsets


class TestLoadSystemPrompt:
    """Tests for load_system_prompt."""

    def test_reads_prompt_file(self):
        prompt = load_system_prompt()
        assert len(prompt) > 0
        assert "Daimon" in prompt

    def test_returns_empty_if_missing(self):
        with patch("gateway.daimon.gateway_hooks._SYSTEM_PROMPT_PATH") as mock_path:
            mock_path.exists.return_value = False
            from gateway.daimon import gateway_hooks
            # Need to call with the patched value
            original = gateway_hooks._SYSTEM_PROMPT_PATH
            try:
                from pathlib import Path
                gateway_hooks._SYSTEM_PROMPT_PATH = Path("/nonexistent/path/file.md")
                result = gateway_hooks.load_system_prompt()
                assert result == ""
            finally:
                gateway_hooks._SYSTEM_PROMPT_PATH = original


class TestSetupAndTeardownToolGate:
    """Tests for setup_tool_gate and teardown_tool_gate."""

    def test_registers_and_unregisters_limiter(self):
        cfg = _make_config(admin_users=["admin1"])
        session_id = "test-session-001"

        # Should not have a limiter initially
        assert get_limiter(session_id) is None

        # Setup registers a limiter
        setup_tool_gate(session_id, cfg)
        limiter = get_limiter(session_id)
        assert limiter is not None

        # Teardown removes it
        teardown_tool_gate(session_id)
        assert get_limiter(session_id) is None


class TestGateToolCallNoLimiter:
    """No limiter registered means tool calls are allowed."""

    def test_returns_none_when_no_limiter(self):
        result = gate_tool_call("nonexistent-session", "terminal")
        assert result is None


class TestGateToolCallBlocked:
    """Tool calls blocked when limit is reached."""

    def test_returns_denial_when_limit_hit(self):
        cfg = _make_config(admin_users=["admin1"], tool_limits={"web_search": 1})
        session_id = "test-session-blocked"

        setup_tool_gate(session_id, cfg)
        try:
            # First call allowed
            result1 = gate_tool_call(session_id, "web_search")
            assert result1 is None

            # Second call blocked
            result2 = gate_tool_call(session_id, "web_search")
            assert result2 is not None
            assert "web_search" in result2
            assert "limit" in result2.lower() or "reached" in result2.lower()
        finally:
            teardown_tool_gate(session_id)

    def test_returns_denial_for_disabled_tool(self):
        cfg = _make_config(admin_users=["admin1"], tool_limits={"cronjob": 0})
        session_id = "test-session-disabled"

        setup_tool_gate(session_id, cfg)
        try:
            result = gate_tool_call(session_id, "cronjob")
            assert result is not None
            assert "disabled" in result.lower() or "cronjob" in result
        finally:
            teardown_tool_gate(session_id)


class TestRedactOutput:
    """Tests for redact_output delegation."""

    def test_redacts_api_key(self):
        text = "Here is a key: sk-proj-abcdefghij1234567890abcd"
        result = redact_output(text)
        assert "sk-proj-" not in result
        assert "[REDACTED" in result

    def test_passes_through_safe_text(self):
        text = "Hello, how can I help you today?"
        result = redact_output(text)
        assert result == text


class TestApplyOverridesUser:
    """apply_overrides for user tier."""

    def test_applies_model_iterations_disabled_prompt(self):
        overrides = AgentOverrides(
            model="openai/gpt-4o-mini",
            max_iterations=30,
            disabled_toolsets=["cronjob", "send_message"],
            tier=Tier.USER,
        )
        result = apply_overrides(
            overrides,
            model="anthropic/claude-sonnet-4.6",
            max_iterations=90,
            disabled_toolsets=None,
        )
        assert result["model"] == "openai/gpt-4o-mini"
        assert result["max_iterations"] == 30
        assert "cronjob" in result["disabled_toolsets"]
        assert "send_message" in result["disabled_toolsets"]
        # User gets system prompt
        assert result["ephemeral_system_prompt"] is not None
        assert "Daimon" in result["ephemeral_system_prompt"]


class TestApplyOverridesAdmin:
    """apply_overrides for admin tier."""

    def test_applies_model_only_no_prompt_no_disabled(self):
        overrides = AgentOverrides(
            model="openai/gpt-4o",
            tier=Tier.ADMIN,
        )
        result = apply_overrides(
            overrides,
            model="anthropic/claude-sonnet-4.6",
            max_iterations=90,
            disabled_toolsets=None,
        )
        assert result["model"] == "openai/gpt-4o"
        assert result["max_iterations"] == 90  # unchanged
        assert result["disabled_toolsets"] is None
        # Admin does NOT get system prompt
        assert result["ephemeral_system_prompt"] is None


class TestApplyOverridesMergeDisabled:
    """apply_overrides merges disabled_toolsets additively."""

    def test_merges_with_existing_disabled_toolsets(self):
        overrides = AgentOverrides(
            model="openai/gpt-4o-mini",
            max_iterations=30,
            disabled_toolsets=["cronjob", "send_message"],
            tier=Tier.USER,
        )
        result = apply_overrides(
            overrides,
            model="anthropic/claude-sonnet-4.6",
            max_iterations=90,
            disabled_toolsets=["image_generate"],
        )
        disabled = result["disabled_toolsets"]
        assert "image_generate" in disabled
        assert "cronjob" in disabled
        assert "send_message" in disabled

    def test_no_duplicates_in_merged(self):
        overrides = AgentOverrides(
            model="openai/gpt-4o-mini",
            max_iterations=30,
            disabled_toolsets=["cronjob", "send_message"],
            tier=Tier.USER,
        )
        result = apply_overrides(
            overrides,
            model="anthropic/claude-sonnet-4.6",
            max_iterations=90,
            disabled_toolsets=["cronjob"],  # already in overrides
        )
        disabled = result["disabled_toolsets"]
        # No duplicates
        assert disabled.count("cronjob") == 1
