"""Tests for the display.credits_notices config gate on _emit_credits_notices.

The toggle suppresses notice EMISSION only — credits state capture and /usage
stay live. Uses the bare-AIAgent pattern (object.__new__) from test_notice_spine.py.
"""
from __future__ import annotations

from unittest.mock import patch

from agent.credits_tracker import CreditsState
from run_agent import AIAgent


def _agent_with_state(*, paid_access: bool = False) -> AIAgent:
    """Bare agent with a depleted-shaped state that would normally emit."""
    agent = object.__new__(AIAgent)
    agent.notice_callback = None
    agent.notice_clear_callback = None
    agent._credits_state = CreditsState(paid_access=paid_access)
    agent.model = ""
    agent.base_url = ""
    return agent


def _cfg(enabled):
    return {"display": {"credits_notices": enabled}}


class TestCreditsNoticesToggle:
    def test_disabled_emits_nothing(self):
        agent = _agent_with_state()
        received = []
        agent.notice_callback = received.append
        with patch("hermes_cli.config.load_config", return_value=_cfg(False)):
            agent._emit_credits_notices()
        assert received == []

    def test_enabled_emits_depleted(self):
        agent = _agent_with_state()
        received = []
        agent.notice_callback = received.append
        with patch("hermes_cli.config.load_config", return_value=_cfg(True)):
            agent._emit_credits_notices()
        assert any(getattr(n, "key", None) == "credits.depleted" for n in received)

    def test_default_missing_key_emits(self):
        """Key absent from config → fail-open True (current behaviour preserved)."""
        agent = _agent_with_state()
        received = []
        agent.notice_callback = received.append
        with patch("hermes_cli.config.load_config", return_value={"display": {}}):
            agent._emit_credits_notices()
        assert any(getattr(n, "key", None) == "credits.depleted" for n in received)

    def test_config_error_fails_open(self):
        agent = _agent_with_state()
        received = []
        agent.notice_callback = received.append
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            agent._emit_credits_notices()
        assert any(getattr(n, "key", None) == "credits.depleted" for n in received)

    def test_toggle_cached_per_agent(self):
        """load_config is consulted once per agent, not once per emission."""
        agent = _agent_with_state()
        agent.notice_callback = lambda n: None
        with patch("hermes_cli.config.load_config", return_value=_cfg(True)) as mock_load:
            agent._emit_credits_notices()
            agent._emit_credits_notices()
        assert mock_load.call_count == 1

    def test_disabled_state_still_cached_for_usage(self):
        """The gate stops emission only — get_credits_state still returns data."""
        agent = _agent_with_state()
        agent.notice_callback = lambda n: None
        agent._credits_session_start_micros = None
        with patch("hermes_cli.config.load_config", return_value=_cfg(False)):
            agent._emit_credits_notices()
        assert agent.get_credits_state() is not None
