"""Regression tests: /reasoning and /fast update agent config in-place.

When a live agent exists, /reasoning <level> and /fast <mode> must update
the agent's config attributes in-place rather than destroying the agent
(self.agent = None).  Destroying the agent triggers:

- MCP server rediscovery + disconnect storm on the next message
- Prompt cache invalidation (per-conversation caching is sacred)
- Visible UX error: "Agent updated — 0 tool(s) available"

Both reasoning_config and service_tier are read dynamically on every API
call (agent/chat_completion_helpers.py and cli_agent_setup_mixin.py), so
an in-place update takes effect immediately without a rebuild.
"""

import os
from unittest.mock import MagicMock

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _Stub(CLICommandsMixin):
    """Minimal carrier for the attributes the commands read/write."""

    def __init__(self, agent=None):
        self.reasoning_config = None
        self.show_reasoning = True
        self.reasoning_full = False
        self.service_tier = None
        self.agent = agent

    def _current_reasoning_callback(self):
        return None

    def _fast_command_available(self):
        return True


def _seed_config(tmp_path, monkeypatch):
    hh = tmp_path / ".hermes"
    hh.mkdir()
    (hh / "config.yaml").write_text("display:\n  show_reasoning: true\n")
    monkeypatch.setenv("HERMES_HOME", str(hh))
    import cli
    monkeypatch.setattr(cli, "_hermes_home", hh, raising=False)
    return hh


# ── /reasoning in-place ──────────────────────────────────────────────


class TestReasoningInPlace:
    """When agent is live, /reasoning updates config in-place."""

    def test_reasoning_level_updates_agent_in_place(self, tmp_path, monkeypatch):
        """Live agent: reasoning_config is set on the agent, not destroyed."""
        _seed_config(tmp_path, monkeypatch)
        agent = MagicMock()
        agent.reasoning_config = {"enabled": True, "effort": "medium"}
        s = _Stub(agent=agent)

        s._handle_reasoning_command("/reasoning high")

        assert s.agent is agent  # agent NOT destroyed
        assert s.reasoning_config == {"enabled": True, "effort": "high"}
        assert agent.reasoning_config == {"enabled": True, "effort": "high"}

    def test_reasoning_none_agent_stays_none(self, tmp_path, monkeypatch):
        """No live agent: agent stays None (backward compatible)."""
        _seed_config(tmp_path, monkeypatch)
        s = _Stub(agent=None)

        s._handle_reasoning_command("/reasoning high")

        assert s.agent is None
        assert s.reasoning_config == {"enabled": True, "effort": "high"}

    def test_reasoning_preserves_agent_identity(self, tmp_path, monkeypatch):
        """The exact same agent object survives across multiple /reasoning calls."""
        _seed_config(tmp_path, monkeypatch)
        agent = MagicMock()
        agent.reasoning_config = {"enabled": True, "effort": "medium"}
        s = _Stub(agent=agent)

        s._handle_reasoning_command("/reasoning high")
        agent_after_first = s.agent

        s._handle_reasoning_command("/reasoning max")
        agent_after_second = s.agent

        assert agent_after_first is agent
        assert agent_after_second is agent
        assert agent.reasoning_config == {"enabled": True, "effort": "max"}


# ── /fast in-place ───────────────────────────────────────────────────


class TestFastInPlace:
    """When agent is live, /fast updates service_tier in-place."""

    def test_fast_on_updates_agent_in_place(self, tmp_path, monkeypatch):
        """Live agent: service_tier is set on the agent, not destroyed."""
        _seed_config(tmp_path, monkeypatch)
        agent = MagicMock()
        agent.service_tier = None
        s = _Stub(agent=agent)

        s._handle_fast_command("/fast fast")

        assert s.agent is agent  # agent NOT destroyed
        assert s.service_tier == "priority"
        assert agent.service_tier == "priority"

    def test_fast_off_updates_agent_in_place(self, tmp_path, monkeypatch):
        """Live agent: turning fast off clears service_tier in-place."""
        _seed_config(tmp_path, monkeypatch)
        agent = MagicMock()
        agent.service_tier = "priority"
        s = _Stub(agent=agent)
        s.service_tier = "priority"

        s._handle_fast_command("/fast normal")

        assert s.agent is agent
        assert s.service_tier is None
        assert agent.service_tier is None

    def test_fast_none_agent_stays_none(self, tmp_path, monkeypatch):
        """No live agent: agent stays None (backward compatible)."""
        _seed_config(tmp_path, monkeypatch)
        s = _Stub(agent=None)

        s._handle_fast_command("/fast fast")

        assert s.agent is None
        assert s.service_tier == "priority"

    def test_fast_preserves_agent_identity(self, tmp_path, monkeypatch):
        """The exact same agent object survives across multiple /fast calls."""
        _seed_config(tmp_path, monkeypatch)
        agent = MagicMock()
        agent.service_tier = None
        s = _Stub(agent=agent)

        s._handle_fast_command("/fast fast")
        agent_after_on = s.agent

        s._handle_fast_command("/fast normal")
        agent_after_off = s.agent

        assert agent_after_on is agent
        assert agent_after_off is agent
