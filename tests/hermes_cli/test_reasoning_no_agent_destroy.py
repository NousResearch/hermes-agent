"""Regression tests: /reasoning must NOT destroy agent (prevents MCP reload regression).

Bug A (fixed in 91d2118ba4, cherry-picked 2026-07-16):
  /reasoning set self.agent = None, destroying the agent and triggering
  MCP rediscovery + prompt cache invalidation on every invocation.
  reasoning_config is read at API-call time, so in-place update suffices.

These tests prevent the regression where agent destruction silently returns.
"""

import yaml

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _StubAgent:
    """Fake agent to verify in-place reasoning_config update."""

    def __init__(self):
        self.reasoning_config = None
        self.reasoning_callback = None


class _Stub(CLICommandsMixin):
    """Minimal carrier for attributes _handle_reasoning_command reads."""

    def __init__(self):
        self.reasoning_config: dict | None = None
        self.show_reasoning: bool = True
        self.reasoning_full: bool = False
        self.agent: _StubAgent | None = None

    def _current_reasoning_callback(self):
        return None


def _seed_config(tmp_path, monkeypatch, config_yaml: str = ""):
    hh = tmp_path / ".hermes"
    hh.mkdir()
    (hh / "config.yaml").write_text(config_yaml or "agent:\n  reasoning_effort: medium\n")
    monkeypatch.setenv("HERMES_HOME", str(hh))
    import cli
    monkeypatch.setattr(cli, "_hermes_home", hh, raising=False)
    return hh


class TestReasoningNoAgentDestroy:
    """Regression: /reasoning must NOT set self.agent = None."""

    def test_reasoning_does_not_destroy_agent(self, tmp_path, monkeypatch):
        """Agent object identity is preserved after /reasoning."""
        _seed_config(tmp_path, monkeypatch)
        s = _Stub()
        agent = _StubAgent()
        s.agent = agent

        s._handle_reasoning_command("/reasoning high")

        assert s.agent is agent  # same object, not destroyed

    def test_reasoning_updates_agent_in_place(self, tmp_path, monkeypatch):
        """reasoning_config is set directly on the existing agent."""
        _seed_config(tmp_path, monkeypatch)
        s = _Stub()
        agent = _StubAgent()
        s.agent = agent

        s._handle_reasoning_command("/reasoning high")

        assert agent.reasoning_config == {"enabled": True, "effort": "high"}

    def test_reasoning_xhigh_does_not_destroy_agent(self, tmp_path, monkeypatch):
        """All effort levels preserve agent identity."""
        _seed_config(tmp_path, monkeypatch)
        s = _Stub()
        agent = _StubAgent()
        s.agent = agent

        for level in ("none", "minimal", "low", "medium", "high", "xhigh"):
            s._handle_reasoning_command(f"/reasoning {level}")
            assert s.agent is agent, f"agent destroyed on /reasoning {level}"

    def test_reasoning_agent_none_is_safe(self, tmp_path, monkeypatch):
        """Does not crash when agent is None (not yet initialized)."""
        _seed_config(tmp_path, monkeypatch)
        s = _Stub()
        s.agent = None  # before first agent init

        # Must not raise
        s._handle_reasoning_command("/reasoning high")

        assert s.agent is None  # still None, no crash
