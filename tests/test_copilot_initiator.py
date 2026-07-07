"""Tests for per-turn Copilot x-initiator header injection (issue #3040).

Copilot bills "premium requests" only when a request is marked as
user-initiated via the ``x-initiator: user`` header. Hermes previously sent
``x-initiator: agent`` on every request (client-level default headers), so
user prompts never consumed premium requests and were throttled as agent
traffic. The fix marks the FIRST API call of each user turn as "user" and
lets tool-loop follow-ups keep the "agent" default.

Salvaged from PR #4097 (@tjp2021); adapted to the post-refactor layout
(conversation_loop.py owns the injection site, the codex transport now
accepts extra_headers).
"""

import pytest

from run_agent import AIAgent


def _tool_defs(*names):
    return [
        {"type": "function", "function": {"name": n, "description": n, "parameters": {}}}
        for n in names
    ]


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _make_agent(monkeypatch, base_url, api_mode="chat_completions"):
    """Create an AIAgent pointing at the given base_url."""
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: _tool_defs("web_search"))
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider="copilot" if "githubcopilot" in base_url else "openrouter",
        api_mode=api_mode,
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _inject(agent, api_kwargs):
    """Mirror the injection block in agent/conversation_loop.py."""
    if getattr(agent, "_is_user_initiated_turn", False) and getattr(agent, "_is_copilot_url", lambda: False)():
        _xh = dict(api_kwargs.get("extra_headers") or {})
        _xh["x-initiator"] = "user"
        api_kwargs["extra_headers"] = _xh
        agent._is_user_initiated_turn = False
    return api_kwargs


class TestIsCopilotUrl:
    """_is_copilot_url() detects GitHub Copilot endpoints."""

    def test_standard_copilot_url(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        assert agent._is_copilot_url() is True

    def test_copilot_url_with_path(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com/v1")
        assert agent._is_copilot_url() is True

    def test_github_models_url(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://models.github.ai/inference")
        assert agent._is_copilot_url() is True

    def test_openrouter_url(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://openrouter.ai/api/v1")
        assert agent._is_copilot_url() is False

    def test_case_insensitive(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://API.GITHUBCOPILOT.COM")
        assert agent._is_copilot_url() is True


class TestUserInitiatedTurnFlag:
    """_is_user_initiated_turn lifecycle."""

    def test_default_is_false(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        assert agent._is_user_initiated_turn is False

    def test_reset_session_clears_flag(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        agent._is_user_initiated_turn = True
        agent.reset_session_state()
        assert agent._is_user_initiated_turn is False


class TestFlagFlipOnInjection:
    """Flag flips immediately on injection so tool-loop calls use 'agent'."""

    def test_first_call_injects_user_initiator(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        agent._is_user_initiated_turn = True
        kwargs = _inject(agent, {})
        assert kwargs["extra_headers"] == {"x-initiator": "user"}
        assert agent._is_user_initiated_turn is False

    def test_second_call_has_no_injection(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        agent._is_user_initiated_turn = True
        kwargs1 = _inject(agent, {})
        kwargs2 = _inject(agent, {})
        assert "extra_headers" in kwargs1
        assert "extra_headers" not in kwargs2

    def test_existing_extra_headers_preserved(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        agent._is_user_initiated_turn = True
        kwargs = _inject(agent, {"extra_headers": {"x-custom": "1"}})
        assert kwargs["extra_headers"]["x-custom"] == "1"
        assert kwargs["extra_headers"]["x-initiator"] == "user"

    def test_non_copilot_flag_not_flipped(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://openrouter.ai/api/v1")
        agent._is_user_initiated_turn = True
        kwargs = _inject(agent, {})
        assert "extra_headers" not in kwargs
        # Flag unchanged — non-Copilot path doesn't touch it
        assert agent._is_user_initiated_turn is True

    def test_missing_is_copilot_url_does_not_crash(self):
        """Guard against intermittent missing _is_copilot_url (#59845)."""
        # Simulate a partially-initialized agent (module-reload / wrapper)
        # that has _is_user_initiated_turn but not _is_copilot_url.
        class PartialAgent:
            _is_user_initiated_turn = True
        agent = PartialAgent()
        kwargs = _inject(agent, {})
        assert "extra_headers" not in kwargs
        # Flag unchanged — guard falls through cleanly
        assert agent._is_user_initiated_turn is True


class TestHeaderValues:
    """copilot_default_headers(is_agent_turn=...) sets x-initiator correctly."""

    def test_default_is_agent(self):
        from hermes_cli.models import copilot_default_headers
        assert copilot_default_headers()["x-initiator"] == "agent"

    def test_user_turn(self):
        from hermes_cli.models import copilot_default_headers
        assert copilot_default_headers(is_agent_turn=False)["x-initiator"] == "user"

    def test_agent_turn_explicit(self):
        from hermes_cli.models import copilot_default_headers
        assert copilot_default_headers(is_agent_turn=True)["x-initiator"] == "agent"
