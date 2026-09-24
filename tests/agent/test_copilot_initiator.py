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
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: _tool_defs("web_search"))
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", _FakeOpenAI)
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




class TestIsCopilotUrl:
    """_is_copilot_url() detects GitHub Copilot endpoints."""

    def test_standard_copilot_url(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        assert agent._is_copilot_url() is True


    def test_github_models_url(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://models.github.ai/inference")
        assert agent._is_copilot_url() is True

    def test_openrouter_url(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://openrouter.ai/api/v1")
        assert agent._is_copilot_url() is False



class TestUserInitiatedTurnFlag:
    """_is_user_initiated_turn lifecycle."""


    def test_reset_session_clears_flag(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        agent._is_user_initiated_turn = True
        agent.reset_session_state()
        assert agent._is_user_initiated_turn is False




class TestHeaderValues:
    """copilot_default_headers(is_agent_turn=...) sets x-initiator correctly."""

    def test_default_is_agent(self):
        from hermes_cli.models import copilot_default_headers
        assert copilot_default_headers()["x-initiator"] == "agent"

    def test_user_turn(self):
        from hermes_cli.models import copilot_default_headers
        assert copilot_default_headers(is_agent_turn=False)["x-initiator"] == "user"

