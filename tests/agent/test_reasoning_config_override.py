"""Per-turn reasoning override plumbing tests."""

from __future__ import annotations

from typing import Any

from agent.chat_completion_helpers import build_api_kwargs


class _CapturingTransport:
    def __init__(self):
        self.params: dict[str, Any] | None = None

    def build_kwargs(self, **kwargs):
        self.params = kwargs
        return kwargs


class _FakeCodexAgent:
    def __init__(self):
        self.api_mode = "codex_responses"
        self.provider = "openai-codex"
        self.model = "gpt-5.5"
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self._base_url_hostname = "chatgpt.com"
        self._base_url_lower = self.base_url
        self.tools = []
        self.max_tokens = None
        self.request_overrides = {}
        self.reasoning_config = {"enabled": True, "effort": "xhigh"}
        self._turn_reasoning_config_override: dict[str, Any] | None = None
        self.session_id = "sess-1"
        self._codex_reasoning_replay_enabled = True
        self.transport = _CapturingTransport()

    def _get_transport(self):
        return self.transport

    def _prepare_messages_for_non_vision_model(self, messages):
        return messages

    def _github_models_reasoning_extra_body(self):
        return None

    def _resolved_api_call_timeout(self):
        return None


def test_build_api_kwargs_prefers_turn_reasoning_override():
    agent = _FakeCodexAgent()
    agent._turn_reasoning_config_override = {"enabled": False}

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert kwargs["reasoning_config"] == {"enabled": False}
    assert agent.transport.params is not None
    assert agent.transport.params["reasoning_config"] == {"enabled": False}


def test_build_api_kwargs_uses_static_reasoning_when_no_turn_override():
    agent = _FakeCodexAgent()

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
