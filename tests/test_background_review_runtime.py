"""Regression tests for background review agent runtime inheritance."""

import threading

from run_agent import AIAgent


class _ImmediateThread:
    def __init__(self, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target is not None:
            self._target()


def test_background_review_inherits_runtime(monkeypatch):
    captured = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self._memory_store = None
            self._memory_enabled = False
            self._user_profile_enabled = False
            self._memory_nudge_interval = 99
            self._skill_nudge_interval = 99

        def run_conversation(self, user_message, conversation_history):
            captured["run"] = {
                "user_message": user_message,
                "conversation_history": conversation_history,
            }

    monkeypatch.setattr("run_agent.AIAgent", FakeReviewAgent)
    monkeypatch.setattr(threading, "Thread", _ImmediateThread)

    agent = AIAgent.__new__(AIAgent)
    agent.model = "qwen3.5-9b-local"
    agent.platform = "cli"
    agent.provider = "openrouter"
    agent.api_key = "sk-doodles"
    agent.base_url = "https://litellm.ferret-beta.ts.net/v1"
    agent.api_mode = "chat_completions"
    agent.acp_command = None
    agent.acp_args = []
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = True

    messages = [{"role": "user", "content": "hello"}]
    agent._spawn_background_review(messages_snapshot=messages, review_memory=True, review_skills=False)

    assert captured["init"]["model"] == "qwen3.5-9b-local"
    assert captured["init"]["provider"] == "openrouter"
    assert captured["init"]["api_key"] == "sk-doodles"
    assert captured["init"]["base_url"] == "https://litellm.ferret-beta.ts.net/v1"
    assert captured["init"]["api_mode"] == "chat_completions"
    assert captured["run"]["conversation_history"] == messages
    assert "memory" in captured["run"]["user_message"].lower()
