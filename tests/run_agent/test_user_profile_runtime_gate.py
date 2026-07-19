"""Runtime enforcement for opt-in USER.md access."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tools.memory_tool import MemoryStore


def _tool_defs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "memory",
                "description": "memory tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _make_agent(tmp_path, monkeypatch, *, user_profile_enabled: bool) -> AIAgent:
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent.tool_delay = 0
    agent._flush_messages_to_session_db = MagicMock()
    agent._memory_store = MemoryStore()
    agent._memory_store.load_from_disk()
    agent._memory_manager = None
    agent._memory_enabled = True
    agent._user_profile_enabled = user_profile_enabled
    return agent


def _memory_call(content: str, call_id: str = "memory-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name="memory",
            arguments=json.dumps(
                {"action": "add", "target": "user", "content": content}
            ),
        ),
    )


def _execute(agent: AIAgent, dispatch_mode: str, content: str) -> dict:
    messages: list[dict] = []
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[_memory_call(content)],
    )
    execute = getattr(agent, f"_execute_tool_calls_{dispatch_mode}")
    execute(assistant_message, messages, "test-task")
    assert len(messages) == 1
    return json.loads(messages[0]["content"])


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_user_profile_target_is_blocked_without_opt_in(
    tmp_path,
    monkeypatch,
    dispatch_mode: str,
):
    agent = _make_agent(tmp_path, monkeypatch, user_profile_enabled=False)

    result = _execute(agent, dispatch_mode, "must not be captured")

    assert result["success"] is False
    assert "memory.user_profile_enabled" in result["error"]
    assert not (tmp_path / "hermes-home" / "memories" / "USER.md").exists()


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_user_profile_target_is_allowed_with_explicit_opt_in(
    tmp_path,
    monkeypatch,
    dispatch_mode: str,
):
    agent = _make_agent(tmp_path, monkeypatch, user_profile_enabled=True)

    result = _execute(agent, dispatch_mode, "explicitly allowed")

    assert result["success"] is True
    user_file = tmp_path / "hermes-home" / "memories" / "USER.md"
    assert "explicitly allowed" in user_file.read_text(encoding="utf-8")


class _ImmediateThread:
    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


def test_background_review_cannot_capture_user_profile_without_opt_in(
    tmp_path,
    monkeypatch,
):
    import run_agent as run_agent_module
    from agent.agent_runtime_helpers import invoke_tool

    parent = _make_agent(tmp_path, monkeypatch, user_profile_enabled=False)
    captured: dict[str, object] = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self.session_id = "review-session"
            self._memory_manager = None
            self._session_messages: list[dict] = []

        def run_conversation(self, **kwargs):
            result = invoke_tool(
                self,
                "memory",
                {
                    "action": "add",
                    "target": "user",
                    "content": "background capture",
                },
                "background-review",
                pre_tool_block_checked=True,
                skip_tool_request_middleware=True,
            )
            captured["result"] = json.loads(result)

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", _ImmediateThread)

    AIAgent._spawn_background_review(
        parent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert captured["result"]["success"] is False
    assert not (tmp_path / "hermes-home" / "memories" / "USER.md").exists()
