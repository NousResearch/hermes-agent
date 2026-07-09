from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _clarify_tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "clarify",
                "description": "Ask for a choice",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _mock_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _mock_tool_response(name: str = "clarify", arguments: str = "{}"):
    tc = SimpleNamespace(
        id="call_clarify_1",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    msg = SimpleNamespace(content="", tool_calls=[tc])
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_clarify_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform="desktop",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    return agent


DEAD_MENU = """## 결과 요약
완료했습니다.

## 다음 추천 작업
- 그대로 두기
- 소스 하드게이트 구현하기
- 보류
"""

ENGLISH_FENCED_MENU = """## Next actions

```select
A. Continue
B. Stop
```
"""


def test_run_conversation_retries_markdown_only_next_action_choices():
    agent = _make_agent()
    calls = []

    def fake_api_call(api_kwargs):
        calls.append(api_kwargs)
        if len(calls) == 1:
            return _mock_response(DEAD_MENU)
        return _mock_response("선택이 필요 없어 기본 권장안으로 마무리합니다.")

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation("진단해줘")

    assert len(calls) == 2
    assert result["completed"] is True
    assert result["final_response"] == "선택이 필요 없어 기본 권장안으로 마무리합니다."
    assert not any(m.get("_clarify_choice_guard_synthetic") for m in result["messages"])
    assert not any(DEAD_MENU in (m.get("content") or "") for m in result["messages"])


def test_run_conversation_cleans_dead_menu_when_retry_turn_calls_clarify_tool():
    agent = _make_agent()
    calls = []

    def fake_api_call(api_kwargs):
        calls.append(api_kwargs)
        if len(calls) == 1:
            return _mock_response(DEAD_MENU)
        if len(calls) == 2:
            return _mock_tool_response(
                arguments='{"question":"어떻게 할까?","choices":["A","B"]}'
            )
        return _mock_response("clarify 도구 호출 뒤 최종 마무리")

    def fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count):
        for tc in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "name": tc.function.name,
                    "tool_call_id": tc.id,
                    "content": "selected: A",
                }
            )

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_execute_tool_calls", side_effect=fake_execute_tool_calls),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation("진단해줘")

    assert len(calls) == 3
    assert result["completed"] is True
    assert result["final_response"] == "clarify 도구 호출 뒤 최종 마무리"
    assert not any(m.get("_clarify_choice_guard_synthetic") for m in result["messages"])
    assert not any(DEAD_MENU in (m.get("content") or "") for m in result["messages"])
    assert not any("Markdown-only next-action" in (m.get("content") or "") for m in result["messages"])


def test_run_conversation_blocks_dead_choices_after_retry_budget():
    agent = _make_agent()

    with (
        patch.object(
            agent,
            "_interruptible_api_call",
            side_effect=[_mock_response(DEAD_MENU), _mock_response(DEAD_MENU)],
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation("진단해줘")

    assert result["completed"] is True
    assert "Clarify hard gate blocked" in result["final_response"]
    assert "## 다음 추천 작업" not in result["final_response"]
    assert not any(DEAD_MENU in (m.get("content") or "") for m in result["messages"])


def test_run_conversation_fails_closed_if_guard_check_crashes_on_dead_menu():
    agent = _make_agent()

    with (
        patch.object(agent, "_interruptible_api_call", return_value=_mock_response(DEAD_MENU)),
        patch("agent.clarify_choice_guard.clarify_choice_guard_action", side_effect=RuntimeError("boom")),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation("진단해줘")

    assert result["completed"] is True
    assert "Clarify hard gate failed" in result["final_response"]
    assert "dead choice UI" in result["final_response"]
    assert "## 다음 추천 작업" not in result["final_response"]
    assert not any(DEAD_MENU in (m.get("content") or "") for m in result["messages"])


def test_run_conversation_fails_closed_if_guard_check_crashes_on_english_fenced_select():
    agent = _make_agent()

    with (
        patch.object(agent, "_interruptible_api_call", return_value=_mock_response(ENGLISH_FENCED_MENU)),
        patch("agent.clarify_choice_guard.clarify_choice_guard_action", side_effect=RuntimeError("boom")),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation("diagnose")

    assert result["completed"] is True
    assert "Clarify hard gate failed" in result["final_response"]
    assert "dead choice UI" in result["final_response"]
    assert "```select" not in result["final_response"]
    assert not any(ENGLISH_FENCED_MENU in (m.get("content") or "") for m in result["messages"])


def test_drop_trailing_scaffolding_removes_clarify_choice_synthetic_tail():
    agent = _make_agent()
    messages = [
        {"role": "user", "content": "진단해줘"},
        {
            "role": "assistant",
            "content": DEAD_MENU,
            "_clarify_choice_guard_synthetic": True,
        },
        {
            "role": "user",
            "content": "[System: call clarify instead of Markdown choices]",
            "_clarify_choice_guard_synthetic": True,
        },
    ]

    agent._drop_trailing_empty_response_scaffolding(messages)

    assert messages == [{"role": "user", "content": "진단해줘"}]
