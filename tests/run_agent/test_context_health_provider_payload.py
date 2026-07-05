from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent


OLD_TASK_A_RAW_NEEDLE = "OLD_TASK_A_RAW_NEEDLE_PHASE3_RED"
OLD_TASK_A_ASSISTANT_NEEDLE = "OLD_TASK_A_ASSISTANT_NEEDLE_PHASE3_RED"
OLD_TASK_A_TOOL_RESULT_NEEDLE = "OLD_TASK_A_TOOL_RESULT_NEEDLE_PHASE3_RED"
PRIOR_PHASE2_RAW_LONG_PROMPT_NEEDLE = "PRIOR_PHASE2_RAW_LONG_PROMPT_NEEDLE_PHASE3_RED"
CURRENT_TASK_B_WCP_MARKER = "CURRENT_TASK_B_WCP_MARKER_PHASE3_RED"
CURRENT_TASK_B_NEEDLE = "CURRENT_TASK_B_NEEDLE_PHASE3_RED"
PRE_API_OLD_TASK_A_NEEDLE = "PRE_API_OLD_TASK_A_NEEDLE_PHASE3_RED"
PRE_API_CURRENT_WCP_MARKER = "PRE_API_CURRENT_WCP_MARKER_PHASE3_RED"
DISABLED_PASS_THROUGH_NEEDLE = "DISABLED_WCP_PASS_THROUGH_NEEDLE_PHASE3_RED"
UNSAFE_WCP_SECRET_NEEDLE = "UNSAFE_WCP_SECRET_NEEDLE_PHASE3_RED"


def _response(content: str = "ok"):
    message = SimpleNamespace(content=content, reasoning=None, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _make_agent(*, wcp_enabled: bool) -> AIAgent:
    cfg = {
        "context_health": {
            "enabled": bool(wcp_enabled),
            "runtime_behavior_enabled": bool(wcp_enabled),
            "pre_model_intake": {"enabled": False},
            "working_context_packet": {"enabled": bool(wcp_enabled)},
        },
        "agent": {"api_max_retries": 1},
    }
    with (
        patch("run_agent.OpenAI"),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=cfg),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    setattr(agent, "context_health", cfg["context_health"])
    agent.client.chat.completions.create.return_value = _response()
    agent._persist_session = lambda *a, **k: None
    agent._save_trajectory = lambda *a, **k: None
    agent._cleanup_task_resources = lambda *a, **k: None
    return agent


def _run_and_capture_provider_kwargs(agent: AIAgent, user_message: str, conversation_history: list[dict]):
    captured: dict[str, object] = {"provider_called": False}
    real_build = agent._build_api_kwargs

    def capture_build(api_messages):
        captured["api_messages_input_to_build_kwargs"] = [dict(m) for m in api_messages]
        api_kwargs = real_build(api_messages)
        captured["api_kwargs_after_build"] = dict(api_kwargs)
        return api_kwargs

    def fake_create(**kwargs):
        captured["provider_called"] = True
        captured["provider_kwargs_at_call"] = dict(kwargs)
        return _response()

    agent._build_api_kwargs = capture_build
    agent.client.chat.completions.create.side_effect = fake_create

    result = agent.run_conversation(
        user_message,
        conversation_history=conversation_history,
        task_id="phase3-red-test-task",
    )
    captured["result"] = result
    return captured


def _payload_text(captured: dict[str, object]) -> str:
    return repr(
        {
            "api_messages": captured.get("api_messages_input_to_build_kwargs"),
            "api_kwargs_after_build": captured.get("api_kwargs_after_build"),
            "provider_kwargs_at_call": captured.get("provider_kwargs_at_call"),
        }
    )


def test_wcp_enabled_excludes_old_task_a_needle_from_provider_kwargs():
    agent = _make_agent(wcp_enabled=True)
    history = [
        {"role": "user", "content": f"old task A body {OLD_TASK_A_RAW_NEEDLE}"},
        {"role": "assistant", "content": f"old task A plan {OLD_TASK_A_ASSISTANT_NEEDLE}"},
    ]

    captured = _run_and_capture_provider_kwargs(
        agent,
        f"new task B request {CURRENT_TASK_B_NEEDLE} {CURRENT_TASK_B_WCP_MARKER}",
        history,
    )

    payload = _payload_text(captured)
    assert CURRENT_TASK_B_NEEDLE in payload
    assert OLD_TASK_A_RAW_NEEDLE not in payload
    assert OLD_TASK_A_ASSISTANT_NEEDLE not in payload


def test_wcp_enabled_phase2_bridge_excludes_prior_raw_long_history_needle_from_provider_kwargs():
    agent = _make_agent(wcp_enabled=True)
    history = [
        {
            "role": "user",
            "content": "prior raw long prompt that should be excluded "
            f"{PRIOR_PHASE2_RAW_LONG_PROMPT_NEEDLE} "
            + ("line " * 80),
        },
        {"role": "assistant", "content": "old task A completed"},
    ]
    safe_current_pointer = (
        "Context Health Intake safe summary/path pointer "
        f"{CURRENT_TASK_B_WCP_MARKER} summary.md task-state.md"
    )

    captured = _run_and_capture_provider_kwargs(agent, safe_current_pointer, history)

    payload = _payload_text(captured)
    assert "Context Health Intake" in payload
    assert CURRENT_TASK_B_WCP_MARKER in payload
    assert PRIOR_PHASE2_RAW_LONG_PROMPT_NEEDLE not in payload


def test_wcp_enabled_unsafe_build_returns_safe_hold_without_provider_call():
    agent = _make_agent(wcp_enabled=True)
    provider_called = False

    def fail_if_called(**_kwargs):
        nonlocal provider_called
        provider_called = True
        raise AssertionError("provider/model call should not happen when WCP build is unsafe")

    agent.client.chat.completions.create.side_effect = fail_if_called
    unsafe_prompt = f"unsafe ambiguous task boundary {UNSAFE_WCP_SECRET_NEEDLE} token=abc123"

    result = agent.run_conversation(
        unsafe_prompt,
        conversation_history=[{"role": "assistant", "content": "old task context"}],
        task_id="phase3-red-unsafe",
    )

    assert result["failed"] is True
    assert result["completed"] is False
    assert result["api_calls"] == 0
    assert provider_called is False
    assert UNSAFE_WCP_SECRET_NEEDLE not in repr(result)
    assert "abc123" not in repr(result)


def test_pre_api_request_observes_wcp_limited_request_messages():
    agent = _make_agent(wcp_enabled=True)
    captured_hook: dict[str, object] = {}

    def fake_has_hook(name):
        return name == "pre_api_request"

    def fake_invoke_hook(name, **kwargs):
        if name == "pre_api_request":
            captured_hook.update(kwargs)
        return []

    history = [
        {"role": "user", "content": f"old task A pre-api text {PRE_API_OLD_TASK_A_NEEDLE}"},
        {"role": "assistant", "content": "old answer"},
    ]

    with (
        patch("hermes_cli.plugins.has_hook", fake_has_hook),
        patch("hermes_cli.plugins.invoke_hook", fake_invoke_hook),
    ):
        _run_and_capture_provider_kwargs(
            agent,
            f"current B {PRE_API_CURRENT_WCP_MARKER}",
            history,
        )

    request_messages = captured_hook.get("request_messages")
    assert request_messages, "pre_api_request hook was not invoked with request_messages"
    request_payload = repr(
        {
            "request_messages": request_messages,
            "request": captured_hook.get("request"),
        }
    )
    assert PRE_API_CURRENT_WCP_MARKER in request_payload
    assert PRE_API_OLD_TASK_A_NEEDLE not in request_payload


def test_wcp_disabled_provider_payload_passes_through_full_history():
    agent = _make_agent(wcp_enabled=False)
    history = [
        {"role": "user", "content": f"old task text {DISABLED_PASS_THROUGH_NEEDLE}"},
        {"role": "assistant", "content": "old task answer"},
    ]

    captured = _run_and_capture_provider_kwargs(agent, "new task while disabled", history)

    payload = _payload_text(captured)
    assert captured["provider_called"] is True
    assert DISABLED_PASS_THROUGH_NEEDLE in payload


def test_wcp_excludes_old_task_tool_chain_without_orphaning_tool_messages():
    agent = _make_agent(wcp_enabled=True)
    history = [
        {"role": "user", "content": "old task A uses a tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_old_a",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"old.txt\"}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_old_a", "content": OLD_TASK_A_TOOL_RESULT_NEEDLE},
        {"role": "assistant", "content": "old task A final"},
    ]

    captured = _run_and_capture_provider_kwargs(
        agent,
        f"current B independent task {CURRENT_TASK_B_WCP_MARKER}",
        history,
    )

    payload = _payload_text(captured)
    assert CURRENT_TASK_B_WCP_MARKER in payload
    assert OLD_TASK_A_TOOL_RESULT_NEEDLE not in payload
    api_messages = captured.get("api_messages_input_to_build_kwargs") or []
    tool_messages = [m for m in api_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert tool_messages == []
