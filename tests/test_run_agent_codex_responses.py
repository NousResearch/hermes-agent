import sys
import types
from types import SimpleNamespace


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run shell commands.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _build_agent(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="codex-token",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    agent._save_session_log = lambda messages: None
    return agent


def _codex_message_response(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
        status="completed",
        model="gpt-5-codex",
    )


def _codex_tool_call_response():
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="terminal",
                arguments="{}",
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
        status="completed",
        model="gpt-5-codex",
    )


def _codex_incomplete_message_response(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="in_progress",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="in_progress",
        model="gpt-5-codex",
    )


def _codex_commentary_message_response(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                phase="commentary",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="completed",
        model="gpt-5-codex",
    )


def _codex_ack_message_response(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="completed",
        model="gpt-5-codex",
    )


def test_api_mode_uses_explicit_provider_when_codex(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        base_url="https://openrouter.ai/api/v1",
        provider="openai-codex",
        api_key="codex-token",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "codex_responses"
    assert agent.provider == "openai-codex"


def test_api_mode_normalizes_provider_case(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        base_url="https://openrouter.ai/api/v1",
        provider="OpenAI-Codex",
        api_key="codex-token",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.provider == "openai-codex"
    assert agent.api_mode == "codex_responses"


def test_api_mode_respects_explicit_openrouter_provider_over_codex_url(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openrouter",
        api_key="test-token",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "chat_completions"
    assert agent.provider == "openrouter"


def test_build_api_kwargs_codex(monkeypatch):
    agent = _build_agent(monkeypatch)
    kwargs = agent._build_api_kwargs(
        [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Ping"},
        ]
    )

    assert kwargs["model"] == "gpt-5-codex"
    assert kwargs["instructions"] == "You are Hermes."
    assert kwargs["store"] is False
    assert isinstance(kwargs["input"], list)
    assert kwargs["input"][0]["role"] == "user"
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["tools"][0]["name"] == "terminal"
    assert kwargs["tools"][0]["strict"] is False
    assert "function" not in kwargs["tools"][0]


def test_run_conversation_codex_plain_text(monkeypatch):
    agent = _build_agent(monkeypatch)
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: _codex_message_response("OK"))

    result = agent.run_conversation("Say OK")

    assert result["completed"] is True
    assert result["final_response"] == "OK"
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == "OK"


def test_run_conversation_codex_tool_round_trip(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [_codex_tool_call_response(), _codex_message_response("done")]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("run a command")

    assert result["completed"] is True
    assert result["final_response"] == "done"
    assert any(msg.get("tool_calls") for msg in result["messages"] if msg.get("role") == "assistant")
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1" for msg in result["messages"])


def test_chat_messages_to_responses_input_uses_fc_id_for_function_call(monkeypatch):
    agent = _build_agent(monkeypatch)
    items = agent._chat_messages_to_responses_input(
        [
            {"role": "user", "content": "Run terminal"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc123", "content": '{"ok":true}'},
        ]
    )

    function_call = next(item for item in items if item.get("type") == "function_call")
    function_output = next(item for item in items if item.get("type") == "function_call_output")

    assert function_call["call_id"] == "call_abc123"
    assert function_call["id"] == "fc_abc123"
    assert function_output["call_id"] == "call_abc123"


def test_chat_messages_to_responses_input_accepts_call_pipe_fc_ids(monkeypatch):
    agent = _build_agent(monkeypatch)
    items = agent._chat_messages_to_responses_input(
        [
            {"role": "user", "content": "Run terminal"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_pair123|fc_pair123",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_pair123|fc_pair123", "content": '{"ok":true}'},
        ]
    )

    function_call = next(item for item in items if item.get("type") == "function_call")
    function_output = next(item for item in items if item.get("type") == "function_call_output")

    assert function_call["call_id"] == "call_pair123"
    assert function_call["id"] == "fc_pair123"
    assert function_output["call_id"] == "call_pair123"


def test_run_conversation_codex_replay_payload_keeps_call_id_and_fc_id(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [_codex_tool_call_response(), _codex_message_response("done")]
    requests = []

    def _fake_api_call(api_kwargs):
        requests.append(api_kwargs)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("run a command")

    assert result["completed"] is True
    assert result["final_response"] == "done"
    assert len(requests) >= 2

    replay_input = requests[1]["input"]
    function_call = next(item for item in replay_input if item.get("type") == "function_call")
    function_output = next(item for item in replay_input if item.get("type") == "function_call_output")
    assert function_call["call_id"] == "call_1"
    assert function_call["id"] == "fc_1"
    assert function_output["call_id"] == "call_1"


def test_run_conversation_codex_continues_after_incomplete_interim_message(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [
        _codex_incomplete_message_response("I'll inspect the repo structure first."),
        _codex_tool_call_response(),
        _codex_message_response("Architecture summary complete."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("analyze repo")

    assert result["completed"] is True
    assert result["final_response"] == "Architecture summary complete."
    assert any(
        msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
        and "inspect the repo structure" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1" for msg in result["messages"])


def test_normalize_codex_response_marks_commentary_only_message_as_incomplete(monkeypatch):
    agent = _build_agent(monkeypatch)
    assistant_message, finish_reason = agent._normalize_codex_response(
        _codex_commentary_message_response("I'll inspect the repository first.")
    )

    assert finish_reason == "incomplete"
    assert "inspect the repository" in (assistant_message.content or "")


def test_run_conversation_codex_continues_after_commentary_phase_message(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [
        _codex_commentary_message_response("I'll inspect the repo structure first."),
        _codex_tool_call_response(),
        _codex_message_response("Architecture summary complete."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("analyze repo")

    assert result["completed"] is True
    assert result["final_response"] == "Architecture summary complete."
    assert any(
        msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
        and "inspect the repo structure" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1" for msg in result["messages"])


def test_run_conversation_codex_continues_after_ack_stop_message(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [
        _codex_ack_message_response(
            "Absolutely — I can do that. I'll inspect ~/openclaw-studio and report back with a walkthrough."
        ),
        _codex_tool_call_response(),
        _codex_message_response("Architecture summary complete."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("look into ~/openclaw-studio and tell me how it works")

    assert result["completed"] is True
    assert result["final_response"] == "Architecture summary complete."
    assert any(
        msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
        and "inspect ~/openclaw-studio" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    assert any(
        msg.get("role") == "user"
        and "Continue now. Execute the required tool calls" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1" for msg in result["messages"])
