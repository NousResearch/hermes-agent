import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


DEFAULT_WEB_BASE = "https://chatgpt.com/backend-api/f"


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _build_agent(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model="gpt-5-thinking",
        provider="chatgpt-web",
        api_mode="chatgpt_web",
        base_url=DEFAULT_WEB_BASE,
        api_key="chatgpt-web-token",
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


def test_build_api_kwargs_chatgpt_web_carries_thread_state(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent._chatgpt_web_conversation_id = "conv_existing"
    agent._chatgpt_web_parent_message_id = "msg_existing"

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello from Hermes."},
    ])

    assert kwargs["model"] == "gpt-5-thinking"
    assert kwargs["conversation_id"] == "conv_existing"
    assert kwargs["parent_message_id"] == "msg_existing"
    assert kwargs["messages"][-1]["content"] == "Hello from Hermes."
    assert "tools" not in kwargs


def test_select_chatgpt_web_tools_prefers_explicit_sequence(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "execute_code", "description": "Run Python code", "parameters": {"type": "object"}}},
    ]

    first = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "First use search_files, then terminal, then execute_code."},
    ])
    second = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "First use search_files, then terminal, then execute_code."},
        {"role": "tool", "tool_name": "search_files", "content": "{}"},
    ])
    third = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "First use search_files, then terminal, then execute_code."},
        {"role": "tool", "tool_name": "search_files", "content": "{}"},
        {"role": "tool", "tool_name": "terminal", "content": "{}"},
    ])

    assert [tool["function"]["name"] for tool in first] == ["search_files"]
    assert [tool["function"]["name"] for tool in second] == ["terminal"]
    assert [tool["function"]["name"] for tool in third] == ["execute_code"]



def test_select_chatgpt_web_tools_prefers_terminal_for_working_directory(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "Use Hermes tools to print the current working directory. Answer only the path."},
    ])

    assert [tool["function"]["name"] for tool in selected] == ["terminal"]



def test_build_api_kwargs_chatgpt_web_with_tools_injects_protocol_and_disables_remote_thread(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent._chatgpt_web_conversation_id = "conv_existing"
    agent._chatgpt_web_parent_message_id = "msg_existing"
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        },
    }]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Use Hermes tools to grep hermes_cli/chatgpt_web.py for stream_chatgpt_web_completion. Answer only yes/no and one matching path."},
    ])

    assert kwargs["conversation_id"] is None
    assert kwargs["parent_message_id"] is None
    assert kwargs["history_and_training_disabled"] is True
    assert "Be concise." in kwargs["instructions"]
    assert "<tool_call>" in kwargs["instructions"]
    assert "<tool_response>" in kwargs["instructions"]
    assert "search_files" in kwargs["instructions"]
    assert "does not support Hermes tool calls" not in kwargs["instructions"]

    rewritten_user = kwargs["messages"][-1]["content"]
    assert rewritten_user.startswith("Original user request:\nUse Hermes tools to grep")
    assert "Hermes has already determined that this turn requires a tool call." in rewritten_user
    assert "Reply now with this exact structure:" in rewritten_user
    assert '"name": "search_files"' in rewritten_user
    assert '"pattern": "stream_chatgpt_web_completion"' in rewritten_user
    assert '"path": "hermes_cli/chatgpt_web.py"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_refreshes_runtime_reminder_after_tool_response(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run Python code",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        },
    }]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": (
                "Original user request:\nUse Hermes tools to run Python that prints 6*7. Answer only the result.\n\n"
                "Runtime reminder:\nThe tool available for this turn is: execute_code. "
                "Hermes has already determined that this turn requires a tool call."
            ),
        },
        {"role": "tool", "content": "42"},
    ])

    rewritten_user = kwargs["messages"][0]["content"]
    assert rewritten_user.startswith("Original user request:\nUse Hermes tools to run Python")
    assert "You have already received at least one <tool_response>." in rewritten_user
    assert "Otherwise, give the final answer directly with no extra tool-call markup." in rewritten_user
    assert "follow the original user's requested output format exactly" in rewritten_user
    assert "Hermes has already determined that this turn requires a tool call." not in rewritten_user


def test_wrap_chatgpt_web_response_extracts_xml_tool_calls(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
        },
    }]

    response = agent._wrap_chatgpt_web_response({
        "content": (
            "I will inspect the repo.\n"
            "<tool_call>\n"
            '{"name":"search_files","arguments":{"pattern":"chatgpt_web"}}\n'
            "</tool_call>"
        ),
        "message_id": "msg_tool_1",
        "model": "gpt-5-thinking",
        "finish_reason": "stop",
    })

    message = response.choices[0].message
    assert "inspect the repo" in message.content
    assert "<tool_call>" not in message.content
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.name == "search_files"
    assert message.tool_calls[0].function.arguments == '{"pattern": "chatgpt_web"}'


def test_wrap_chatgpt_web_response_extracts_tool_calls_with_missing_opening_angle(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
        },
    }]

    response = agent._wrap_chatgpt_web_response({
        "content": (
            "tool_call>\n"
            '{"name":"search_files","arguments":{"pattern":"run_agent.py","target":"files"}}\n'
            "</tool_call>"
        ),
        "message_id": "msg_tool_2",
        "model": "gpt-5-thinking",
        "finish_reason": "stop",
    })

    message = response.choices[0].message
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.name == "search_files"
    assert message.tool_calls[0].function.arguments == '{"pattern": "run_agent.py", "target": "files"}'



def test_wrap_chatgpt_web_response_forces_selected_tool_when_model_refuses(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
        },
    }]
    agent._chatgpt_web_forced_tool_call = {
        "name": "search_files",
        "arguments": {"pattern": "stream_chatgpt_web_completion", "target": "content", "path": "hermes_cli/chatgpt_web.py"},
    }

    response = agent._wrap_chatgpt_web_response({
        "content": "I can't access the tool right now.",
        "message_id": "msg_tool_3",
        "model": "gpt-5-thinking",
        "finish_reason": "stop",
    })

    message = response.choices[0].message
    assert message.content == ""
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.name == "search_files"
    assert message.tool_calls[0].function.arguments == '{"pattern": "stream_chatgpt_web_completion", "target": "content", "path": "hermes_cli/chatgpt_web.py"}'
    assert agent._chatgpt_web_forced_tool_call is None



def test_interruptible_api_call_chatgpt_web_returns_openai_like_response(monkeypatch):
    agent = _build_agent(monkeypatch)

    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.stream_chatgpt_web_completion",
        lambda **kwargs: {
            "content": "Hello from ChatGPT web.",
            "conversation_id": "conv_123",
            "parent_message_id": "msg_456",
            "message_id": "msg_456",
            "model": "gpt-5-thinking",
            "finish_reason": "stop",
        },
    )

    response = agent._interruptible_api_call({
        "model": "gpt-5-thinking",
        "messages": [{"role": "user", "content": "hi"}],
        "conversation_id": None,
        "parent_message_id": None,
        "instructions": "Be concise.",
    })

    assert response.choices[0].message.content == "Hello from ChatGPT web."
    assert response.choices[0].finish_reason == "stop"
    assert agent._chatgpt_web_conversation_id == "conv_123"
    assert agent._chatgpt_web_parent_message_id == "msg_456"


def test_interruptible_streaming_api_call_chatgpt_web_emits_deltas(monkeypatch):
    agent = _build_agent(monkeypatch)
    deltas = []
    agent.stream_delta_callback = deltas.append

    def _fake_stream(**kwargs):
        on_delta = kwargs.get("on_delta")
        assert on_delta is not None
        on_delta("Hel")
        on_delta("lo")
        return {
            "content": "Hello",
            "conversation_id": "conv_stream",
            "parent_message_id": "msg_stream",
            "message_id": "msg_stream",
            "model": "gpt-5-thinking",
            "finish_reason": "stop",
        }

    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.stream_chatgpt_web_completion",
        _fake_stream,
    )

    response = agent._interruptible_streaming_api_call({
        "model": "gpt-5-thinking",
        "messages": [{"role": "user", "content": "hi"}],
        "conversation_id": None,
        "parent_message_id": None,
        "instructions": "Be concise.",
    })

    assert deltas == ["Hel", "lo"]
    assert response.choices[0].message.content == "Hello"
    assert agent._chatgpt_web_conversation_id == "conv_stream"
    assert agent._chatgpt_web_parent_message_id == "msg_stream"


def test_interruptible_streaming_api_call_chatgpt_web_fires_first_delta_without_text_arg(monkeypatch):
    agent = _build_agent(monkeypatch)
    deltas = []
    first_delta_calls = []
    agent.stream_delta_callback = deltas.append

    def _fake_stream(**kwargs):
        on_delta = kwargs.get("on_delta")
        assert on_delta is not None
        on_delta("Hel")
        on_delta("lo")
        return {
            "content": "Hello",
            "conversation_id": "conv_first",
            "parent_message_id": "msg_first",
            "message_id": "msg_first",
            "model": "gpt-5-thinking",
            "finish_reason": "stop",
        }

    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.stream_chatgpt_web_completion",
        _fake_stream,
    )

    response = agent._interruptible_streaming_api_call(
        {
            "model": "gpt-5-thinking",
            "messages": [{"role": "user", "content": "hi"}],
            "conversation_id": None,
            "parent_message_id": None,
            "instructions": "Be concise.",
        },
        on_first_delta=lambda: first_delta_calls.append("fired"),
    )

    assert first_delta_calls == ["fired"]
    assert deltas == ["Hel", "lo"]
    assert response.choices[0].message.content == "Hello"


def test_interruptible_api_call_chatgpt_web_closes_request_client_on_interrupt(monkeypatch):
    agent = _build_agent(monkeypatch)
    entered = threading.Event()
    close_seen = threading.Event()
    created_clients = []
    close_reasons = []

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True
            close_seen.set()

    def _fake_client_factory(*args, **kwargs):
        client = _FakeClient(*args, **kwargs)
        created_clients.append(client)
        return client

    def _fake_close_request_client(client, *, reason):
        close_reasons.append(reason)
        client.close()

    def _fake_stream(**kwargs):
        client = kwargs.get("client")
        assert client is created_clients[0]
        entered.set()
        while not client.closed:
            time.sleep(0.01)
        raise RuntimeError("aborted")

    monkeypatch.setattr("httpx.Client", _fake_client_factory)
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.stream_chatgpt_web_completion",
        _fake_stream,
    )
    agent._close_request_openai_client = _fake_close_request_client

    def _interrupt_once_stream_starts():
        assert entered.wait(timeout=2)
        agent._interrupt_requested = True

    interrupter = threading.Thread(target=_interrupt_once_stream_starts, daemon=True)
    interrupter.start()

    with pytest.raises(InterruptedError, match="Agent interrupted during API call"):
        agent._interruptible_api_call({
            "model": "gpt-5-thinking",
            "messages": [{"role": "user", "content": "hi"}],
            "conversation_id": None,
            "parent_message_id": None,
            "instructions": "Be concise.",
        })

    assert close_seen.wait(timeout=2)
    assert created_clients
    assert "interrupt_abort" in close_reasons
    time.sleep(0.05)
    assert agent._chatgpt_web_conversation_id is None
    assert agent._chatgpt_web_parent_message_id is None
