import json
import sys
import types
from types import SimpleNamespace

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent import codex_runtime


@pytest.fixture(autouse=True)
def _isolated_hermes_home(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """
model:
  provider: sub2api-test
  default: gpt-5.4
  base_url: https://sub2api.tegical.com
  api_mode: codex_responses
providers:
  sub2api-test:
    base_url: https://sub2api.tegical.com
    transport: codex_responses
    codex_responses_websocket: true
agent:
  task_completion_guidance: false
skills:
  enabled: false
memory:
  enabled: false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(run_agent, "_hermes_home", hermes_home)


@pytest.fixture(autouse=True)
def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


class _HermesResponsesWebSocket:
    def __init__(self, connection):
        self.connection = connection
        self.events = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.connection["closed"] = True
        return False

    def close(self):
        self.connection["closed"] = True

    def send(self, payload):
        sent = json.loads(payload)
        call = {
            "id": f"resp_{len(self.connection['sends']) + 1}",
            "sent": sent,
        }
        self.connection["sends"].append(call)
        response_text = self._response_for(sent)
        self.events = [
            {"type": "response.created", "response": {"id": call["id"], "status": "in_progress"}},
            {"type": "response.output_text.delta", "delta": response_text},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": response_text}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": call["id"],
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "total_tokens": 13,
                    },
                },
            },
        ]

    def recv(self, timeout=None):
        if not self.events:
            raise AssertionError("Hermes consumed past terminal WebSocket event")
        event = self.events.pop(0)
        self.connection.setdefault("events", []).append(event["type"])
        return json.dumps(event)

    @staticmethod
    def _response_for(sent):
        input_text = json.dumps(sent.get("input", []), ensure_ascii=False)
        if "Append -done" in input_text:
            return "cobalt-otter-17-done"
        if "What nonce" in input_text:
            return "cobalt-otter-17"
        if "Remember the nonce" in input_text:
            return "noted"
        return "unexpected"


def test_run_conversation_codex_responses_websocket_multi_turn_e2e(monkeypatch):
    connections = []

    def fake_connect(uri, **kwargs):
        connection = {
            "uri": uri,
            "headers": kwargs.get("additional_headers") or {},
            "sends": [],
        }
        connections.append(connection)
        return _HermesResponsesWebSocket(connection)

    monkeypatch.setattr(codex_runtime, "_connect_responses_websocket", fake_connect)

    agent = run_agent.AIAgent(
        model="gpt-5.4",
        provider="sub2api-test",
        api_mode="codex_responses",
        base_url="https://sub2api.tegical.com",
        api_key="sk-test",
        quiet_mode=True,
        max_iterations=3,
        enabled_toolsets=[],
        disabled_toolsets=["terminal", "web", "browser", "memory", "todo"],
        skip_context_files=True,
        skip_memory=True,
        load_soul_identity=False,
        session_id="codex-responses-ws-e2e",
        reasoning_config={"enabled": False},
        max_tokens=80,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    assert codex_runtime.codex_responses_websocket_enabled(agent) is True

    class _FailResponsesClient:
        def create(self, *args, **kwargs):
            raise AssertionError("codex_responses WebSocket path must not use responses.create()")

    class _FailOpenAIClient:
        responses = _FailResponsesClient()

        def close(self):
            pass

    agent._create_request_openai_client = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("codex_responses WebSocket path must not create an HTTP request client")
        )
    )
    agent._ensure_primary_openai_client = lambda *args, **kwargs: _FailOpenAIClient()

    history = None
    turn_specs = [
        (
            "Remember the nonce cobalt-otter-17. Reply exactly: noted",
            "noted",
        ),
        (
            "What nonce did I ask you to remember? Reply only the nonce.",
            "cobalt-otter-17",
        ),
        (
            "Append -done to that nonce and reply only the result.",
            "cobalt-otter-17-done",
        ),
    ]

    for prompt, expected in turn_specs:
        chunks = []
        result = agent.run_conversation(
            prompt,
            conversation_history=history,
            stream_callback=chunks.append,
        )
        assert result["completed"] is True
        assert result["api_calls"] == 1
        assert result["final_response"] == expected
        assert "".join(chunks) == expected
        history = result["messages"]

    assert len(connections) == 1
    connection = connections[0]
    assert connection["uri"] == "wss://sub2api.tegical.com/responses"
    assert connection["headers"].get("Authorization") == "Bearer sk-test"

    sends = connection["sends"]
    assert len(sends) == 3
    assert all(call["sent"]["type"] == "response.create" for call in sends)
    assert all("stream" not in call["sent"] for call in sends)
    assert all("background" not in call["sent"] for call in sends)

    assert "previous_response_id" not in sends[0]["sent"]
    assert sends[1]["sent"]["previous_response_id"] == "resp_1"
    assert sends[2]["sent"]["previous_response_id"] == "resp_2"

    first_input = json.dumps(sends[0]["sent"]["input"], ensure_ascii=False)
    second_input = json.dumps(sends[1]["sent"]["input"], ensure_ascii=False)
    third_input = json.dumps(sends[2]["sent"]["input"], ensure_ascii=False)
    assert "cobalt-otter-17" in first_input
    assert "cobalt-otter-17" not in second_input
    assert "noted" not in second_input
    assert "Remember the nonce" not in second_input
    assert "cobalt-otter-17" not in third_input
    assert "noted" not in third_input
    assert "What nonce" not in third_input
    assert "cobalt-otter-17-done" not in third_input

    agent.release_clients()
    assert connection.get("closed") is True


class _HermesToolResponsesWebSocket:
    def __init__(self, connection):
        self.connection = connection
        self.events = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.connection["closed"] = True
        return False

    def close(self):
        self.connection["closed"] = True

    def send(self, payload):
        sent = json.loads(payload)
        index = len(self.connection["sends"])
        call = {
            "id": f"resp_tool_{index + 1}",
            "sent": sent,
        }
        self.connection["sends"].append(call)
        if index == 0:
            self.events = [
                {"type": "response.created", "response": {"id": call["id"], "status": "in_progress"}},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "terminal",
                        "arguments": "{\"cmd\":\"pwd\"}",
                    },
                },
                {"type": "response.completed", "response": {"id": call["id"], "status": "completed"}},
            ]
        elif index == 1:
            self.events = [
                {"type": "response.created", "response": {"id": call["id"], "status": "in_progress"}},
                {"type": "response.output_text.delta", "delta": "done"},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "done"}],
                    },
                },
                {"type": "response.completed", "response": {"id": call["id"], "status": "completed"}},
            ]
        else:
            raise AssertionError("unexpected third WebSocket response.create")

    def recv(self, timeout=None):
        if not self.events:
            raise AssertionError("Hermes consumed past terminal WebSocket event")
        event = self.events.pop(0)
        self.connection.setdefault("events", []).append(event["type"])
        return json.dumps(event)


def test_run_conversation_codex_responses_websocket_tool_round_trip_uses_incremental_input(monkeypatch):
    connections = []

    def fake_connect(uri, **kwargs):
        connection = {
            "uri": uri,
            "headers": kwargs.get("additional_headers") or {},
            "sends": [],
        }
        connections.append(connection)
        return _HermesToolResponsesWebSocket(connection)

    monkeypatch.setattr(codex_runtime, "_connect_responses_websocket", fake_connect)
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

    agent = run_agent.AIAgent(
        model="gpt-5.4",
        provider="sub2api-test",
        api_mode="codex_responses",
        base_url="https://sub2api.tegical.com",
        api_key="sk-test",
        quiet_mode=True,
        max_iterations=3,
        enabled_toolsets=[],
        disabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
        load_soul_identity=False,
        session_id="codex-responses-ws-tool-e2e",
        reasoning_config={"enabled": False},
        max_tokens=80,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    agent._create_request_openai_client = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("codex_responses WebSocket path must not create an HTTP request client")
        )
    )

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"cwd":"/workspace"}',
                }
            )

    agent._execute_tool_calls = _fake_execute_tool_calls

    result = agent.run_conversation("run pwd")

    assert result["completed"] is True
    assert result["final_response"] == "done"
    assert len(connections) == 1

    sends = connections[0]["sends"]
    assert len(sends) == 2
    assert "previous_response_id" not in sends[0]["sent"]
    assert sends[1]["sent"]["previous_response_id"] == "resp_tool_1"

    replay_input = sends[1]["sent"]["input"]
    assert replay_input == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"cwd":"/workspace"}',
        }
    ]


class _FakeCreateStream:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


def _make_codex_responses_ws_agent(session_id, *, max_iterations=2, disabled_toolsets=None):
    if disabled_toolsets is None:
        disabled_toolsets = ["terminal", "web", "browser", "memory", "todo"]
    agent = run_agent.AIAgent(
        model="gpt-5.4",
        provider="sub2api-test",
        api_mode="codex_responses",
        base_url="https://sub2api.tegical.com",
        api_key="sk-test",
        quiet_mode=True,
        max_iterations=max_iterations,
        enabled_toolsets=[],
        disabled_toolsets=disabled_toolsets,
        skip_context_files=True,
        skip_memory=True,
        load_soul_identity=False,
        session_id=session_id,
        reasoning_config={"enabled": False},
        max_tokens=80,
    )
    agent._api_max_retries = 1
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    return agent


def test_run_conversation_codex_responses_does_not_use_websocket_without_provider_switch(monkeypatch):
    def fail_if_websocket_used(*args, **kwargs):
        raise AssertionError("WebSocket must not be used without provider opt-in")

    monkeypatch.setattr(codex_runtime, "_connect_responses_websocket", fail_if_websocket_used)

    http_calls = []

    class _HTTPResponses:
        def create(self, **kwargs):
            http_calls.append(kwargs)
            return _FakeCreateStream(
                [
                    {"type": "response.output_text.delta", "delta": "http-ok"},
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "http-ok"}],
                        },
                    },
                    {
                        "type": "response.completed",
                        "response": {"id": "resp_http_1", "status": "completed"},
                    },
                ]
            )

    class _HTTPOpenAIClient:
        responses = _HTTPResponses()

        def close(self):
            pass

    agent = run_agent.AIAgent(
        model="gpt-5.4",
        provider="plain-provider",
        api_mode="codex_responses",
        base_url="https://plain.example.com/v1",
        api_key="sk-test",
        quiet_mode=True,
        max_iterations=2,
        enabled_toolsets=[],
        disabled_toolsets=["terminal", "web", "browser", "memory", "todo"],
        skip_context_files=True,
        skip_memory=True,
        load_soul_identity=False,
        session_id="codex-responses-no-ws-e2e",
        reasoning_config={"enabled": False},
        max_tokens=80,
    )
    agent._create_request_openai_client = lambda *args, **kwargs: _HTTPOpenAIClient()
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None

    assert codex_runtime.codex_responses_websocket_enabled(agent) is False

    result = agent.run_conversation("ping")

    assert result["completed"] is True
    assert result["final_response"] == "http-ok"
    assert len(http_calls) == 1
    assert http_calls[0]["stream"] is True


def test_run_conversation_codex_responses_websocket_connect_failure_falls_back_to_http_same_provider(
    monkeypatch,
):
    ws_attempts = []

    def fake_connect(uri, **kwargs):
        ws_attempts.append({"uri": uri, "headers": kwargs.get("additional_headers") or {}})
        raise ConnectionError("websocket connect failed")

    monkeypatch.setattr(codex_runtime, "_connect_responses_websocket", fake_connect)

    http_calls = []

    class _HTTPResponses:
        def create(self, **kwargs):
            http_calls.append(kwargs)
            return _FakeCreateStream(
                [
                    {"type": "response.output_text.delta", "delta": "http-ok"},
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "http-ok"}],
                        },
                    },
                    {
                        "type": "response.completed",
                        "response": {"id": "resp_http_1", "status": "completed"},
                    },
                ]
            )

    class _HTTPOpenAIClient:
        responses = _HTTPResponses()

        def close(self):
            pass

    agent = _make_codex_responses_ws_agent("codex-responses-ws-http-fallback")
    agent._create_request_openai_client = lambda *args, **kwargs: _HTTPOpenAIClient()

    result = agent.run_conversation("Reply http-ok.")

    assert result["completed"] is True
    assert result["final_response"] == "http-ok"
    assert len(ws_attempts) == 1
    assert len(http_calls) == 1
    assert http_calls[0]["stream"] is True
    assert "previous_response_id" not in http_calls[0]
    request_input = json.dumps(http_calls[0]["input"], ensure_ascii=False)
    assert "Reply http-ok" in request_input


def test_run_conversation_codex_responses_websocket_fallback_keeps_remaining_turn_on_full_http(
    monkeypatch,
):
    ws_attempts = []

    def fake_connect(uri, **kwargs):
        ws_attempts.append(uri)
        raise ConnectionError("websocket recv failed before output")

    monkeypatch.setattr(codex_runtime, "_connect_responses_websocket", fake_connect)
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

    http_calls = []

    class _HTTPResponses:
        def create(self, **kwargs):
            http_calls.append(kwargs)
            if len(http_calls) == 1:
                return _FakeCreateStream(
                    [
                        {
                            "type": "response.output_item.done",
                            "item": SimpleNamespace(
                                type="function_call",
                                id="fc_http_1",
                                call_id="call_http_1",
                                name="terminal",
                                arguments="{\"cmd\":\"pwd\"}",
                            ),
                        },
                        {
                            "type": "response.completed",
                            "response": {"id": "resp_http_1", "status": "completed"},
                        },
                    ]
                )
            if len(http_calls) == 2:
                return _FakeCreateStream(
                    [
                        {"type": "response.output_text.delta", "delta": "done"},
                        {
                            "type": "response.output_item.done",
                            "item": {
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": "done"}],
                            },
                        },
                        {
                            "type": "response.completed",
                            "response": {"id": "resp_http_2", "status": "completed"},
                        },
                    ]
                )
            raise AssertionError("unexpected third HTTP fallback request")

    class _HTTPOpenAIClient:
        responses = _HTTPResponses()

        def close(self):
            pass

    agent = _make_codex_responses_ws_agent(
        "codex-responses-ws-http-fallback-tool",
        max_iterations=3,
        disabled_toolsets=[],
    )
    agent._create_request_openai_client = lambda *args, **kwargs: _HTTPOpenAIClient()

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"cwd":"/workspace"}',
                }
            )

    agent._execute_tool_calls = _fake_execute_tool_calls

    result = agent.run_conversation("run pwd")

    assert result["completed"] is True
    assert result["final_response"] == "done"
    assert len(ws_attempts) == 1
    assert len(http_calls) == 2
    assert all("previous_response_id" not in call for call in http_calls)

    second_input = json.dumps(http_calls[1]["input"], ensure_ascii=False)
    assert "run pwd" in second_input
    assert "call_http_1" in second_input
    assert any(
        item.get("type") == "function_call_output"
        and item.get("call_id") == "call_http_1"
        and json.loads(item.get("output", "{}")) == {"cwd": "/workspace"}
        for item in http_calls[1]["input"]
    )


class _PartialThenFailResponsesWebSocket:
    def __init__(self, connection):
        self.connection = connection
        self.events = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.connection["closed"] = True
        return False

    def close(self):
        self.connection["closed"] = True

    def send(self, payload):
        sent = json.loads(payload)
        self.connection["sends"].append({"sent": sent})
        self.events = [
            {"type": "response.created", "response": {"id": "resp_partial", "status": "in_progress"}},
            {"type": "response.output_text.delta", "delta": "partial-"},
        ]

    def recv(self, timeout=None):
        if self.events:
            return json.dumps(self.events.pop(0))
        raise ConnectionError("websocket failed after committed text")


def test_run_conversation_codex_responses_websocket_does_not_http_fallback_after_committed_text(
    monkeypatch,
):
    connections = []

    def fake_connect(uri, **kwargs):
        connection = {"uri": uri, "sends": []}
        connections.append(connection)
        return _PartialThenFailResponsesWebSocket(connection)

    monkeypatch.setattr(codex_runtime, "_connect_responses_websocket", fake_connect)

    http_calls = []

    class _HTTPResponses:
        def create(self, **kwargs):
            http_calls.append(kwargs)
            raise AssertionError("HTTP fallback must not run after text was streamed")

    class _HTTPOpenAIClient:
        responses = _HTTPResponses()

        def close(self):
            pass

    agent = _make_codex_responses_ws_agent("codex-responses-ws-no-fallback-after-text")
    agent._create_request_openai_client = lambda *args, **kwargs: _HTTPOpenAIClient()

    chunks = []
    result = agent.run_conversation("stream partial then fail", stream_callback=chunks.append)

    assert result["completed"] is False
    assert chunks == ["partial-"]
    assert http_calls == []


class _RestoredResponsesWebSocket:
    def __init__(self, connection):
        self.connection = connection
        self.events = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.connection["closed"] = True
        return False

    def close(self):
        self.connection["closed"] = True

    def send(self, payload):
        sent = json.loads(payload)
        index = len(self.connection["sends"])
        response_id = f"resp_ws_restored_{index + 1}"
        self.connection["sends"].append({"id": response_id, "sent": sent})
        input_text = json.dumps(sent.get("input", []), ensure_ascii=False)
        response_text = "ws-third" if "Third turn" in input_text else "ws-restored"
        self.events = [
            {"type": "response.created", "response": {"id": response_id, "status": "in_progress"}},
            {"type": "response.output_text.delta", "delta": response_text},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": response_text}],
                },
            },
            {"type": "response.completed", "response": {"id": response_id, "status": "completed"}},
        ]

    def recv(self, timeout=None):
        if not self.events:
            raise AssertionError("Hermes consumed past terminal WebSocket event")
        return json.dumps(self.events.pop(0))


def test_run_conversation_codex_responses_websocket_restores_with_full_input_after_http_fallback(
    monkeypatch,
):
    ws_connect_attempts = []
    connections = []

    def fake_connect(uri, **kwargs):
        ws_connect_attempts.append(uri)
        if len(ws_connect_attempts) == 1:
            raise ConnectionError("initial websocket unavailable")
        connection = {
            "uri": uri,
            "headers": kwargs.get("additional_headers") or {},
            "sends": [],
        }
        connections.append(connection)
        return _RestoredResponsesWebSocket(connection)

    monkeypatch.setattr(codex_runtime, "_connect_responses_websocket", fake_connect)

    http_calls = []

    class _HTTPResponses:
        def create(self, **kwargs):
            http_calls.append(kwargs)
            return _FakeCreateStream(
                [
                    {"type": "response.output_text.delta", "delta": "http-first"},
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "http-first"}],
                        },
                    },
                    {
                        "type": "response.completed",
                        "response": {"id": "resp_http_first", "status": "completed"},
                    },
                ]
            )

    class _HTTPOpenAIClient:
        responses = _HTTPResponses()

        def close(self):
            pass

    agent = _make_codex_responses_ws_agent("codex-responses-ws-restore-after-http", max_iterations=2)
    agent._create_request_openai_client = lambda *args, **kwargs: _HTTPOpenAIClient()

    first = agent.run_conversation("Initial turn. Reply http-first.")
    assert first["completed"] is True
    assert first["final_response"] == "http-first"

    second = agent.run_conversation(
        "Second turn. Reply ws-restored.",
        conversation_history=first["messages"],
    )
    assert second["completed"] is True
    assert second["final_response"] == "ws-restored"

    third = agent.run_conversation(
        "Third turn. Reply ws-third.",
        conversation_history=second["messages"],
    )
    assert third["completed"] is True
    assert third["final_response"] == "ws-third"

    assert len(http_calls) == 1
    assert len(ws_connect_attempts) == 2
    assert len(connections) == 1
    sends = connections[0]["sends"]
    assert len(sends) == 2

    restored_payload = sends[0]["sent"]
    restored_input = json.dumps(restored_payload["input"], ensure_ascii=False)
    assert "previous_response_id" not in restored_payload
    assert "Initial turn" in restored_input
    assert "http-first" in restored_input
    assert "Second turn" in restored_input

    incremental_payload = sends[1]["sent"]
    incremental_input = json.dumps(incremental_payload["input"], ensure_ascii=False)
    assert incremental_payload["previous_response_id"] == "resp_ws_restored_1"
    assert "Third turn" in incremental_input
    assert "Initial turn" not in incremental_input
    assert "http-first" not in incremental_input
    assert "Second turn" not in incremental_input


class _RecoveringResponsesWebSocket:
    def __init__(self, connection):
        self.connection = connection
        self.events = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.connection["closed"] = True
        return False

    def close(self):
        self.connection["closed"] = True

    def send(self, payload):
        sent = json.loads(payload)
        call = {
            "id": f"{self.connection['name']}_resp_{len(self.connection['sends']) + 1}",
            "sent": sent,
        }
        self.connection["sends"].append(call)
        if self.connection["name"] == "primary" and len(self.connection["sends"]) == 2:
            self.events = [
                {
                    "type": "error",
                    "status": 400,
                    "error": {
                        "code": "previous_response_not_found",
                        "message": "Previous response with id 'primary_resp_1' not found.",
                        "param": "previous_response_id",
                    },
                }
            ]
            return

        input_text = json.dumps(sent.get("input", []), ensure_ascii=False)
        text = "recovered-ok" if "Second turn" in input_text else "first-ok"
        self.events = [
            {"type": "response.created", "response": {"id": call["id"], "status": "in_progress"}},
            {"type": "response.output_text.delta", "delta": text},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text}],
                },
            },
            {"type": "response.completed", "response": {"id": call["id"], "status": "completed"}},
        ]

    def recv(self, timeout=None):
        if not self.events:
            raise AssertionError("Hermes consumed past terminal WebSocket event")
        return json.dumps(self.events.pop(0))


def test_run_conversation_codex_responses_websocket_recovers_with_full_input_when_previous_id_is_missing(
    monkeypatch,
):
    connections = []

    def fake_connect(uri, **kwargs):
        connection = {
            "name": "primary" if not connections else f"reconnect-{len(connections)}",
            "uri": uri,
            "headers": kwargs.get("additional_headers") or {},
            "sends": [],
        }
        connections.append(connection)
        return _RecoveringResponsesWebSocket(connection)

    monkeypatch.setattr(codex_runtime, "_connect_responses_websocket", fake_connect)

    agent = run_agent.AIAgent(
        model="gpt-5.4",
        provider="sub2api-test",
        api_mode="codex_responses",
        base_url="https://sub2api.tegical.com",
        api_key="sk-test",
        quiet_mode=True,
        max_iterations=2,
        enabled_toolsets=[],
        disabled_toolsets=["terminal", "web", "browser", "memory", "todo"],
        skip_context_files=True,
        skip_memory=True,
        load_soul_identity=False,
        session_id="codex-responses-ws-recover-e2e",
        reasoning_config={"enabled": False},
        max_tokens=80,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    agent._create_request_openai_client = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("codex_responses WebSocket path must not create an HTTP request client")
        )
    )

    first = agent.run_conversation("First turn. Reply first-ok.")
    assert first["completed"] is True
    assert first["final_response"] == "first-ok"

    second = agent.run_conversation(
        "Second turn. Reply recovered-ok.",
        conversation_history=first["messages"],
    )

    assert second["completed"] is True
    assert second["final_response"] == "recovered-ok"
    assert len(connections) == 2
    assert connections[0].get("closed") is True

    failed_continuation = connections[0]["sends"][1]["sent"]
    assert failed_continuation["previous_response_id"] == "primary_resp_1"
    assert "First turn" not in json.dumps(failed_continuation["input"])

    recovered_payload = connections[1]["sends"][0]["sent"]
    recovered_input = json.dumps(recovered_payload["input"], ensure_ascii=False)
    assert "previous_response_id" not in recovered_payload
    assert "First turn" in recovered_input
    assert "first-ok" in recovered_input
    assert "Second turn" in recovered_input
