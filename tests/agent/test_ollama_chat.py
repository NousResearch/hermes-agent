"""Behaviour contracts for the local Ollama chat adapter."""

import json
from types import SimpleNamespace

from agent.ollama_chat import (
    OllamaChatStream,
    _choice,
    _native_messages,
    _native_payload,
    ollama_chat_url,
)


def test_openai_base_url_maps_to_native_chat_route():
    assert ollama_chat_url("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434/api/chat"
    assert ollama_chat_url("http://ollama:11434/proxy/v1/") == "http://ollama:11434/proxy/api/chat"


def test_tool_continuation_maps_tool_call_and_result_to_ollama_messages():
    messages = _native_messages([
        {"role": "user", "content": "run uname"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "terminal", "arguments": '{"command":"uname -a"}',
            },
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Linux host"},
    ])

    assert messages == [
        {"role": "user", "content": "run uname"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "type": "function", "function": {"name": "terminal", "arguments": {"command": "uname -a"}},
        }]},
        {"role": "tool", "content": "Linux host", "tool_name": "terminal"},
    ]


def test_payload_sets_supported_num_ctx_and_preserves_tool_schema():
    tools = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}]
    payload = _native_payload({
        "model": "qwen3.8:latest", "messages": [{"role": "user", "content": "hello"}],
        "tools": tools, "temperature": 0.2, "extra_body": {"options": {"seed": 7}},
    }, num_ctx=6144, stream=True)

    assert payload["options"] == {"seed": 7, "num_ctx": 6144, "temperature": 0.2}
    assert payload["tools"] == tools
    assert payload["stream"] is True


def test_nonstream_tool_finish_reason_and_arguments_are_openai_shaped():
    choice = _choice({"content": "", "tool_calls": [{
        "function": {"name": "terminal", "arguments": {"command": "uname -a"}},
    }]}, "stop")

    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls[0].function.name == "terminal"
    assert json.loads(choice.message.tool_calls[0].function.arguments) == {"command": "uname -a"}


def test_stream_translates_native_message_and_terminal_finish_reason():
    lines = [
        json.dumps({"model": "qwen3.8:latest", "message": {"content": "Done"}, "done": False}),
        json.dumps({"model": "qwen3.8:latest", "message": {}, "done": True,
                    "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 3}),
    ]

    class Response:
        def iter_lines(self):
            return iter(lines)

    class Client:
        pass

    chunks = list(OllamaChatStream(Client(), Response()))
    assert chunks[0].choices[0].delta.content == "Done"
    assert chunks[1].choices[0].finish_reason == "stop"
    assert chunks[2].usage.total_tokens == 15


def test_stream_keeps_tool_call_finish_reason_when_terminal_chunk_has_no_message():
    lines = [
        json.dumps({"message": {"tool_calls": [{"function": {
            "name": "terminal", "arguments": {"command": "uname -a"},
        }}]}, "done": False}),
        json.dumps({"message": {}, "done": True, "done_reason": "stop"}),
    ]

    class Response:
        def iter_lines(self):
            return iter(lines)

    chunks = list(OllamaChatStream(object(), Response()))
    assert chunks[0].choices[0].delta.tool_calls[0].function.name == "terminal"
    assert chunks[1].choices[0].finish_reason == "tool_calls"
