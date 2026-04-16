"""Tests for agent.ollama_adapter — Ollama native /api/chat adapter."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.ollama_adapter import (
    OllamaAPIError,
    OllamaAuthenticationError,
    OllamaConnectionError,
    build_ollama_kwargs,
    convert_messages_to_ollama,
    convert_tools_to_ollama,
    normalize_ollama_response,
    normalize_ollama_url,
    stream_ollama_with_callbacks,
)


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


class TestNormalizeOllamaUrl:
    def test_bare_host_gets_api_chat_appended(self):
        result = normalize_ollama_url("http://localhost:11434")
        assert result.endswith("/api/chat")

    def test_trailing_slash_handled(self):
        result = normalize_ollama_url("http://localhost:11434/")
        assert result.endswith("/api/chat")
        assert "//" not in result.replace("://", "")

    def test_api_only_gets_chat_appended(self):
        result = normalize_ollama_url("http://localhost:11434/api")
        assert result.endswith("/api/chat")

    def test_already_full_url_unchanged(self):
        url = "http://localhost:11434/api/chat"
        assert normalize_ollama_url(url) == url

    def test_https_remote_host(self):
        result = normalize_ollama_url("https://ollama.com")
        assert result.endswith("/api/chat")
        assert result.startswith("https://ollama.com")


# ---------------------------------------------------------------------------
# Message conversion: OpenAI/Hermes → Ollama
# ---------------------------------------------------------------------------


class TestConvertMessagesToOllama:
    def test_system_message_preserved(self):
        msgs = [{"role": "system", "content": "You are helpful"}]
        result = convert_messages_to_ollama(msgs)
        assert result == [{"role": "system", "content": "You are helpful"}]

    def test_user_message_preserved(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = convert_messages_to_ollama(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_with_tool_calls_preserves_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
            }
        ]
        result = convert_messages_to_ollama(msgs)
        assert len(result) == 1
        entry = result[0]
        assert entry["role"] == "assistant"
        tc = entry["tool_calls"][0]
        # Arguments should be a dict (Ollama native format)
        assert isinstance(tc["function"]["arguments"], dict)
        assert tc["function"]["arguments"] == {"city": "London"}
        assert tc["function"]["name"] == "get_weather"

    def test_assistant_tool_calls_with_string_arguments_parsed(self):
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "calc",
                            "arguments": '{"x": 1, "y": 2}',
                        }
                    }
                ],
            }
        ]
        result = convert_messages_to_ollama(msgs)
        tc = result[0]["tool_calls"][0]
        assert tc["function"]["arguments"] == {"x": 1, "y": 2}

    def test_tool_result_mapped_correctly(self):
        msgs = [
            {
                "role": "tool",
                "content": "result data",
                "tool_call_id": "call_123",
            }
        ]
        result = convert_messages_to_ollama(msgs)
        assert result == [
            {"role": "tool", "content": "result data", "tool_call_id": "call_123"}
        ]

    def test_tool_result_without_id(self):
        msgs = [{"role": "tool", "content": "result"}]
        result = convert_messages_to_ollama(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "result"
        assert "tool_call_id" not in result[0]

    def test_user_with_images_preserved(self):
        msgs = [
            {
                "role": "user",
                "content": "What is this?",
                "images": ["base64data=="],
            }
        ]
        result = convert_messages_to_ollama(msgs)
        assert result[0]["images"] == ["base64data=="]
        assert result[0]["content"] == "What is this?"

    def test_user_with_image_url_content_parts(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                ],
            }
        ]
        result = convert_messages_to_ollama(msgs)
        entry = result[0]
        assert entry["content"] == "Describe this image"
        assert "abc123" in entry["images"]

    def test_empty_messages_list(self):
        assert convert_messages_to_ollama([]) == []


# ---------------------------------------------------------------------------
# Tool conversion: OpenAI → Ollama
# ---------------------------------------------------------------------------


class TestConvertToolsToOllama:
    def test_openai_format_tool_passthrough(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ]
        result = convert_tools_to_ollama(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["description"] == "Search the web"

    def test_empty_tools_list_returns_empty(self):
        assert convert_tools_to_ollama([]) == []

    def test_none_entry_skipped(self):
        tools = [None, {"type": "function", "function": {"name": "fn"}}]
        result = convert_tools_to_ollama(tools)
        # None is skipped; valid tool is included
        assert len(result) == 1

    def test_tool_without_name_skipped(self):
        tools = [{"type": "function", "function": {"description": "no name"}}]
        result = convert_tools_to_ollama(tools)
        assert result == []

    def test_multiple_tools_all_converted(self):
        tools = [
            {"type": "function", "function": {"name": "a", "description": "A"}},
            {"type": "function", "function": {"name": "b", "description": "B"}},
        ]
        result = convert_tools_to_ollama(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"


# ---------------------------------------------------------------------------
# Response normalization: Ollama → OpenAI-compatible SimpleNamespace
# ---------------------------------------------------------------------------


class TestNormalizeOllamaResponse:
    def _text_response(self, **overrides):
        base = {
            "message": {"role": "assistant", "content": "Hello"},
            "model": "llama3",
            "prompt_eval_count": 10,
            "eval_count": 20,
            "done": True,
        }
        base.update(overrides)
        return base

    def test_text_response_content(self):
        result = normalize_ollama_response(self._text_response())
        assert result.choices[0].message.content == "Hello"

    def test_usage_prompt_tokens(self):
        result = normalize_ollama_response(self._text_response())
        assert result.usage.prompt_tokens == 10

    def test_usage_completion_tokens(self):
        result = normalize_ollama_response(self._text_response())
        assert result.usage.completion_tokens == 20

    def test_usage_total_tokens(self):
        result = normalize_ollama_response(self._text_response())
        assert result.usage.total_tokens == 30

    def test_model_field(self):
        result = normalize_ollama_response(self._text_response())
        assert result.model == "llama3"

    def test_finish_reason_stop_for_text(self):
        result = normalize_ollama_response(self._text_response())
        assert result.choices[0].finish_reason == "stop"

    def test_tool_call_response_generates_id(self):
        data = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}
                ],
            },
            "model": "llama3",
            "prompt_eval_count": 5,
            "eval_count": 15,
            "done": True,
        }
        result = normalize_ollama_response(data)
        tc = result.choices[0].message.tool_calls[0]
        assert tc.id.startswith("call_")
        assert tc.function.name == "get_weather"

    def test_tool_call_arguments_serialized_to_json_string(self):
        data = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "fn", "arguments": {"key": "val"}}}
                ],
            },
            "model": "llama3",
            "prompt_eval_count": 0,
            "eval_count": 0,
            "done": True,
        }
        result = normalize_ollama_response(data)
        args = result.choices[0].message.tool_calls[0].function.arguments
        assert isinstance(args, str)
        assert json.loads(args) == {"key": "val"}

    def test_tool_call_already_string_arguments_preserved(self):
        data = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "fn", "arguments": '{"key": "val"}'}}
                ],
            },
            "model": "llama3",
            "prompt_eval_count": 0,
            "eval_count": 0,
            "done": True,
        }
        result = normalize_ollama_response(data)
        args = result.choices[0].message.tool_calls[0].function.arguments
        assert isinstance(args, str)
        assert json.loads(args) == {"key": "val"}

    def test_empty_content_with_tool_calls_is_none(self):
        data = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "fn", "arguments": {}}}
                ],
            },
            "model": "llama3",
            "prompt_eval_count": 0,
            "eval_count": 0,
            "done": True,
        }
        result = normalize_ollama_response(data)
        assert result.choices[0].message.content is None

    def test_finish_reason_tool_calls_when_tools_present(self):
        data = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "fn", "arguments": {}}}],
            },
            "model": "llama3",
            "prompt_eval_count": 0,
            "eval_count": 0,
            "done": True,
        }
        result = normalize_ollama_response(data)
        assert result.choices[0].finish_reason == "tool_calls"

    def test_thinking_reasoning_populated(self):
        data = {
            "message": {
                "role": "assistant",
                "content": "Answer",
                "thinking": "I need to think carefully...",
            },
            "model": "qwq",
            "prompt_eval_count": 5,
            "eval_count": 50,
            "done": True,
        }
        result = normalize_ollama_response(data)
        assert result.choices[0].message.reasoning == "I need to think carefully..."

    def test_missing_thinking_is_none(self):
        result = normalize_ollama_response(self._text_response())
        assert result.choices[0].message.reasoning is None

    def test_missing_token_counts_default_to_zero(self):
        data = {
            "message": {"role": "assistant", "content": "Hi"},
            "model": "llama3",
            "done": True,
        }
        result = normalize_ollama_response(data)
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0

    def test_no_tool_calls_returns_none(self):
        result = normalize_ollama_response(self._text_response())
        assert result.choices[0].message.tool_calls is None


# ---------------------------------------------------------------------------
# Build kwargs
# ---------------------------------------------------------------------------


class TestBuildOllamaKwargs:
    def test_stream_false_in_output(self):
        payload = build_ollama_kwargs("llama3", [])
        assert payload["stream"] is False

    def test_model_field_set(self):
        payload = build_ollama_kwargs("llama3.2", [])
        assert payload["model"] == "llama3.2"

    def test_num_ctx_passed_through(self):
        payload = build_ollama_kwargs("llama3", [], options={"num_ctx": 8192})
        assert payload["options"]["num_ctx"] == 8192

    def test_max_tokens_maps_to_num_predict(self):
        payload = build_ollama_kwargs("llama3", [], max_tokens=512)
        assert payload["options"]["num_predict"] == 512

    def test_temperature_in_options(self):
        payload = build_ollama_kwargs("llama3", [], temperature=0.7)
        assert payload["options"]["temperature"] == 0.7

    def test_none_fields_omitted(self):
        payload = build_ollama_kwargs("llama3", [])
        assert "options" not in payload
        assert "keep_alive" not in payload
        assert "think" not in payload
        assert "tools" not in payload

    def test_tools_included_when_provided(self):
        tools = [{"type": "function", "function": {"name": "fn"}}]
        payload = build_ollama_kwargs("llama3", [], tools=tools)
        assert payload["tools"] == tools

    def test_empty_tools_list_omitted(self):
        payload = build_ollama_kwargs("llama3", [], tools=[])
        assert "tools" not in payload

    def test_keep_alive_included(self):
        payload = build_ollama_kwargs("llama3", [], keep_alive="5m")
        assert payload["keep_alive"] == "5m"

    def test_think_flag_included(self):
        payload = build_ollama_kwargs("llama3", [], think=True)
        assert payload["think"] is True

    def test_options_merged_with_max_tokens(self):
        payload = build_ollama_kwargs(
            "llama3", [], options={"num_ctx": 4096}, max_tokens=256
        )
        assert payload["options"]["num_ctx"] == 4096
        assert payload["options"]["num_predict"] == 256


# ---------------------------------------------------------------------------
# NDJSON stream parsing
# ---------------------------------------------------------------------------


class TestStreamOllamaWithCallbacks:
    def _make_lines(self, chunks):
        """Convert list of dicts to NDJSON lines."""
        return [json.dumps(c) for c in chunks]

    def test_normal_stream_collects_content(self):
        lines = self._make_lines([
            {"message": {"role": "assistant", "content": "Hello"}, "done": False},
            {"message": {"role": "assistant", "content": " world"}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "model": "llama3",
                "prompt_eval_count": 10,
                "eval_count": 20,
            },
        ])
        result = stream_ollama_with_callbacks(iter(lines))
        assert result.choices[0].message.content == "Hello world"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20

    def test_text_delta_callback_fired(self):
        lines = self._make_lines([
            {"message": {"role": "assistant", "content": "Hi"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True,
             "prompt_eval_count": 0, "eval_count": 0},
        ])
        deltas = []
        stream_ollama_with_callbacks(iter(lines), on_text_delta=deltas.append)
        assert deltas == ["Hi"]

    def test_error_mid_stream_raises_ollama_api_error(self):
        lines = self._make_lines([
            {"message": {"role": "assistant", "content": "Hello"}, "done": False},
            {"error": "model not found"},
        ])
        with pytest.raises(OllamaAPIError):
            stream_ollama_with_callbacks(iter(lines))

    def test_missing_final_chunk_graceful(self):
        lines = self._make_lines([
            {"message": {"role": "assistant", "content": "Hello"}, "done": False},
        ])
        result = stream_ollama_with_callbacks(iter(lines))
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.choices[0].message.content == "Hello"

    def test_empty_lines_skipped(self):
        lines = [
            "",
            "   ",
            json.dumps({"message": {"role": "assistant", "content": "Hi"}, "done": False}),
            "",
            json.dumps({"message": {"role": "assistant", "content": ""},
                        "done": True, "prompt_eval_count": 1, "eval_count": 2}),
        ]
        result = stream_ollama_with_callbacks(iter(lines))
        assert result.choices[0].message.content == "Hi"
        assert result.usage.prompt_tokens == 1

    def test_tool_call_in_stream(self):
        lines = self._make_lines([
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "search", "arguments": {"q": "test"}}}],
                },
                "done": False,
            },
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 10,
            },
        ])
        result = stream_ollama_with_callbacks(iter(lines))
        tc = result.choices[0].message.tool_calls[0]
        assert tc.function.name == "search"
        assert json.loads(tc.function.arguments) == {"q": "test"}
        assert tc.id.startswith("call_")
        assert result.choices[0].finish_reason == "tool_calls"

    def test_tool_start_callback_fired(self):
        lines = self._make_lines([
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "fn", "arguments": {"k": "v"}}}],
                },
                "done": False,
            },
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 0, "eval_count": 0},
        ])
        tool_starts = []
        stream_ollama_with_callbacks(
            iter(lines), on_tool_start=lambda n, a: tool_starts.append((n, a))
        )
        assert tool_starts == [("fn", {"k": "v"})]

    def test_reasoning_collected(self):
        lines = self._make_lines([
            {"message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": False},
            {"message": {"role": "assistant", "content": "answer"}, "done": False},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 0, "eval_count": 0},
        ])
        result = stream_ollama_with_callbacks(iter(lines))
        assert result.choices[0].message.reasoning == "hmm"
        assert result.choices[0].message.content == "answer"

    def test_reasoning_callback_fired(self):
        lines = self._make_lines([
            {"message": {"thinking": "let me think"}, "done": False},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 0, "eval_count": 0},
        ])
        reasoning_deltas = []
        stream_ollama_with_callbacks(
            iter(lines), on_reasoning_delta=reasoning_deltas.append
        )
        assert reasoning_deltas == ["let me think"]

    def test_interrupt_check_stops_iteration(self):
        call_count = 0

        def interrupt():
            nonlocal call_count
            call_count += 1
            return call_count >= 2

        lines = self._make_lines([
            {"message": {"content": "chunk1"}, "done": False},
            {"message": {"content": "chunk2"}, "done": False},
            {"message": {"content": "chunk3"}, "done": True, "prompt_eval_count": 0, "eval_count": 0},
        ])
        result = stream_ollama_with_callbacks(iter(lines), on_interrupt_check=interrupt)
        # Only chunk1 processed before interrupt on 2nd check
        assert result.choices[0].message.content == "chunk1"

    def test_non_json_lines_skipped(self):
        lines = [
            "not json at all",
            json.dumps({"message": {"content": "Hi"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True,
                        "prompt_eval_count": 1, "eval_count": 1}),
        ]
        result = stream_ollama_with_callbacks(iter(lines))
        assert result.choices[0].message.content == "Hi"
