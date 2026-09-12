"""Regression tests for the Chat Completions SDK request-transform bypass (#106776).

``chat.completions.create`` walks the entire ``messages`` and ``tools`` trees
against pydantic type unions with the GIL held — multi-MB conversations in
auxiliary tasks (compression, summaries, etc.) stall CPU execution pre-network.
Bulk wire-format fields are therefore routed through ``extra_body``, which the
SDK merges into the JSON body *after* the transform, producing a byte-identical request.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import OpenAI

from agent.chat_completion_helpers import bypass_chat_sdk_request_transform
from agent.codex_runtime import _is_plain_json_data


def _sample_chat_kwargs():
    return {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }


class TestBypassChatSdkRequestTransform:
    def test_moves_messages_and_tools_to_extra_body(self):
        kwargs = _sample_chat_kwargs()
        original_messages = kwargs["messages"]
        original_tools = kwargs["tools"]

        bypassed = bypass_chat_sdk_request_transform(kwargs)

        # Bulk fields moved to extra_body
        assert bypassed["messages"] == []
        assert "tools" not in bypassed
        assert bypassed["extra_body"]["messages"] is original_messages
        assert bypassed["extra_body"]["tools"] is original_tools

        # Scalar configuration preserved at top level
        assert bypassed["model"] == "gpt-4o"
        assert bypassed["temperature"] == 0.7
        assert bypassed["max_tokens"] == 512

        # Original caller dictionary remains unmutated
        assert kwargs["messages"] is original_messages
        assert "tools" in kwargs
        assert "extra_body" not in kwargs

    def test_merges_with_existing_extra_body_and_preserves_caller_precedence(self):
        kwargs = _sample_chat_kwargs()
        caller_extra = {"custom_field": "val", "messages": [{"role": "user", "content": "override"}]}
        kwargs["extra_body"] = caller_extra

        bypassed = bypass_chat_sdk_request_transform(kwargs)

        assert bypassed["extra_body"]["messages"] == [{"role": "user", "content": "override"}]
        assert bypassed["extra_body"]["custom_field"] == "val"
        assert bypassed["extra_body"]["tools"] == kwargs["tools"]

    def test_non_json_messages_stay_on_typed_path(self):
        kwargs = _sample_chat_kwargs()
        kwargs["messages"] = [{"role": "user", "content": object()}]

        bypassed = bypass_chat_sdk_request_transform(kwargs)

        assert bypassed["messages"] == kwargs["messages"]
        assert bypassed["extra_body"]["tools"] == kwargs["tools"]
        assert "messages" not in bypassed["extra_body"]

    def test_env_escape_hatch_restores_passthrough(self, monkeypatch):
        monkeypatch.setenv("HERMES_CHAT_SDK_TRANSFORM", "1")
        kwargs = _sample_chat_kwargs()

        assert bypass_chat_sdk_request_transform(kwargs) is kwargs

    def test_non_sdk_custom_adapter_is_passthrough(self):
        kwargs = _sample_chat_kwargs()
        custom_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))

        # Non-SDK client (no _post method) is returned untouched
        assert bypass_chat_sdk_request_transform(kwargs, client=custom_client) is kwargs


class TestWirePayloadByteIdentity:
    def test_bypassed_request_produces_identical_wire_body(self):
        captured_requests = []

        class MockTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured_requests.append(json.loads(request.content.decode("utf-8")))
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-test",
                        "choices": [
                            {"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                )

        client = OpenAI(
            api_key="mock-key",
            base_url="https://api.openai.com/v1",
            http_client=httpx.Client(transport=MockTransport()),
        )

        kwargs = _sample_chat_kwargs()

        # 1. Standard plain call without bypass
        client.chat.completions.create(**kwargs)

        # 2. Bypassed call
        bypassed_kwargs = bypass_chat_sdk_request_transform(kwargs, client=client)
        client.chat.completions.create(**bypassed_kwargs)

        assert len(captured_requests) == 2
        # Wire bodies must be 100% structurally and byte identical
        assert captured_requests[0] == captured_requests[1]


class TestSummaryAttemptBypassIntegration:
    def test_summary_attempt_uses_bypassed_kwargs(self, monkeypatch):
        from agent import chat_completion_helpers as cch
        from run_agent import AIAgent

        mock_client = MagicMock()
        mock_client.chat.completions._post = MagicMock()  # Mark as OpenAI SDK client
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary text"))]
        )

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="deepseek/deepseek-chat",
            provider="openrouter",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._transport = None
        monkeypatch.setattr(agent, "_ensure_primary_openai_client", lambda reason: mock_client)

        attempt_fn = cch._chat_summary_attempt(agent, [{"role": "user", "content": "long context"}], "req-1")
        summary = attempt_fn(0)

        assert summary == "summary text"
        assert mock_client.chat.completions.create.called
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == []
        assert "messages" in call_kwargs["extra_body"]
        assert call_kwargs["extra_body"]["messages"][0]["content"] == "long context"
