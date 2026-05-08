"""Tests for the BedrockTransport."""

import json
import pytest
from types import SimpleNamespace

from agent.transports import get_transport
from agent.transports.types import NormalizedResponse, ToolCall


@pytest.fixture
def transport():
    import agent.transports.bedrock  # noqa: F401
    return get_transport("bedrock_converse")


class TestBedrockBasic:

    def test_api_mode(self, transport):
        assert transport.api_mode == "bedrock_converse"

    def test_registered(self, transport):
        assert transport is not None


class TestBedrockBuildKwargs:

    def test_basic_kwargs(self, transport):
        msgs = [{"role": "user", "content": "Hello"}]
        kw = transport.build_kwargs(model="anthropic.claude-3-5-sonnet-20241022-v2:0", messages=msgs)
        assert kw["modelId"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert kw["__bedrock_converse__"] is True
        assert kw["__bedrock_region__"] == "us-east-1"
        assert "messages" in kw

    def test_custom_region(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=msgs,
            region="eu-west-1",
        )
        assert kw["__bedrock_region__"] == "eu-west-1"

    def test_max_tokens(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=msgs,
            max_tokens=8192,
        )
        assert kw["inferenceConfig"]["maxTokens"] == 8192


class TestBedrockConvertTools:

    def test_convert_tools(self, transport):
        tools = [{
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run commands",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        }]
        result = transport.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["toolSpec"]["name"] == "terminal"


class TestBedrockValidate:

    def test_none(self, transport):
        assert transport.validate_response(None) is False

    def test_raw_dict_valid(self, transport):
        assert transport.validate_response({"output": {"message": {}}}) is True

    def test_raw_dict_invalid(self, transport):
        assert transport.validate_response({"error": "fail"}) is False

    def test_normalized_valid(self, transport):
        r = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
        assert transport.validate_response(r) is True


class TestBedrockMapFinishReason:

    def test_end_turn(self, transport):
        assert transport.map_finish_reason("end_turn") == "stop"

    def test_tool_use(self, transport):
        assert transport.map_finish_reason("tool_use") == "tool_calls"

    def test_max_tokens(self, transport):
        assert transport.map_finish_reason("max_tokens") == "length"

    def test_guardrail(self, transport):
        assert transport.map_finish_reason("guardrail_intervened") == "content_filter"

    def test_unknown(self, transport):
        assert transport.map_finish_reason("unknown") == "stop"


class TestBedrockNormalize:

    def _make_bedrock_response(self, text="Hello", tool_calls=None, stop_reason="end_turn"):
        """Build a raw Bedrock converse response dict."""
        content = []
        if text:
            content.append({"text": text})
        if tool_calls:
            for tc in tool_calls:
                content.append({
                    "toolUse": {
                        "toolUseId": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"],
                    }
                })
        return {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": stop_reason,
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }

    def test_text_response(self, transport):
        raw = self._make_bedrock_response(text="Hello world")
        nr = transport.normalize_response(raw)
        assert isinstance(nr, NormalizedResponse)
        assert nr.content == "Hello world"
        assert nr.finish_reason == "stop"

    def test_tool_call_response(self, transport):
        raw = self._make_bedrock_response(
            text=None,
            tool_calls=[{"id": "tool_1", "name": "terminal", "input": {"command": "ls"}}],
            stop_reason="tool_use",
        )
        nr = transport.normalize_response(raw)
        assert nr.finish_reason == "tool_calls"
        assert len(nr.tool_calls) == 1
        assert nr.tool_calls[0].name == "terminal"

    def test_raw_reasoning_content_response(self, transport):
        raw = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"reasoningContent": {"text": "Let me think..."}},
                        {"text": "Answer."},
                    ],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }
        nr = transport.normalize_response(raw)
        assert nr.reasoning == "Let me think..."
        assert nr.content == "Answer."

    def test_already_normalized_response(self, transport):
        """Test normalize_response handles already-normalized SimpleNamespace (from dispatch site)."""
        pre_normalized = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello from Bedrock",
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        nr = transport.normalize_response(pre_normalized)
        assert isinstance(nr, NormalizedResponse)
        assert nr.content == "Hello from Bedrock"
        assert nr.finish_reason == "stop"
        assert nr.usage is not None
        assert nr.usage.prompt_tokens == 10
