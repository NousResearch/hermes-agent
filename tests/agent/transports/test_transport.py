"""Tests for the transport ABC, registry, and AnthropicTransport."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage
from agent.transports import get_transport, register_transport, _REGISTRY


# ── ABC contract tests ──────────────────────────────────────────────────

class TestProviderTransportABC:
    """Verify the ABC contract is enforceable."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ProviderTransport()

    def test_concrete_must_implement_all_abstract(self):
        class Incomplete(ProviderTransport):
            @property
            def api_mode(self):
                return "test"
        with pytest.raises(TypeError):
            Incomplete()

    def test_minimal_concrete(self):
        class Minimal(ProviderTransport):
            @property
            def api_mode(self):
                return "test_minimal"
            def convert_messages(self, messages, **kw):
                return messages
            def convert_tools(self, tools):
                return tools
            def build_kwargs(self, model, messages, tools=None, **params):
                return {"model": model, "messages": messages}
            def normalize_response(self, response, **kw):
                return NormalizedResponse(content="ok", tool_calls=None, finish_reason="stop")

        t = Minimal()
        assert t.api_mode == "test_minimal"
        assert t.validate_response(None) is True  # default
        assert t.extract_cache_stats(None) is None  # default
        assert t.map_finish_reason("end_turn") == "end_turn"  # default passthrough


# ── Registry tests ───────────────────────────────────────────────────────

class TestTransportRegistry:

    def test_get_unregistered_returns_none(self):
        assert get_transport("nonexistent_mode") is None

    def test_anthropic_registered_on_import(self):
        import agent.transports.anthropic  # noqa: F401
        t = get_transport("anthropic_messages")
        assert t is not None
        assert t.api_mode == "anthropic_messages"

    def test_register_and_get(self):
        class DummyTransport(ProviderTransport):
            @property
            def api_mode(self):
                return "dummy_test"
            def convert_messages(self, messages, **kw):
                return messages
            def convert_tools(self, tools):
                return tools
            def build_kwargs(self, model, messages, tools=None, **params):
                return {}
            def normalize_response(self, response, **kw):
                return NormalizedResponse(content=None, tool_calls=None, finish_reason="stop")

        register_transport("dummy_test", DummyTransport)
        t = get_transport("dummy_test")
        assert t.api_mode == "dummy_test"
        # Cleanup
        _REGISTRY.pop("dummy_test", None)


# ── AnthropicTransport tests ────────────────────────────────────────────

class TestAnthropicTransport:

    @pytest.fixture
    def transport(self):
        import agent.transports.anthropic  # noqa: F401
        return get_transport("anthropic_messages")

    def test_api_mode(self, transport):
        assert transport.api_mode == "anthropic_messages"

    def test_convert_tools_simple(self, transport):
        tools = [{
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test",
                "parameters": {"type": "object", "properties": {}},
            }
        }]
        result = transport.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "test_tool"
        assert "input_schema" in result[0]

    def test_validate_response_none(self, transport):
        assert transport.validate_response(None) is False

    def test_validate_response_empty_content(self, transport):
        r = SimpleNamespace(content=[])
        assert transport.validate_response(r) is False

    def test_validate_response_valid(self, transport):
        r = SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")])
        assert transport.validate_response(r) is True

    def test_map_finish_reason(self, transport):
        assert transport.map_finish_reason("end_turn") == "stop"
        assert transport.map_finish_reason("tool_use") == "tool_calls"
        assert transport.map_finish_reason("max_tokens") == "length"
        assert transport.map_finish_reason("stop_sequence") == "stop"
        assert transport.map_finish_reason("refusal") == "content_filter"
        assert transport.map_finish_reason("model_context_window_exceeded") == "length"
        assert transport.map_finish_reason("unknown") == "stop"

    def test_extract_cache_stats_none_usage(self, transport):
        r = SimpleNamespace(usage=None)
        assert transport.extract_cache_stats(r) is None

    def test_extract_cache_stats_with_cache(self, transport):
        usage = SimpleNamespace(cache_read_input_tokens=100, cache_creation_input_tokens=50)
        r = SimpleNamespace(usage=usage)
        result = transport.extract_cache_stats(r)
        assert result == {"cached_tokens": 100, "creation_tokens": 50}

    def test_extract_cache_stats_zero(self, transport):
        usage = SimpleNamespace(cache_read_input_tokens=0, cache_creation_input_tokens=0)
        r = SimpleNamespace(usage=usage)
        assert transport.extract_cache_stats(r) is None

    def test_normalize_response_text(self, transport):
        """Test normalization of a simple text response."""
        r = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Hello world")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            model="claude-sonnet-4-6",
        )
        nr = transport.normalize_response(r)
        assert isinstance(nr, NormalizedResponse)
        assert nr.content == "Hello world"
        assert nr.tool_calls is None or nr.tool_calls == []
        assert nr.finish_reason == "stop"

    def test_normalize_response_tool_calls(self, transport):
        """Test normalization of a tool-use response."""
        r = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_123",
                    name="terminal",
                    input={"command": "ls"},
                ),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            model="claude-sonnet-4-6",
        )
        nr = transport.normalize_response(r)
        assert nr.finish_reason == "tool_calls"
        assert len(nr.tool_calls) == 1
        tc = nr.tool_calls[0]
        assert tc.name == "terminal"
        assert tc.id == "toolu_123"
        assert '"command"' in tc.arguments

    def test_normalize_response_thinking(self, transport):
        """Test normalization preserves thinking content."""
        r = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="Let me think..."),
                SimpleNamespace(type="text", text="The answer is 42"),
            ],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=15),
            model="claude-sonnet-4-6",
        )
        nr = transport.normalize_response(r)
        assert nr.content == "The answer is 42"
        assert nr.reasoning == "Let me think..."

    def test_build_kwargs_returns_dict(self, transport):
        """Test build_kwargs produces a usable kwargs dict."""
        messages = [{"role": "user", "content": "Hello"}]
        kw = transport.build_kwargs(
            model="claude-sonnet-4-6",
            messages=messages,
            max_tokens=1024,
        )
        assert isinstance(kw, dict)
        assert "model" in kw
        assert "max_tokens" in kw
        assert "messages" in kw

    def test_convert_messages_extracts_system(self, transport):
        """Test convert_messages separates system from messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, msgs = transport.convert_messages(messages)
        # System should be extracted
        assert system is not None
        # Messages should only have user
        assert len(msgs) >= 1
