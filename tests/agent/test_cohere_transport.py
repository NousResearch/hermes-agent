"""Tests for the Cohere v2 chat transport.

Covers:
  - registration in the transport registry
  - convert_messages: tool-result document wrapping + assistant tool-call shape
  - convert_tools: stripping unsupported OpenAI fields
  - build_kwargs: native-knob plumbing (safety_mode, citation_options,
    documents, connectors, force_single_step, thinking via reasoning_config)
  - normalize_response: text + tool_plan reasoning + tool_calls + citations
  - validate_response: empty-content + TOOL_CALL finish_reason is valid
  - map_finish_reason: COMPLETE/TOOL_CALL/MAX_TOKENS → OpenAI vocabulary
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.transports import get_transport
from agent.transports.cohere import CohereTransport, _convert_message, _strip_tool_fields


# ── Registry ────────────────────────────────────────────────────────────


class TestCohereTransportRegistry:
    def test_registered_on_import(self):
        import agent.transports.cohere  # noqa: F401
        t = get_transport("cohere_chat")
        assert t is not None
        assert isinstance(t, CohereTransport)
        assert t.api_mode == "cohere_chat"


# ── convert_messages ────────────────────────────────────────────────────


class TestConvertMessages:
    def test_user_message_passthrough(self):
        t = CohereTransport()
        out = t.convert_messages([{"role": "user", "content": "hi"}])
        assert out == [{"role": "user", "content": "hi"}]

    def test_system_message_passthrough(self):
        t = CohereTransport()
        out = t.convert_messages([{"role": "system", "content": "be helpful"}])
        assert out == [{"role": "system", "content": "be helpful"}]

    def test_tool_result_string_wrapped_as_document(self):
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "the answer is 42"}
        out = _convert_message(msg)
        assert out["role"] == "tool"
        assert out["tool_call_id"] == "call_1"
        assert isinstance(out["content"], list)
        assert out["content"][0] == {
            "type": "document",
            "document": {"data": "the answer is 42"},
        }

    def test_tool_result_list_flattened(self):
        msg = {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": [
                {"type": "text", "text": "part one"},
                {"type": "text", "text": "part two"},
            ],
        }
        out = _convert_message(msg)
        assert out["content"][0]["document"]["data"] == "part one\npart two"

    def test_assistant_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "",
            "reasoning": "I should call the weather tool",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                }
            ],
        }
        out = _convert_message(msg)
        assert out["role"] == "assistant"
        assert out["tool_plan"] == "I should call the weather tool"
        assert out["tool_calls"][0]["id"] == "tc_1"
        assert out["tool_calls"][0]["function"]["name"] == "get_weather"
        assert out["tool_calls"][0]["function"]["arguments"] == '{"city": "Tokyo"}'

    def test_assistant_tool_calls_with_dict_arguments_serialized(self):
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_x",
                    "function": {"name": "do_thing", "arguments": {"k": "v"}},
                }
            ],
        }
        out = _convert_message(msg)
        args_str = out["tool_calls"][0]["function"]["arguments"]
        assert json.loads(args_str) == {"k": "v"}

    def test_drops_non_dict_entries(self):
        t = CohereTransport()
        out = t.convert_messages([{"role": "user", "content": "hi"}, "junk", None])
        assert len(out) == 1


# ── convert_tools ───────────────────────────────────────────────────────


class TestConvertTools:
    def test_empty_tools(self):
        t = CohereTransport()
        assert t.convert_tools([]) == []
        assert t.convert_tools(None) == []

    def test_passthrough_basic(self):
        t = CohereTransport()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        out = t.convert_tools(tools)
        assert out[0]["function"]["name"] == "search"

    def test_strips_unsupported_fields(self):
        tools = [
            {
                "type": "function",
                "strict": True,
                "cache_control": {"type": "ephemeral"},
                "function": {
                    "name": "f",
                    "strict": True,
                    "cache_control": {"type": "ephemeral"},
                    "parameters": {"type": "object"},
                },
            }
        ]
        out = _strip_tool_fields(tools)
        assert "strict" not in out[0]
        assert "cache_control" not in out[0]
        assert "strict" not in out[0]["function"]
        assert "cache_control" not in out[0]["function"]


# ── build_kwargs ────────────────────────────────────────────────────────


class TestBuildKwargs:
    def test_minimal(self):
        t = CohereTransport()
        kw = t.build_kwargs(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert kw["model"] == "command-a-03-2025"
        assert kw["messages"] == [{"role": "user", "content": "hi"}]
        assert "tools" not in kw

    def test_tools_included_when_present(self):
        t = CohereTransport()
        kw = t.build_kwargs(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {"type": "object"}},
                }
            ],
        )
        assert kw["tools"][0]["function"]["name"] == "f"

    def test_max_tokens_and_temperature(self):
        t = CohereTransport()
        kw = t.build_kwargs(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
        )
        assert kw["max_tokens"] == 512
        assert kw["temperature"] == pytest.approx(0.3)
        assert kw["p"] == pytest.approx(0.9)

    def test_native_knobs_direct_params(self):
        t = CohereTransport()
        kw = t.build_kwargs(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            safety_mode="strict",
            citation_options={"mode": "ACCURATE"},
            documents=[{"data": "doc one"}],
            connectors=[{"id": "web-search"}],
            force_single_step=True,
        )
        assert kw["safety_mode"] == "STRICT"
        assert kw["citation_options"] == {"mode": "ACCURATE"}
        assert kw["documents"] == [{"data": "doc one"}]
        assert kw["connectors"] == [{"id": "web-search"}]
        assert kw["force_single_step"] is True

    def test_empty_collections_are_dropped(self):
        t = CohereTransport()
        kw = t.build_kwargs(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            documents=[],
            connectors=[],
            citation_options={},
        )
        assert "documents" not in kw
        assert "connectors" not in kw
        assert "citation_options" not in kw

    def test_profile_drives_thinking_for_reasoning_model(self):
        """When a CohereProfile is passed, reasoning_config + reasoning model
        should produce a Cohere ``thinking`` field with a token budget."""
        from providers import get_provider_profile

        profile = get_provider_profile("cohere")
        assert profile is not None, "CohereProfile must be auto-discovered"
        t = CohereTransport()
        kw = t.build_kwargs(
            model="command-a-reasoning-08-2025",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "high"},
            provider_profile=profile,
        )
        assert "thinking" in kw
        assert kw["thinking"]["type"] == "enabled"
        assert kw["thinking"]["token_budget"] == 16384

    def test_profile_no_thinking_for_non_reasoning_model(self):
        from providers import get_provider_profile

        profile = get_provider_profile("cohere")
        assert profile is not None
        t = CohereTransport()
        kw = t.build_kwargs(
            model="command-r-08-2024",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "high"},
            provider_profile=profile,
        )
        assert "thinking" not in kw

    def test_explicit_thinking_budget_overrides_effort(self):
        from providers import get_provider_profile

        profile = get_provider_profile("cohere")
        assert profile is not None
        t = CohereTransport()
        kw = t.build_kwargs(
            model="command-a-reasoning-08-2025",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"},
            thinking_token_budget=4096,
            provider_profile=profile,
        )
        assert kw["thinking"]["token_budget"] == 4096


# ── normalize_response ──────────────────────────────────────────────────


def _make_text_content_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _make_tool_call(id_: str, name: str, args: str):
    return SimpleNamespace(
        id=id_,
        type="function",
        function=SimpleNamespace(name=name, arguments=args),
    )


class TestNormalizeResponse:
    def test_text_only(self):
        t = CohereTransport()
        resp = SimpleNamespace(
            message=SimpleNamespace(
                content=[_make_text_content_block("hello world")],
                tool_plan=None,
                tool_calls=None,
                citations=None,
            ),
            finish_reason="COMPLETE",
            usage=SimpleNamespace(
                tokens=SimpleNamespace(input_tokens=12, output_tokens=4)
            ),
        )
        out = t.normalize_response(resp)
        assert out.content == "hello world"
        assert out.tool_calls is None
        assert out.finish_reason == "stop"
        assert out.usage.prompt_tokens == 12
        assert out.usage.completion_tokens == 4
        assert out.usage.total_tokens == 16

    def test_tool_calls_with_tool_plan(self):
        t = CohereTransport()
        resp = SimpleNamespace(
            message=SimpleNamespace(
                content=[],
                tool_plan="I need to check the weather",
                tool_calls=[_make_tool_call("tc1", "get_weather", '{"city": "Tokyo"}')],
                citations=None,
            ),
            finish_reason="TOOL_CALL",
            usage=None,
        )
        out = t.normalize_response(resp)
        assert out.content is None
        assert out.finish_reason == "tool_calls"
        assert out.reasoning == "I need to check the weather"
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].name == "get_weather"
        assert out.tool_calls[0].arguments == '{"city": "Tokyo"}'

    def test_max_tokens_finish_reason(self):
        t = CohereTransport()
        resp = SimpleNamespace(
            message=SimpleNamespace(
                content=[_make_text_content_block("partial...")],
                tool_plan=None,
                tool_calls=None,
                citations=None,
            ),
            finish_reason="MAX_TOKENS",
            usage=None,
        )
        out = t.normalize_response(resp)
        assert out.finish_reason == "length"

    def test_citations_stashed_in_provider_data(self):
        t = CohereTransport()
        citation = {"start": 0, "end": 5, "text": "hello", "sources": ["doc-1"]}
        resp = SimpleNamespace(
            message=SimpleNamespace(
                content=[_make_text_content_block("hello world")],
                tool_plan=None,
                tool_calls=None,
                citations=[citation],
            ),
            finish_reason="COMPLETE",
            usage=None,
        )
        out = t.normalize_response(resp)
        assert out.provider_data is not None
        assert out.provider_data["citations"] == [citation]

    def test_tool_call_with_dict_arguments_is_serialized(self):
        t = CohereTransport()
        tc = {
            "id": "tc1",
            "type": "function",
            "function": {"name": "f", "arguments": {"k": "v"}},
        }
        resp = SimpleNamespace(
            message=SimpleNamespace(
                content=[], tool_plan=None, tool_calls=[tc], citations=None
            ),
            finish_reason="TOOL_CALL",
            usage=None,
        )
        out = t.normalize_response(resp)
        assert out.tool_calls[0].arguments == '{"k": "v"}'


# ── validate_response ───────────────────────────────────────────────────


class TestValidateResponse:
    def test_none_invalid(self):
        t = CohereTransport()
        assert t.validate_response(None) is False

    def test_no_message_invalid(self):
        t = CohereTransport()
        assert t.validate_response(SimpleNamespace()) is False

    def test_text_content_valid(self):
        t = CohereTransport()
        resp = SimpleNamespace(
            message=SimpleNamespace(content=[_make_text_content_block("hi")]),
            finish_reason="COMPLETE",
        )
        assert t.validate_response(resp) is True

    def test_empty_content_with_tool_call_valid(self):
        t = CohereTransport()
        resp = SimpleNamespace(
            message=SimpleNamespace(
                content=[],
                tool_calls=[_make_tool_call("tc1", "f", "{}")],
            ),
            finish_reason="TOOL_CALL",
        )
        assert t.validate_response(resp) is True

    def test_empty_content_with_tool_call_finish_reason_valid(self):
        """Empty content + finish_reason=TOOL_CALL (even without tool_calls
        populated) is treated as valid — Cohere occasionally yields this
        shape during streaming."""
        t = CohereTransport()
        resp = SimpleNamespace(
            message=SimpleNamespace(content=[], tool_calls=None),
            finish_reason="TOOL_CALL",
        )
        assert t.validate_response(resp) is True

    def test_empty_content_no_tool_call_invalid(self):
        t = CohereTransport()
        resp = SimpleNamespace(
            message=SimpleNamespace(content=[], tool_calls=None),
            finish_reason="COMPLETE",
        )
        assert t.validate_response(resp) is False


# ── map_finish_reason ───────────────────────────────────────────────────


class TestMapFinishReason:
    def test_complete(self):
        t = CohereTransport()
        assert t.map_finish_reason("COMPLETE") == "stop"

    def test_max_tokens(self):
        t = CohereTransport()
        assert t.map_finish_reason("MAX_TOKENS") == "length"

    def test_tool_call(self):
        t = CohereTransport()
        assert t.map_finish_reason("TOOL_CALL") == "tool_calls"

    def test_error_toxic(self):
        t = CohereTransport()
        assert t.map_finish_reason("ERROR_TOXIC") == "content_filter"

    def test_unknown_falls_back_to_stop(self):
        t = CohereTransport()
        assert t.map_finish_reason("WHATEVER") == "stop"
