"""Promote native ``<tool_call>`` text into structured tool_calls (#107080).

Ollama's OpenAI-compat ``/v1`` can drop Gemma/trajectory XML tool calls instead
of emitting structured ``tool_calls``. Chat Completions normalize must promote
well-formed ``<tool_call>`` JSON, and Ollama-local requests must omit wire
``tools`` so the XML survives in ``content``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.transports import get_transport


GEMMA_XML = (
    '<tool_call>\n'
    '{"name": "ha_call_service", "args": {"action": "turn_off", "device_id": "light.salon"}}\n'
    "</tool_call>"
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ha_call_service",
            "description": "Call a Home Assistant service",
            "parameters": {"type": "object"},
        },
    }
]


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401

    return get_transport("chat_completions")


def _response(content, tool_calls=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def test_gemma_native_shape_promoted_via_normalize_response(transport):
    normalized = transport.normalize_response(
        _response(GEMMA_XML, tool_calls=None, finish_reason="stop")
    )

    assert normalized.tool_calls is not None
    assert len(normalized.tool_calls) == 1
    assert normalized.tool_calls[0].name == "ha_call_service"
    assert json.loads(normalized.tool_calls[0].arguments)["action"] == "turn_off"
    assert json.loads(normalized.tool_calls[0].arguments)["device_id"] == "light.salon"
    assert normalized.finish_reason == "tool_calls"
    assert not (normalized.content or "").strip()
    assert "<tool_call>" not in (normalized.content or "")


def test_name_arguments_shape_and_openai_xml_also_promoted(transport):
    trajectory = (
        '<tool_call>{"name": "lookup", "arguments": {"city": "Paris"}}</tool_call>'
    )
    normalized = transport.normalize_response(_response(trajectory))
    assert normalized.tool_calls[0].name == "lookup"
    assert json.loads(normalized.tool_calls[0].arguments)["city"] == "Paris"
    assert normalized.finish_reason == "tool_calls"

    openai_xml = (
        '<tool_call>{"id": "c1", "type": "function", '
        '"function": {"name": "memory", "arguments": "{\\"action\\": \\"add\\"}"}}'
        "</tool_call>"
    )
    openai_nr = transport.normalize_response(_response(openai_xml))
    assert openai_nr.tool_calls[0].name == "memory"
    assert openai_nr.tool_calls[0].id == "c1"
    assert json.loads(openai_nr.tool_calls[0].arguments) == {"action": "add"}


def test_fail_open_empty_content_does_not_invent_tool_calls(transport):
    for content in (None, "", "   ", "\n"):
        normalized = transport.normalize_response(
            _response(content, tool_calls=None, finish_reason="stop")
        )
        assert not normalized.tool_calls
        assert normalized.finish_reason == "stop"


def test_fail_open_structured_tool_calls_not_overwritten(transport):
    structured = SimpleNamespace(
        id="call_structured",
        function=SimpleNamespace(name="already_there", arguments='{"ok": true}'),
    )
    normalized = transport.normalize_response(
        _response(GEMMA_XML, tool_calls=[structured], finish_reason="tool_calls")
    )
    assert len(normalized.tool_calls) == 1
    assert normalized.tool_calls[0].name == "already_there"
    assert normalized.tool_calls[0].id == "call_structured"
    assert normalized.content == GEMMA_XML


def test_fail_open_invalid_or_nameless_xml_left_as_content(transport):
    invalid = "<tool_call>{not json}</tool_call>please retry"
    normalized = transport.normalize_response(_response(invalid))
    assert not normalized.tool_calls
    assert normalized.finish_reason == "stop"
    assert normalized.content == invalid

    nameless = '<tool_call>{"args": {"x": 1}}</tool_call>'
    nameless_nr = transport.normalize_response(_response(nameless))
    assert not nameless_nr.tool_calls
    assert nameless_nr.content == nameless


def test_bare_json_without_tool_call_wrapper_is_not_promoted(transport):
    bare = '{"name": "ha_call_service", "args": {"action": "turn_off"}}'
    normalized = transport.normalize_response(_response(bare))
    assert not normalized.tool_calls
    assert normalized.finish_reason == "stop"
    assert normalized.content == bare


def test_ollama_local_omits_wire_tools_and_injects_text_bridge(transport):
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "turn off the salon light"},
    ]
    kw = transport.build_kwargs(
        model="gemma3n",
        messages=messages,
        tools=_TOOLS,
        provider_name="ollama",
        base_url="http://127.0.0.1:11434/v1",
    )
    assert "tools" not in kw
    assert "tool_choice" not in kw
    system = kw["messages"][0]["content"]
    assert "<tool_call>" in system
    assert "ha_call_service" in system
    assert "{name, args}" in system or "{name,args}" in system or '"args"' in system
    # Original history is not mutated.
    assert messages[0]["content"] == "You are helpful."


def test_non_ollama_keeps_structured_tools(transport):
    messages = [{"role": "user", "content": "hi"}]
    kw = transport.build_kwargs(
        model="gpt-4o",
        messages=messages,
        tools=_TOOLS,
        provider_name="openai",
        base_url="https://api.openai.com/v1",
    )
    assert kw["tools"] == _TOOLS
    assert all("<tool_call>" not in str(m.get("content") or "") for m in kw["messages"])


def test_local_non_ollama_keeps_structured_tools(transport):
    """LM Studio / vLLM need the OpenAI tools channel; do not omit indiscriminately."""
    for provider_name, base_url in (
        ("lmstudio", "http://127.0.0.1:1234/v1"),
        ("custom", "http://127.0.0.1:8000/v1"),
        ("ollama-cloud", "https://ollama.com/v1"),
    ):
        kw = transport.build_kwargs(
            model="local-model",
            messages=[{"role": "user", "content": "hi"}],
            tools=_TOOLS,
            provider_name=provider_name,
            base_url=base_url,
        )
        assert kw["tools"] == _TOOLS, (provider_name, base_url)
