"""Tests for issue #76997: non-streaming Gemini tool-turn narration relocation.

Vertex's OpenAI-compat layer for Gemini 3.x returns the model's pre-tool
narration in plain ``content`` on tool-call turns. ``normalize_response``
relocates it to ``reasoning_content`` so it is neither delivered as normal
assistant content nor persisted into session history.
"""

from types import SimpleNamespace

from agent.transports.chat_completions import ChatCompletionsTransport


def _make_choice(message, finish_reason="tool_calls"):
    return SimpleNamespace(index=0, message=message, finish_reason=finish_reason)


def _make_tool_call(name="read_file", arguments='{"path":"/tmp/a"}'):
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_message(content, tool_calls):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
        model_extra=None,
        refusal=None,
    )


def test_gemini_tool_turn_content_relocated_to_reasoning():
    transport = ChatCompletionsTransport()
    msg = _make_message("Locating Files", [_make_tool_call()])
    response = SimpleNamespace(
        choices=[_make_choice(msg)],
        model="gemini-3.5-flash",
        usage=None,
    )

    normalized = transport.normalize_response(response, model="gemini-3.5-flash")

    assert normalized.content is None
    assert normalized.tool_calls is not None
    assert normalized.provider_data.get("reasoning_content") == "Locating Files"


def test_gemini_tool_turn_reasoning_merged():
    """Existing reasoning_content is preserved and narration appended."""
    transport = ChatCompletionsTransport()
    msg = _make_message("Locating Files", [_make_tool_call()])
    msg.reasoning_content = "previous thought"
    response = SimpleNamespace(
        choices=[_make_choice(msg)],
        model="gemini-3.5-flash",
        usage=None,
    )

    normalized = transport.normalize_response(response, model="gemini-3.5-flash")

    assert normalized.content is None
    assert normalized.provider_data.get("reasoning_content") == "previous thoughtLocating Files"


def test_non_gemini_tool_turn_unchanged():
    transport = ChatCompletionsTransport()
    msg = _make_message("Reading file...", [_make_tool_call()])
    response = SimpleNamespace(
        choices=[_make_choice(msg)],
        model="gpt-5.5",
        usage=None,
    )

    normalized = transport.normalize_response(response, model="gpt-5.5")

    assert normalized.content == "Reading file..."
    assert normalized.tool_calls is not None
    assert normalized.provider_data is None or "reasoning_content" not in normalized.provider_data


def test_gemini_text_turn_unchanged():
    """No tool calls -> content is the real answer; never relocated."""
    transport = ChatCompletionsTransport()
    msg = _make_message("Here is the answer.", None)
    response = SimpleNamespace(
        choices=[_make_choice(msg, finish_reason="stop")],
        model="gemini-3.5-flash",
        usage=None,
    )

    normalized = transport.normalize_response(response, model="gemini-3.5-flash")

    assert normalized.content == "Here is the answer."
