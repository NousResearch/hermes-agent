"""Regression tests: Anthropic Messages transport must recover tool calls that
the model leaked as ``<invoke name=...>`` XML inside a *text* block.

Background (verified against a real session state.db):

Some models served behind ``api_mode: anthropic_messages`` (e.g. a private
Vertex/Bedrock-style gateway fronting Claude) intermittently degenerate: instead
of emitting a native ``tool_use`` content block, they emit the tool call as the
Anthropic *prompt-format* XML —

    call
    <invoke name="patch">
    <parameter name="mode">replace</parameter>
    <parameter name="new_string">X</parameter>
    <parameter name="path">/a/b.md</parameter>
    </invoke>

— inside an ordinary ``text`` block, with ``stop_reason="end_turn"``.

Before the fix, ``AnthropicTransport.normalize_response`` only built
``tool_calls`` from ``block.type == "tool_use"``, so:
  * ``tool_calls`` came back ``None`` (the tool never ran), and
  * the raw XML leaked into ``content`` and was shown to the user (Telegram, CLI).

This is platform-independent — the leak was observed on both ``telegram`` and
``cli`` sources. The fix lives in the transport (platform-agnostic), not the
Telegram adapter.
"""

from __future__ import annotations

from types import SimpleNamespace


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, block_id: str = "tc_1", input_data: dict | None = None):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=input_data or {})


def _make_response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        model="claude-opus-4-8",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


def _get_transport():
    from agent.transports.anthropic import AnthropicTransport
    return AnthropicTransport()


# The exact shape captured from the real leaking session.
_LEAKED = (
    'GPU.md 写得很扎实但 NVIDIA-only。我做针对性修改。\n\n'
    'call\n'
    '<invoke name="patch">\n'
    '<parameter name="mode">replace</parameter>\n'
    '<parameter name="new_string">new body text</parameter>\n'
    '<parameter name="old_string">old body text</parameter>\n'
    '<parameter name="path">/home/zhenc/sparsed/SOURCE-SPEC.md</parameter>\n'
    '</invoke>'
)


class TestAnthropicXmlToolLeakRecovery:
    def test_leaked_invoke_block_is_recovered_as_tool_call(self):
        transport = _get_transport()
        response = _make_response(_text_block(_LEAKED))

        result = transport.normalize_response(response)

        assert result.tool_calls, "leaked <invoke> XML must be recovered into a structured tool call"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.name == "patch"

        import json
        args = json.loads(tc.arguments)
        assert args["mode"] == "replace"
        assert args["new_string"] == "new body text"
        assert args["old_string"] == "old body text"
        assert args["path"] == "/home/zhenc/sparsed/SOURCE-SPEC.md"

    def test_finish_reason_becomes_tool_calls(self):
        transport = _get_transport()
        response = _make_response(_text_block(_LEAKED), stop_reason="end_turn")

        result = transport.normalize_response(response)

        assert result.finish_reason == "tool_calls"

    def test_leaked_xml_is_stripped_from_visible_content(self):
        transport = _get_transport()
        response = _make_response(_text_block(_LEAKED))

        result = transport.normalize_response(response)

        content = result.content or ""
        assert "<invoke" not in content
        assert "<parameter" not in content
        # The model's genuine prose preamble should be preserved.
        assert "针对性修改" in content

    def test_native_tool_use_block_still_wins_and_no_double_count(self):
        """When a real tool_use block is present, we must NOT also re-parse any
        XML-looking text — the structured block is authoritative."""
        transport = _get_transport()
        response = _make_response(
            _text_block("Here is the call:\n" + _LEAKED),
            _tool_use_block("patch", input_data={"mode": "replace", "path": "/x"}),
            stop_reason="tool_use",
        )

        result = transport.normalize_response(response)

        assert len(result.tool_calls) == 1
        # The authoritative one is the structured tool_use block.
        assert result.tool_calls[0].id == "tc_1"
        assert result.tool_calls[0].name == "patch"

    def test_plain_text_with_angle_brackets_is_not_misparsed(self):
        """Prose that merely mentions <invoke> without a well-formed block must
        not be turned into a tool call."""
        transport = _get_transport()
        response = _make_response(_text_block(
            "I noticed Telegram returned a call <invoke> in the message. Is that a bug?"
        ))

        result = transport.normalize_response(response)

        assert not result.tool_calls
        assert result.finish_reason == "stop"
        assert "<invoke>" in (result.content or "")

    def test_multiple_leaked_blocks_all_recovered(self):
        transport = _get_transport()
        two = _LEAKED + "\n\n现在改第二处。\n\n" + (
            'call\n'
            '<invoke name="read_file">\n'
            '<parameter name="path">/tmp/x.md</parameter>\n'
            '</invoke>'
        )
        response = _make_response(_text_block(two))

        result = transport.normalize_response(response)

        assert len(result.tool_calls) == 2
        names = [tc.name for tc in result.tool_calls]
        assert names == ["patch", "read_file"]
