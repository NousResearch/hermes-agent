"""Tests for fallback XML tool call extraction in ChatCompletionsTransport."""

import json
from types import SimpleNamespace
import pytest

from agent.transports import get_transport
from agent.transports.types import ToolCall


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


class TestChatCompletionsXMLToolCalls:
    def test_extract_dots_function_call_from_reasoning_content(self, transport):
        raw_xml = """Thinking through the problem:
<dots_function_call>
<invoke name="execute_code">
<parameter name="code">
import subprocess
print("testing dots tool call")
</parameter>
</invoke>
</dots_function_call>
Finished thinking."""

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=None,
                        reasoning_content=raw_xml,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

        normalized = transport.normalize_response(response)

        assert normalized.tool_calls is not None
        assert len(normalized.tool_calls) == 1
        tc = normalized.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.name == "execute_code"
        args = json.loads(tc.arguments)
        assert "import subprocess" in args["code"]
        assert normalized.finish_reason == "tool_calls"
        # Reasoning content should have the XML block stripped
        assert "<dots_function_call>" not in (normalized.reasoning_content or "")
        assert "Thinking through the problem:" in (normalized.reasoning_content or "")

    def test_extract_dots_patch_call_multiline(self, transport):
        raw_xml = """<dots_function_call>
<invoke name="patch">
<parameter name="path">C:/Users/Larry/cli.py</parameter>
<parameter name="old_string">foo = 1</parameter>
<parameter name="new_string">foo = 2 ✅</parameter>
</invoke>
</dots_function_call>"""

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=None,
                        reasoning=raw_xml,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

        normalized = transport.normalize_response(response)

        assert normalized.tool_calls is not None
        assert len(normalized.tool_calls) == 1
        tc = normalized.tool_calls[0]
        assert tc.name == "patch"
        args = json.loads(tc.arguments)
        assert args["path"] == "C:/Users/Larry/cli.py"
        assert args["old_string"] == "foo = 1"
        assert args["new_string"] == "foo = 2 ✅"
        assert normalized.finish_reason == "tool_calls"

    def test_extract_xml_tool_call_from_content(self, transport):
        raw_xml = """<tool_call>
<invoke name="read_file">
<parameter name="path">settings.json</parameter>
</invoke>
</tool_call>"""

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=raw_xml,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

        normalized = transport.normalize_response(response)

        assert normalized.tool_calls is not None
        assert len(normalized.tool_calls) == 1
        assert normalized.tool_calls[0].name == "read_file"
        assert json.loads(normalized.tool_calls[0].arguments)["path"] == "settings.json"
        assert normalized.finish_reason == "tool_calls"

    def test_standard_tool_calls_precedence(self, transport):
        # When wire tool_calls is already present, it should not be replaced
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_wire",
                                function=SimpleNamespace(name="wire_tool", arguments='{"k":"v"}'),
                            )
                        ],
                        reasoning_content="<dots_function_call><invoke name=\"xml_tool\"><parameter name=\"x\">1</parameter></invoke></dots_function_call>",
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        )

        normalized = transport.normalize_response(response)

        assert normalized.tool_calls is not None
        assert len(normalized.tool_calls) == 1
        assert normalized.tool_calls[0].name == "wire_tool"
        assert normalized.tool_calls[0].id == "call_wire"

    def test_unclosed_truncated_invoke(self, transport):
        raw_xml = """<dots_function_call>
<invoke name="execute_code">
<parameter name="code">
x = 42
print(x)"""

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=None,
                        reasoning_content=raw_xml,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

        normalized = transport.normalize_response(response)

        assert normalized.tool_calls is not None
        assert len(normalized.tool_calls) == 1
        assert normalized.tool_calls[0].name == "execute_code"
        assert "x = 42" in json.loads(normalized.tool_calls[0].arguments)["code"]
