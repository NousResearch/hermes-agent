"""DeepSeek DSML tool-call fallback in ChatCompletionsTransport.

DeepSeek models sometimes emit tool calls as their own XML markup inside
``message.content`` instead of the OpenAI ``tool_calls`` array.  Without the
fallback the agent prints the markup as assistant prose and halts at the prompt
with no tool executed -- the model looks alive and does nothing.

Two shapes matter and they need opposite handling:

* a COMPLETE block is parsed into real tool calls (``finish_reason`` becomes
  ``tool_calls``), and
* a block the provider cut off mid call is reported as ``length`` so the
  existing truncation recovery re-issues the call, instead of the dead markup
  being surfaced as the turn's final answer.
"""

from types import SimpleNamespace

import pytest

from agent.transports import get_transport
from agent.transports.chat_completions import (
    _has_truncated_dsml,
    _parse_dsml_tool_calls,
)

# The models emit the fullwidth bar (U+FF5C); relays sometimes transliterate to ASCII.
BAR = "\uff5c"


def dsml(body: str) -> str:
    return f"<{BAR}{BAR}DSML{BAR}{BAR} calls>{body}</{BAR}{BAR}DSML{BAR}{BAR} calls>"


def invoke(name: str, params: str = "") -> str:
    return f"<{BAR}{BAR}invoke name=\"{name}\">{params}</{BAR}{BAR}invoke>"


def param(name: str, value: str, string: str | None = "true") -> str:
    attr = f" string=\"{string}\"" if string is not None else ""
    return f"<{BAR}{BAR}parameter name=\"{name}\"{attr}>{value}</{BAR}{BAR}parameter>"


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401

    return get_transport("chat_completions")


def response(content, *, finish_reason="stop", tool_calls=None):
    """Minimal ChatCompletion shape ``normalize_response`` reads."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, refusal=None, reasoning=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=msg, finish_reason=finish_reason)],
        usage=None,
    )


class TestParseComplete:
    def test_single_call(self):
        calls, clean = _parse_dsml_tool_calls(dsml(invoke("read_file", param("path", "orca.yaml"))))
        assert calls == [{"name": "read_file", "arguments": '{"path": "orca.yaml"}'}]
        assert clean == ""

    def test_prose_before_call_is_preserved(self):
        src = "Let me check.\n" + dsml(invoke("terminal", param("command", "ls")))
        calls, clean = _parse_dsml_tool_calls(src)
        assert len(calls) == 1
        assert clean == "Let me check."

    def test_two_calls_in_one_block(self):
        src = dsml(invoke("a", param("x", "1")) + invoke("b", param("y", "2")))
        calls, _ = _parse_dsml_tool_calls(src)
        assert [c["name"] for c in calls] == ["a", "b"]

    def test_non_string_parameter_is_json_decoded(self):
        src = dsml(invoke("c", param("n", "[1, 2]", string="false")))
        calls, _ = _parse_dsml_tool_calls(src)
        assert calls[0]["arguments"] == '{"n": [1, 2]}'

    def test_malformed_non_string_parameter_falls_back_to_text(self):
        src = dsml(invoke("c", param("n", "{not json", string="false")))
        calls, _ = _parse_dsml_tool_calls(src)
        assert calls[0]["arguments"] == '{"n": "{not json"}'

    def test_parameter_without_string_attribute(self):
        src = dsml(invoke("d", param("p", "hello", string=None)))
        calls, _ = _parse_dsml_tool_calls(src)
        assert calls[0]["arguments"] == '{"p": "hello"}'

    def test_ascii_bar_transliteration(self):
        src = '<||DSML|| calls><||invoke name="e"><||parameter name="k" string="true">v</||parameter></||invoke></||DSML|| calls>'
        calls, _ = _parse_dsml_tool_calls(src)
        assert calls == [{"name": "e", "arguments": '{"k": "v"}'}]

    def test_prose_mentioning_dsml_is_not_a_call(self):
        src = "The model emits DSML markup instead of tool_calls, which is the bug."
        calls, clean = _parse_dsml_tool_calls(src)
        assert calls == []
        assert clean == src


class TestTruncationDetection:
    def test_cut_mid_parameter(self):
        src = f"<{BAR}{BAR}DSML{BAR}{BAR} calls>\n<{BAR}{BAR}invoke name=\"terminal\">\n<{BAR}{BAR}parameter name=\"command\" string=\"true\">ls -la"
        assert _has_truncated_dsml(src) is True

    def test_cut_after_invoke_open(self):
        src = f"<{BAR}{BAR}DSML{BAR}{BAR} calls>\n<{BAR}{BAR}invoke name=\"terminal\">"
        assert _has_truncated_dsml(src) is True

    def test_complete_block_is_not_truncated(self):
        assert _has_truncated_dsml(dsml(invoke("read_file", param("path", "a")))) is False

    def test_complete_block_followed_by_a_cut_one(self):
        src = dsml(invoke("a", param("x", "1"))) + f"<{BAR}{BAR}DSML{BAR}{BAR} calls><{BAR}{BAR}invoke name=\"b\">"
        assert _has_truncated_dsml(src) is True

    def test_prose_mentioning_dsml_is_not_truncated(self):
        assert _has_truncated_dsml("DeepSeek emits DSML rather than tool_calls.") is False

    def test_plain_text_is_not_truncated(self):
        assert _has_truncated_dsml("ordinary answer") is False


class TestNormalizeResponse:
    def test_complete_block_becomes_tool_calls(self, transport):
        out = transport.normalize_response(response(dsml(invoke("read_file", param("path", "orca.yaml")))))
        assert out.finish_reason == "tool_calls"
        assert [tc.name for tc in out.tool_calls] == ["read_file"]
        assert out.tool_calls[0].arguments == '{"path": "orca.yaml"}'
        assert out.content is None

    def test_call_ids_are_unique(self, transport):
        src = dsml(invoke("a", param("x", "1")) + invoke("b", param("y", "2")))
        out = transport.normalize_response(response(src))
        assert len({tc.id for tc in out.tool_calls}) == 2

    def test_truncated_block_reports_length(self, transport):
        src = f"<{BAR}{BAR}DSML{BAR}{BAR} calls>\n<{BAR}{BAR}invoke name=\"terminal\">\n<{BAR}{BAR}parameter name=\"command\" string=\"true\">ls"
        out = transport.normalize_response(response(src))
        assert out.finish_reason == "length"
        assert not out.tool_calls

    def test_native_tool_calls_win_over_the_fallback(self, transport):
        native = [SimpleNamespace(
            id="call_native", type="function",
            function=SimpleNamespace(name="read_file", arguments="{}"),
        )]
        out = transport.normalize_response(
            response(dsml(invoke("other", param("p", "v"))), finish_reason="tool_calls", tool_calls=native))
        assert [tc.name for tc in out.tool_calls] == ["read_file"]

    def test_ordinary_content_is_untouched(self, transport):
        out = transport.normalize_response(response("Just an answer."))
        assert out.finish_reason == "stop"
        assert out.content == "Just an answer."
        assert not out.tool_calls

    def test_non_string_content_is_ignored(self, transport):
        out = transport.normalize_response(response(None))
        assert out.finish_reason == "stop"
        assert not out.tool_calls
