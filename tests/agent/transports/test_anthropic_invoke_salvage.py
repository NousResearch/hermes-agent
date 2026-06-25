"""Tests for AnthropicTransport bare-invoke tool-call salvage.

Regression coverage for the opus-4-8 ``antml:`` namespace-drop bug:
Anthropic OAuth promotes tool calls to structured ``tool_use`` blocks ONLY
when the model emits the namespaced invoke markup. Some Claude builds
intermittently drop the ``antml:`` prefix and emit a bare invoke block,
which the API returns as a plain ``text`` block with
``stop_reason="end_turn"`` and NO ``tool_use``. The agent loop then treats
the turn as a final answer and halts WITHOUT running the tool (reproduced
as round 2-A in the model-routing test matrix).

``normalize_response`` now salvages a complete invoke block from assistant
text, re-promotes it to a real ``ToolCall``, and re-maps the finish reason
to ``tool_calls`` so the loop keeps executing.

The markup tokens are assembled from ``_TOK`` fragments rather than written
as literals, so the test source itself never carries raw invoke markup that
a tool-call parser could accidentally grab.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.transports import get_transport
from agent.transports.types import NormalizedResponse
from agent.transports.anthropic import (
    salvage_tool_calls_from_text,
    _coerce_salvaged_value,
)


# ── Markup token builders (avoid literal invoke markup in source) ────────
_LT, _GT, _SL, _NS = "<", ">", "/", "antml:"


def _inv(name, body, ns=False):
    pre = _NS if ns else ""
    return f"{_LT}{pre}invoke name=\"{name}\"{_GT}{body}{_LT}{_SL}{pre}invoke{_GT}"


def _param(name, val, ns=False):
    pre = _NS if ns else ""
    return f"{_LT}{pre}parameter name=\"{name}\"{_GT}{val}{_LT}{_SL}{pre}parameter{_GT}"


# ── Pure-function tests: salvage_tool_calls_from_text ────────────────────

class TestSalvageFromText:

    def test_bare_invoke_single_param(self):
        text = "Reading the file now.\n" + _inv(
            "read_file", _param("path", "/etc/os-release")
        )
        calls, cleaned = salvage_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"] == {"path": "/etc/os-release"}
        assert "invoke" not in cleaned
        assert "Reading the file now." in cleaned

    def test_namespaced_invoke_also_salvaged(self):
        text = _inv("terminal", _param("command", "ls -la", ns=True), ns=True)
        calls, cleaned = salvage_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "terminal"
        assert calls[0]["arguments"] == {"command": "ls -la"}
        assert cleaned == ""

    def test_multiple_params_with_type_coercion(self):
        body = (
            _param("path", "/home/x/config.yaml")
            + _param("offset", "61")
            + _param("limit", "90")
        )
        text = _inv("read_file", body)
        calls, _ = salvage_tool_calls_from_text(text)
        assert calls[0]["arguments"] == {
            "path": "/home/x/config.yaml",
            "offset": 61,
            "limit": 90,
        }
        # offset/limit must be real ints, mirroring a structured tool_use.
        assert isinstance(calls[0]["arguments"]["offset"], int)

    def test_multiple_invoke_blocks(self):
        text = (
            _inv("read_file", _param("path", "/a"))
            + "\nand then\n"
            + _inv("terminal", _param("command", "echo hi"))
        )
        calls, cleaned = salvage_tool_calls_from_text(text)
        assert [c["name"] for c in calls] == ["read_file", "terminal"]
        assert "and then" in cleaned

    def test_no_markup_returns_text_unchanged(self):
        text = "Just a normal answer, no tools here."
        calls, cleaned = salvage_tool_calls_from_text(text)
        assert calls == []
        assert cleaned == text

    def test_truncated_block_not_salvaged(self):
        # Open tag with no closing </invoke> — must NOT execute a partial call.
        text = "Working: " + _LT + "invoke name=\"terminal\"" + _GT + _param(
            "command", "rm -rf /"
        )
        calls, cleaned = salvage_tool_calls_from_text(text)
        assert calls == []
        assert cleaned == text

    def test_empty_param_value(self):
        text = _inv("todo", _param("status", ""))
        calls, _ = salvage_tool_calls_from_text(text)
        assert calls[0]["arguments"] == {"status": ""}

    def test_invoke_with_no_params(self):
        text = _inv("session_search", "")
        calls, _ = salvage_tool_calls_from_text(text)
        assert calls == [{"name": "session_search", "arguments": {}}]


# ── _coerce_salvaged_value ───────────────────────────────────────────────

class TestCoerceSalvagedValue:

    def test_int(self):
        assert _coerce_salvaged_value("61") == 61

    def test_bool(self):
        assert _coerce_salvaged_value("true") is True

    def test_json_object(self):
        assert _coerce_salvaged_value('{"a": 1}') == {"a": 1}

    def test_plain_path_string(self):
        assert _coerce_salvaged_value("/etc/passwd") == "/etc/passwd"

    def test_shell_command_string(self):
        assert _coerce_salvaged_value("ls -la /tmp") == "ls -la /tmp"

    def test_whitespace_trimmed(self):
        assert _coerce_salvaged_value("  hello  ") == "hello"

    def test_empty(self):
        assert _coerce_salvaged_value("   ") == ""


# ── End-to-end: normalize_response promotes salvaged calls ───────────────

class TestNormalizeResponseSalvage:

    @pytest.fixture
    def transport(self):
        import agent.transports.anthropic  # noqa: F401
        return get_transport("anthropic_messages")

    def test_bare_invoke_in_text_promoted_to_tool_call(self, transport):
        """The core 2-A repro: end_turn + text-only invoke -> real tool call."""
        leaked = "Let me read it.\n" + _inv(
            "read_file", _param("path", "/etc/os-release")
        )
        r = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=leaked)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            model="claude-opus-4-8",
        )
        nr = transport.normalize_response(r)
        assert nr.tool_calls is not None
        assert len(nr.tool_calls) == 1
        tc = nr.tool_calls[0]
        assert tc.name == "read_file"
        assert json.loads(tc.arguments) == {"path": "/etc/os-release"}
        # Finish reason re-mapped so the agent loop keeps going.
        assert nr.finish_reason == "tool_calls"
        # Raw markup scrubbed from user-visible content.
        assert nr.content is None or "invoke" not in nr.content
        assert nr.content is None or "Let me read it." in nr.content

    def test_structured_tool_use_unaffected(self, transport):
        """Healthy path: a real tool_use block is untouched by salvage."""
        r = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use", id="toolu_1", name="terminal",
                    input={"command": "ls"},
                ),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            model="claude-opus-4-8",
        )
        nr = transport.normalize_response(r)
        assert len(nr.tool_calls) == 1
        assert nr.tool_calls[0].id == "toolu_1"
        assert nr.finish_reason == "tool_calls"

    def test_plain_text_answer_unaffected(self, transport):
        """A genuine final text answer must NOT be mangled."""
        r = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="The answer is 42.")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            model="claude-opus-4-8",
        )
        nr = transport.normalize_response(r)
        assert nr.tool_calls is None or nr.tool_calls == []
        assert nr.content == "The answer is 42."
        assert nr.finish_reason == "stop"

    def test_salvage_skipped_when_structured_calls_present(self, transport):
        """If the API already gave a tool_use, leftover text markup is left
        as-is (we only salvage when there are zero structured calls)."""
        leaked = "extra " + _inv("read_file", _param("path", "/x"))
        r = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text=leaked),
                SimpleNamespace(
                    type="tool_use", id="toolu_9", name="terminal",
                    input={"command": "pwd"},
                ),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=5, output_tokens=5),
            model="claude-opus-4-8",
        )
        nr = transport.normalize_response(r)
        # Only the structured call survives; no salvage double-count.
        assert len(nr.tool_calls) == 1
        assert nr.tool_calls[0].name == "terminal"
