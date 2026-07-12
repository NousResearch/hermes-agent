"""Tests for `_sanitize_tool_error` in model_tools.

Ported from ironclaw#1639 — defense-in-depth on tool exception strings before
they enter the model's `tool` message content. Note that `json.dumps()` in
`handle_function_call` already handles quote/backslash escaping at the wire
layer; this helper exists to strip structural framing tokens the model
itself might react to (XML role tags, CDATA, markdown code fences) and to
cap pathological lengths.
"""
from __future__ import annotations

from model_tools import _sanitize_tool_error, _TOOL_ERROR_MAX_LEN


class TestRoleTagStripping:
    def test_strips_tool_call_tags(self):
        out = _sanitize_tool_error("bad <tool_call>injected</tool_call> happened")
        assert "<tool_call>" not in out
        assert "</tool_call>" not in out
        assert "bad injected happened" in out

    def test_strips_function_call_tags(self):
        out = _sanitize_tool_error("<function_call>x</function_call>")
        assert "<function_call>" not in out
        assert "</function_call>" not in out

    def test_strips_role_tags(self):
        # Each of these should be stripped
        for tag in ("system", "assistant", "user", "result", "response", "output", "input"):
            raw = f"prefix <{tag}>hi</{tag}> suffix"
            out = _sanitize_tool_error(raw)
            assert f"<{tag}>" not in out, f"failed to strip <{tag}>"
            assert f"</{tag}>" not in out, f"failed to strip </{tag}>"

    def test_role_tag_strip_is_case_insensitive(self):
        out = _sanitize_tool_error("<TOOL_CALL>x</Tool_Call>")
        assert "<" not in out.replace("[TOOL_ERROR]", "")  # only the prefix bracket survives

    def test_unrelated_xml_kept(self):
        # We intentionally only strip the role-like tag whitelist, not all XML
        out = _sanitize_tool_error("Error parsing <ParseError>line 5</ParseError>")
        assert "<ParseError>" in out


class TestCDATAStripping:
    def test_strips_cdata(self):
        out = _sanitize_tool_error("error: <![CDATA[malicious]]> here")
        assert "<![CDATA[" not in out
        assert "]]>" not in out

    def test_strips_multiline_cdata(self):
        out = _sanitize_tool_error("a\n<![CDATA[line1\nline2]]>\nb")
        assert "CDATA" not in out
        assert "a" in out and "b" in out


class TestCodeFenceStripping:
    def test_strips_leading_fence_with_lang(self):
        out = _sanitize_tool_error("```json\n{\"x\": 1}")
        assert not out.replace("[TOOL_ERROR] ", "").startswith("```")

    def test_strips_trailing_fence(self):
        out = _sanitize_tool_error("payload\n```")
        assert not out.rstrip().endswith("```")

    def test_strips_bare_fence(self):
        out = _sanitize_tool_error("```\nstuff")
        assert "```" not in out.split("\n")[0]


class TestTruncation:
    def test_caps_long_input(self):
        long = "A" * (_TOOL_ERROR_MAX_LEN * 2)
        out = _sanitize_tool_error(long)
        # Total length is prefix + truncated body
        body = out[len("[TOOL_ERROR] "):]
        assert len(body) == _TOOL_ERROR_MAX_LEN
        assert body.endswith("...")

    def test_does_not_truncate_short_input(self):
        msg = "short error"
        out = _sanitize_tool_error(msg)
        assert "..." not in out
        assert msg in out


class TestEnvelope:
    def test_wraps_with_prefix(self):
        out = _sanitize_tool_error("oh no")
        assert out.startswith("[TOOL_ERROR] ")

    def test_empty_input(self):
        out = _sanitize_tool_error("")
        assert out == "[TOOL_ERROR] "

    def test_preserves_normal_error_text(self):
        msg = "Error executing read_file: FileNotFoundError: /tmp/missing"
        out = _sanitize_tool_error(msg)
        assert msg in out


class TestHandleFunctionCallIntegration:
    """Verify handle_function_call routes exception-path errors through the sanitizer.

    Note: the "Unknown tool: ..." early-return in tools/registry.py is a
    *different* code path from `except Exception` in handle_function_call —
    that one returns directly without sanitization (and there's nothing to
    sanitize in a hardcoded format string anyway). This test exercises the
    real exception path by passing args that make a known tool raise.
    """

    def test_exception_path_error_is_sanitized(self):
        import json
        from model_tools import handle_function_call
        from tools.registry import registry as _registry

        # Force a known tool to raise with a payload containing role tags.
        def boom(_args, **_kwargs):
            raise RuntimeError("<tool_call>injected</tool_call> boom")

        all_tools = _registry.get_all_tool_names()
        assert all_tools, "no tools registered — test environment broken"
        target = all_tools[0]
        original = _registry._tools[target].handler
        _registry._tools[target].handler = boom
        try:
            result_str = handle_function_call(target, {})
        finally:
            _registry._tools[target].handler = original

        payload = json.loads(result_str)
        assert "error" in payload, payload
        assert payload["error"].startswith("[TOOL_ERROR] "), payload["error"]
        # Role-tag stripping carried through
        assert "<tool_call>" not in payload["error"]
        assert "</tool_call>" not in payload["error"]
        assert "boom" in payload["error"]
