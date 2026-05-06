"""Tests for agent/codeact_dispatcher.py."""

import json
import pytest
from unittest.mock import MagicMock

from agent.codeact_dispatcher import extract_code, dispatch, CodeActParseError


# ---------------------------------------------------------------------------
# extract_code — envelope mode
# ---------------------------------------------------------------------------

class TestExtractCodeEnvelope:
    def test_valid_envelope(self):
        response = json.dumps({"thoughts": "I'll search.", "code": "result = web_search(query='hi')"})
        thoughts, code = extract_code(response, envelope_mode=True)
        assert thoughts == "I'll search."
        assert "web_search" in code

    def test_envelope_with_leading_text(self):
        """Model sometimes prepends prose before the JSON."""
        response = 'Sure, here is the code:\n{"thoughts": "ok", "code": "x = 1"}'
        thoughts, code = extract_code(response, envelope_mode=True)
        assert code == "x = 1"

    def test_envelope_missing_code_key_falls_back(self):
        """Malformed envelope without 'code' key falls back to fence parse."""
        code_block = "```python\ny = 2\n```"
        response = json.dumps({"thoughts": "..."}) + "\n" + code_block
        _, code = extract_code(response, envelope_mode=True)
        assert code == "y = 2"

    def test_empty_code_field_falls_back(self):
        response = json.dumps({"thoughts": "ok", "code": ""}) + "\n```python\nz = 3\n```"
        _, code = extract_code(response, envelope_mode=True)
        assert code == "z = 3"

    def test_multiline_code_in_envelope(self):
        code = "a = 1\nb = 2\nprint(a + b)"
        response = json.dumps({"thoughts": "multi", "code": code})
        _, extracted = extract_code(response, envelope_mode=True)
        assert extracted == code

    def test_nested_braces_in_code(self):
        """Code with dict literals must not confuse the JSON parser."""
        code = "d = {'key': 'value', 'nested': {'a': 1}}"
        response = json.dumps({"thoughts": "dict test", "code": code})
        _, extracted = extract_code(response, envelope_mode=True)
        assert extracted == code


# ---------------------------------------------------------------------------
# extract_code — fence fallback
# ---------------------------------------------------------------------------

class TestExtractCodeFence:
    def test_python_fence(self):
        response = "Here you go:\n```python\nprint('hello')\n```"
        _, code = extract_code(response, envelope_mode=False)
        assert code == "print('hello')"

    def test_plain_fence(self):
        response = "```\nx = 42\n```"
        _, code = extract_code(response, envelope_mode=False)
        assert code == "x = 42"

    def test_fence_multiline(self):
        response = "```python\na = 1\nb = 2\nc = a + b\n```"
        _, code = extract_code(response, envelope_mode=False)
        assert "c = a + b" in code

    def test_bare_python_import(self):
        response = "import os\nprint(os.getcwd())"
        _, code = extract_code(response, envelope_mode=False)
        assert "import os" in code

    def test_bare_python_def(self):
        response = "def foo():\n    return 42"
        _, code = extract_code(response, envelope_mode=False)
        assert "def foo" in code

    def test_bare_python_print(self):
        response = "print('hello world')"
        _, code = extract_code(response, envelope_mode=False)
        assert code == "print('hello world')"

    def test_bare_python_pass(self):
        response = "pass"
        _, code = extract_code(response, envelope_mode=False)
        assert code == "pass"

    def test_bare_python_assignment(self):
        response = "answer = 42"
        _, code = extract_code(response, envelope_mode=False)
        assert code == "answer = 42"

    def test_raises_when_no_code_found(self):
        response = "This is just a sentence with no code whatsoever."
        with pytest.raises(CodeActParseError):
            extract_code(response, envelope_mode=False)

    def test_raises_on_empty_response(self):
        with pytest.raises(CodeActParseError):
            extract_code("", envelope_mode=False)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def _make_kernel(self, return_value="output"):
        kernel = MagicMock()
        kernel.execute.return_value = return_value
        return kernel

    def test_dispatch_envelope(self):
        kernel = self._make_kernel("42")
        response = json.dumps({"thoughts": "compute", "code": "print(6 * 7)"})
        result = dispatch(response, kernel, envelope_mode=True)
        assert result == "42"
        kernel.execute.assert_called_once_with("print(6 * 7)")

    def test_dispatch_fence_fallback(self):
        kernel = self._make_kernel("done")
        response = "```python\nx = 1\n```"
        result = dispatch(response, kernel, envelope_mode=True)
        assert result == "done"
        kernel.execute.assert_called_once()

    def test_dispatch_parse_error_returns_message(self):
        """On parse failure, dispatch returns an error string (does not raise)."""
        kernel = self._make_kernel()
        result = dispatch("just plain text no code", kernel, envelope_mode=True)
        assert "CodeAct parse error" in result or "parse error" in result.lower()
        kernel.execute.assert_not_called()

    def test_dispatch_thoughts_not_executed(self):
        """Only the 'code' field should be sent to kernel.execute."""
        kernel = self._make_kernel("ok")
        thoughts = "I need to search for something important."
        code = "result = web_search(query='test')"
        response = json.dumps({"thoughts": thoughts, "code": code})
        dispatch(response, kernel, envelope_mode=True)
        kernel.execute.assert_called_once_with(code)
