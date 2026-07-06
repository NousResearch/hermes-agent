"""Tests for the Telegram output-sanitization guard (#59291).

When a local model (Ollama, vLLM — e.g. gpt-oss:120b) emits a
tool-call-shaped JSON envelope directly in plain assistant ``content``
instead of routing it through ``tool_calls``, the literal JSON would
otherwise reach end users verbatim.  The fix is a deterministic
structural guard in :mod:`agent.message_sanitization` that:

  * Strips tool-call-shaped envelopes from plain assistant text.
  * Preserves any JSON the user *intentionally* placed in fenced or
    inline code regions.
  * Logs each suppression at DEBUG level.

These tests cover the four required cases for the Telegram adapter
pipeline:

  1. Plain text -> forwarded as-is.
  2. Tool-call-shaped JSON envelope -> stripped (default; replaced
     with a friendly placeholder when one is supplied).
  3. JSON inside a fenced code block -> preserved (user-intended).
  4. Empty / None input -> handled gracefully (no exception).
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.message_sanitization import (
    _is_tool_call_envelope,
    _protect_code_regions,
    _restore_code_regions,
    _strip_tool_call_envelope,
)


# ---------------------------------------------------------------------------
# Mock the telegram package so the adapter can be imported.
# ---------------------------------------------------------------------------

def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    return TelegramAdapter(config)


# =========================================================================
# 1) Plain text -> forwarded as-is
# =========================================================================


class TestPlainTextForwarded:
    """Plain assistant content must NOT be mutated by the sanitizer."""

    def test_plain_prose_passes_through_unchanged(self):
        text = "Hello, world! Here is a normal sentence."
        assert _strip_tool_call_envelope(text) == text

    def test_multiline_prose_passes_through_unchanged(self):
        text = "Line one.\nLine two.\nLine three with **bold** and `code`."
        assert _strip_tool_call_envelope(text) == text

    def test_empty_string_returned_unchanged(self):
        assert _strip_tool_call_envelope("") == ""

    def test_none_returned_as_empty_string(self):
        # ``None`` is coerced to ``""`` so callers can pass
        # ``response.choices[0].message.content`` directly without
        # isinstance checks.
        assert _strip_tool_call_envelope(None) == ""

    def test_non_string_input_does_not_raise(self):
        # Non-string inputs are returned unchanged rather than crashing
        # — callers should always pass strings, but a stray ``int`` or
        # ``list`` must not blow up the conversation loop.
        assert _strip_tool_call_envelope(0) == 0  # type: ignore[arg-type]
        assert _strip_tool_call_envelope([]) == []  # type: ignore[arg-type]
        # The string-only fast-path still applies for str inputs.
        assert _strip_tool_call_envelope("") == ""


# =========================================================================
# 2) Tool-call-shaped JSON envelope -> stripped / replaced
# =========================================================================


class TestEnvelopeSuppression:
    """Tool-call-shaped JSON must be removed from the visible output."""

    def test_pseudo_style_action_envelope_stripped(self):
        # The exact shape from the bug report — invented ``action`` key.
        raw = '{"action":"clarify","question":"Which color?","choices":["red","blue"]}'
        result = _strip_tool_call_envelope(raw)
        assert result == ""
        assert "action" not in result
        assert "clarify" not in result

    def test_openai_style_name_arguments_stripped(self):
        raw = '{"name":"clarify","arguments":{"question":"q","choices":["a","b"]}}'
        result = _strip_tool_call_envelope(raw)
        assert result == ""

    def test_function_wrapper_stripped(self):
        raw = '{"function":{"name":"foo","arguments":"{}"}}'
        result = _strip_tool_call_envelope(raw)
        assert result == ""

    def test_envelope_embedded_in_prose_is_stripped(self):
        text = (
            "Sure, let me ask you:\n"
            '{"action":"clarify","question":"Which color?","choices":["red","blue"]}\n'
            "Hope that helps!"
        )
        result = _strip_tool_call_envelope(text)
        assert "action" not in result
        assert "clarify" not in result
        assert "Which color?" not in result
        assert "Sure, let me ask you:" in result
        assert "Hope that helps!" in result

    def test_multiple_envelopes_all_stripped(self):
        text = (
            '{"action":"clarify","question":"q1"}\n'
            "Some prose.\n"
            '{"name":"foo","arguments":{"x":1}}\n'
        )
        result = _strip_tool_call_envelope(text)
        assert "action" not in result
        assert "clarify" not in result
        assert '"name"' not in result
        assert "Some prose." in result

    def test_replacement_parameter_is_used(self):
        raw = '{"action":"clarify","question":"q"}'
        result = _strip_tool_call_envelope(raw, replacement="(handled)")
        assert result == "(handled)"

    def test_envelope_strip_logs_at_debug_level(self, caplog):
        raw = '{"action":"clarify","question":"q","choices":["a"]}'
        with caplog.at_level(logging.DEBUG, logger="agent.message_sanitization"):
            _strip_tool_call_envelope(raw)
        assert any(
            "tool-call-shaped JSON envelope" in rec.message
            for rec in caplog.records
        ), f"expected DEBUG suppression log; got: {[r.message for r in caplog.records]}"

    def test_no_log_when_nothing_stripped(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="agent.message_sanitization"):
            _strip_tool_call_envelope("Plain text, no envelopes here.")
        assert not any(
            "tool-call-shaped JSON envelope" in rec.message
            for rec in caplog.records
        )

    def test_non_envelope_json_object_left_alone(self):
        # A plain JSON object without any tool-call-shaped keys should
        # not be stripped — it may be a deliberate user message
        # (e.g. ``{"hello": "world"}``).
        text = 'Here is data: {"hello": "world", "n": 42}'
        result = _strip_tool_call_envelope(text)
        assert result == text


# =========================================================================
# 3) JSON inside code blocks -> preserved (user-intended)
# =========================================================================


class TestCodeBlockPreservation:
    """User-intended JSON samples inside code regions must survive."""

    def test_json_in_fenced_code_block_preserved(self):
        text = (
            "Here is an example:\n"
            "```json\n"
            '{"name":"foo","arguments":{"q":"x"}}\n'
            "```\n"
        )
        result = _strip_tool_call_envelope(text)
        assert result == text
        assert '{"name":"foo","arguments":{"q":"x"}}' in result

    def test_json_in_inline_code_preserved(self):
        text = 'Use the `{"action":"x","y":1}` pattern for that.'
        result = _strip_tool_call_envelope(text)
        assert result == text

    def test_plain_envelope_stripped_but_code_block_preserved(self):
        text = (
            "Intro text.\n"
            '{"action":"clarify","question":"q"}\n'
            "Then a code block:\n"
            "```json\n"
            '{"action":"also_clarify","question":"q2"}\n'
            "```\n"
            "End text."
        )
        result = _strip_tool_call_envelope(text)
        # Outside-code envelope gone.
        assert '{"action":"clarify","question":"q"}' not in result
        # Inside-code envelope preserved.
        assert '{"action":"also_clarify","question":"q2"}' in result
        assert "End text." in result
        assert "Then a code block:" in result

    def test_multiple_fenced_blocks_all_preserved(self):
        text = (
            "```json\n"
            '{"name":"a","arguments":{}}\n'
            "```\n"
            "between\n"
            "```\n"
            '{"function":{"name":"b","arguments":"{}"}}\n'
            "```\n"
        )
        result = _strip_tool_call_envelope(text)
        assert result == text


# =========================================================================
# 4) Edge cases: empty / None / non-dict JSON
# =========================================================================


class TestEmptyAndEdgeCases:
    """Empty / None / non-object JSON must not crash or misbehave."""

    def test_whitespace_only_returns_unchanged(self):
        # Whitespace-only input passes through — the function never
        # mutates content that has no JSON object to suppress.
        text = "   \n\t  "
        assert _strip_tool_call_envelope(text) == text

    def test_unbalanced_braces_do_not_crash(self):
        text = 'Some text with a stray { but no closing brace'
        assert _strip_tool_call_envelope(text) == text

    def test_json_array_not_treated_as_envelope(self):
        # Arrays are tool-call-shaped only when wrapped in a dict; a
        # bare array must not trigger the suppressor.
        text = 'Here is a list: ["a", "b", {"name": "x", "arguments": {}}]'
        result = _strip_tool_call_envelope(text)
        # The dict *inside* the array will be stripped (it's still a
        # tool-call envelope when extracted as a balanced object).
        # The bare list shape itself isn't mistaken for an envelope.
        assert "[\"a\"" in result or "['a'" in result or result.startswith("Here is a list")

    def test_partial_json_object_is_not_misidentified(self):
        # Truncated JSON that doesn't parse should pass through.
        text = 'Incomplete: {"name":"foo","argum'
        assert _strip_tool_call_envelope(text) == text


# =========================================================================
# Telegram adapter integration
# =========================================================================


class TestTelegramAdapterFormatMessage:
    """The Telegram adapter's ``format_message`` must apply the guard."""

    def test_adapter_strips_envelope(self, adapter):
        raw = '{"action":"clarify","question":"Which color?","choices":["red","blue"]}'
        result = adapter.format_message(raw)
        # Default replacement is empty string — output is blank.
        assert result == ""

    def test_adapter_preserves_plain_prose(self, adapter):
        # Avoid MarkdownV2 special characters in the test string — the
        # adapter is expected to escape ``!``, ``.``, etc. for
        # MarkdownV2.  We just want to confirm the sanitizer itself
        # does not mutate the underlying content.
        text = "Hello world"
        result = adapter.format_message(text)
        # Strip the MarkdownV2 escapes that ``format_message`` adds —
        # ``!``, ``.``, ``(``, etc. — by replacing ``\!`` / ``\.`` etc.
        # back to their unescaped form.  Easier: just check the
        # substantive characters survive unchanged.
        assert "Hello world" in result

    def test_adapter_preserves_json_in_code_block(self, adapter):
        text = (
            "Example:\n"
            "```json\n"
            '{"name":"foo","arguments":{"q":"x"}}\n'
            "```\n"
        )
        result = adapter.format_message(text)
        # The fenced code block survives intact (with MarkdownV2
        # escaping applied inside the body).
        assert "```" in result
        assert "name" in result
        assert "foo" in result


# =========================================================================
# Internal helpers (defensive coverage)
# =========================================================================


class TestEnvelopeDetection:
    """Direct unit coverage of :func:`_is_tool_call_envelope`."""

    def test_empty_dict_is_not_envelope(self):
        assert _is_tool_call_envelope({}) is False

    def test_non_dict_is_not_envelope(self):
        assert _is_tool_call_envelope("string") is False  # type: ignore[arg-type]
        assert _is_tool_call_envelope(None) is False  # type: ignore[arg-type]
        assert _is_tool_call_envelope([1, 2]) is False  # type: ignore[arg-type]

    def test_openai_style_name_with_arguments(self):
        assert _is_tool_call_envelope({"name": "foo", "arguments": {}}) is True

    def test_openai_style_name_with_args(self):
        assert _is_tool_call_envelope({"name": "foo", "args": {}}) is True

    def test_function_wrapper(self):
        assert _is_tool_call_envelope({"function": {"name": "foo"}}) is True

    def test_invented_action_key(self):
        assert _is_tool_call_envelope({"action": "clarify", "question": "q"}) is True

    def test_tool_key(self):
        assert _is_tool_call_envelope({"tool": "foo"}) is True

    def test_unrelated_dict_is_not_envelope(self):
        assert _is_tool_call_envelope({"hello": "world", "n": 42}) is False


class TestProtectRestoreCodeRegions:
    """Round-trip the protect / restore helpers."""

    def test_fenced_code_is_stashed_and_restored(self):
        text = 'pre\n```json\n{"a": 1}\n```\npost'
        protected, phs = _protect_code_regions(text)
        assert "```" not in protected
        assert '{"a": 1}' not in protected
        assert phs, "expected at least one placeholder"
        restored = _restore_code_regions(protected, phs)
        assert restored == text

    def test_inline_code_is_stashed_and_restored(self):
        text = "Use `foo` here."
        protected, phs = _protect_code_regions(text)
        assert "foo" not in protected or "`foo`" not in protected
        restored = _restore_code_regions(protected, phs)
        assert restored == text

    def test_restore_with_empty_placeholders_is_noop(self):
        assert _restore_code_regions("hello", {}) == "hello"