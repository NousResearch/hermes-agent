"""Tests for normalize_model_name dot-to-hyphen conversion (#14528).

The blanket .replace(".", "-") corrupted non-Claude model names on
third-party Anthropic-compatible endpoints.  The fix restricts
conversion to digit-dot-digit version positions only.
"""
import pytest
from agent.anthropic_adapter import normalize_model_name


class TestNormalizeModelNameDots:
    """Dot-to-hyphen conversion must only affect version numbers (#14528)."""

    @pytest.mark.parametrize("input_name,expected", [
        # Claude models: dots between digits should convert
        ("claude-opus-4.6", "claude-opus-4-6"),
        ("claude-sonnet-4.5", "claude-sonnet-4-5"),
        ("claude-3.5-sonnet-20241022", "claude-3-5-sonnet-20241022"),
        ("anthropic/claude-opus-4.6", "claude-opus-4-6"),
        # Non-Claude models: dots NOT between digits must be preserved
        ("llama3.2-vision", "llama3.2-vision"),
        ("qwen3.5-plus", "qwen3.5-plus"),
        ("gpt-4o", "gpt-4o"),
        # No dots at all
        ("claude-3-opus", "claude-3-opus"),
        ("test-model", "test-model"),
    ])
    def test_dot_conversion(self, input_name, expected):
        assert normalize_model_name(input_name) == expected

    def test_preserve_dots_flag(self):
        """preserve_dots=True should skip all conversion."""
        assert normalize_model_name("claude-opus-4.6", preserve_dots=True) == "claude-opus-4.6"

    @pytest.mark.parametrize("input_name", [
        "llama3.2-vision",
        "mixtral-8x7b.v0.1",
        "model.name.with.dots",
    ])
    def test_non_version_dots_preserved(self, input_name):
        """Dots that are NOT between two digits must be preserved (#14528)."""
        result = normalize_model_name(input_name)
        # Count non-digit-adjacent dots — they should all survive
        import re
        non_version_dots_input = len(re.findall(r'(?<!\d)\.|\.(?!\d)', input_name.removeprefix("anthropic/")))
        non_version_dots_output = len(re.findall(r'(?<!\d)\.|\.(?!\d)', result))
        assert non_version_dots_output == non_version_dots_input
