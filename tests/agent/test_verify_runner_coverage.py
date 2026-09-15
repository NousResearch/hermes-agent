"""Test coverage for agent/verify/runner.py — pure helper functions."""
import pytest

from agent.verify.runner import _tail


class TestTail:
    def test_short_text_unchanged(self):
        assert _tail("short", limit=100) == "short"

    def test_long_text_truncated(self):
        text = "x" * 2000
        result = _tail(text, limit=100)
        assert len(result) <= 100
        assert result.endswith("x" * 50)

    def test_empty_text(self):
        assert _tail("", limit=100) == ""

    def test_exact_limit(self):
        text = "a" * 100
        assert _tail(text, limit=100) == text
