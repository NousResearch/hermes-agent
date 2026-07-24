"""Regression tests for _humanize_image_error empty error string guard.

Refs: #62678
"""
from agent.pet.generate.orchestrate import _humanize_image_error


def test_humanize_empty_string():
    """Empty error string must not crash (regression test for #62678)."""
    result = _humanize_image_error("")
    assert result == ""


def test_humanize_whitespace_only():
    """Whitespace-only error string must not crash."""
    result = _humanize_image_error("   ")
    assert result == ""


def test_humanize_normal_error():
    """Normal multi-line error is unchanged."""
    result = _humanize_image_error("something went wrong\n  details here  ")
    assert result == "something went wrong"


def test_humanize_single_line():
    """Single-line error works correctly."""
    result = _humanize_image_error("just a single line error")
    assert result == "just a single line error"


def test_humanize_empty_lines():
    """Error with only newlines / empty lines must not crash."""
    result = _humanize_image_error("\n\n")
    assert result == ""
