"""Regression tests for pet generation helpers."""

from agent.pet.generate.orchestrate import _humanize_image_error


def test_humanize_empty_error_string():
    """Verify that _humanize_image_error handles empty strings without
    raising IndexError."""
    result = _humanize_image_error("")
    assert result == ""


def test_humanize_error_with_only_whitespace():
    """Verify that whitespace-only errors return empty after stripping."""
    result = _humanize_image_error("   \n  \n   ")
    assert result == ""


def test_humanize_error_single_line():
    """Verify that a single line error returns trimmed to 200 chars."""
    error = "Connection refused"
    result = _humanize_image_error(error)
    assert result == "Connection refused"


def test_humanize_error_multiline_trims_to_first_line():
    """Verify that multiline errors return only the first line, trimmed."""
    error = "Error: something broke\n  at module.py:42\n  in function()"
    result = _humanize_image_error(error)
    assert result == "Error: something broke"


def test_humanize_rate_limit_errors():
    """Verify that rate limit errors get a friendly message."""
    error = "ProviderError: 429 Too Many Requests - you hit the rate limit"
    result = _humanize_image_error(error)
    assert "rate-limiting" in result


def test_humanize_error_truncated_at_200():
    """Verify that very long first lines are truncated to 200 chars."""
    error = "A" * 300
    result = _humanize_image_error(error)
    assert len(result) == 200
    assert result == "A" * 200
