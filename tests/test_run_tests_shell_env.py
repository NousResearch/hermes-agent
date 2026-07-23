"""Windows regression tests for the canonical hermetic runner."""

import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows home resolution is the behavior under test")
def test_runner_preserves_windows_home_resolution():
    """Path.home() must resolve when run under scripts/run_tests.sh."""
    actual = Path.home()
    expected = os.environ.get("USERPROFILE")
    assert expected, "canonical runner dropped USERPROFILE"
    assert actual == Path(expected)
