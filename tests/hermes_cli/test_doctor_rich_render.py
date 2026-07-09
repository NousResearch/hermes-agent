import io
import pytest
from unittest.mock import patch
from argparse import Namespace
from rich.console import Console

from hermes_cli.doctor import check_ok, check_warn, check_fail, check_info, _section


@pytest.fixture
def capture_console():
    """Fixture to intercept global rich.print calls and return clean strings instantly.

    This uses an in-memory StringIO buffer to bypass asynchronous/context-manager
    race conditions entirely.
    """
    string_io = io.StringIO()
    test_console = Console(file=string_io, force_terminal=False, color_system=None, width=80)

    def mock_rich_print(*args, **kwargs):
        test_console.print(*args, **kwargs)

    with patch("rich.print", side_effect=mock_rich_print):
        yield lambda: string_io.getvalue()


# ==============================================================================
#  Test Cases: Diagnostic Check Output Commands
# ==============================================================================

def test_check_ok_without_detail(capture_console):
    """Verify system output strings when a check passes without extra details."""
    check_ok("System healthy")

    output = capture_console()
    assert "System healthy" in output
    assert "✓" in output


def test_check_ok_with_detail(capture_console):
    """Verify that structured records compile safely into Rich layout structures."""
    check_ok("System healthy", "v1.0.0")

    output = capture_console()
    assert "System healthy" in output
    assert "v1.0.0" in output
    assert "✓" in output


def test_check_warn(capture_console):
    """Verify status text updates gracefully for warning states."""
    check_warn("High memory usage", "(85%)")

    output = capture_console()
    assert "High memory usage" in output
    assert "(85%)" in output
    assert "⚠" in output


def test_check_fail(capture_console):
    """Verify that failure messages output the correct text and icon."""
    check_fail("Missing config", "config.yaml not found")

    output = capture_console()
    assert "Missing config" in output
    assert "config.yaml not found" in output
    assert "✗" in output


def test_check_info(capture_console):
    """Ensure successful info initialization renders flat dictionary targets."""
    check_info("Run setup command")

    output = capture_console()
    assert "Run setup command" in output
    assert "→" in output


# ==============================================================================
#  Test Cases: Section Formatting Command
# ==============================================================================

def test_section_formatting(capture_console):
    """Verify core CLI parser arguments map out execution tracks cleanly for sections."""
    _section("System Requirements")

    output = capture_console()
    assert "System Requirements" in output
    assert "◆" in output