import io
import pytest
from unittest.mock import patch
from rich.console import Console
import re

from hermes_cli.gui_uninstall import log_info, log_success, log_warn

def strip_rich_formatting(text: str) -> str:
    """Removes ANSI escape sequences and non-breaking spaces from Rich output."""
    ansi_escape = re.compile(r'\x1b\[([0-9]+)(;[0-9]+)*m')
    cleaned = ansi_escape.sub('', text)
    return cleaned.replace('\xa0', ' ')

@pytest.fixture
def capture_console():
    """Fixture to intercept global rich.print calls and return strings with ANSI rendering.

    Forces terminal color capabilities and overrides internal color configuration flags.
    """
    string_io = io.StringIO()
    test_console = Console(file=string_io, force_terminal=True, color_system="truecolor", width=80)

    def mock_rich_print(*args, **kwargs):
        test_console.print(*args, **kwargs)

    with patch("rich.print", side_effect=mock_rich_print), \
            patch("hermes_cli.colors.should_use_color", return_value=True):
        yield lambda: string_io.getvalue()

def test_info_rendering(capture_console):
    """Test that the info log function returns the correct rich Text object
        with the proper symbols and color attributes applied.
    """
    with patch("hermes_cli.colors.should_use_color", return_value=True):
        log_info("info")
        info_output = capture_console()
        assert "→ info" in strip_rich_formatting(info_output), "Expected arrow, space, and log message"
        assert "\x1b[" in info_output, "Expected ANSI styling sequences for Cyan color"

def test_success_rendering(capture_console):
    """Test that the success log function returns the correct rich Text object
        with the proper symbols and color attributes applied.
    """
    with patch("hermes_cli.colors.should_use_color", return_value=True):
        log_success("success")
        success_output = capture_console()
        assert "✓ success" in strip_rich_formatting(success_output), "Expected check mark, space, and log message"
        assert "\x1b[" in success_output, "Expected ANSI styling sequences for Green color"

def test_warning_rendering(capture_console):
    """Test that the warn log function returns the correct rich Text object
        with the proper symbols and color attributes applied.
    """
    with patch("hermes_cli.colors.should_use_color", return_value=True):
        log_warn("warning")
        warning_output = capture_console()
        assert "⚠ warning" in strip_rich_formatting(warning_output), "Expected warning, space, and log message"
        assert "\x1b[" in warning_output, "Expected ANSI styling sequences for Yellow color"