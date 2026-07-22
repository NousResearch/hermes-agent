import io
import re
import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from hermes_cli.cli_output import (
    print_info,
    print_success,
    print_warning,
    print_error,
    print_header,
    prompt,
    prompt_yes_no
)

def strip_rich_formatting(text: str) -> str:
    """Removes ANSI escape sequences and non-breaking spaces from Rich output."""
    ansi_escape = re.compile(r'\x1b\[([0-9]+)(;[0-9]+)*m')
    cleaned = ansi_escape.sub('', text)
    return cleaned.replace('\xa0', ' ')


@pytest.fixture
def capture_console():
    """
        Fixture to intercept global rich.print calls in the uninstall module
        and return strings with ANSI rendering applied.
    """
    string_io = io.StringIO()
    test_console = Console(file=string_io, force_terminal=True, color_system="truecolor", width=100)

    def mock_rich_print(*args, **kwargs):
        test_console.print(*args, **kwargs)

    with patch("hermes_cli.uninstall.rich.print", side_effect=mock_rich_print), \
            patch("hermes_cli.colors.should_use_color", return_value=True):
        yield lambda: string_io.getvalue()

def test_info_rendering(capture_console):
    """Test that the info helper applies the correct symbols and ANSI sequences."""
    print_info("info")
    output = capture_console()
    assert "  info" in strip_rich_formatting(output)
    assert "\x1b[" in output

def test_success_rendering(capture_console):
    """Test that the success helper applies the correct symbols and ANSI sequences."""
    print_success("success")
    output = capture_console()
    assert "✓ success" in strip_rich_formatting(output)
    assert "\x1b[" in output

def test_warning_rendering(capture_console):
    """Test that the warning helper applies the correct symbols and ANSI sequences."""
    print_warning("warning")
    output = capture_console()
    assert "⚠ warning" in strip_rich_formatting(output)
    assert "\x1b[" in output

def test_error_rendering(capture_console):
    """Test that the error helper applies the correct symbols and ANSI sequences."""
    print_error("error")
    output = capture_console()
    assert "✗ error" in strip_rich_formatting(output)
    assert "\x1b[" in output

def test_header_rendering(capture_console):
    """Test that the header helper applies the correct symbols and ANSI sequences."""
    print_header("header")
    output = capture_console()
    assert "\n  header" in strip_rich_formatting(output)
    assert "\x1b[" in output

def test_non_pass_prompt(monkeypatch, capsys):
    """Test that the prompt helper applies the correct symbols and ANSI sequences."""
    string_io = io.StringIO()
    test_console = Console(file=string_io, force_terminal=True, color_system="truecolor")
    monkeypatch.setattr("builtins.input", lambda *args: "1234ABcd")
    with patch("rich.prompt.get_console", return_value=test_console), \
            patch("hermes_cli.colors.should_use_color", return_value=True):
        non_pass = prompt("No Password Test")
    output = string_io.getvalue()
    assert strip_rich_formatting(output) == "  No Password Test: "
    assert non_pass == "1234ABcd"
    assert "\x1b[" in output

def test_yes_no_prompt(monkeypatch, capsys):
    """Test that the Y/N prompt helper applies the correct symbols and ANSI sequences."""
    string_io = io.StringIO()
    test_console = Console(file=string_io, force_terminal=True, color_system="truecolor")
    monkeypatch.setattr("builtins.input", lambda *args: "Y")
    with patch("rich.prompt.get_console", return_value=test_console), \
            patch("hermes_cli.colors.should_use_color", return_value=True):
        res = prompt_yes_no("Y/N: ")
    output = string_io.getvalue()
    assert "Y/N: " in strip_rich_formatting(output)
    assert res
    assert "\x1b[" in output