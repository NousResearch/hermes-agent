import io
import re
import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
import rich
from rich.console import Console

from hermes_cli.curses_ui import _radio_numbered_fallback, _numbered_single_fallback, _numbered_fallback

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

def test_radio_numbered_fallback(monkeypatch, capture_console):
    monkeypatch.setattr("builtins.input", lambda *args: "0")
    _radio_numbered_fallback("Test", ["A", "B", "C"], 1, 0)
    output = capture_console()
    assert "Test" in output
    assert "A" in output
    assert "B" in output
    assert "C" in output
    assert "\x1b[" in output

def test_numbered_single_fallback(monkeypatch, capture_console):
    monkeypatch.setattr("builtins.input", lambda *args: "0")
    _numbered_single_fallback("Test", ["A", "B", "C"], 0)
    output = capture_console()
    assert "Test" in output
    assert "A" in output
    assert "B" in output
    assert "C" in output
    assert "." in output
    assert "\x1b[" in output

def test_numbered_fallback(monkeypatch, capture_console):
    monkeypatch.setattr("builtins.input", lambda *args: "A")
    _numbered_fallback("Test", ["A", "B", "C"], {1}, {0})
    output = capture_console()
    assert "Test" in output
    assert "A" in output
    assert "B" in output
    assert "C" in output
    assert "✓" in output
    assert "[" in output
    assert "\x1b[" in output
