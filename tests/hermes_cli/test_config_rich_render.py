import io
import re
import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
import rich
from rich.console import Console

from hermes_cli.config import redact_key, show_config, config_command

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

def test_redact_rendering(capture_console):
    rich.print(redact_key(""))
    output = capture_console()
    assert "(not set)" in strip_rich_formatting(output)
    assert "\x1b[" in output

def test_show_config_rendering(capture_console):
    show_config()
    output = capture_console()
    assert "\x1b[" in output

def test_config_command(monkeypatch, capture_console):
    config_command(argparse.Namespace(config_command="check"))
    output = capture_console()
    assert "Configuration Status" in strip_rich_formatting(output)
    assert "\x1b[" in output