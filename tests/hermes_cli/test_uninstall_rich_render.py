import io
import re
import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from hermes_cli.uninstall import (
    log_info,
    log_success,
    log_warn,
    _print_uninstall_dry_run,
    run_uninstall,
    run_gui_uninstall,
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


def test_log_helpers_rendering(capture_console):
    """Test that the log helpers apply the correct symbols and ANSI sequences."""

    log_info("Testing info")
    log_success("Testing success")
    log_warn("Testing warning")

    raw_output = capture_console()
    clean_output = strip_rich_formatting(raw_output)

    assert "→ Testing info" in clean_output
    assert "✓ Testing success" in clean_output
    assert "⚠ Testing warning" in clean_output

    assert "\x1b[" in raw_output


def test_dry_run_rendering(capture_console):
    """Test that the dry run summary renders lists and color highlights correctly."""

    mock_root = Path("/mock/root/hermes-agent")
    mock_home = Path("/mock/home/.hermes")

    _print_uninstall_dry_run(
        project_root=mock_root,
        hermes_home=mock_home,
        full_uninstall=True
    )

    raw_output = capture_console()
    clean_output = strip_rich_formatting(raw_output)

    assert "Dry run: no files, services, or environment entries will be changed." in clean_output
    assert "Would inspect/remove:" in clean_output
    assert "• Gateway services" in clean_output

    assert str(mock_root) in clean_output
    assert str(mock_home) in clean_output

    assert "\x1b[" in raw_output


@patch("hermes_cli.uninstall.input", side_effect=EOFError)
@patch("hermes_cli.uninstall.get_project_root", return_value=Path("/mock/root"))
@patch("hermes_cli.uninstall.get_hermes_home", return_value=Path("/mock/home"))
@patch("hermes_cli.uninstall._is_default_hermes_home", return_value=True)
@patch("hermes_cli.uninstall._discover_named_profiles", return_value=[])
def test_run_uninstall_ui_rendering(mock_profiles, mock_is_default, mock_home, mock_root, mock_input, capture_console):
    """Test that the main uninstall menu prints the bordered title and correct options."""

    args = argparse.Namespace(dry_run=False, yes=False, full=False)

    run_uninstall(args)

    raw_output = capture_console()
    clean_output = strip_rich_formatting(raw_output)

    assert "┌─────────────────────────────────────────────────────────┐" in clean_output
    assert "⚕ Hermes Agent Uninstaller" in clean_output
    assert "└─────────────────────────────────────────────────────────┘" in clean_output

    assert "Current Installation:" in clean_output
    assert str(Path("/mock/root")) in clean_output
    assert str(Path("/mock/home")) in clean_output

    assert "1) " in clean_output
    assert "Keep data" in clean_output
    assert "2) " in clean_output
    assert "Full uninstall" in clean_output
    assert "3) " in clean_output

    assert "Cancelled." in clean_output

    assert "\x1b[" in raw_output


@patch("hermes_cli.uninstall.input", side_effect=EOFError)
@patch("hermes_cli.uninstall.get_hermes_home", return_value=Path("/mock/home"))
@patch("hermes_cli.gui_uninstall.gui_install_summary")
@patch("hermes_cli.gui_uninstall.agent_is_installed", return_value=True)
def test_run_gui_uninstall_ui_rendering(mock_installed, mock_summary, mock_home, mock_input, capture_console):
    """Test the GUI uninstaller frame and layout rendering."""

    mock_summary.return_value = {
        "gui_installed": True,
        "source_built_artifacts": [Path("/mock/artifacts")],
        "packaged_app_paths": [Path("/mock/app")],
        "userdata_exists": True,
        "userdata_dir": Path("/mock/userdata")
    }

    args = argparse.Namespace(yes=False)
    run_gui_uninstall(args)

    raw_output = capture_console()
    clean_output = strip_rich_formatting(raw_output)

    assert "┌─────────────────────────────────────────────────────────┐" in clean_output
    assert "⚕ Hermes Chat GUI Uninstaller" in clean_output

    assert "Will remove:" in clean_output
    assert f"• {Path('/mock/artifacts')}" in clean_output
    assert f"• {Path('/mock/app')}" in clean_output
    assert f"• {Path('/mock/userdata')}" in clean_output
    assert "Kept intact:" in clean_output

    assert "\x1b[" in raw_output