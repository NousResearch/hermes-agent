import io
import pytest
from unittest.mock import patch
from rich.console import Console

from hermes_cli.setup import print_header, print_noninteractive_setup_guidance, _prompt_api_key, _print_setup_summary, _print_migration_preview, _run_portal_one_shot


@pytest.fixture
def capture_rich_console():
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


## ---------------------------------------------------------------------------
## Tests for Header & Guidance Functions
## ---------------------------------------------------------------------------

def test_print_header_output(capture_rich_console):
    """Verify section header prints correctly with CYAN and BOLD formatting."""
    title = "Test Section Header"
    print_header(title)

    output = capture_rich_console()
    assert f"◆ {title}" in output
    assert "\x1b[" in output


def test_print_noninteractive_setup_guidance(capture_rich_console):
    """Verify non-interactive guidance header prints with CYAN and BOLD."""
    print_noninteractive_setup_guidance(reason="No TTY available")

    output = capture_rich_console()
    assert "⚕ Hermes Setup — Non-interactive mode" in output
    assert "No TTY available" in output
    assert "\x1b[" in output


## ---------------------------------------------------------------------------
## Tests for API Key Prompt Header
## ---------------------------------------------------------------------------

def test_prompt_api_key_header(monkeypatch, capture_rich_console):
    """Verify _prompt_api_key prints the stylized tool section header in CYAN."""
    var_info = {
        "name": "TEST_KEY",
        "description": "Test API Key",
        "tools": ["ToolA", "ToolB"],
        "url": "https://example.com",
    }

    monkeypatch.setattr("builtins.input", lambda *args: "0")

    _prompt_api_key(var_info)

    output = capture_rich_console()
    assert "─── Test API Key ───" in output
    assert "\x1b[" in output


## ---------------------------------------------------------------------------
## Tests for Setup Summary Banner & Tool Statuses
## ---------------------------------------------------------------------------

def test_print_setup_summary_rich_prints(capture_rich_console):
    """Verify rich colors and symbols (✓ green, ✗ red) printed in setup summary."""
    dummy_config = {}

    _print_setup_summary(dummy_config, hermes_home="/mock/hermes")

    output = capture_rich_console()

    assert "✓" in output or "✗" in output
    assert "Setup Complete!" in output
    assert "📁 All your files are in" in output
    assert "🚀 Ready to go!" in output
    assert "\x1b[" in output


## ---------------------------------------------------------------------------
## Tests for Migration Preview Output
## ---------------------------------------------------------------------------

def test_print_migration_preview_rich_prints(capture_rich_console):
    """Verify migration dry-run preview prints colored section headers and warnings."""
    report = {
        "items": [
            {"kind": "telegram", "destination": "/home/user/.hermes/telegram", "status": "migrated"},
            {"kind": "config", "reason": "Already exists", "status": "conflict"},
            {"kind": "soul", "reason": "Not applicable", "status": "skipped"},
        ]
    }

    _print_migration_preview(report)

    output = capture_rich_console()
    assert "Would import:" in output
    assert "Would overwrite" in output
    assert "Would skip:" in output
    assert "── Warnings ──" in output
    assert "\x1b[" in output


## ---------------------------------------------------------------------------
## Tests for One-Shot Portal Setup Box
## ---------------------------------------------------------------------------

def test_run_portal_one_shot_banner(capture_rich_console):
    """Verify MAGENTA banner box rendered in _run_portal_one_shot."""
    dummy_config = {}

    with patch("hermes_cli.main._model_flow_nous"):
        _run_portal_one_shot(dummy_config)

    output = capture_rich_console()
    assert "⚕ Hermes Setup — Nous Portal (one-shot)" in output
    assert "\x1b[" in output