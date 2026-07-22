import io
import argparse
from unittest.mock import patch, MagicMock

import pytest
from rich.console import Console
from hermes_cli.status import show_status, check_mark


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


def test_check_mark_rendering():
    """Test that the check mark helper returns the correct rich Text object

    with the proper characters and color attributes applied.
    """
    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=True, color_system="truecolor")

    with patch("hermes_cli.colors.should_use_color", return_value=True):
        true_mark = check_mark(True)
        assert true_mark.plain == "✓", "Expected a check mark character for True"
        console.print(true_mark)
        assert "\x1b[" in string_io.getvalue(), "Expected ANSI styling sequences for Green color"

        string_io.seek(0)
        string_io.truncate(0)

        false_mark = check_mark(False)
        assert false_mark.plain == "✗", "Expected an X character for False"
        console.print(false_mark)
        assert "\x1b[" in string_io.getvalue(), "Expected ANSI styling sequences for Red color"


@patch("hermes_cli.status.load_config")
@patch("hermes_cli.status.get_env_path")
def test_show_status_sections_rendered(mock_get_env_path, mock_load_config, capture_console):
    """Test that the base show_status command calls rich.print with the

    correctly formatted section headers, title frames, and ANSI colors.
    """
    mock_get_env_path.return_value.exists.return_value = False
    mock_load_config.return_value = {}

    args = argparse.Namespace(deep=False)
    show_status(args)

    joined_output = capture_console()

    assert "⚕ Hermes Agent Status" in joined_output
    assert "\x1b[" in joined_output, "Expected ANSI color sequences in the title or layout frame"

    assert "◆ Environment" in joined_output
    assert "◆ API Keys" in joined_output
    assert "◆ Auth Providers" in joined_output
    assert "◆ API-Key Providers" in joined_output
    assert "◆ Terminal Backend" in joined_output
    assert "◆ Messaging Platforms" in joined_output
    assert "◆ Gateway Service" in joined_output
    assert "◆ Scheduled Jobs" in joined_output
    assert "◆ Sessions" in joined_output

    assert "◆ Deep Checks" not in joined_output


@patch("hermes_cli.status.load_config")
@patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-mock-key"})
def test_show_status_deep_checks_rendered(mock_load_config, capture_console):
    """Test that the deep checks section is rendered exclusively when the

    deep parameter is passed in the args, complete with ANSI rendering.
    """
    mock_load_config.return_value = {}
    args = argparse.Namespace(deep=True)

    with patch("httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        show_status(args)

    joined_output = capture_console()

    assert "◆ Deep Checks" in joined_output
    assert "OpenRouter:" in joined_output
    assert "Port \x1b" in joined_output


def test_show_status_footer_rendered(capture_console):
    """Test that the diagnostic footers are correctly appended at the very end with style."""
    args = argparse.Namespace(deep=False)
    show_status(args)

    joined_output = capture_console()

    assert "Run 'hermes doctor' for detailed diagnostics" in joined_output
    assert "Run 'hermes setup' to configure" in joined_output
    assert "\x1b[" in joined_output, "Expected ANSI escape characters inside the dimmed footer section"