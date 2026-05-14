"""Tests for ``read_secret_line()`` in hermes_cli.cli_output.

Guards against the regression where ``getpass.getpass()`` on Windows
leaks ``\\x00`` + scan-code sequences (from ``msvcrt.getwch()``) into
the returned string, producing API-key values that cannot be assigned
to ``os.environ`` (``ValueError: embedded null character``) or sent as
HTTP headers (ASCII encode error).
"""

from unittest.mock import patch

from hermes_cli.cli_output import read_secret_line


def test_strips_null_and_scan_code_pair():
    """Left-Arrow on Windows console emits ``\\x00K`` — both bytes must go."""
    with patch("hermes_cli.cli_output.getpass.getpass", return_value="\x00Ksk-test-abc"):
        assert read_secret_line("prompt: ") == "Ksk-test-abc"


def test_strips_all_ascii_control_chars():
    """Every ``\\x00``-``\\x1f`` byte should be removed."""
    raw = "".join(chr(i) for i in range(32)) + "real-value"
    with patch("hermes_cli.cli_output.getpass.getpass", return_value=raw):
        assert read_secret_line("prompt: ") == "real-value"


def test_preserves_normal_input():
    """Regular printable input must pass through untouched."""
    with patch("hermes_cli.cli_output.getpass.getpass", return_value="sk-ant-xyz123"):
        assert read_secret_line("prompt: ") == "sk-ant-xyz123"


def test_passes_prompt_through():
    """The prompt argument must reach ``getpass.getpass`` verbatim."""
    with patch("hermes_cli.cli_output.getpass.getpass", return_value="x") as mock:
        read_secret_line("API key: ")
        mock.assert_called_once_with("API key: ")
