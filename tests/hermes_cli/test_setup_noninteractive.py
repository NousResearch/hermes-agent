"""Tests for non-interactive setup wizard behavior."""
import pytest
from unittest.mock import patch, MagicMock


def _make_args(**kwargs):
    args = MagicMock()
    args.non_interactive = kwargs.get("non_interactive", False)
    args.section = kwargs.get("section", None)
    args.reset = kwargs.get("reset", False)
    return args


class TestNonInteractiveSetup:
    """Verify setup wizard exits cleanly in non-interactive environments."""

    def test_non_interactive_flag_skips_wizard(self, capsys):
        """--non-interactive flag should print help and return without hanging."""
        from hermes_cli.setup import run_setup_wizard
        args = _make_args(non_interactive=True)

        with patch("hermes_cli.setup.ensure_hermes_home"), \
             patch("hermes_cli.setup.load_config", return_value={}), \
             patch("hermes_cli.setup.get_hermes_home", return_value="/tmp/.hermes"):
            run_setup_wizard(args)

        out = capsys.readouterr().out
        assert "hermes config set" in out

    def test_no_tty_skips_wizard(self, capsys):
        """When stdin has no TTY, wizard should exit with helpful message."""
        from hermes_cli.setup import run_setup_wizard
        args = _make_args(non_interactive=False)

        with patch("hermes_cli.setup.ensure_hermes_home"), \
             patch("hermes_cli.setup.load_config", return_value={}), \
             patch("hermes_cli.setup.get_hermes_home", return_value="/tmp/.hermes"), \
             patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            run_setup_wizard(args)

        out = capsys.readouterr().out
        assert "hermes config set" in out
