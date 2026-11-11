"""Tests for ``HermesCLI._prompt_text_input`` without a running app.

With the app running the answer is typed into the composer
(tests/hermes_cli/test_cli_free_text_prompt.py); without one it is a plain
``input()`` line.
"""

from unittest.mock import MagicMock, patch


def _make_cli():
    """Minimal HermesCLI shell exposing prompt fallback helpers."""
    import cli as cli_mod

    obj = object.__new__(cli_mod.HermesCLI)
    obj._app = MagicMock()
    obj._status_bar_visible = True
    return obj


class TestPromptTextInputThreadSafety:

    def test_no_app_uses_direct_input(self):
        """Without an active prompt_toolkit app, always call input() directly."""
        cli = _make_cli()
        cli._app = None

        with patch("builtins.input", return_value="cancel") as mock_input:
            result = cli._prompt_text_input("Choice: ")

        assert mock_input.called
        assert result == "cancel"

    def test_eof_returns_none(self):
        """EOFError from input() yields None, not an unhandled exception."""
        cli = _make_cli()
        cli._app = None

        with patch("builtins.input", side_effect=EOFError()):
            result = cli._prompt_text_input("Choice: ")

        assert result is None
