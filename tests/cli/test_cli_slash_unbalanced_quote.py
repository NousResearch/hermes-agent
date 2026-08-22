"""Slash-command handlers must survive an unbalanced quote in user input.

``/cron`` and ``/curator`` tokenize the typed command with ``shlex.split()``,
which raises ``ValueError`` on an unbalanced quote (e.g. ``/cron add "do x``).
The REPL dispatch (``cli.process_command``) is wrapped only in a
``try/except KeyboardInterrupt``, so an escaping ``ValueError`` unwinds to the
prompt_toolkit loop and kills the whole session. Both handlers now fall back to
a naive whitespace split, matching the existing ``/tools`` handler.
"""

from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli() -> HermesCLI:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "sess-shlex-test"
    cli_obj._pending_input = MagicMock()
    cli_obj._app = None
    return cli_obj


def test_curator_command_survives_unbalanced_quote():
    # shlex.split('/curator "x') raises ValueError; the handler must degrade to
    # a naive split instead of letting it escape and kill the session.
    cli_obj = _make_cli()
    with patch("hermes_cli.curator.cli_main") as mock_main:
        cli_obj._handle_curator_command('/curator "unterminated')
    mock_main.assert_called_once()
    assert mock_main.call_args[0][0] == ['"unterminated']


def test_cron_command_survives_unbalanced_quote():
    # Must not raise ValueError out of the handler. The unknown subcommand
    # ("unterminated) just prints usage; cronjob is patched defensively so no
    # real scheduling happens.
    cli_obj = _make_cli()
    with patch("tools.cronjob_tools.cronjob", return_value='{"status": "ok"}'):
        cli_obj._handle_cron_command('/cron "unterminated')
    # Reaching here without an exception is the assertion.


def test_curator_balanced_quote_still_groups_tokens():
    # The fast path (valid shlex) must keep grouping quoted arguments so the
    # naive-split fallback never degrades normal usage.
    cli_obj = _make_cli()
    with patch("hermes_cli.curator.cli_main") as mock_main:
        cli_obj._handle_curator_command('/curator run "two words"')
    mock_main.assert_called_once()
    assert mock_main.call_args[0][0] == ["run", "two words"]
