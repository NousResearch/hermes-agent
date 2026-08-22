"""An unbalanced quote in a slash command must be refused, not executed.

``/journey``, ``/cron`` and ``/curator`` tokenize the typed line with
``shlex.split``, which raises ``ValueError: No closing quotation`` when a quote
is left open (``/cron add 30m "partial``).  The interactive dispatch in
``cli.py`` wraps ``process_command`` in an ``except KeyboardInterrupt`` only, so
that ``ValueError`` unwinds to the outer prompt_toolkit loop and takes the
session — and the conversation — with it.

The fix refuses the line: print a quoting hint and return.  It is deliberately
*not* a ``cmd.split()`` fallback, because a naive split keeps the session alive
while letting the malformed line run — ``/cron add 30m "partial`` still splits
into a well-formed ``add`` whose schedule (``30m``) validates, so a scheduled
job gets created from input the user never finished typing.  These tests pin
both halves: nothing escapes, and nothing runs.
"""

from unittest.mock import patch

import pytest

from cli import HermesCLI

_CRON_OK = (
    '{"success": true, "job_id": "job-1", "schedule": "every 2h",'
    ' "next_run_at": "in 2h"}'
)


def _make_cli() -> HermesCLI:
    """A bare ``HermesCLI``.

    ``__init__`` is skipped on purpose: none of the three handlers reads
    instance state on these paths, so an uninitialized instance exercises the
    real MRO (``HermesCLI`` -> ``CLICommandsMixin``) without standing up a
    session, an agent or a config file.
    """
    return HermesCLI.__new__(HermesCLI)


# --------------------------------------------------------------------------
# /cron — the effectful case teknium1 called out on #43503
# --------------------------------------------------------------------------


def test_cron_unbalanced_quote_does_not_create_a_job(capsys):
    cli_obj = _make_cli()
    # A JSON return keeps the mock usable so a call would surface as the
    # assertion below rather than as an unrelated decode error.
    with patch("tools.cronjob_tools.cronjob", return_value=_CRON_OK) as mock_cronjob:
        cli_obj._handle_cron_command('/cron add 30m "partial')
    mock_cronjob.assert_not_called()
    out = capsys.readouterr().out
    assert "/cron" in out and "No closing quotation" in out


def test_cron_balanced_quotes_still_group_schedule_and_prompt(capsys):
    """The refusal path must not degrade a correctly quoted command."""
    cli_obj = _make_cli()
    with patch("tools.cronjob_tools.cronjob", return_value=_CRON_OK) as mock_cronjob:
        cli_obj._handle_cron_command('/cron add "every 2h" "Check server status"')
    mock_cronjob.assert_called_once()
    kwargs = mock_cronjob.call_args.kwargs
    assert kwargs["action"] == "create"
    assert kwargs["schedule"] == "every 2h"
    assert kwargs["prompt"] == "Check server status"
    assert "No closing quotation" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# /curator — must not delegate the malformed line
# --------------------------------------------------------------------------


def test_curator_unbalanced_quote_does_not_delegate(capsys):
    cli_obj = _make_cli()
    with patch("hermes_cli.curator.cli_main") as mock_main:
        cli_obj._handle_curator_command('/curator "unterminated')
    mock_main.assert_not_called()
    assert "/curator" in capsys.readouterr().out


def test_curator_balanced_quotes_still_group_tokens():
    cli_obj = _make_cli()
    with patch("hermes_cli.curator.cli_main") as mock_main:
        cli_obj._handle_curator_command('/curator note "a title"')
    mock_main.assert_called_once()
    assert mock_main.call_args[0][0] == ["note", "a title"]


def test_curator_bare_command_still_defaults_to_status():
    """The ``if cmd:`` guard must keep the bare-``/curator`` default intact."""
    cli_obj = _make_cli()
    with patch("hermes_cli.curator.cli_main") as mock_main:
        cli_obj._handle_curator_command("")
    mock_main.assert_called_once()
    assert mock_main.call_args[0][0] == ["status"]


# --------------------------------------------------------------------------
# /journey — the third site, which #43503 does not cover
# --------------------------------------------------------------------------


def test_journey_unbalanced_quote_does_not_dispatch(capsys):
    cli_obj = _make_cli()
    with patch("hermes_cli.journey._cmd_delete") as mock_delete, \
            patch("hermes_cli.journey._cmd_show") as mock_show:
        cli_obj._handle_journey_command('/journey delete "my entry')
    mock_delete.assert_not_called()
    mock_show.assert_not_called()
    assert "/journey" in capsys.readouterr().out


def test_journey_balanced_quotes_still_group_tokens():
    cli_obj = _make_cli()
    with patch("hermes_cli.journey._cmd_delete") as mock_delete:
        cli_obj._handle_journey_command('/journey delete "my entry"')
    mock_delete.assert_called_once()
    assert mock_delete.call_args[0][0].node == "my entry"


# --------------------------------------------------------------------------
# The session-survival contract the REPL cannot provide for itself
# --------------------------------------------------------------------------


def test_refusal_hint_uses_the_prompt_toolkit_safe_printer():
    """The hint must survive ``patch_stdout``.

    A bare ``print`` is swallowed while the prompt_toolkit Application owns the
    terminal — exactly the situation this guard exists for — so the refusal has
    to go through ``cli._cli_visible_print`` or it is silent where it matters.
    """
    cli_obj = _make_cli()
    with patch("cli._cli_visible_print") as mock_print, \
            patch("hermes_cli.curator.cli_main"):
        cli_obj._handle_curator_command('/curator "unterminated')
    printed = " ".join(str(call[0][0]) for call in mock_print.call_args_list)
    assert "No closing quotation" in printed
    assert "/curator" in printed


@pytest.mark.parametrize(
    "handler_name, line",
    [
        ("_handle_journey_command", '/journey delete "my entry'),
        ("_handle_cron_command", '/cron add 30m "partial'),
        ("_handle_curator_command", '/curator "unterminated'),
    ],
)
def test_slash_handlers_survive_unbalanced_quote(handler_name, line):
    cli_obj = _make_cli()
    with patch("tools.cronjob_tools.cronjob", return_value=_CRON_OK), \
            patch("hermes_cli.curator.cli_main"), \
            patch("hermes_cli.journey._cmd_delete"), \
            patch("hermes_cli.journey._cmd_show"):
        # No exception may escape: cli.py's slash dispatch catches only
        # KeyboardInterrupt, so anything else ends the interactive session.
        getattr(cli_obj, handler_name)(line)
