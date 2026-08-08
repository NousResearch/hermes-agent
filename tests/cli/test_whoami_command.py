"""Tests for the /whoami CLI command.

/whoami is registered in COMMAND_REGISTRY but used to have no dispatch branch
in HermesCLI.process_command — typing it printed "Unknown command:
/whoami". These tests lock in the dispatch wiring and the handler behavior.
"""

from __future__ import annotations

import getpass
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli() -> HermesCLI:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = None
    cli_obj._pending_input = MagicMock()
    cli_obj._app = None
    cli_obj._pending_resume_sessions = None
    return cli_obj


class TestWhoamiDispatch:
    """The command must route to its handler — not fall through to "Unknown"."""

    def test_whoami_dispatches_to_handler(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_handle_whoami_command") as mock_handler:
            result = cli_obj.process_command("/whoami")

        mock_handler.assert_called_once_with()
        assert result is True

    def test_whoami_is_not_unknown_command(self):
        cli_obj = _make_cli()
        with (
            patch("cli._cprint") as mock_cprint,
            patch.object(cli_obj, "_handle_whoami_command"),
        ):
            result = cli_obj.process_command("/whoami")

        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "Unknown command" not in printed
        assert result is True


class TestHandleWhoamiCommand:
    def test_reports_local_admin_tier(self, capsys):
        cli_obj = _make_cli()
        cli_obj._handle_whoami_command()

        out = capsys.readouterr().out
        assert "local CLI" in out
        assert getpass.getuser() in out
        assert "Tier: admin" in out
        assert "all available" in out
