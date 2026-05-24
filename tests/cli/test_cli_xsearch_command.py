from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("prompt_toolkit")

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj._busy_command = lambda _status: nullcontext()
    return cli_obj


def test_process_command_dispatches_xsearch_through_handler():
    cli_obj = _make_cli()

    with patch.object(cli_obj, "_handle_xsearch_command") as mock_handler:
        assert cli_obj.process_command("/xsearch status") is True

    mock_handler.assert_called_once_with("/xsearch status")
