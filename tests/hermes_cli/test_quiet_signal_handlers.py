import signal
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_quiet_signal_handler_does_not_emit_interrupted_end_for_sigint():
    from hermes_cli.cli_single_query import _install_single_query_signal_handlers

    cli = SimpleNamespace(agent=object())
    with (
        patch("signal.signal") as mock_signal,
        patch("cli._arm_exit_watchdog_on_shutdown_signal"),
        patch("cli._interrupt_agent_for_signal") as mock_interrupt,
        patch("cli._emit_interrupted_session_end") as mock_emit,
    ):
        _install_single_query_signal_handlers(cli)
        handler = next(
            call.args[1]
            for call in mock_signal.call_args_list
            if call.args[0] == signal.SIGINT
        )

        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)

    mock_emit.assert_not_called()
    mock_interrupt.assert_called_once_with(cli.agent, signal.SIGINT)