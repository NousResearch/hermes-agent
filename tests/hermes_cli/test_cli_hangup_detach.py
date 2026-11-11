"""A lost terminal is a viewer disconnect, not ``/exit`` (gateway-backed classic CLI).

SIGHUP (closed tab, dropped SSH / VS Code remote link) and the emulator's SIGTERM used to run the
same shutdown as ``/exit``: the CLI detached with ``end_reason='cli_close'``, and the owner's viewer
close interrupted the session's turn and every background subagent
(gateway/session_viewer_close.py::close_viewer_session). A signal now leaves the session as a
dropped socket does: no ``end_session``, no ``session.interrupt``; the owner's orphan close
(``ws_orphan_reap``) ends the idle row later without stopping delegations. User exits are unchanged
(tests/hermes_cli/test_gateway_session_end_ownership.py).
"""
import signal
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_session_store import GatewaySessionStore


@pytest.fixture
def cli_mod():
    import cli as cli_mod
    cli_mod._cleanup_done = False
    yield cli_mod
    cli_mod._active_agent_ref = None
    cli_mod._cleanup_done = False


def _hang_up(cli_mod, cli, signum):
    with patch.object(cli_mod, "_arm_exit_watchdog_on_shutdown_signal"), \
            patch.object(cli_mod, "_interrupt_agent_for_signal") as interrupt, \
            pytest.raises(KeyboardInterrupt):  # no prompt_toolkit app in a test: the fallback unwind
        cli._tui_signal_handler(signum, None)
    with patch.object(cli_mod, "_run_cleanup"), patch.object(cli, "_print_exit_summary"), \
            patch.object(cli, "_release_active_session"):
        cli._tui_shutdown()
    return interrupt


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGTERM])
@patch("hermes_cli.lifecycle.invoke_hook")
def test_hangup_mid_turn_leaves_the_gateway_session_running(_hook, cli_mod, signum):
    cli = cli_mod.HermesCLI()
    cli._session_db = MagicMock(spec=GatewaySessionStore)
    cli.agent = GatewayAgent(connection=MagicMock(), session_id="gw-session")
    cli._agent_running = True
    interrupt = _hang_up(cli_mod, cli, signum)
    interrupt.assert_not_called()
    assert "session.interrupt" not in [c.args[0] for c in cli.agent.connection.call.call_args_list]
    cli._session_db.end_session.assert_not_called()


@patch("hermes_cli.lifecycle.invoke_hook")
def test_hangup_at_an_idle_prompt_does_not_close_the_gateway_session(_hook, cli_mod):
    cli = cli_mod.HermesCLI()
    cli._session_db = MagicMock(spec=GatewaySessionStore)
    cli.agent = GatewayAgent(connection=MagicMock(), session_id="gw-session")
    cli._agent_running = False
    _hang_up(cli_mod, cli, signal.SIGHUP)
    cli._session_db.end_session.assert_not_called()
    cli._session_db.delete_session.assert_not_called()


@patch("hermes_cli.lifecycle.invoke_hook")
def test_hangup_in_cli_local_mode_still_interrupts_and_ends_the_session(_hook, cli_mod):
    """No gateway owns a CLI-local session: a lost terminal still stops and closes it."""
    cli = cli_mod.HermesCLI()
    cli._session_db = MagicMock()
    cli.agent = MagicMock(session_id="local-session")
    cli._agent_running = True
    with patch("hermes_cli.cli_tui_runtime_mixin.request_hard_interrupt"):
        interrupt = _hang_up(cli_mod, cli, signal.SIGHUP)
    interrupt.assert_called_once_with(cli.agent, signal.SIGHUP)
    cli._session_db.end_session.assert_called_once_with("local-session", "cli_close")
