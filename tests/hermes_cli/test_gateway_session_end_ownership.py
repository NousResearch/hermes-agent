"""A gateway-backed classic CLI leaves the end of its session to the gateway.

Regression for #134889 LC-2. The CLI's ``end_session`` (exit, ``/new``) detaches with an ``end_reason``
and the gateway, which owns the agent, the memory manager and the Relay scope, runs
``on_session_finalize`` (tests/gateway/test_viewer_close_lifecycle.py). Firing it in the CLI process as
well finalized every session twice, once in a process that holds none of that state; and the CLI's
mid-turn ``on_session_end`` duplicated the one the gateway's interrupted turn fires.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.gateway_agent import GatewayAgent


@pytest.fixture
def cli_mod():
    import cli as cli_mod
    cli_mod._cleanup_done = False
    cli_mod._single_query_finalize_attempted_session_ids.clear()
    cli_mod._handed_off_session_ids.clear()
    yield cli_mod
    cli_mod._active_agent_ref = None
    cli_mod._cleanup_done = False


def _gateway_agent(session_id):
    return GatewayAgent(connection=MagicMock(), session_id=session_id)


@patch("hermes_cli.lifecycle.finalize_session")
def test_exit_cleanup_does_not_finalize_a_gateway_session(mock_finalize, cli_mod):
    cli_mod._active_agent_ref = _gateway_agent("gw-session")
    cli_mod._run_cleanup()
    mock_finalize.assert_not_called()


@patch("hermes_cli.lifecycle.invoke_hook")
@patch("hermes_cli.lifecycle.finalize_session")
def test_new_does_not_finalize_a_gateway_session_in_the_cli(mock_finalize, mock_invoke_hook, cli_mod):
    cli = cli_mod.HermesCLI()
    cli.agent = _gateway_agent("gw-session")
    cli.new_session(silent=True)
    mock_finalize.assert_not_called()


@patch("hermes_cli.lifecycle.invoke_hook")
@patch("hermes_cli.lifecycle.finalize_session")
def test_new_leaves_on_session_reset_to_the_gateway(mock_finalize, mock_invoke_hook, cli_mod):
    cli = cli_mod.HermesCLI()
    cli.agent = _gateway_agent("gw-session")
    cli.new_session(silent=True)
    # The detach's ``new_session`` reason makes the owner fire it (tests/gateway/test_viewer_close_lifecycle.py).
    assert not [c for c in mock_invoke_hook.call_args_list if c.args == ("on_session_reset",)]


@patch("hermes_cli.lifecycle.invoke_hook")
def test_exit_mid_turn_leaves_on_session_end_to_the_gateway_turn(mock_invoke_hook, cli_mod):
    cli = cli_mod.HermesCLI()
    cli.agent = _gateway_agent("gw-session")
    cli._agent_running = True
    cli._session_db = None
    with patch.object(cli_mod, "_run_cleanup"), patch.object(cli, "_print_exit_summary"):
        cli._tui_shutdown()
    assert not [c for c in mock_invoke_hook.call_args_list if c.args == ("on_session_end",)]


@patch("hermes_cli.lifecycle.finalize_session")
def test_one_shot_ends_a_gateway_session_through_the_gateway_only(mock_finalize, cli_mod, monkeypatch):
    store = MagicMock()
    agent = GatewayAgent(connection=MagicMock(), session_db=store, session_id="gw-session")
    released = []
    one_shot = SimpleNamespace(agent=agent, session_id="gw-session", conversation_history=[], _session_db=store,
                               _release_active_session=lambda: released.append(True))
    monkeypatch.setattr(cli_mod, "_wait_for_oneshot_background_completions", lambda _cli: None)
    monkeypatch.setattr(cli_mod, "_run_cleanup", lambda **_kwargs: None)

    cli_mod._finalize_single_query(one_shot)

    # The detach carries the end reason, so the gateway runs the lifecycle; the CLI never does.
    store.end_session.assert_called_once_with("gw-session", "cli_close")
    mock_finalize.assert_not_called()
    assert released == [True]


@patch("hermes_cli.lifecycle.invoke_hook")
def test_interrupted_one_shot_leaves_on_session_end_to_the_gateway_turn(mock_invoke_hook, cli_mod):
    one_shot = SimpleNamespace(agent=_gateway_agent("gw-session"), session_id="gw-session")
    cli_mod._emit_interrupted_session_end(one_shot)
    assert not [c for c in mock_invoke_hook.call_args_list if c.args == ("on_session_end",)]
