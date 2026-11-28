"""Tests for gateway-mode detection behind /goal, /heartbeat and /loop.

The Desktop app serves slash commands from ``tui_gateway.slash_worker``, a
shadow CLI that holds a ``GatewaySessionStore`` but never opens a gateway
session. When gateway mode was inferred from the store type alone, the shadow
CLI was reported as "on the gateway with no connection", and every
session-bound manager answered "unavailable (no active session)" in the
Desktop app.
"""

import types

from hermes_cli.cli_loops_mixin import CLILoopsMixin
from hermes_cli.gateway_session_store import GatewaySessionStore


class _FakeStore(GatewaySessionStore):
    """A GatewaySessionStore without touching the shared gateway connection."""

    def __init__(self, connection):
        self.connection = connection
        self.snapshots = {}
        self.current = None


def _cli(session_db=None, gateway_session_open=False):
    """A stub carrying only the two attributes the lookup reads."""
    cli = types.SimpleNamespace(
        _session_db=session_db,
        _gateway_session_open=gateway_session_open,
    )
    cli._automation_connection = CLILoopsMixin._automation_connection.__get__(cli)
    cli._session_bound_manager = CLILoopsMixin._session_bound_manager.__get__(cli)
    cli._automation_manager = CLILoopsMixin._automation_manager.__get__(cli)
    return cli


# -- gateway-mode detection ---------------------------------------------------


def test_shadow_cli_with_a_gateway_store_is_not_gateway_mode():
    """The TUI/Desktop slash worker: holds the store, never opens the session."""
    cli = _cli(session_db=_FakeStore(object()), gateway_session_open=False)
    assert cli._automation_connection() == (False, None)


def test_open_gateway_session_is_gateway_mode():
    connection = object()
    cli = _cli(session_db=_FakeStore(connection), gateway_session_open=True)
    assert cli._automation_connection() == (True, connection)


def test_non_gateway_store_is_not_gateway_mode():
    cli = _cli(session_db=object(), gateway_session_open=True)
    assert cli._automation_connection() == (False, None)


def test_no_session_store_is_not_gateway_mode():
    cli = _cli(session_db=None, gateway_session_open=True)
    assert cli._automation_connection() == (False, None)


# -- the managers behind /goal, /heartbeat and /loop -------------------------


def test_shadow_cli_gets_the_local_manager_not_none():
    """The regression: this used to return None -> "unavailable (no active session)"."""
    cli = _cli(session_db=_FakeStore(object()), gateway_session_open=False)
    cli.session_id = "sess-1"

    def local():
        return lambda sid: ("local", sid)

    def remote(sid, connection):
        return ("remote", sid)

    mgr = cli._automation_manager("_manager", "manager", local, remote)
    assert mgr is not None
    assert mgr == ("local", "sess-1")


def test_open_gateway_session_gets_the_remote_manager():
    connection = object()
    cli = _cli(session_db=_FakeStore(connection), gateway_session_open=True)
    cli.session_id = "sess-2"

    def local():
        return lambda sid: ("local", sid)

    def remote(sid, conn):
        return ("remote", sid, conn)

    mgr = cli._automation_manager("_manager", "manager", local, remote)
    assert mgr is not None
    assert mgr == ("remote", "sess-2", connection)
