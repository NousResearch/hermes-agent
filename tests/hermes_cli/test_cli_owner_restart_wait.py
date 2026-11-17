"""The classic CLI waits out a gateway restart instead of exiting on ``runtime_draining``."""
import asyncio
import types
from contextlib import asynccontextmanager

import pytest

from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_session_store import GatewaySessionStore


def _snapshot(sid):
    return {"session_id": sid, "stored_session_id": sid, "messages": [], "revision": 1,
            "execution_generation": 0, "subscription_id": "sub-" + sid, "info": {"model": "m"}}


class DrainingThenReady:
    """An owner that refuses ``session.create`` while it drains, then (its successor) answers."""

    def __init__(self, refusals, reason="runtime_draining"):
        self.events = asyncio.Queue()
        self.refusals, self.reason = refusals, reason
        self.creates = 0

    async def rpc(self, method, *, _timeout=None, **params):
        if method == "session.create":
            self.creates += 1
            if self.creates <= self.refusals:
                raise GatewayRPCError(self.reason, -32000, {"reason": self.reason})
            return _snapshot("local-after-restart")
        if method == "session.resume":
            return _snapshot(params["session_id"])
        return {"session_create": {"parameters": []}}


@pytest.fixture
def owner(monkeypatch):
    import hermes_cli.gateway_owner_restart as restart
    monkeypatch.setattr(restart, "restart_wait_budget", lambda: 30.0)
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.01)
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    client = DrainingThenReady(refusals=2)

    @asynccontextmanager
    async def connect():
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    yield GatewaySessionStore(conn), client
    conn.close()


def _host(store):
    return types.SimpleNamespace(session_id="local-old", _resumed=False, _session_db=store, model="m",
                                 _explicit_model_override=False, requested_provider=None, enabled_toolsets=["file"],
                                 _explicit_toolsets=False, _gateway_session_open=True)


def test_new_session_waits_for_the_restarted_owner_and_says_so_once(owner, capsys):
    """A draining owner's refusal is not the user's error: one plain line, then the session opens
    on the successor, and the reason code is never printed."""
    from hermes_cli.cli_init_mixin import CLIInitMixin
    store, client = owner
    sid = CLIInitMixin._new_session_id(_host(store), None)
    assert sid == "local-after-restart" and client.creates == 3
    err = capsys.readouterr().err
    assert err.count("Hermes is restarting") == 1
    assert "runtime_draining" not in err


def test_a_final_refusal_is_still_raised_at_once(owner, capsys):
    from hermes_cli.cli_init_mixin import CLIInitMixin
    store, client = owner
    client.reason = "invalid_params"
    with pytest.raises(GatewayRPCError, match="invalid_params"):
        CLIInitMixin._new_session_id(_host(store), None)
    assert client.creates == 1 and "Hermes is restarting" not in capsys.readouterr().err


def test_the_wait_is_bounded_and_ctrl_c_stops_it():
    import threading
    from hermes_cli.gateway_owner_restart import SuccessorWaitStopped, wait_for_successor

    def refuse(_remaining):
        raise GatewayRPCError("runtime_retired")
    now = [0.0]

    def clock():
        now[0] += 10
        return now[0]
    with pytest.raises(GatewayClientError) as timed_out:
        wait_for_successor(refuse, announce=lambda: None, budget=25, clock=clock)
    assert "runtime_retired" not in str(timed_out.value) and not isinstance(timed_out.value, GatewayRPCError)
    stop = threading.Event()
    stop.set()
    with pytest.raises(SuccessorWaitStopped):
        wait_for_successor(refuse, announce=lambda: None, wake=stop, budget=60)
