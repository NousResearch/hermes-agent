"""A Kanban worker never re-enters an owner's restart after-turn wait.

The wait cuts every socket opened before it so their clients go to the NEXT owner. A worker that
redials into the wait and keeps polling ``prompt.receipt`` stays alive, so ``_watch_submitter``
never ends the card's turn and the restart (``hermes update``) is held until
``agent.restart_after_turn_timeout``. The wait still admits interactive clients (#139041).
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_client, gateway_runtime, kanban_worker_client
from gateway.drain_control import RESTART_WAIT_REASON

GUE = gateway_client.GatewayUnavailableError


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)


def _restart_wait_verdict(*_, **__):
    return gateway_runtime.GatewayDiscovery(
        "draining", gateway_runtime.GatewayEndpoint("p", "i", 1, 1, "http://127.0.0.1:1", "none", frozenset()),
        reason_code=RESTART_WAIT_REASON)


@pytest.mark.asyncio
async def test_a_machine_client_is_refused_by_a_restart_wait_and_mints_no_ticket(monkeypatch):
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", _restart_wait_verdict)
    minted = []

    def ticket(*args, **kwargs):
        minted.append(args)
        raise RuntimeError("ticket minted")
    monkeypatch.setattr(gateway_client, "_session_ticket", ticket)

    with pytest.raises(GUE) as raised, gateway_client.outside_restart_wait():
        async with gateway_client.connect_gateway():
            pass
    assert (raised.value.state, raised.value.reason_code) == ("draining", RESTART_WAIT_REASON)
    assert minted == []

    # An interactive client still attaches to answer a parked prompt.
    with pytest.raises(RuntimeError, match="ticket minted"):
        async with gateway_client.connect_gateway():
            pass
    assert len(minted) == 1


class _RestartingOwner:
    """First connection: the claim is admitted and pending, then the restart wait cuts the socket.
    Every later dial lands in the wait, where the turn stays pending (its LLM call is held)."""

    def __init__(self):
        self.dials = 0
        self.receipts_in_wait = 0

    def connect(self):
        owner = self

        @asynccontextmanager
        async def _connect(*_, **__):
            owner.dials += 1
            in_wait = owner.dials > 1
            # The verdict the real connect_gateway gives a restart-wait owner (tested above).
            refuse = getattr(gateway_client, "_REFUSE_RESTART_WAIT", None)
            if in_wait and refuse is not None and refuse.get():
                raise GUE(f"Gateway draining: {RESTART_WAIT_REASON}", "draining", RESTART_WAIT_REASON)

            class _Client:
                events = asyncio.Queue()

                async def rpc(self, method, **params):
                    if method == "kanban.run":
                        return {"session_id": "s1", "receipt": {"status": "pending", "admission_id": "a1"}}
                    if not in_wait:
                        raise gateway_client.GatewayClientError("Gateway disconnected; turn outcome is unknown")
                    owner.receipts_in_wait += 1
                    if owner.receipts_in_wait > 50:
                        raise AssertionError("the worker keeps polling its card inside the restart wait")
                    return {"status": "pending"}

            yield _Client()

        return _connect


@pytest.mark.asyncio
async def test_a_worker_cut_by_a_restart_wait_waits_outside_it(monkeypatch, tmp_path):
    clock = [0.0]
    monkeypatch.setattr(kanban_worker_client, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def _sleep(seconds):
        clock[0] += seconds
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _sleep)
    owner = _RestartingOwner()
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())

    # The worker waits for the next owner without reading anything from the restarting one, and an
    # owner that never comes back still fails it at OWNER_WAIT_S (which ends the turn it held).
    with pytest.raises(gateway_client.GatewayClientError, match="owner silent"):
        await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))
    assert owner.receipts_in_wait == 0
    assert owner.dials > 2


class _Attached(Exception):
    """The owner admitted the socket into its handler."""


def _owner_rig(tmp_path, monkeypatch, *, old_owner=False):
    """A ready owner whose restart wait opens while the client mints its ticket: after discovery
    said ``ready`` and before the WebSocket dials. Tickets come from the real control-socket verb
    and the dial is admitted (or refused) by the real ``GatewayRuntimeAPI``."""
    import json

    from gateway.control_socket import GatewayControlServer
    from gateway.run_api import GatewayRuntimeAPI
    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli import web_server as web
    from hermes_cli import web_server_chat

    profile = str(tmp_path)
    tickets = TicketStore("owner", [profile])
    control = GatewayControlServer(tmp_path)
    control.ticket_store = tickets
    runner = SimpleNamespace(_draining=False, _running=True, session_runtime_descriptor={"state": "ready"},
                             session_authority=SimpleNamespace(db=SimpleNamespace(db_path=tmp_path / "state.db"),
                                                               profile_id=profile, runner=None),
                             session_ticket_store=tickets)
    endpoint = gateway_runtime.GatewayEndpoint(profile, "owner", 1, 1, "http://127.0.0.1:1", "none", frozenset())
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime",
                        lambda *a, **k: gateway_runtime.GatewayDiscovery("ready", endpoint))

    def control_query(home, request, timeout=5):
        params = json.loads(request)["params"]
        if old_owner and "restart_wait" in params:  # an owner from before the policy existed
            return json.dumps({"ok": False, "error": gateway_client._UNKNOWN_TICKET_PARAMS, "protocol": 1,
                               "id": 1}).encode() + b"\n"
        answer = control.handle_request_line(request, "uid:test")
        runner._draining = True  # the restart begins: its after-turn wait is open by the dial
        runner.session_runtime_descriptor.update(state="draining", drain_reason=RESTART_WAIT_REASON)
        return answer
    monkeypatch.setattr(gateway_client, "private_control_query", control_query)

    async def handle(ws, **kwargs):
        raise _Attached

    async def fallback(*args):
        pytest.fail("a ticketed native socket must not enter the legacy fallback")
    monkeypatch.setattr("tui_gateway.ws.handle_ws", handle)
    monkeypatch.setattr(web, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(web_server_chat, "_ws_request_is_allowed", lambda ws: True)
    api = GatewayRuntimeAPI(fallback, runner, web.app)

    @asynccontextmanager
    async def dial(url, *, subprotocols, **_):
        closed = []

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            closed.append(message)
        scope = {"type": "websocket", "path": "/api/ws", "scheme": "ws", "query_string": b"",
                 "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 1),
                 "headers": [(b"sec-websocket-protocol", ", ".join(subprotocols).encode())],
                 "subprotocols": subprotocols}
        await api(scope, receive, send)
        assert [(m["type"], m.get("code")) for m in closed] == [("websocket.close", 1013)], closed
        raise OSError("owner refused the upgrade (1013)")
        yield
    monkeypatch.setattr("websockets.asyncio.client.connect", dial)


@pytest.mark.asyncio
async def test_a_restart_that_begins_after_discovery_still_refuses_the_worker_at_admission(tmp_path, monkeypatch):
    _owner_rig(tmp_path, monkeypatch)
    with pytest.raises(gateway_client.GatewayClientError, match="connection failed"), \
            gateway_client.outside_restart_wait():
        async with gateway_client.connect_gateway():
            pytest.fail("attached inside the restart wait")

    # An interactive client dialing the same race still attaches to answer a parked prompt (#139041).
    _owner_rig(tmp_path, monkeypatch)
    with pytest.raises(_Attached):
        async with gateway_client.connect_gateway():
            pass


@pytest.mark.asyncio
async def test_an_owner_older_than_the_policy_still_mints_the_worker_a_ticket(tmp_path, monkeypatch):
    # The code on disk was updated under a running owner: the worker still attaches to it (the
    # pre-dial check is all that owner can honour) instead of being stranded until it restarts.
    _owner_rig(tmp_path, monkeypatch, old_owner=True)
    with pytest.raises(_Attached), gateway_client.outside_restart_wait():
        async with gateway_client.connect_gateway():
            pass
