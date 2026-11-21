"""Loopback keepalive regression (#154910); receipt-only recovery after loss (#141971).

The stalled server is a disposable wire fixture, not the user's gateway or a real
LLM compaction. Reconciliation uses the existing real session-authority fixture.
"""
import asyncio
import json
import queue
import threading
from types import SimpleNamespace

import pytest
from websockets.asyncio.client import connect as websocket_connect
from websockets.asyncio.server import serve
from websockets.protocol import State

from hermes_cli import gateway_client
from tests.gateway.fixtures.replay_gap_owner import Owner


def local_endpoint(origin):
    return SimpleNamespace(api_origin=origin, profile_id="owned", instance_id="owner")


@pytest.fixture
def ticket(monkeypatch):
    # Bootstrap/update recovery is outside this transport test and must not inspect the live install.
    monkeypatch.setattr("hermes_cli._early_recovery.restore_interrupted_pull", lambda root: False)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *a, **k: "fixture-ticket")


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", [
    "http://127.0.0.1:9119", "http://127.8.0.1:9119", "http://[::1]:9119",
    "http://localhost:9119", "http://LOCALHOST.:9119",
    "https://gateway.example.com", "http://192.168.1.20:9119",
    "http://localhost.example.com:9119",
])
@pytest.mark.parametrize("explicit", [False, True])
async def test_dial_preserves_remote_defaults_and_disables_only_canonical_local_pings(
    monkeypatch, ticket, origin, explicit,
):
    captured = {}

    def dial(url, **kwargs):
        captured.update(kwargs)
        raise OSError("fixture does not dial")

    monkeypatch.setattr("websockets.asyncio.client.connect", dial)
    endpoint = local_endpoint(origin)
    if explicit:
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", origin.replace("http", "ws") + "/api/ws")
    with pytest.raises(gateway_client.GatewayClientError, match="no local fallback"):
        async with gateway_client.connect_gateway(None if explicit else endpoint, max_size=12345):
            pytest.fail("a refused dial cannot yield a client")
    local = origin in {
        "http://127.0.0.1:9119", "http://127.8.0.1:9119", "http://[::1]:9119",
        "http://localhost:9119", "http://LOCALHOST.:9119",
    }
    if local and not explicit:
        assert captured["ping_interval"] is None and captured["ping_timeout"] is None
    else:
        assert "ping_interval" not in captured and "ping_timeout" not in captured
    assert captured["open_timeout"] == 10 and captured["max_size"] == 12345
    assert captured["proxy"] is None
    assert captured["subprotocols"] == (None if explicit else [
        gateway_client.GATEWAY_WS_PROTOCOL, gateway_client.GATEWAY_WS_TICKET_PREFIX + "fixture-ticket",
    ])


@pytest.mark.asyncio
async def test_discovered_owner_uses_the_same_local_policy(monkeypatch, ticket):
    from hermes_cli import gateway_runtime
    endpoint = local_endpoint("http://127.0.0.1:9119")
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", lambda *a, **k: SimpleNamespace(
        state="ready", endpoint=endpoint,
    ))
    monkeypatch.setattr(gateway_runtime, "attachable", lambda result: True)
    captured = {}

    def dial(url, **kwargs):
        captured.update(kwargs)
        raise OSError("fixture does not dial")

    monkeypatch.setattr("websockets.asyncio.client.connect", dial)
    with pytest.raises(gateway_client.GatewayClientError):
        async with gateway_client.connect_gateway():
            pytest.fail("fixture refused the dial")
    assert captured["ping_interval"] is None and captured["ping_timeout"] is None


@pytest.mark.asyncio
async def test_gateway_loop_stall_during_compaction_keeps_the_active_local_turn(monkeypatch, ticket):
    ready = queue.Queue()
    release = threading.Event()
    entered = threading.Event()
    submissions = []

    async def peer(ws):
        async for raw in ws:
            request = json.loads(raw)
            method = request["method"]
            if method == "prompt.submit":
                submissions.append(request["params"])
                result = {"admission_id": "fixture-admission", "status": "started"}
            elif method == "session.compress":
                entered.set()
                if not release.wait(10):  # intentionally stall only the fixture gateway's loop
                    raise AssertionError("test did not release the stalled gateway")
                result = {"compressed": True}
            else:
                raise AssertionError(method)
            await ws.send(json.dumps({"id": request["id"], "result": result}))

    async def server_main():
        async with serve(peer, "127.0.0.1", 0, ping_interval=None,
                         subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as server:
            stopped = asyncio.Event()
            ready.put((next(iter(server.sockets)).getsockname()[1], asyncio.get_running_loop(), stopped))
            await stopped.wait()

    worker = threading.Thread(target=lambda: asyncio.run(server_main()), name="keepalive-test-server")
    worker.start()
    port, loop, stopped = await asyncio.to_thread(ready.get, True, 10)

    def accelerated_connect(*args, **kwargs):
        # Same library and caller: shorten only unspecified defaults for a fast contrasting control.
        kwargs.setdefault("ping_interval", 0.05)
        kwargs.setdefault("ping_timeout", 0.05)
        kwargs.setdefault("close_timeout", 0.05)
        return websocket_connect(*args, **kwargs)

    monkeypatch.setattr("websockets.asyncio.client.connect", accelerated_connect)
    compacting = None
    try:
        async with gateway_client.connect_gateway(local_endpoint(f"http://127.0.0.1:{port}")) as client:
            async with websocket_connect(f"ws://127.0.0.1:{port}", proxy=None,
                                         ping_interval=0.05, ping_timeout=0.05, close_timeout=0.05,
                                         subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as control:
                assert (await client.rpc("prompt.submit", session_id="s", text="research"))["status"] == "started"
                assert control.state is State.OPEN
                compacting = asyncio.create_task(client.rpc(
                    "session.compress", session_id="s", _timeout=gateway_client.COMPRESS_RPC_TIMEOUT,
                ))
                assert await asyncio.to_thread(entered.wait, 5)
                await asyncio.wait_for(control.wait_closed(), 5)
                assert control.protocol.close_sent.reason == "keepalive ping timeout"
                assert client.websocket.state is State.OPEN and not client.reader.done()
                assert not compacting.done()
                release.set()
                assert await asyncio.wait_for(compacting, 5) == {"compressed": True}
                assert client.websocket.state is State.OPEN
                assert submissions == [{"session_id": "s", "text": "research"}]
    finally:
        release.set()
        if compacting is not None:
            compacting.cancel()
            await asyncio.gather(compacting, return_exceptions=True)
        loop.call_soon_threadsafe(stopped.set)
        await asyncio.to_thread(worker.join, 10)
        assert not worker.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", ["lost_ack", "active_turn"])
async def test_genuine_loss_resumes_the_existing_receipt_without_resubmission(tmp_path, ticket, loss, monkeypatch):
    from gateway import session_settlement_recovery
    from tests.gateway.fixtures.replay_gap_owner import FINAL
    finish = threading.Event()
    original_commit = session_settlement_recovery.commit_and_publish

    def commit(*args, **kwargs):
        assert finish.wait(10), "test did not release the active turn"
        return original_commit(*args, **kwargs)

    if loss == "active_turn":
        monkeypatch.setattr(session_settlement_recovery, "commit_and_publish", commit)
    owner = Owner(tmp_path / "state.db", pause=None)
    admissions = []

    async def peer(ws):
        async for raw in ws:
            request = json.loads(raw)
            result = await owner.rpc(request["method"], **request["params"])
            if request["method"] == "prompt.submit":
                admissions.append(result["admission_id"])
                if loss == "lost_ack":
                    await asyncio.to_thread(owner.turns[-1].join, 5)
                    assert not owner.turns[-1].is_alive()
                    ws.transport.abort()
                    return
                await ws.send(json.dumps({"id": request["id"], "result": result}))
                ws.transport.abort()
                return
            await ws.send(json.dumps({"id": request["id"], "result": result}))

    try:
        async with serve(peer, "127.0.0.1", 0, ping_interval=None,
                         subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as server:
            endpoint = local_endpoint(f"http://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}")
            async with gateway_client.connect_gateway(endpoint) as first:
                await first.rpc("session.resume", session_id="s")
                if loss == "lost_ack":
                    with pytest.raises(gateway_client.GatewayClientError, match="do not resend"):
                        await first.rpc("prompt.submit", session_id="s", text="once", input_id="one-input")
                else:
                    await first.rpc("prompt.submit", session_id="s", text="once", input_id="one-input")
                assert isinstance(await asyncio.wait_for(first.events.get(), 5), gateway_client.GatewayClientError)
            async with gateway_client.connect_gateway(endpoint) as resumed:
                snapshot = await resumed.rpc("session.resume", session_id="s")
                assert snapshot["stored_session_id"] == "s"
                receipt = await resumed.rpc("prompt.receipt", session_id="s",
                                            admission_id=admissions[0], include_result=True)
                assert receipt["status"] == ("started" if loss == "active_turn" else "terminal")
                finish.set()
                owner.running.set()
                await asyncio.to_thread(owner.turns[-1].join, 5)
                assert not owner.turns[-1].is_alive()
                receipt = await resumed.rpc("prompt.receipt", session_id="s",
                                            admission_id=admissions[0], include_result=True)
                assert receipt["status"] == "terminal" and receipt["outcome"] == "completed"
                assert receipt["result"]["final_response"] == FINAL
                assert len(admissions) == len(owner.turns) == 1
    finally:
        finish.set()
        owner.finish()


@pytest.mark.asyncio
async def test_local_keepalive_policy_still_rejects_protocol_mismatch(ticket):
    async def peer(ws):
        await ws.wait_closed()

    async with serve(peer, "127.0.0.1", 0, ping_interval=None) as server:
        endpoint = local_endpoint(f"http://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}")
        with pytest.raises(gateway_client.GatewayClientError, match="protocol mismatch"):
            async with gateway_client.connect_gateway(endpoint):
                pytest.fail("an unauthenticated protocol cannot yield a client")
