"""``pre_gateway_dispatch`` on relay deliveries runs at admission, before the commit, but never on the
relay socket reader: a hook may await ``adapter.send()``, whose ``outbound_result`` only that reader
delivers, or wait for another inbound message. A cancelled admission releases the media its
preparation copied unless the admission write already began (that write may still commit)."""
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig


def _hook(callback):
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    manager._discovered = True
    manager._hooks.setdefault("pre_gateway_dispatch", []).append(callback)


def _inbound(message_id, chat_id="relay-chat", **fields):
    return {"text": "hello", "message_id": message_id, **fields,
            "source": {"platform": "telegram", "chat_id": chat_id, "chat_type": "dm", "user_id": "relay-user"}}


@asynccontextmanager
async def _relay(tmp_path, monkeypatch, media=None):
    """A real relay adapter, transport and reader against a fixture connector that answers every
    outbound frame. Yields ``(adapter, ws, outgoing, authority, admissions)``: ``admissions`` are the
    tasks ``admit_producer`` ran in."""
    websockets = pytest.importorskip("websockets")
    import gateway.platforms.webhook_ingress as ingress
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.media import RelayMediaClient
    from gateway.relay.ws_transport import WebSocketRelayTransport
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority

    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)  # rows stay queued
    outgoing, ready, admissions = [], asyncio.Queue(), []
    descriptor = CapabilityDescriptor(1, "telegram", "Fixture", 4096, False, True, False, "plain", "chars")
    admit = ingress.admit_producer

    async def observed(adapter, event):
        admissions.append(asyncio.current_task())
        return await admit(adapter, event)

    monkeypatch.setattr(ingress, "admit_producer", observed)

    async def connector(ws):
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "descriptor", "descriptor": json.loads(descriptor.to_json())}) + "\n")
        await ready.put(ws)
        async for line in ws:
            frame = json.loads(line)
            outgoing.append(frame)
            if frame["type"] == "outbound":
                await ws.send(json.dumps({"type": "outbound_result", "requestId": frame["requestId"],
                                          "result": {"success": True, "message_id": "sent"}}) + "\n")

    async with websockets.serve(connector, "127.0.0.1", 0) as relay:
        # A short outbound timeout bounds a send the reader cannot answer.
        transport = WebSocketRelayTransport(
            f"ws://127.0.0.1:{relay.sockets[0].getsockname()[1]}/relay", "telegram", "fixture-bot",
            gateway_id="fixture-gateway", upgrade_secret="fixture-secret", reconnect=True,
            reconnect_backoff_s=.05, outbound_timeout_s=3)
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        authority = await initialize_session_authority(runner, profile_id="default", instance_id="relay-reader")
        adapter = RelayAdapter(PlatformConfig(enabled=True), descriptor, transport)
        if media is not None:
            adapter._media_client = RelayMediaClient(f"http://127.0.0.1:{media.server_port}", None, None)
        adapter.gateway_runner = runner
        runner.adapters[Platform.RELAY] = adapter
        runner._wire_adapter_handlers(adapter)
        try:
            assert await adapter.connect()
            yield adapter, await asyncio.wait_for(ready.get(), 5), outgoing, authority, admissions
        finally:
            await adapter.disconnect()
            authority.db.close()


@asynccontextmanager
async def _media_server():
    from tests.gateway.test_native_media_redelivery import _Media

    media = ThreadingHTTPServer(("127.0.0.1", 0), _Media)
    threading.Thread(target=media.serve_forever, daemon=True).start()
    try:
        yield media, f"http://127.0.0.1:{media.server_port}/photo.png"
    finally:
        media.shutdown()
        media.server_close()


def _acked(outgoing):
    return [frame["bufferId"] for frame in outgoing if frame.get("type") == "inbound_ack"]


@pytest.mark.asyncio
async def test_a_hook_that_sends_a_reply_gets_its_outbound_result_and_the_ack_follows_the_commit(
        tmp_path, monkeypatch):
    from tests.waiting import async_wait_until

    sent = []

    async def hook(event, gateway, **_kwargs):
        sent.append(await gateway.adapters[Platform.RELAY].send(event.source.chat_id, "one moment"))

    _hook(hook)
    async with _relay(tmp_path, monkeypatch) as (_adapter, ws, outgoing, authority, _admissions):
        await ws.send(json.dumps({"type": "inbound", "event": _inbound("r1"), "bufferId": "buf-1"}) + "\n")
        await async_wait_until(lambda: _acked(outgoing), timeout=20, what="buf-1 acknowledged")
        rows = authority.db._read_all("SELECT request_id FROM session_admissions")
    assert [result.success for result in sent] == [True], "the hook's send never got its outbound_result"
    assert [row["request_id"] for row in rows] == ["r1"]
    kinds = [frame["type"] for frame in outgoing]
    assert kinds.index("outbound") < kinds.index("inbound_ack")


@pytest.mark.asyncio
async def test_a_hook_waiting_for_another_chat_does_not_block_it_and_one_chat_still_admits_in_order(
        tmp_path, monkeypatch):
    """The hook for chat A's first message waits for chat B's message: B's hook runs meanwhile, A's
    second message waits for A's first, and A's rows commit in arrival order."""
    from tests.waiting import async_wait_until

    fired, b_arrived = [], asyncio.Event()

    async def hook(event, **_kwargs):
        fired.append(event.message_id)
        if event.message_id == "a1":
            await b_arrived.wait()
        if event.message_id == "b1":
            b_arrived.set()

    async def deliver(buffer_id, message_id, chat_id):
        event = _inbound(message_id, chat_id)
        await ws.send(json.dumps({"type": "inbound", "event": event, "bufferId": buffer_id}) + "\n")

    _hook(hook)
    async with _relay(tmp_path, monkeypatch) as (adapter, ws, outgoing, authority, _admissions):
        assert getattr(adapter, "_admission_lanes", None) == {}  # declared adapter state, not added lazily
        await deliver("buf-1", "a1", "chat-a")
        await deliver("buf-2", "a2", "chat-a")
        # B is sent once A's first hook waits for it: sent together, B's lane may reach the hook first.
        await async_wait_until(lambda: fired, timeout=20, what="chat A's first hook")
        assert fired == ["a1"]
        await deliver("buf-3", "b1", "chat-b")
        await async_wait_until(lambda: len(_acked(outgoing)) == 3, timeout=20, what="all three acknowledged")
        rows = authority.db._read_all("SELECT request_id FROM session_admissions ORDER BY rowid")
        assert adapter._admission_lanes == {}  # each chat's lane is dropped once its last admission is done
    assert fired == ["a1", "b1", "a2"]
    assert [row["request_id"] for row in rows if row["request_id"].startswith("a")] == ["a1", "a2"]


@pytest.mark.asyncio
async def test_an_admission_cancelled_while_the_hook_waits_releases_its_prepared_media(tmp_path, monkeypatch):
    """A relay disconnect or shutdown cancels the admission while the hook waits: no admission owns
    the copy preparing it made, so it is released, and the cancellation still propagates."""
    from gateway.session_ingress_media import _media_root
    from tests.waiting import async_wait_until

    entered = asyncio.Event()

    async def hook(**_kwargs):
        entered.set()
        await asyncio.Event().wait()  # never answers

    _hook(hook)
    async with _media_server() as (media, url), _relay(tmp_path, monkeypatch, media) as (
            _adapter, ws, outgoing, authority, admissions):
        event = _inbound("r1", message_type="photo", media_urls=[url], media=[{"url": url, "mime": "image/png"}])
        await ws.send(json.dumps({"type": "inbound", "event": event, "bufferId": "buf-1"}) + "\n")
        await asyncio.wait_for(entered.wait(), 20)
        copied = sorted(_media_root().rglob("*.png"))
        [admission] = admissions
        admission.cancel()
        await asyncio.wait([admission])
        await async_wait_until(lambda: not list(_media_root().rglob("*.png")), timeout=10,
                               what="the prepared copy released")
        rows = authority.db._read_all("SELECT 1 FROM session_admissions")
    assert copied and admission.cancelled()
    assert (rows, _acked(outgoing)) == ([], [])


@pytest.mark.asyncio
async def test_an_admission_cancelled_after_its_write_began_keeps_the_media_the_write_commits(
        tmp_path, monkeypatch):
    """Cancelled once the shielded admission write is running, the write still commits: the bytes it
    commits stay on disk."""
    from gateway.session_authority import SessionAuthority
    from tests.waiting import async_wait_until

    writing, proceed = threading.Event(), threading.Event()
    write = SessionAuthority._admit_native_write

    def held_write(self, **kwargs):
        writing.set()
        proceed.wait(20)
        return write(self, **kwargs)

    monkeypatch.setattr(SessionAuthority, "_admit_native_write", held_write)
    _hook(lambda **_kwargs: None)
    async with _media_server() as (media, url), _relay(tmp_path, monkeypatch, media) as (
            _adapter, ws, outgoing, authority, admissions):
        event = _inbound("r1", message_type="photo", media_urls=[url], media=[{"url": url, "mime": "image/png"}])
        await ws.send(json.dumps({"type": "inbound", "event": event, "bufferId": "buf-1"}) + "\n")
        assert await asyncio.to_thread(writing.wait, 20), "the admission write never began"
        [admission] = admissions
        admission.cancel()
        await asyncio.wait([admission])
        proceed.set()
        rows = await async_wait_until(
            lambda: authority.db._read_all("SELECT payload_json FROM session_admissions"), timeout=20,
            what="the shielded write committed")
    [reference] = json.loads(rows[0]["payload_json"])["native_text_v1"]["media"]
    assert admission.cancelled() and _acked(outgoing) == []
    assert Path(reference["path"]).is_file(), "a cancellation deleted media its admission committed"
