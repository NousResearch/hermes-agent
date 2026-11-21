"""Every relay ``hello`` names the gateway process that sent it (``instance_id``).

A prompt's id carries an owner nonce, so after a gateway restart the new process
drops answers to prompts the old one left open. The connector cannot see the
restart on its own: a re-dial looks the same. ``instance_id`` is one opaque value
per gateway process, on every hello, unchanged across re-dials and transport
rebuilds, and new after a restart, so a connector can tell the two apart.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import gateway.relay.ws_transport as wst
from gateway.relay.ws_transport import WEBSOCKETS_AVAILABLE, WebSocketRelayTransport
from tests.waiting import async_wait_until

pytestmark = pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")

if WEBSOCKETS_AVAILABLE:
    import websockets

_REPO_ROOT = Path(__file__).resolve().parents[3]
_IDENTITIES = [("discord", "appShared"), ("telegram", "bot-9")]
_DESCRIPTOR = {
    "contract_version": 1, "platform": "discord", "label": "Discord", "max_message_length": 2000,
    "supports_draft_streaming": False, "supports_edit": True, "supports_threads": True,
    "markdown_dialect": "discord", "len_unit": "chars",
}


class _HelloServer:
    """Connector stub: records every hello per connection, answers the first with a
    descriptor, and drops the first ``drops`` connections once all hellos are in
    (with ``close_code``, e.g. 4401 with no close reason)."""

    def __init__(self, drops: int = 0, close_code: int = 1000) -> None:
        self.hellos: list[list[dict]] = []
        self._drops = drops
        self._close_code = close_code
        self._server = None
        self.url = ""

    async def __aenter__(self) -> _HelloServer:
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{next(iter(self._server.sockets)).getsockname()[1]}"
        return self

    async def __aexit__(self, *_exc) -> None:
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, ws) -> None:
        hellos: list[dict] = []
        self.hellos.append(hellos)
        drop = len(self.hellos) <= self._drops
        async for raw in ws:
            for line in str(raw).split("\n"):
                frame = json.loads(line) if line.strip() else {}
                if frame.get("type") != "hello":
                    continue
                hellos.append(frame)
                if len(hellos) == 1:
                    await ws.send(json.dumps({"type": "descriptor", "descriptor": _DESCRIPTOR}) + "\n")
                if drop and len(hellos) == len(_IDENTITIES):
                    await ws.close(code=self._close_code)
                    return


def _transport(url: str, **kw) -> WebSocketRelayTransport:
    return WebSocketRelayTransport(url, "discord", "appShared", identities=_IDENTITIES, **kw)


def _instance_ids(hellos: list[dict]) -> set:
    return {h["instance_id"] for h in hellos}


@pytest.mark.asyncio
async def test_every_hello_carries_one_instance_id_that_survives_a_reconnect():
    async with _HelloServer(drops=1) as srv:
        t = _transport(srv.url, reconnect=True, reconnect_backoff_s=0.05)
        try:
            await t.connect()
            await t.handshake()
            await async_wait_until(
                lambda: len(srv.hellos) >= 2 and len(srv.hellos[1]) == len(_IDENTITIES),
                what="re-dial after the dropped connection to send its hellos",
            )
        finally:
            await t.disconnect()

    first, second = srv.hellos[0], srv.hellos[1]
    assert [h["platform"] for h in first] == [h["platform"] for h in second] == ["discord", "telegram"]
    (instance_id,) = _instance_ids(first)
    assert isinstance(instance_id, str) and instance_id
    assert _instance_ids(second) == {instance_id}


@pytest.mark.asyncio
async def test_the_fresh_token_redial_after_a_4401_keeps_the_instance_id():
    # A 4401 after the handshake without the "expired" close reason (a stale token or a
    # revoked secret) is retried at once with a freshly minted token, bypassing the 5 s
    # supervisor backoff below.
    async with _HelloServer(drops=1, close_code=4401) as srv:
        t = _transport(
            srv.url, gateway_id="gw-x", upgrade_secret="secret-x", reconnect=True, reconnect_backoff_s=5.0
        )
        try:
            await t.connect()
            await t.handshake()
            await async_wait_until(
                lambda: len(srv.hellos) >= 2 and len(srv.hellos[1]) == len(_IDENTITIES),
                timeout=3.0, what="the fresh-token re-dial to send its hellos",
            )
            assert t.auth_revoked is False
        finally:
            await t.disconnect()

    (instance_id,) = _instance_ids(srv.hellos[0])
    assert _instance_ids(srv.hellos[1]) == {instance_id}


@pytest.mark.asyncio
async def test_a_rebuilt_transport_in_the_same_process_keeps_the_instance_id():
    # The gateway's reconnect watcher builds a fresh adapter + transport in the
    # same process; that is not a restart.
    async with _HelloServer() as srv:
        for _ in range(2):
            t = _transport(srv.url)
            try:
                await t.connect()
                await t.handshake()
            finally:
                await t.disconnect()

    assert len(_instance_ids(srv.hellos[0] + srv.hellos[1])) == 1


async def _hello_instance_ids_from_one_dial() -> list[str]:
    """Dial once without a socket and return each hello's ``instance_id``."""
    sent: list[dict] = []

    async def _send(frame: dict) -> None:
        sent.append(frame)

    async def _connect(_url, **_kw):
        return SimpleNamespace()

    idle = asyncio.Event()
    t = _transport("wss://connector.example/relay")
    t._send = _send
    t._read_loop = idle.wait
    with mock.patch.object(wst, "websockets", SimpleNamespace(connect=_connect)):
        await t._dial_and_start()
    t._reader.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await t._reader
    return [f["instance_id"] for f in sent if f["type"] == "hello"]


def _instance_id_of_a_new_process() -> str:
    code = (
        "import asyncio\n"
        "from tests.gateway.relay.test_relay_hello_instance_id import _hello_instance_ids_from_one_dial\n"
        "print(*asyncio.run(_hello_instance_ids_from_one_dial()))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=_REPO_ROOT,
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.split()
    assert len(out) == len(_IDENTITIES) and len(set(out)) == 1
    return out[0]


@pytest.mark.asyncio
async def test_a_restarted_process_sends_a_new_instance_id():
    (here,) = set(await _hello_instance_ids_from_one_dial())
    restarts = {_instance_id_of_a_new_process() for _ in range(2)}
    assert len(restarts) == 2
    assert here not in restarts
