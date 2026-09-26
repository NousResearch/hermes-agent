"""A stalled remote preview must not stall subsequent requests on the RPC WebSocket."""

import asyncio
import time

from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient


def test_thumbnail_timeout_does_not_block_rpc_reader(tmp_path, monkeypatch):
    from hermes_cli.config import atomic_config_write
    from tui_gateway import server
    from tui_gateway.ws import handle_ws

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Background process services are unrelated to this connection's dispatch.
    for name in ("_ensure_skin_watcher", "_ensure_lease_watcher",
                 "_start_backend_heartbeat_refresher", "_schedule_startup_orphan_sweep"):
        monkeypatch.setattr(server, name, lambda: None)
    app = FastAPI()

    @app.websocket("/rpc")
    async def rpc(ws: WebSocket):
        await handle_ws(ws)

    async def run():
        disconnected = asyncio.Event()

        async def stall(reader, writer):
            try:
                assert await reader.read() == b""
            finally:
                writer.close()
                await writer.wait_closed()
                disconnected.set()

        async with await asyncio.start_server(stall, "127.0.0.1", 0) as listener:
            endpoint = f"127.0.0.1:{listener.sockets[0].getsockname()[1]}"
            atomic_config_write(tmp_path / "config.yaml", {"bot_desktop": {
                "remote_endpoint": endpoint, "remote_allow_loopback": True, "remote_password": "secret"}})

            def exchange():
                with TestClient(app) as client, client.websocket_connect("/rpc") as ws:
                    assert ws.receive_json()["params"]["type"] == "gateway.ready"
                    started = time.monotonic()
                    ws.send_json({"jsonrpc": "2.0", "id": 1, "method": "display.thumbnail", "params": {}})
                    ws.send_json({"jsonrpc": "2.0", "id": 2, "method": "display.status", "params": {}})
                    replies = []
                    while len(replies) < 2:
                        frame = ws.receive_json()
                        if "id" in frame:
                            replies.append(frame)
                            if frame["id"] == 2:
                                status_elapsed = time.monotonic() - started
                    # Ordering, rather than a fragile wall-clock bound, proves the reader kept serving.
                    assert replies[0]["id"] == 2
                    assert replies[0]["result"]["remote"] == endpoint
                    assert replies[1]["id"] == 1
                    thumbnail_elapsed = time.monotonic() - started
                    assert status_elapsed < 5 and thumbnail_elapsed >= 8
                    error = replies[1]["error"]
                    assert error["code"] == 5300
                    assert "timed out" in error["message"] and "authentication failed" not in error["message"]
                    assert "secret" not in error["message"]

            await asyncio.wait_for(asyncio.to_thread(exchange), 30)
            await asyncio.wait_for(disconnected.wait(), 5)
    asyncio.run(run())
