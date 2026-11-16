"""The Webapp listener follows saved-gateway edits another writer (native Desktop) makes to the file.

Contracts: with a relayed socket open, a rename written to the store by another writer leaves it
open; re-pointing the entry (its URL) or removing it closes it (4404) within a few polls. Desktop's write is stood in for by replacing the file whole,
as its writer does (the TS module's live proof is in the PR's FIXES notes).
"""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketDisconnect

from hermes_cli import gateway_connections as store
from hermes_cli.gateway_connections_watch import watch_store
from hermes_cli.web_routers import webapp_relay_proxy
from tests.hermes_cli import test_webapp_relay_proxy as relay_tests
from tests.waiting import wait_until

webapp, live_other = relay_tests.webapp, relay_tests.live_other
_ORIGIN, _relay_ticket = relay_tests._ORIGIN, relay_tests._relay_ticket


@pytest.fixture
def watching():
    loop = asyncio.new_event_loop()
    started, stop, held = threading.Event(), asyncio.Event(), []

    async def run():
        async with watch_store(interval=0.05) as watch:
            held.append(watch)
            started.set()
            await stop.wait()

    thread = threading.Thread(target=loop.run_until_complete, args=(run(),), daemon=True)
    thread.start()
    started.wait(10)
    yield held[0]
    loop.call_soon_threadsafe(stop.set)
    thread.join(10)


def _other_writer(edit) -> None:
    doc = json.loads(store.store_path().read_text())
    edit(doc)
    doc["revision"] += 1
    tmp = store.store_path().with_suffix(".desk")
    tmp.write_text(json.dumps(doc))
    tmp.replace(store.store_path())


def _entry(doc, connection_id):
    return next(c for c in doc["connections"] if c["id"] == connection_id)


@pytest.mark.parametrize("edit", ["re-point", "remove"])
def test_another_writers_re_point_or_remove_closes_the_relayed_socket_and_a_rename_does_not(
        live_other, watching, edit):
    served, connection_id, _seen, _stop = live_other
    url = f"ws://localhost/api/relay/{connection_id}/api/audio/speak-stream?relay_ticket={_relay_ticket(served.browser)}"
    with served.browser.websocket_connect(url, headers=_ORIGIN) as ws:
        ws.send_text(json.dumps({"text": "one"}))
        assert ws.receive_json() == {"echo": {"text": "one"}}

        _other_writer(lambda doc: _entry(doc, connection_id).update(label="Renamed on Desktop"))
        renamed = store.read_registry()["revision"]
        wait_until(lambda: watching.revision == renamed, timeout=5, what="the watcher reading the rename")
        assert [gid for _ref, _loop, gid in webapp_relay_proxy._live.values()] == [connection_id]

        if edit == "re-point":
            _other_writer(lambda doc: _entry(doc, connection_id).update(url="http://127.0.0.1:9"))
        else:
            _other_writer(lambda doc: doc.update(connections=[c for c in doc["connections"] if c["id"] != connection_id]))
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
        assert (closed.value.code, closed.value.reason) == (4404, "Gateway removed")


def test_one_entrys_failed_teardown_still_tears_down_the_others(monkeypatch):
    """An SSH host that stopped answering raises from its teardown; the next entry is still torn down."""
    from hermes_cli import gateway_connections_reach as reach
    from hermes_cli import gateway_connections_watch as watch_mod
    from hermes_cli.gateway_connections_ssh_transport import SshReachError

    closed, forgotten = [], []
    monkeypatch.setattr(webapp_relay_proxy, "close_gateway_sockets", closed.append)

    def forget(connection_id):
        forgotten.append(connection_id)
        if connection_id == "a":
            raise SshReachError("unreachable", "host stopped answering")

    monkeypatch.setattr(reach, "forget", forget)
    watch_mod._teardown([], ["a", "b"])
    assert closed == ["a", "b"]
    assert forgotten == ["a", "b"]


def test_the_webapp_lifespan_ending_closes_every_ssh_tunnel_it_opened(tmp_path, monkeypatch):
    """Stopping the listener tears down its tunnels (forward, owned remote serve, master)."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from hermes_cli import gateway_connections_ssh as ssh
    from hermes_cli.web_server_app import http_lifespan

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    torn: list[str] = []

    class Master:
        def cancel_forward(self, local_port, remote_port):
            torn.append(f"forward {local_port}->{remote_port}")

        def close(self):
            torn.append("master")

    monkeypatch.setattr(ssh, "disconnect_remote", lambda _ssh, oid: torn.append(f"serve {oid}"))
    connected = SimpleNamespace(local_port=41000, remote_port=9119)
    monkeypatch.setitem(ssh._tunnels, "box", ssh._Tunnel(Master(), "oid", connected, "box.lan", ()))

    app = FastAPI(lifespan=http_lifespan)
    app.state.ui_surface = "webapp"
    with TestClient(app):
        assert torn == []
    assert torn == ["forward 41000->9119", "serve oid", "master"]
    assert "box" not in ssh._tunnels
