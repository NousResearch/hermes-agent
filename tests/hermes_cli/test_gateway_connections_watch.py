"""The Webapp listener follows saved-gateway edits another writer (native Desktop) makes to the file.

Contracts: with a relayed socket open, a rename written to the store by another writer leaves it
open; re-pointing the entry (its URL) or removing it closes it (4404) within a few polls. Desktop's write is stood in for by replacing the file whole,
as its writer does (the TS module's live proof is in the PR's FIXES notes).
"""
import asyncio
import json
import threading

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
