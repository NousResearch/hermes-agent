from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from hermes_cli import web_server


class Recorder:
    def __init__(self):
        self.sent: list[str] = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)


def _app():
    return SimpleNamespace(state=SimpleNamespace())


def test_runtime_channel_registry_tracks_active_publishers_subscribers_and_profiles():
    app = _app()

    web_server._runtime_channel_connected(app, "chan-a", role="publisher", profile="by-nature-cto")
    web_server._runtime_channel_connected(app, "chan-a", role="subscriber")
    snapshot = web_server._runtime_channel_snapshot(app)

    assert snapshot == [
        {
            "channel": "chan-a",
            "publisher_count": 1,
            "subscriber_count": 1,
            "profiles": ["by-nature-cto"],
            "last_seen_at": snapshot[0]["last_seen_at"],
        }
    ]

    web_server._runtime_channel_disconnected(app, "chan-a", role="publisher", profile="by-nature-cto")
    after = web_server._runtime_channel_snapshot(app)
    assert after[0]["publisher_count"] == 0
    assert after[0]["subscriber_count"] == 1
    assert after[0]["profiles"] == []


def test_runtime_global_bus_receives_every_channel_frame_with_channel_metadata():
    app = _app()
    channel_sub = Recorder()
    global_sub = Recorder()
    frame = json.dumps({
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "tool.start",
            "session_id": "sid-1",
            "payload": {"name": "terminal"},
        },
    })

    async def run():
        event_channels, event_lock = web_server._get_event_state(app)
        async with event_lock:
            event_channels.setdefault("chan-a", set()).add(channel_sub)
        global_subs, global_lock = web_server._get_runtime_global_state(app)
        async with global_lock:
            global_subs.add(global_sub)
        await web_server._broadcast_event(app, "chan-a", frame)

    asyncio.run(run())

    assert channel_sub.sent == [frame]
    assert len(global_sub.sent) == 1
    relayed = json.loads(global_sub.sent[0])
    assert relayed["method"] == "event"
    assert relayed["params"]["type"] == "tool.start"
    assert relayed["params"]["channel"] == "chan-a"
    assert relayed["params"]["payload"] == {"name": "terminal"}
