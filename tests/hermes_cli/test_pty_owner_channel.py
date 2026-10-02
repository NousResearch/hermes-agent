"""The sidebar must follow the publisher of the PTY actually reattached."""
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

import pytest

spec = importlib.util.spec_from_file_location("owner_fixture", Path(__file__).with_name("test_pty_resume_owner.py"))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
live_pty_client = fixture_module.live_pty_client
_url = fixture_module._url
pytestmark = pytest.mark.platforms("posix")


def attachment(ws):
    marker = f"attachment-sync-{uuid4().hex}\n".encode()
    output, metadata = b"", None
    sent = False
    while not sent or marker not in output:
        message = ws.receive()
        if "bytes" in message:
            output += message["bytes"]
        elif "text" in message:
            frame = json.loads(message["text"])
            if frame.get("type") == "pty.attached":
                metadata = frame
        else:
            pytest.fail(f"unexpected websocket frame: {message}")
        if not sent and b"READY\n" in output:
            ws.send_bytes(marker)
            sent = True
    assert metadata is not None, "attachment omitted the existing owner's publisher channel"
    return metadata


def channel_url(path, channel):
    return "ws://127.0.0.1" + path + "?" + urlencode({
        "token": fixture_module.web_server._SESSION_TOKEN, "channel": channel,
    })


def test_owner_channel_survives_new_chat_return_and_page_refresh(live_pty_client, monkeypatch):
    client, registry = live_pty_client
    resolve = fixture_module.chat._resolve_chat_argv_async
    fresh_ids = iter(["session-a", "session-b"])

    async def distinct_fresh_id(**kwargs):
        argv, cwd, env = await resolve(**kwargs)
        if not kwargs.get("resume"):
            env["HERMES_TUI_RESUME"] = next(fresh_ids)
        return argv, cwd, env

    monkeypatch.setattr(fixture_module.chat, "_resolve_chat_argv_async", distinct_fresh_id)
    with client.websocket_connect(_url(attach="terminal-a", tab="tab", channel="publisher-a")) as ws:
        assert attachment(ws)["channel"] == "publisher-a"
        owner = next(iter(registry._sessions.values()))
    with client.websocket_connect(_url(attach="terminal-b", tab="tab", channel="publisher-b", fresh="1")) as ws:
        assert attachment(ws)["channel"] == "publisher-b"

    for attempted_channel in ("selection-channel", "refresh-channel"):
        with client.websocket_connect(_url(attach="terminal-b", tab="tab", channel=attempted_channel, resume="session-a")) as ws:
            metadata = attachment(ws)
            assert metadata["channel"] == "publisher-a"
            assert owner.attached and len(registry._sessions) == 2
            with client.websocket_connect(channel_url("/api/events", metadata["channel"])) as subscriber:
                with client.websocket_connect(channel_url("/api/pub", "publisher-b")) as other_pub:
                    other_pub.send_text('{"type":"session.info","payload":{"title":"B-only"}}')
                with client.websocket_connect(channel_url("/api/pub", "publisher-a")) as owner_pub:
                    payload = {"type": "session.info", "payload": {"title": attempted_channel}}
                    owner_pub.send_json(payload)
                    assert subscriber.receive_json() == payload
                    boundary = {"type": "dashboard.new_session_requested", "payload": {}}
                    owner_pub.send_json(boundary)
                    assert subscriber.receive_json() == boundary
    assert not owner.attached and owner.bridge.is_alive()
