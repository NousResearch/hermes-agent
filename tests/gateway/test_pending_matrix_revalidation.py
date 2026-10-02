"""Restored native Matrix input requires current authority and original cache bytes."""

from copy import deepcopy
from unittest.mock import MagicMock
from typing import Any

import pytest
from mautrix.api import HTTPAPI
from mautrix.client.api import ClientAPI
from mautrix.types import UserID

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageType
from gateway.run import GatewayRunner
from gateway.shutdown_pending import PendingQueueSnapshot
from gateway.shutdown_pending_codec import decode_pending_event
from hermes_constants import get_hermes_home
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,state",
    [
        ("m.text", "current"),
        ("m.text", "edited"),
        ("m.emote", "current"),
        ("m.image", "current"),
        ("m.sticker", "current"),
        ("m.audio", "current"),
        ("m.image", "changed-cache"),
        ("m.image", "missing-cache"),
        ("m.image", "changed-native"),
        ("m.text", "redacted"),
        ("m.text", "left-room"),
        ("m.text", "wrong-author"),
        ("m.text", "disallowed-room"),
        ("m.text", "missing-mention"),
    ],
)
async def test_restoration_requires_current_native_authority_and_cached_identity(
    monkeypatch, kind, state
):
    monkeypatch.setenv("MATRIX_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    runner = GatewayRunner(GatewayConfig())
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@hermes:example.org",
            },
        )
    )
    runner.adapters[Platform.MATRIX] = adapter
    adapter.gateway_runner = runner
    source = adapter.build_source(
        chat_id="!room:example.org", user_id="@alice:example.org", chat_type="group"
    )
    content: dict[str, Any] = {"msgtype": kind, "body": "authored"}
    extras: dict[str, Any] = {}
    paths = []
    if kind in {"m.image", "m.sticker", "m.audio"}:
        path = get_hermes_home() / "cache" / "native.dat"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"original attachment")
        paths = [str(path)]
        content.update({
            "url": "mxc://example.org/original",
            "info": {"mimetype": "audio/ogg" if kind == "m.audio" else "image/png"},
        })
        extras = {
            "media_msgtype": kind,
            "media_urls": paths,
            "media_types": [content["info"]["mimetype"]],
            "message_type": MessageType.VOICE
            if kind == "m.audio"
            else MessageType.PHOTO,
        }
    event = await adapter._build_inbound_event(
        source.chat_id,
        "@alice:example.org",
        "$original",
        "authored",
        content,
        {},
        ctx=("authored", False, "group", None, "Alice", False, source),
        **extras,
    )
    assert event is not None
    assert adapter._canonicalize(source) is not None
    restored = decode_pending_event(
        PendingQueueSnapshot.capture(
            runner._session_key_for_source(source), [event]
        ).events[0],
        adapter=adapter,
    )
    current: dict[str, Any] = {
        "event_id": "$original",
        "room_id": source.chat_id,
        "sender": source.user_id,
        "type": "m.sticker" if kind == "m.sticker" else "m.room.message",
        "content": deepcopy(content),
    }
    if state == "edited":
        current["unsigned"] = {
            "m.relations": {
                "m.replace": {
                    "event_id": "$edit",
                    "sender": source.user_id,
                    "type": "m.room.message",
                    "content": {
                        "m.relates_to": {
                            "rel_type": "m.replace",
                            "event_id": "$original",
                        },
                        "m.new_content": {
                            "msgtype": "m.text",
                            "body": "edited authored",
                        },
                    },
                }
            }
        }
    if state == "redacted":
        current["content"] = {}
        current["unsigned"] = {"redacted_because": {"event_id": "$redaction"}}
    if state == "wrong-author":
        current["sender"] = "@other:example.org"
    if state == "changed-native":
        current["content"]["url"] = "mxc://example.org/replaced"
    if state == "changed-cache":
        path.write_bytes(b"different cached bytes")
    if state == "missing-cache":
        path.unlink()
    if state == "disallowed-room":
        adapter._allowed_rooms = {"!different:example.org"}
    if state == "missing-mention":
        adapter._require_mention = True
    api = HTTPAPI(
        "https://matrix.example.org", token="test", client_session=MagicMock()
    )

    async def request(method, request_path, **kwargs):
        url = str(request_path)
        if "/event/" in url:
            return current
        if "/state/m.room.member/" in url:
            return {
                "membership": "leave" if state == "left-room" else "join",
                "displayname": "Alice",
            }
        if "/joined_members" in url:
            return {
                "joined": {
                    "@hermes:example.org": {},
                    "@alice:example.org": {},
                    "@third:example.org": {},
                }
            }
        return {}

    monkeypatch.setattr(api, "request", request)
    adapter._client = ClientAPI(UserID("@hermes:example.org"), api=api)
    verified = await adapter.revalidate_pending_event(restored)
    expected = None
    if state in {"current", "edited"}:
        expected = (
            "edited authored" if state == "edited" else event.text,
            paths,
            source.chat_id,
            source.user_id,
            event.message_type,
            {"msgtype": kind, "body": "edited authored"}
            if state == "edited"
            else content,
        )
    actual = (
        None
        if verified is None
        else (
            verified.text,
            verified.media_urls,
            verified.source.chat_id,
            verified.source.user_id,
            verified.message_type,
            verified.raw_message,
        )
    )
    assert actual == expected
