"""Private coordination transport and authenticated ingress (issue #51532)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import re
import time

import pytest

pytest.importorskip("mautrix")
from mautrix.types import Event, EventType

from gateway.config import PlatformConfig
from gateway.conversation import ConversationMiddleware
from plugins.platforms.matrix.adapter import MatrixAdapter


KIND = "org.hermes.groupchat.coordination"
ROOM = "!room:example.test"
PEER = "@peer:example.test"
OWN = "@agent:example.test"


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Transport:
    def __init__(self, **kwargs):
        self.mxid = kwargs.get("mxid", OWN)
        self.handlers = {}
        self.sent = []
        self.fail = False

    def add_dispatcher(self, dispatcher):
        pass

    def add_event_handler(self, kind, handler, **kwargs):
        self.handlers[kind] = handler

    async def send_message_event(self, room_id, kind, content):
        def validate(value):
            assert not isinstance(value, float), "Matrix canonical JSON forbids floats"
            if isinstance(value, dict):
                for child in value.values():
                    validate(child)
            if isinstance(value, list):
                for child in value:
                    validate(child)
        validate(content)
        if self.fail:
            raise OSError("transport unavailable")
        self.sent.append((room_id, kind, content))
        return "$sent"


def event(sender=PEER, room=ROOM, content=None, timestamp=None):
    return Event.deserialize({
        "type": KIND, "room_id": room, "sender": sender,
        "event_id": "$coordination", "origin_server_ts": timestamp if timestamp is not None else int(time.time() * 1000),
        "content": content if content is not None else {"kind": "claim", "message_id": "$source"},
    })


def setup_adapter(monkeypatch, handlers):
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("MATRIX_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("MATRIX_ALLOWED_ROOMS", raising=False)
    monkeypatch.delenv("MATRIX_IGNORE_USER_PATTERNS", raising=False)
    adapter = MatrixAdapter(PlatformConfig(extra={
        "user_id": OWN, "homeserver": "https://example.test", "e2ee_mode": "off",
        "allowed_users": PEER, "allowed_rooms": ROOM,
    }))
    adapter._handle_message_after_conversation = AsyncMock()
    monkeypatch.setattr("hermes_cli.plugins.invoke_middleware", lambda *a, **kw: handlers)
    adapter._conversation_middleware = ConversationMiddleware(adapter)
    adapter._joined_rooms.add(ROOM)
    return adapter


def handler(callback=None):
    result = SimpleNamespace(bind=lambda dispatch: None, receive=AsyncMock())
    if callback is not None:
        result.signal = callback
    return result


@pytest.mark.anyio
async def test_custom_event_roundtrip_uses_authenticated_sender_and_never_chat(monkeypatch):
    received = AsyncMock()
    consumer = handler(received)
    adapter = setup_adapter(monkeypatch, [consumer])
    monkeypatch.setattr("mautrix.client.Client", Transport)
    monkeypatch.setattr("plugins.platforms.matrix.adapter._create_matrix_session", lambda *a: None)
    adapter._connect_authenticate = AsyncMock(return_value=True)
    adapter._connect_initial_sync = AsyncMock()
    adapter._sync_loop = AsyncMock()
    adapter._wire_plugin_handlers = lambda client: None
    assert await adapter.connect()
    await adapter._sync_task
    transport = adapter._client
    payload = {"kind": "claim", "message_id": "$source", "sender": "@forged:example.test",
               "started": time.time(),
               "nested": {"items": [{"value": 1}]}}
    assert await adapter.conversation_send_signal(ROOM, payload) is True
    room, kind, content = transport.sent[0]
    assert room == ROOM and kind == EventType.find(KIND)
    assert kind != EventType.ROOM_MESSAGE
    assert content == {**payload, "started": str(payload["started"])}
    # Real mautrix deserialization yields nested Obj/List, not plain dicts.
    # Syncer normalizes timeline event classes before selecting a handler.
    incoming = event(content=content)
    incoming.type = incoming.type.with_class(EventType.Class.MESSAGE)
    await transport.handlers[incoming.type](incoming)
    received.assert_awaited_once_with(ROOM, PEER, payload)
    assert type(received.await_args.args[2]) is dict
    assert type(received.await_args.args[2]["nested"]) is dict
    consumer.receive.assert_not_awaited()
    adapter._handle_message_after_conversation.assert_not_awaited()
    transport.fail = True
    assert await adapter.conversation_send_signal(ROOM, payload) is False


@pytest.mark.anyio
async def test_acl_joined_room_gates_and_optional_callback_failure_isolation(monkeypatch):
    received = AsyncMock()
    broken = AsyncMock(side_effect=RuntimeError("plugin failed"))
    sync_received = []
    adapter = setup_adapter(monkeypatch, [
        handler(), handler(broken), handler(lambda *args: sync_received.append(args)), handler(received),
    ])
    adapter._client = Transport()
    adapter._dm_rooms["!blocked:example.test"] = False
    adapter._joined_rooms.add("!blocked:example.test")
    for rejected in (event(sender="@stranger:example.test"), event(sender=OWN),
                     event(room="!unjoined:example.test"), event(room="!blocked:example.test"),
                     event(timestamp=int((time.time() - 180) * 1000)), event(timestamp=0)):
        await adapter._on_coordination_signal(rejected)
    adapter._startup_ts = time.time()
    await adapter._on_coordination_signal(event(timestamp=int((time.time() - 10) * 1000)))
    adapter._ignored_user_patterns = [re.compile(re.escape(PEER))]
    await adapter._on_coordination_signal(event())
    adapter._ignored_user_patterns = []
    await adapter._on_coordination_signal(SimpleNamespace(
        room_id=ROOM, sender=PEER, timestamp=time.time(), content=[]))
    received.assert_not_awaited()
    broken.assert_not_awaited()
    await adapter._on_coordination_signal(event())
    received.assert_awaited_once_with(ROOM, PEER, {"kind": "claim", "message_id": "$source"})
    assert sync_received == [received.await_args.args]
    # A room leaving sync must revoke admission even if it remains allowlisted.
    await adapter._absorb_sync(adapter._client, {"rooms": {"leave": {ROOM: {}}}})
    await adapter._on_coordination_signal(event())
    assert received.await_count == 1
    assert await adapter.conversation_send_signal(ROOM, {}) is False
    assert adapter._client.sent == []
    adapter._handle_message_after_conversation.assert_not_awaited()
