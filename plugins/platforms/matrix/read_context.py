"""Bounded Matrix history reads for a live Matrix session."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote

from hermes_constants import get_hermes_home

from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reply_context import _effective_content, _label_body, _own_text

try:
    from mautrix.api import Method
except ImportError:
    class Method(str, Enum):
        GET = "GET"


def _raw_event(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    serialize = getattr(event, "serialize", None)
    return serialize() if callable(serialize) else {}


class MatrixSessionError(Exception):
    def __init__(self, message: str, event_id: str | None = None):
        super().__init__(message)
        self.event_id = event_id


@dataclass(frozen=True)
class MatrixSessionAccess:
    adapter: Any
    client: Any
    room_id: str
    requester: str
    user_id: str
    crypto: Any
    store_dir: Path | None
    owner_profile: str | None
    runtime_home: Path
    participation_home: Path
    client_user_id: str
    client_device_id: str | None
    api: Any
    homeserver: str
    access_token: str | None = field(repr=False)
    state_store: Any
    crypto_store: Any
    interrupted: Callable[[], bool] | None

    @classmethod
    def capture(
        cls, adapter: Any, room_id: str, requester: str, *, interrupted: Callable[[], bool] | None = None,
    ) -> MatrixSessionAccess:
        client = adapter._client
        if client is None:
            raise MatrixSessionError("Matrix client is disconnected")
        crypto = getattr(client, "crypto", None)
        api = getattr(client, "api", None)
        return cls(
            adapter=adapter, client=client, room_id=room_id, requester=requester,
            user_id=adapter._user_id, crypto=crypto,
            store_dir=getattr(adapter, "_store_dir", None),
            owner_profile=getattr(adapter, "_owner_profile", None), runtime_home=get_hermes_home(),
            participation_home=getattr(adapter, "_thread_home", get_hermes_home()),
            client_user_id=getattr(client, "mxid", adapter._user_id),
            client_device_id=getattr(client, "device_id", None), api=api,
            homeserver=str(getattr(api, "base_url", "")),
            access_token=getattr(api, "token", None), state_store=getattr(client, "state_store", None),
            crypto_store=getattr(crypto, "crypto_store", None),
            interrupted=interrupted,
        )

    def check(self, event_id: str | None = None) -> None:
        adapter = self.adapter
        if (adapter._client is not self.client or adapter._user_id != self.user_id
                or getattr(self.client, "mxid", self.user_id) != self.client_user_id
                or getattr(self.client, "device_id", None) != self.client_device_id
                or getattr(self.client, "api", None) is not self.api
                or str(getattr(self.api, "base_url", "")) != self.homeserver
                or getattr(self.api, "token", None) != self.access_token
                or getattr(self.client, "state_store", None) is not self.state_store
                or getattr(self.client, "crypto", None) is not self.crypto
                or getattr(self.crypto, "crypto_store", None) is not self.crypto_store
                or getattr(adapter, "_store_dir", None) != self.store_dir
                or getattr(adapter, "_owner_profile", None) != self.owner_profile
                or getattr(adapter, "_thread_home", self.participation_home) != self.participation_home
                or get_hermes_home() != self.runtime_home or getattr(adapter, "_closing", False)):
            raise MatrixSessionError("Matrix client ownership changed", event_id)
        if self.room_id not in adapter._joined_rooms:
            raise MatrixSessionError("Matrix room is not allowed or joined", event_id)
        if event_id is None and self.interrupted is not None and self.interrupted():
            raise MatrixSessionError("Matrix thread creation interrupted", event_id)

    async def admit(self) -> str:
        self.check()
        allowed = await self.adapter._is_allowed_matrix_room_event(self.room_id)
        self.check()
        if not allowed:
            raise MatrixSessionError("Matrix room is not allowed or joined")
        is_dm = await self.adapter._is_dm_room(self.room_id)
        self.check()
        chat_type = "dm" if is_dm else "group"
        if self.adapter._is_sender_authorized(self.requester, chat_type=chat_type, chat_id=self.room_id) is not True:
            raise MatrixSessionError("Matrix requester is not authorized for this room")
        return chat_type

    async def decrypt(self, raw: dict[str, Any]) -> dict[str, Any]:
        self.check()
        if raw.get("type") != "m.room.encrypted":
            return raw
        if self.crypto is None:
            raise MatrixSessionError("missing decryption keys")
        from mautrix.types import Event
        try:
            event = await asyncio.wait_for(self.crypto.decrypt_megolm_event(Event.deserialize(raw)), timeout=10.0)
        except Exception as exc:
            self.check()
            error = "missing decryption keys" if type(exc).__name__ == "SessionNotFound" else "decryption failed"
            raise MatrixSessionError(error) from exc
        self.check()
        if event is None:
            raise MatrixSessionError("missing decryption keys")
        return _raw_event(event)

    async def require_delivery_keys(self) -> bool:
        self.check()
        encrypted = await self.client.state_store.is_encrypted(self.room_id)
        self.check()
        if encrypted is None:
            from mautrix.errors import MNotFound
            from mautrix.types import EventType
            try:
                await asyncio.wait_for(self.client.get_state_event(self.room_id, EventType.ROOM_ENCRYPTION), timeout=10.0)
                encrypted = True
            except MNotFound:
                encrypted = False
            self.check()
        if encrypted and self.crypto is None:
            raise MatrixSessionError("missing encryption keys")
        return encrypted

    async def send_message(
        self, content: dict[str, Any], *, before_request: Callable[[], None] | None = None,
    ) -> str:
        from mautrix.types import EventType, RoomID

        event_type = EventType.ROOM_MESSAGE
        if await self.require_delivery_keys():
            content = await self.client.encrypt(RoomID(self.room_id), event_type, content)
            self.check()
            event_type = EventType.ROOM_ENCRYPTED
        await self.admit()
        if event_type == EventType.ROOM_MESSAGE and await self.require_delivery_keys():
            content = await self.client.encrypt(RoomID(self.room_id), event_type, content)
            self.check()
            event_type = EventType.ROOM_ENCRYPTED
            await self.admit()
        if before_request is not None:
            before_request()
        return await self.client.send_message_event(
            RoomID(self.room_id), event_type, content, disable_encryption=True,
        )


async def _visible_event(access: MatrixSessionAccess, raw: dict[str, Any], room_id: str, chat_type: str) -> tuple[dict | None, dict | None]:
    adapter = access.adapter
    event_id = raw.get("event_id")
    try:
        event = await access.decrypt(raw)
    except MatrixSessionError as exc:
        return None, {"event_id": event_id, "error": str(exc)}

    content, edited = _effective_content(event)
    if not content.get("msgtype"):
        return None, None
    body = content.get("body")
    if not isinstance(body, str):
        body = ""
    body = body.strip()
    if edited and body.startswith("* "):
        body = body[2:].strip()
    body = _label_body(str(content.get("msgtype")), _own_text(body))[:1200]
    relation = MatrixRelation.from_content(content.get("m.relates_to"))
    sender = str(raw.get("sender") or "")
    authorized = sender == adapter._user_id or adapter._is_sender_authorized(
        sender, chat_type=chat_type, chat_id=room_id
    ) is True
    return {
        "event_id": event_id,
        "sender": sender,
        "body": body,
        "msgtype": str(content.get("msgtype")),
        "thread_id": relation.thread_root,
        "timestamp": raw.get("origin_server_ts"),
        "sender_authorized": authorized,
    }, None


async def read_matrix_context(
    adapter: Any, kind: str, room_id: str, event_id: str | None, limit: int,
    *, requester: str,
) -> dict[str, Any]:
    try:
        access = MatrixSessionAccess.capture(adapter, room_id, requester)
        chat_type = await access.admit()
    except MatrixSessionError as exc:
        return {"error": str(exc)}
    client = access.client

    root: dict[str, Any] | None = None
    if kind == "thread":
        try:
            root = _raw_event(await asyncio.wait_for(client.get_event(room_id, event_id), timeout=10.0))
            access.check()
            if root.get("event_id") != event_id:
                root = None
        except Exception:
            root = None

    try:
        access.check()
        if kind == "event":
            raw = _raw_event(await asyncio.wait_for(client.get_event(room_id, event_id), timeout=10.0))
            chunk = [raw]
        else:
            room = quote(room_id, safe="")
            if kind == "thread":
                path = f"/_matrix/client/v1/rooms/{room}/relations/{quote(event_id or '', safe='')}/m.thread"
                query = {"dir": "b", "limit": str(limit - 1 if root is not None else limit)}
            else:
                token = await asyncio.wait_for(client.sync_store.get_next_batch(), timeout=10.0)
                access.check()
                if not token:
                    return {"error": "Matrix history is unavailable until the first sync completes"}
                path = f"/_matrix/client/v3/rooms/{room}/messages"
                query = {"from": token, "dir": "b", "limit": str(limit)}
            if kind == "thread" and root is not None and limit == 1:
                chunk = []
            else:
                response = await asyncio.wait_for(client.api.request(Method.GET, path, query_params=query), timeout=10.0)
                chunk = response.get("chunk", []) if isinstance(response, dict) else []
        access.check()
    except Exception as exc:
        return {"error": f"Matrix read failed: {type(exc).__name__}"}

    events: list[dict] = []
    errors: list[dict] = []
    for raw in ([root] if root is not None else []) + chunk[:limit - bool(root)]:
        if not isinstance(raw, dict):
            continue
        visible, error = await _visible_event(access, raw, room_id, chat_type)
        try:
            access.check()
        except MatrixSessionError as exc:
            return {"error": str(exc)}
        if error is not None:
            errors.append(error)
        if visible is None:
            continue
        if kind == "thread" and visible["event_id"] != event_id and visible["thread_id"] != event_id:
            continue
        events.append(visible)

    return {"events": events, "errors": errors}
