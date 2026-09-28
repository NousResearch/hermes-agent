"""Bounded Matrix history reads for a live Matrix session."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from hermes_constants import get_hermes_home

from plugins.platforms.matrix.client_events import (
    Method, UndecryptableEvent, decrypt_history_event, raw_event,
)
from plugins.platforms.matrix.effective_event import effective_event, event_content
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reaction_context import fetch_reactions_for_events
from plugins.platforms.matrix.reply_context import MatrixEventContext, _label_body, _own_text
from plugins.platforms.matrix.room_access import RoomClientChanged, RoomClientOwner

_MESSAGE_FILTER = json.dumps({"types": ["m.room.message", "m.room.encrypted", "m.sticker"]})


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
        allowed = await self.adapter._is_allowed_matrix_room_event(
            self.room_id, owner=RoomClientOwner(self.adapter, self.client), require_classification=True,
        )
        self.check()
        if not allowed:
            raise MatrixSessionError("Matrix room is not allowed or joined")
        is_dm = await self.adapter._is_dm_room(
            self.room_id, owner=RoomClientOwner(self.adapter, self.client), require_classification=True,
        )
        self.check()
        chat_type = "dm" if is_dm else "group"
        _, _, error = _current_read_access(
            self.adapter, self.room_id, self.requester, chat_type,
        )
        if error is not None:
            raise MatrixSessionError(error["error"])
        return chat_type

    async def decrypt(self, raw: dict[str, Any]) -> dict[str, Any]:
        self.check()
        try:
            event = await decrypt_history_event(self.client, raw)
        except UndecryptableEvent as exc:
            self.check()
            raise MatrixSessionError(str(exc)) from exc
        self.check()
        if event is None:
            raise MatrixSessionError("missing decryption keys")
        return raw_event(event)

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
        chat_type = await self.admit()
        if event_type == EventType.ROOM_MESSAGE and await self.require_delivery_keys():
            content = await self.client.encrypt(RoomID(self.room_id), event_type, content)
            self.check()
            event_type = EventType.ROOM_ENCRYPTED
            chat_type = await self.admit()
        _, _, error = _current_read_access(
            self.adapter, self.room_id, self.requester, chat_type,
        )
        if error is not None:
            raise MatrixSessionError(error["error"])
        if before_request is not None:
            before_request()
        return await self.client.send_message_event(
            RoomID(self.room_id), event_type, content, disable_encryption=True,
        )


async def _visible_event(
    adapter: Any, raw: dict[str, Any], room_id: str, chat_type: str,
    *, before: MatrixEventContext | None,
) -> tuple[dict | None, dict | None, str | None]:
    event_id = raw.get("event_id")
    if raw.get("room_id", room_id) != room_id:
        return None, None, None
    cache = adapter._event_context_cache
    unsigned = raw.get("unsigned")
    if isinstance(event_id, str) and isinstance(unsigned, dict) and unsigned.get("redacted_because"):
        cache.redact(room_id, event_id)
    if MatrixRelation.from_content(event_content(raw).get("m.relates_to")).is_edit:
        return None, None, None
    retained = cache.retain(room_id, event_id) if isinstance(event_id, str) else None
    if before is None and retained is not None and not retained.text and not retained.redacted and not retained.state_error:
        before = retained
    state = await effective_event(adapter._client, raw, cache=cache, room_id=room_id)
    content = state.content
    if state.redacted and isinstance(event_id, str):
        cache.redact(room_id, event_id)
    if content is None:
        if isinstance(event_id, str) and state.error is not None:
            cache.store_resolved(room_id, event_id, MatrixEventContext(
                str(raw.get("sender") or ""), "", state_error=state.error["error"],
            ), before)
        return None, state.error, None
    if not state.redacted and not content.get("msgtype") and not state.error:
        return None, None, None
    body = content.get("body")
    if not isinstance(body, str):
        body = ""
    body = body.strip()
    text = _label_body(str(content.get("msgtype") or ""), _own_text(body, content))
    body = "[redacted]" if state.redacted else text[:1200]
    relation = MatrixRelation.from_content(state.original_content.get("m.relates_to"))
    sender = str(raw.get("sender") or "")
    authorized = sender == adapter._user_id or adapter._is_sender_authorized(
        sender, chat_type=chat_type, chat_id=room_id
    ) is True
    visible = {
        "event_id": event_id,
        "sender": sender,
        "body": body,
        "msgtype": None if state.redacted or not content.get("msgtype") else str(content.get("msgtype")),
        "thread_id": relation.thread_root,
        "timestamp": raw.get("origin_server_ts"),
        "sender_authorized": authorized,
    }
    if state.edited:
        visible["edited"] = True
    if state.redacted:
        visible["redacted"] = True
    if state.error is not None and before is not None and before.state_error:
        visible.update(body="[event content unavailable]", msgtype=None)
        visible.pop("edited", None)
    if not state.redacted and cache.history_entry(room_id, event_id) is not before:
        visible.update(body="[event content unavailable]", msgtype=None)
        visible.pop("edited", None)
        return visible, {"event_id": event_id, "error": "event content changed"}, state.replacement_id
    if isinstance(event_id, str) and not state.redacted:
        if (before is None or before.state_error or state.error or before.text != text
                or before.replacement_id != state.replacement_id
                or before.media_content != MatrixEventContext.image_content(content)):
            cache.store_resolved(room_id, event_id, MatrixEventContext(
                sender, text, is_image=content.get("msgtype") == "m.image", replacement_id=state.replacement_id,
                media_content=MatrixEventContext.image_content(content),
                state_error=state.error["error"] if state.error else None,
            ), before)
    return visible, state.error, state.replacement_id


@dataclass
class MatrixReadEvent:
    raw: dict[str, Any]
    visible: dict[str, Any] | None
    error: dict[str, str] | None
    replacement_id: str | None
    cache_entry: MatrixEventContext | None

    async def refresh(self, adapter: Any, room_id: str, chat_type: str) -> None:
        cache = adapter._event_context_cache
        event_id = self.raw["event_id"]
        current = cache.history_entry(room_id, event_id)
        if (current is self.cache_entry and not cache.is_redacted(room_id, self.replacement_id)
                and not (self.error and self.error["error"] == "event content changed")):
            return
        self.cache_entry = current
        raw = self.raw
        if current is None or not current.redacted:
            try:
                path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id, safe='')}"
                fresh = raw_event(await asyncio.wait_for(adapter._client.api.request(Method.GET, path), timeout=10.0))
            except Exception:
                fresh = {}
            if fresh.get("event_id") == event_id and fresh.get("room_id", room_id) == room_id:
                raw = fresh
            else:
                self.invalidate(
                    "replacement was redacted" if cache.is_redacted(room_id, self.replacement_id) else
                    current.state_error if current and current.state_error else "event content changed"
                )
                return
        visible, error, replacement_id = await _visible_event(adapter, raw, room_id, chat_type, before=current)
        self.raw, self.visible, self.error, self.replacement_id = raw, visible, error, replacement_id
        if error is not None and self.visible is not None:
            self.visible.update(body="[event content unavailable]", msgtype=None)
            self.visible.pop("edited", None)
        self.cache_entry = cache.history_entry(room_id, event_id)

    def invalidate(self, error: str) -> None:
        if self.visible is not None:
            self.visible.update(body="[event content unavailable]", msgtype=None)
            self.visible.pop("edited", None)
        self.error = {"event_id": self.raw["event_id"], "error": error}

    def recheck(self, adapter: Any, room_id: str, chat_type: str) -> None:
        cache = adapter._event_context_cache
        event_id = self.raw["event_id"]
        current = cache.history_entry(room_id, event_id)
        if current is not None and current.redacted:
            if self.visible is None:
                sender = str(self.raw.get("sender") or "")
                self.visible = {
                    "event_id": event_id, "sender": sender,
                    "thread_id": MatrixRelation.from_content(event_content(self.raw).get("m.relates_to")).thread_root,
                    "timestamp": self.raw.get("origin_server_ts"),
                    "sender_authorized": sender == adapter._user_id or adapter._is_sender_authorized(
                        sender, chat_type=chat_type, chat_id=room_id,
                    ) is True,
                }
            self.visible.update(body="[redacted]", msgtype=None, redacted=True)
            self.error = None
        elif cache.is_redacted(room_id, self.replacement_id):
            self.invalidate("replacement was redacted")
        elif current is not self.cache_entry:
            self.invalidate(current.state_error if current and current.state_error else "event content changed")
        else:
            return
        if self.visible is not None:
            self.visible.pop("edited", None)
            self.visible.pop("reactions", None)
            self.visible.pop("reactions_truncated", None)


def _current_read_access(
    adapter: Any, room_id: str, requester: str, chat_type: str,
    *, joined_rooms: Collection[str] | None = None,
) -> tuple[Any, str | None, dict | None]:
    membership = adapter._joined_rooms if joined_rooms is None else joined_rooms
    if room_id not in membership or not adapter._is_allowed_matrix_room(room_id, chat_type):
        return None, None, {"error": "Matrix room is not allowed or joined"}
    if adapter._is_sender_authorized(requester, chat_type=chat_type, chat_id=room_id) is not True:
        return None, None, {"error": "Matrix requester is not authorized for this room"}
    client = adapter._client
    if client is None:
        return None, None, {"error": "Matrix client is disconnected"}
    return client, chat_type, None


async def _read_access(
    adapter: Any, room_id: str, requester: str, *, joined_rooms: Collection[str] | None = None,
    require_classification: bool = False,
) -> tuple[Any, str | None, dict | None]:
    owner = RoomClientOwner(adapter, adapter._client)
    membership = adapter._joined_rooms if joined_rooms is None else joined_rooms
    if room_id not in membership:
        return None, None, {"error": "Matrix room is not allowed or joined"}
    try:
        owner.check()
        allowed = await adapter._is_allowed_matrix_room_event(
            room_id, owner=owner, require_classification=require_classification,
        )
        owner.check()
        if not allowed:
            return None, None, {"error": "Matrix room is not allowed or joined"}
        is_dm = await adapter._is_dm_room(
            room_id, owner=owner, require_classification=require_classification,
        )
        owner.check()
    except RoomClientChanged:
        return None, None, {"error": "Matrix client changed"}
    return _current_read_access(
        adapter, room_id, requester, "dm" if is_dm else "group", joined_rooms=joined_rooms,
    )


async def read_matrix_context(
    adapter: Any, kind: str, room_id: str, event_id: str | None, limit: int,
    *, requester: str,
) -> dict[str, Any]:
    client, chat_type, error = await _read_access(adapter, room_id, requester)
    if error is not None:
        return error
    try:
        access = MatrixSessionAccess.capture(adapter, room_id, requester)
        access.check()
    except MatrixSessionError as exc:
        return {"error": str(exc)}

    try:
        if kind != "room" and event_id is None:
            return {"error": "event_id is required for thread and event reads"}

        cache = adapter._event_context_cache
        cached = cache.snapshot(room_id)
        if event_id and kind in {"event", "thread"}:
            cached.setdefault(event_id, cache.retain(room_id, event_id))

        root: dict[str, Any] | None = None
        if kind == "thread":
            try:
                path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id or '', safe='')}"
                root = raw_event(await asyncio.wait_for(client.api.request(Method.GET, path), timeout=10.0))
                access.check()
                if root.get("event_id") != event_id:
                    root = None
                if root is not None:
                    for target, dependency in cache.retain_events(room_id, [root]).items():
                        cached.setdefault(target, dependency)
            except Exception:
                root = None

        access.check()
        remaining = limit - (root is not None)
        try:
            if kind == "event":
                path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id or '', safe='')}"
                raw = raw_event(await asyncio.wait_for(client.api.request(Method.GET, path), timeout=10.0))
                access.check()
                if raw.get("event_id") != event_id:
                    return {"error": "Matrix event not found in this room"}
                chunk = [raw]
            elif remaining == 0:
                chunk = []
            else:
                room = quote(room_id, safe="")
                if kind == "thread":
                    path = f"/_matrix/client/v1/rooms/{room}/relations/{quote(event_id or '', safe='')}/m.thread"
                    query = {"dir": "b", "limit": str(remaining)}
                else:
                    token = await asyncio.wait_for(client.sync_store.get_next_batch(), timeout=10.0)
                    access.check()
                    if not token:
                        return {"error": "Matrix history is unavailable until the first sync completes"}
                    path = f"/_matrix/client/v3/rooms/{room}/messages"
                    query = {"from": token, "dir": "b", "limit": str(remaining), "filter": _MESSAGE_FILTER}
                response = await asyncio.wait_for(client.api.request(Method.GET, path, query_params=query), timeout=10.0)
                access.check()
                newest_first = response.get("chunk") if isinstance(response, dict) else None
                chunk = list(reversed(newest_first[:remaining])) if isinstance(newest_first, list) else []
        except Exception as exc:
            return {"error": f"Matrix read failed: {type(exc).__name__}"}

        events: list[dict] = []
        errors: list[dict] = []
        resolved: list[MatrixReadEvent] = []
        candidates = ([root] if root is not None else []) + chunk
        returned = [raw for raw in candidates if isinstance(raw, dict)]
        skipped = len(candidates) - len(returned)
        _retained = cache.retain_events(room_id, returned)
        for raw in returned:
            visible, error, replacement_id = await _visible_event(
                adapter, raw, room_id, chat_type, before=cached.get(raw.get("event_id")),
            )
            access.check()
            if visible is None and error is None:
                skipped += 1
                continue
            relation = MatrixRelation.from_content(event_content(raw).get("m.relates_to"))
            if visible is not None and kind == "thread" and raw.get("event_id") != event_id and relation.thread_root != event_id:
                skipped += 1
                continue
            if visible is not None:
                events.append(visible)
            resolved.append(MatrixReadEvent(
                raw, visible, error, replacement_id,
                adapter._event_context_cache.history_entry(room_id, raw.get("event_id")),
            ))

        targets = [event for event in events if isinstance(event["event_id"], str) and not event.get("redacted")]
        snapshots = await fetch_reactions_for_events(
            client, room_id, [event["event_id"] for event in targets],
            limit=50 if kind == "event" else 8, cache=adapter._event_context_cache,
        )
        access.check()
        by_id = {event["event_id"]: snapshot for event, snapshot in zip(targets, snapshots)}
        events = []
        for snapshot in resolved:
            await snapshot.refresh(adapter, room_id, chat_type)
            access.check()
        for snapshot in resolved:
            snapshot.recheck(adapter, room_id, chat_type)
            if snapshot.visible is not None:
                events.append(snapshot.visible)
            if snapshot.error is not None:
                errors.append(snapshot.error)

        for event in events:
            if event.get("redacted") or event["msgtype"] is None:
                continue
            snapshot = by_id.get(event["event_id"])
            if snapshot is None:
                continue
            reactions = [
                reaction for reaction in snapshot.reactions
                if not adapter._event_context_cache.is_redacted(room_id, reaction.event_id)
            ]
            if reactions:
                event["reactions"] = [
                    reaction.to_dict(sender_authorized=(
                        reaction.sender == adapter._user_id or adapter._is_sender_authorized(
                            reaction.sender, chat_type=chat_type, chat_id=room_id,
                        ) is True
                    ))
                    for reaction in reactions
                ]
            if snapshot.truncated:
                event["reactions_truncated"] = True
            for reaction in snapshot.undecryptable:
                errors.append({
                    "event_id": event["event_id"], "reaction_event_id": reaction.event_id,
                    "error": f"reaction {reaction.error}",
                })
            if snapshot.error:
                errors.append({"event_id": event["event_id"], "error": snapshot.error})

        if kind == "event" and not events and not errors:
            return {"error": "Matrix event has no message content"}
        return {"events": events, "errors": errors, "skipped": skipped}
    except (MatrixSessionError, RoomClientChanged) as exc:
        return {"error": str(exc) or "Matrix client changed"}
