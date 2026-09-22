"""Current Matrix room state, members, permissions and pinned messages."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import quote

from plugins.platforms.matrix.client_events import Method, raw_event
from plugins.platforms.matrix.read_context import (
    MatrixReadEvent, _current_read_access, _read_access, _visible_event,
)
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache


def _content(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        content = value.get("content", value)
        return content if isinstance(content, dict) else {}
    serialize = getattr(value, "serialize", None)
    if callable(serialize):
        return _content(serialize())
    return {}


def _state_path(room_id: str, event_type: str, state_key: str = "") -> str:
    return (
        f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}"
        f"/state/{quote(event_type, safe='')}/{quote(state_key, safe='')}"
    )


def _is_missing_state(exc: Exception) -> bool:
    return getattr(exc, "errcode", None) == "M_NOT_FOUND" or type(exc).__name__ == "MNotFound"


async def _state_event(
    context: _InspectionContext, event_type: str, query: dict[str, str] | None = None,
) -> Any:
    # Read raw JSON instead of using mautrix's get_state_event. Its typed
    # contents fill in mautrix defaults, which differ from the spec for
    # `invite`, and it raises when a server ignores `format=event` and
    # returns the content only.
    path = _state_path(context.room_id, event_type)
    return await context.request(
        lambda: context.client.api.request(Method.GET, path, query_params=query),
    )


async def _state(context: _InspectionContext, event_type: str) -> dict[str, Any] | None:
    try:
        value = await _state_event(context, event_type)
    except Exception as exc:
        if _is_missing_state(exc):
            return None
        raise
    return value if isinstance(value, dict) else {}


def _text(content: dict[str, Any], field: str) -> str | None:
    value = content.get(field)
    return value[:1200] if isinstance(value, str) and value else None


def _numeric_level(value: Any, default: int, legacy_strings: bool) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if legacy_strings and isinstance(value, str) and re.fullmatch(r"\s*[+-]?[0-9]+\s*", value):
        return int(value)
    return default


def _level(content: dict[str, Any], key: str, default: int, legacy_strings: bool) -> int:
    return _numeric_level(content.get(key), default, legacy_strings)


def _user_level(content: dict[str, Any], user_id: str, legacy_strings: bool) -> int:
    users = content.get("users")
    users = users if isinstance(users, dict) else {}
    default = _level(content, "users_default", 0, legacy_strings)
    return _numeric_level(users.get(user_id), default, legacy_strings)


@dataclass(frozen=True)
class _RoomCreate:
    numeric_version: int | None
    creators: frozenset[str]
    creators_known: bool

    @classmethod
    def parse(cls, response: Any) -> _RoomCreate:
        response = response if isinstance(response, dict) else {}
        full_event = response.get("type") == "m.room.create" and isinstance(response.get("content"), dict)
        content = response["content"] if full_event else response
        room_version = _text(content, "room_version") or "1"
        numeric_version = int(room_version) if room_version.isdecimal() else None
        creator = response.get("sender") if full_event else None
        if creator is None and numeric_version is not None and numeric_version < 11:
            creator = content.get("creator")
        creators = {creator} if isinstance(creator, str) else set()
        additional = content.get("additional_creators")
        if numeric_version is not None and numeric_version >= 12 and isinstance(additional, list):
            creators.update(value for value in additional if isinstance(value, str))
        return cls(numeric_version, frozenset(creators), isinstance(creator, str))

    @property
    def legacy_string_levels(self) -> bool:
        return self.numeric_version is not None and self.numeric_version <= 9

    @property
    def creator_override(self) -> bool:
        return self.numeric_version is not None and self.numeric_version >= 12

    def is_creator(self, user_id: str) -> bool | None:
        if user_id in self.creators:
            return True
        return False if self.creators_known else None


def _user_permissions(
    user_id: str, power: dict[str, Any] | None, create: _RoomCreate,
) -> tuple[int | None, bool | None]:
    is_creator = create.is_creator(user_id)
    if create.creator_override:
        return _user_level(power or {}, user_id, create.legacy_string_levels), is_creator
    if power is not None:
        return _user_level(power, user_id, create.legacy_string_levels), False
    if is_creator is None:
        return None, False
    return (100 if is_creator else 0), False


async def _permissions(context: _InspectionContext) -> dict[str, Any]:
    encryption = await _state(context, "m.room.encryption") or {}
    create = _RoomCreate.parse(await _state_event(context, "m.room.create", {"format": "event"}))
    power = await _state(context, "m.room.power_levels")
    return _permission_levels(context.requester, context.owner.bot_id, power, encryption, create)


def _permission_levels(
    requester: str, bot: str, power: dict[str, Any] | None, encryption: dict[str, Any], create: _RoomCreate,
) -> dict[str, Any]:
    legacy_strings = create.legacy_string_levels
    levels = power or {}
    events = levels.get("events")
    events = events if isinstance(events, dict) else {}
    pin_level = _numeric_level(
        events.get("m.room.pinned_events"),
        _level(levels, "state_default", 0 if power is None else 50, legacy_strings), legacy_strings,
    )
    send_event_type = "m.room.encrypted" if _text(encryption, "algorithm") else "m.room.message"
    message_level = _numeric_level(
        events.get(send_event_type),
        _level(levels, "events_default", 0, legacy_strings), legacy_strings,
    )
    requester_level, requester_override = _user_permissions(requester, power, create)
    bot_level, bot_override = _user_permissions(bot, power, create)
    bot_can_edit_pins: bool | None = False
    if bot_override or (bot_level is not None and bot_level >= pin_level):
        bot_can_edit_pins = True
    elif bot_level is None or bot_override is None:
        bot_can_edit_pins = None
    return {
        "requester": {"user_id": requester, "level": requester_level,
                      "creator_override": requester_override},
        "bot": {"user_id": bot, "level": bot_level, "creator_override": bot_override},
        "required": {
            "send_message": message_level,
            "send_event_type": send_event_type,
            "edit_pins": pin_level,
            "invite": _level(levels, "invite", 0, legacy_strings),
            "kick": _level(levels, "kick", 50, legacy_strings),
            "ban": _level(levels, "ban", 50, legacy_strings),
            "redact_other": _level(levels, "redact", 50, legacy_strings),
            "send_redaction": _numeric_level(
                events.get("m.room.redaction"), _level(levels, "events_default", 0, legacy_strings), legacy_strings,
            ),
        },
        "bot_can_edit_pins": bot_can_edit_pins,
    }


class _InspectionRejected(Exception):
    def __init__(self, error: dict[str, str]) -> None:
        self.error = error
        super().__init__(error["error"])


@dataclass(frozen=True)
class _InspectionOwner:
    adapter: Any
    client: Any
    cache: MatrixEventContextCache | None
    bot_id: str
    account_id: str
    device_id: str
    api: Any
    api_url: str
    api_token: str | None
    http_session: Any
    crypto: Any
    crypto_store: Any
    store_dir: Any

    @classmethod
    def capture(cls, adapter: Any) -> _InspectionOwner:
        client = adapter._client
        api = getattr(client, "api", None)
        crypto = getattr(client, "crypto", None)
        return cls(
            adapter, client, getattr(adapter, "_event_context_cache", None), str(adapter._user_id or ""),
            str(getattr(client, "mxid", "") or ""), str(getattr(client, "device_id", "") or ""),
            api, str(getattr(api, "base_url", "") or ""), getattr(api, "token", None),
            getattr(api, "session", None), crypto, getattr(crypto, "crypto_store", None),
            getattr(adapter, "_store_dir", None),
        )

    def check(self) -> None:
        if getattr(self.adapter, "_closing", False) or self.adapter._client is None:
            raise _InspectionRejected({"error": "Matrix client is disconnected"})
        current = self.capture(self.adapter)
        if (
            current.client is not self.client or current.cache is not self.cache
            or current.api is not self.api or current.http_session is not self.http_session
            or current.crypto is not self.crypto or current.crypto_store is not self.crypto_store
            or current.bot_id != self.bot_id or current.account_id != self.account_id
            or current.device_id != self.device_id or current.api_url != self.api_url
            or current.api_token != self.api_token or current.store_dir != self.store_dir
        ):
            raise _InspectionRejected({"error": "Matrix room inspection context changed"})

    async def access(self, room_id: str, requester: str, chat_type: str | None = None) -> str:
        self.check()
        client, current_chat_type, error = await asyncio.wait_for(
            _read_access(self.adapter, room_id, requester), timeout=10.0,
        )
        self.check()
        if error is None:
            assert current_chat_type is not None
            client, current_chat_type, error = _current_read_access(
                self.adapter, room_id, requester, current_chat_type,
            )
        if error is not None:
            raise _InspectionRejected(error)
        if client is not self.client or (chat_type is not None and current_chat_type != chat_type):
            raise _InspectionRejected({"error": "Matrix room inspection context changed"})
        assert current_chat_type is not None
        return current_chat_type


@dataclass(frozen=True)
class _InspectionContext:
    adapter: Any
    client: Any
    room_id: str
    chat_type: str
    requester: str
    limit: int
    owner: _InspectionOwner
    dependencies: dict[str, MatrixEventContext] = field(default_factory=dict, compare=False)

    async def check_access(self) -> None:
        await self.owner.access(self.room_id, self.requester, self.chat_type)

    async def request(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        await self.check_access()
        try:
            result = await asyncio.wait_for(operation(), timeout=10.0)
        except Exception:
            await self.check_access()
            raise
        await self.check_access()
        return result


async def _pinned_event(context: _InspectionContext, event_id: str) -> MatrixReadEvent:
    cache = context.owner.cache
    assert cache is not None
    before = context.dependencies[event_id]
    raw = {"event_id": event_id, "room_id": context.room_id}
    try:
        await context.check_access()
        path = (
            f"/_matrix/client/v3/rooms/{quote(context.room_id, safe='')}"
            f"/event/{quote(event_id, safe='')}"
        )
        raw = raw_event(await asyncio.wait_for(context.client.api.request(Method.GET, path), timeout=10.0))
        context.dependencies.update(cache.retain_events(context.room_id, [raw]))
        await context.check_access()
    except _InspectionRejected:
        raise
    except Exception as exc:
        return MatrixReadEvent(raw, None, {
            "event_id": event_id, "error": f"Matrix event read failed: {type(exc).__name__}",
        }, None, cache.history_entry(context.room_id, event_id))
    if raw.get("event_id") != event_id:
        return MatrixReadEvent({"event_id": event_id, "room_id": context.room_id}, None, {
            "event_id": event_id, "error": "Matrix event was not returned",
        }, None, cache.history_entry(context.room_id, event_id))
    visible, error, replacement_id = await _visible_event(
        context.adapter, raw, context.room_id, context.chat_type, before=before,
    )
    await context.check_access()
    if visible is None and error is None:
        error = {"event_id": event_id, "error": "event has no visible message"}
    return MatrixReadEvent(raw, visible, error, replacement_id, cache.history_entry(context.room_id, event_id))


async def _inspect_state(context: _InspectionContext) -> dict[str, Any]:
    fields = {
        "name": ("m.room.name", "name"),
        "topic": ("m.room.topic", "topic"),
        "canonical_alias": ("m.room.canonical_alias", "alias"),
        "join_rule": ("m.room.join_rules", "join_rule"),
        "history_visibility": ("m.room.history_visibility", "history_visibility"),
        "encryption": ("m.room.encryption", "algorithm"),
    }
    result = {"room_id": context.room_id}
    for key, (event_type, field) in fields.items():
        result[key] = _text(await _state(context, event_type) or {}, field)
    return result


async def _inspect_members(context: _InspectionContext) -> dict[str, Any]:
    profiles = await context.request(lambda: context.client.get_joined_members(context.room_id))
    members = []
    for user_id, profile in sorted(profiles.items(), key=lambda item: str(item[0]))[:context.limit]:
        content = _content(profile)
        display_name = _text(content, "displayname") or getattr(profile, "displayname", None)
        avatar_url = _text(content, "avatar_url") or getattr(profile, "avatar_url", None)
        members.append({
            "user_id": str(user_id),
            "display_name": str(display_name)[:1200] if display_name else None,
            "avatar_url": str(avatar_url)[:1200] if avatar_url else None,
        })
    return {"members": members, "total": len(profiles), "truncated": len(profiles) > context.limit}


async def _inspect_permissions(context: _InspectionContext) -> dict[str, Any]:
    return await _permissions(context)


async def _inspect_pins(context: _InspectionContext) -> dict[str, Any]:
    cache = context.owner.cache
    assert cache is not None
    cached = cache.snapshot(context.room_id)
    pinned = await _state(context, "m.room.pinned_events") or {}
    event_ids = pinned.get("pinned")
    if not isinstance(event_ids, list) or not all(isinstance(value, str) for value in event_ids):
        event_ids = []
    selected = event_ids[:context.limit]
    for event_id in selected:
        context.dependencies[event_id] = cached.get(event_id) or cache.retain(context.room_id, event_id)
    semaphore = asyncio.Semaphore(10)

    async def fetch(event_id: str) -> MatrixReadEvent:
        async with semaphore:
            return await _pinned_event(context, event_id)

    resolved = await asyncio.gather(*(fetch(event_id) for event_id in selected))
    for snapshot in resolved:
        await context.check_access()
        await snapshot.refresh(context.adapter, context.room_id, context.chat_type)
    await context.check_access()
    for snapshot in resolved:
        snapshot.recheck(context.adapter, context.room_id, context.chat_type)
    return {
        "events": [snapshot.visible for snapshot in resolved if snapshot.visible is not None],
        "total": len(event_ids), "truncated": len(event_ids) > context.limit,
        "errors": [snapshot.error for snapshot in resolved if snapshot.error is not None],
    }


_INSPECTION_HANDLERS: dict[str, Callable[[_InspectionContext], Awaitable[dict[str, Any]]]] = {
    "state": _inspect_state,
    "members": _inspect_members,
    "permissions": _inspect_permissions,
    "pins": _inspect_pins,
}


async def inspect_matrix_room(
    adapter: Any, kind: str, room_id: str, limit: int, *, requester: str,
) -> dict[str, Any]:
    handler = _INSPECTION_HANDLERS.get(kind)
    if handler is None:
        return {"error": "kind must be state, members, permissions, or pins"}
    owner = _InspectionOwner.capture(adapter)
    try:
        chat_type = await owner.access(room_id, requester)
        context = _InspectionContext(adapter, owner.client, room_id, chat_type, requester, limit, owner)
        return await handler(context)
    except _InspectionRejected as exc:
        return exc.error
    except Exception as exc:
        return {"error": f"Matrix room inspection failed: {type(exc).__name__}"}


def _is_event_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("$") and len(value) > 1


@dataclass(frozen=True)
class _PinChange:
    context: _InspectionContext
    action: str
    event_id: str


async def change_matrix_pin(
    adapter: Any, action: str, room_id: str, event_id: str, *, requester: str,
    interrupt_check: Callable[[], bool], before_write: Callable[[], None],
    recheck_before_write: Callable[[], Awaitable[None]] | None = None, expected_client: Any = None,
) -> dict[str, Any]:
    owner = _InspectionOwner.capture(adapter)
    try:
        chat_type = await owner.access(room_id, requester)
    except _InspectionRejected as exc:
        return exc.error
    if expected_client is not None and owner.client is not expected_client:
        return {"error": "Matrix session or client ownership changed"}
    if action not in {"pin", "unpin"}:
        return {"error": "action must be pin or unpin"}

    context = _InspectionContext(adapter, owner.client, room_id, chat_type, requester, 0, owner)
    async with adapter._pin_state_lock:
        return await _change_pin_state(
            _PinChange(context, action, event_id), interrupt_check, before_write, recheck_before_write,
        )


async def _change_pin_state(
    change: _PinChange, interrupt_check: Callable[[], bool], before_write: Callable[[], None],
    recheck_before_write: Callable[[], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    context, event_id = change.context, change.event_id
    if interrupt_check():
        return {"error": "Matrix pin update interrupted"}

    dispatched = False
    try:
        state = await _state(context, "m.room.pinned_events") or {}
        permissions = await _permissions(context)
        actor, required = permissions["requester"], permissions["required"]["edit_pins"]
        if not actor["creator_override"] and (actor["level"] is None or actor["level"] < required):
            return {"error": "Matrix requester lacks permission to change pins",
                    "required": required, "level": actor["level"]}

        pinned = state.get("pinned")
        if pinned is None:
            pinned = []
        if not isinstance(pinned, list):
            return {"error": "Matrix pinned events state is invalid"}
        updated = list(pinned)
        if change.action == "pin" and event_id not in updated:
            updated.append(event_id)
        if change.action == "unpin":
            updated = [value for value in updated if value != event_id]
        event_ids = [value for value in updated if _is_event_id(value)]
        if updated == pinned:
            return {"pinned": event_ids, "unchanged": True}

        if recheck_before_write is not None:
            await recheck_before_write()
        if interrupt_check():
            return {"error": "Matrix pin update interrupted"}

        before_write()
        dispatched = True
        state_event_id = await asyncio.wait_for(
            context.client.send_state_event(context.room_id, "m.room.pinned_events", {**state, "pinned": updated}),
            timeout=10.0,
        )
    except _InspectionRejected as exc:
        return exc.error
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        errcode = getattr(exc, "errcode", None)
        if errcode == "M_FORBIDDEN":
            return {"error": "Matrix pin update was rejected", "errcode": "M_FORBIDDEN",
                    "message": str(exc)}
        if dispatched and errcode is None:
            return {"error": "Matrix pin update failed after the change was sent to the homeserver",
                    "outcome": "unknown",
                    "next_step": "Read the current pins with matrix_read kind=pins before retrying"}
        return {"error": f"Matrix pin update failed: {type(exc).__name__}"}

    return {"pinned": event_ids, "state_event_id": str(state_event_id)}
