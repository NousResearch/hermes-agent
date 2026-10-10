"""Bounded discovery through the receiving Matrix adapter's live client."""

from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from plugins.platforms.matrix.client_events import Method
from plugins.platforms.matrix.read_context import _read_access
from plugins.platforms.matrix.room_inspection import _content, _text
from plugins.platforms.matrix.room_access import (
    RoomClassificationUnavailable,
    RoomClientChanged,
    RoomClientOwner,
)


MAX_ROOM_CANDIDATES = 50
DISCOVERY_TIMEOUT = 30.0
_REQUEST_TIMEOUT = 10.0

DiscoveryKind = Literal["joined_rooms", "joined_spaces", "users"]
_KINDS: dict[str, DiscoveryKind] = {
    kind: kind for kind in ("joined_rooms", "joined_spaces", "users")
}


@dataclass(frozen=True)
class DiscoveryRequest:
    kind: DiscoveryKind
    limit: int
    search_term: str | None

    @classmethod
    def parse(cls, kind: str, limit: int, search_term: str | None) -> DiscoveryRequest:
        parsed_kind = _KINDS.get(kind) if isinstance(kind, str) else None
        if parsed_kind is None:
            raise ValueError("kind must be joined_rooms, joined_spaces, or users")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 50
        ):
            raise ValueError("limit must be between 1 and 50")
        term = None
        if parsed_kind == "users":
            if (
                not isinstance(search_term, str)
                or not 1 <= len(search_term.strip()) <= 200
            ):
                raise ValueError(
                    "search_term must contain between 1 and 200 characters for user discovery"
                )
            term = search_term.strip()
        return cls(parsed_kind, limit, term)


class _RoomSummary(TypedDict):
    room_id: str
    room_type: str | None
    name: str | None
    topic: str | None
    canonical_alias: str | None


class _DirectoryUser(TypedDict):
    user_id: str
    display_name: str | None
    avatar_url: str | None


@dataclass
class _DiscoveryResult:
    rooms: list[_RoomSummary] = field(default_factory=list)
    users: list[_DirectoryUser] = field(default_factory=list)
    truncated: bool = False
    errors: list[dict[str, str]] = field(default_factory=list)

    def serialize(self, kind: DiscoveryKind) -> dict[str, Any]:
        return {
            "users" if kind == "users" else "rooms": self.users
            if kind == "users"
            else self.rooms,
            "truncated": self.truncated,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class _DiscoveryContext:
    adapter: Any
    client: Any
    room_id: str
    requester: str
    request: DiscoveryRequest

    @property
    def owner(self) -> RoomClientOwner:
        return RoomClientOwner(self.adapter, self.client)

    def check(self) -> None:
        self.owner.check()


async def _state(context: _DiscoveryContext, room_id: str, event_type: str) -> dict[str, Any]:
    event = await context.adapter._read_room_state_event(room_id, event_type, owner=context.owner)
    return _content(event)


async def _room_summary(
    context: _DiscoveryContext, room_id: str, room_type: str | None
) -> _RoomSummary:
    values: dict[str, str | None] = {}
    for key, event_type, state_field in (
        ("name", "m.room.name", "name"),
        ("topic", "m.room.topic", "topic"),
        ("canonical_alias", "m.room.canonical_alias", "alias"),
    ):
        context.check()
        values[key] = _text(
            await _state(context, room_id, event_type), state_field
        )
    context.check()
    return {
        "room_id": room_id,
        "room_type": room_type,
        "name": values["name"],
        "topic": values["topic"],
        "canonical_alias": values["canonical_alias"],
    }


async def _joined_membership(
    context: _DiscoveryContext,
) -> tuple[set[str], dict[str, str] | None]:
    joined = await asyncio.wait_for(
        context.client.get_joined_rooms(), timeout=_REQUEST_TIMEOUT
    )
    context.check()
    if not isinstance(joined, list) or not all(
        isinstance(room, str) and room.startswith("!") and 1 < len(room) <= 255
        for room in joined
    ):
        return set(), {"error": "Matrix joined rooms returned an invalid response"}
    if context.room_id not in joined:
        return set(), {"error": "Matrix session room is no longer joined"}
    return set(joined), None


async def _requester_joined(context: _DiscoveryContext, room_id: str) -> bool:
    identity = await context.adapter._resolve_room_identity(
        room_id, owner=context.owner
    )
    if identity.joined_members is None:
        raise RoomClassificationUnavailable
    return context.requester in identity.joined_members


async def _joined_rooms(
    context: _DiscoveryContext, result: _DiscoveryResult, candidates: set[str]
) -> None:
    selected = heapq.nsmallest(MAX_ROOM_CANDIDATES, candidates)
    result.truncated = len(candidates) > len(selected)
    for index, room_id in enumerate(selected):
        context.check()
        try:
            client, _, error = await _read_access(
                context.adapter,
                room_id,
                context.requester,
                joined_rooms=candidates,
                require_classification=True,
            )
            context.check()
            if error is not None:
                continue
            if client is not context.client:
                raise RoomClientChanged
            if not await _requester_joined(context, room_id):
                continue
        except RoomClassificationUnavailable:
            context.check()
            result.truncated = True
            result.errors.append({"error": "Matrix room classification is unavailable"})
            continue
        try:
            create = await _state(context, room_id, "m.room.create")
            context.check()
            room_type = _text(create, "type")
            if (room_type == "m.space") != (context.request.kind == "joined_spaces"):
                continue
            result.rooms.append(await _room_summary(context, room_id, room_type))
        except (TimeoutError, RoomClientChanged):
            raise
        except Exception as exc:
            result.truncated = True
            result.errors.append({
                "room_id": room_id,
                "error": f"Matrix room discovery failed: {type(exc).__name__}",
            })
        if len(result.rooms) >= context.request.limit:
            result.truncated |= index + 1 < len(selected)
            return


async def _users_sharing_a_room(
    context: _DiscoveryContext,
    result: _DiscoveryResult,
    joined: set[str],
    users: list[_DirectoryUser],
) -> list[_DirectoryUser]:
    wanted = {user["user_id"] for user in users}
    shared: set[str] = set()
    unavailable = False
    selected = heapq.nsmallest(MAX_ROOM_CANDIDATES, joined)
    for room_id in selected:
        if wanted <= shared:
            break
        members = await context.adapter._get_room_members(room_id, owner=context.owner)
        if members is None:
            unavailable = True
        elif context.requester in members:
            shared |= wanted & members
    visible = [user for user in users if user["user_id"] in shared]
    if len(visible) == len(users):
        return visible
    if unavailable:
        result.truncated = True
        result.errors.append({"error": "Matrix room membership is unavailable"})
    result.truncated |= len(selected) < len(joined)
    return visible


async def _directory(
    context: _DiscoveryContext, result: _DiscoveryResult, joined: set[str]
) -> None:
    response = await asyncio.wait_for(
        context.client.api.request(
            Method.POST,
            "/_matrix/client/v3/user_directory/search",
            {
                "search_term": context.request.search_term,
                "limit": context.request.limit,
            },
            retry_count=0,
        ),
        timeout=_REQUEST_TIMEOUT,
    )
    context.check()
    if (
        not isinstance(response, dict)
        or not isinstance(response.get("results"), list)
        or type(response.get("limited")) is not bool
    ):
        result.truncated = True
        result.errors.append({
            "error": "Matrix user directory returned an invalid response"
        })
        return
    rows = response["results"]
    users: list[_DirectoryUser] = []
    for row in rows[: context.request.limit]:
        user_id = row.get("user_id") if isinstance(row, dict) else None
        if (
            not isinstance(user_id, str)
            or not user_id.startswith("@")
            or ":" not in user_id
            or len(user_id) > 255
        ):
            result.truncated = True
            result.errors.append({
                "error": "Matrix user directory returned an invalid response"
            })
            return
        users.append({
            "user_id": user_id,
            "display_name": _text(row, "display_name"),
            "avatar_url": _text(row, "avatar_url"),
        })
    result.users = await _users_sharing_a_room(context, result, joined, users)
    result.truncated |= response["limited"] or len(rows) > context.request.limit


async def discover_matrix(
    adapter: Any,
    kind: str,
    room_id: str,
    limit: int,
    *,
    requester: str,
    search_term: str | None = None,
) -> dict[str, Any]:
    try:
        request = DiscoveryRequest.parse(kind, limit, search_term)
    except ValueError as exc:
        return {"error": str(exc)}
    if adapter._client is None:
        return {"error": "Matrix client is disconnected"}
    context = _DiscoveryContext(adapter, adapter._client, room_id, requester, request)
    result = _DiscoveryResult()
    try:
        async with asyncio.timeout(DISCOVERY_TIMEOUT):
            client, _, error = await _read_access(
                adapter,
                room_id,
                requester,
                require_classification=True,
            )
            context.check()
            if error is not None:
                return error
            if client is not context.client:
                raise RoomClientChanged
            joined, error = await _joined_membership(context)
            if error is not None:
                return error
            if request.kind == "users":
                await _directory(context, result, joined)
            else:
                await _joined_rooms(context, result, joined)
            context.check()
    except RoomClientChanged:
        return {"error": "Matrix discovery client changed"}
    except RoomClassificationUnavailable:
        result.truncated = True
        result.errors.append({"error": "Matrix room classification is unavailable"})
    except TimeoutError:
        result.truncated = True
        result.errors.append({"error": "Matrix discovery timed out"})
    except Exception as exc:
        result.truncated = True
        operation = "user directory" if request.kind == "users" else "discovery"
        result.errors.append({
            "error": f"Matrix {operation} failed: {type(exc).__name__}"
        })
    if adapter._client is not context.client:
        return {"error": "Matrix discovery client changed"}
    return result.serialize(request.kind)
