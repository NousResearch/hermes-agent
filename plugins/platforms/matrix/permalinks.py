"""matrix.to permalinks for the event that triggered a turn."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
from dataclasses import dataclass, field
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlencode

MAX_VIA_SERVERS = 3
MIN_ANCHOR_POWER_LEVEL = 50
ACL_READ_TIMEOUT = 10.0
logger = logging.getLogger("plugins.platforms.matrix.adapter")


@dataclass(frozen=True)
class MatrixServerACL:
    allow: tuple[re.Pattern[str], ...]
    deny: tuple[re.Pattern[str], ...] = ()

    @classmethod
    def from_content(cls, content: Mapping[str, Any]) -> MatrixServerACL:
        if not isinstance(content, Mapping):
            raise ValueError("Invalid Matrix server ACL content")

        def patterns(key: str) -> tuple[re.Pattern[str], ...]:
            values = content.get(key, [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"Invalid Matrix server ACL {key} patterns")
            return tuple(re.compile(re.escape(value).replace(r"\*", ".*").replace(r"\?", "."), re.IGNORECASE)
                         for value in values)

        return cls(patterns("allow"), patterns("deny"))

    def permits(self, server: str) -> bool:
        host = server.partition(":")[0]
        return not any(pattern.fullmatch(host) for pattern in self.deny) and any(
            pattern.fullmatch(host) for pattern in self.allow
        )


@dataclass
class _RoomRouting:
    policy: MatrixServerACL | None = None
    revision: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MatrixPermalinkRouting:
    def __init__(self) -> None:
        self._client: Any = None
        self._rooms: dict[str, _RoomRouting] = {}

    def _bind_client(self, client: Any) -> None:
        if client is self._client:
            return
        self._client = client
        self._rooms.clear()

    def observe_sync(self, client: Any, sync_data: Mapping[str, Any]) -> None:
        self._bind_client(client)
        rooms = sync_data.get("rooms", {})
        for room_id in rooms.get("leave", {}):
            self._rooms.pop(room_id, None)
        for room_id, room in rooms.get("join", {}).items():
            for section in ("state", "timeline"):
                for event in room.get(section, {}).get("events", []):
                    if event.get("type") != "m.room.server_acl" or event.get("state_key") != "":
                        continue
                    routing = self._rooms.setdefault(room_id, _RoomRouting())
                    routing.revision += 1
                    try:
                        routing.policy = MatrixServerACL.from_content(event["content"])
                    except (KeyError, TypeError, ValueError):
                        routing.policy = None

    async def resolve(self, client: Any, room_id: str) -> MatrixServerACL | None:
        self._bind_client(client)
        routing = self._rooms.setdefault(room_id, _RoomRouting())
        async with routing.lock:
            if self._client is not client or self._rooms.get(room_id) is not routing:
                return None
            if routing.policy is not None:
                return routing.policy
            if client is None:
                return None
            from mautrix.errors import MNotFound
            from mautrix.types import EventType, RoomID

            revision = routing.revision
            try:
                content = await asyncio.wait_for(
                    client.get_state_event(RoomID(room_id), EventType.find("m.room.server_acl", EventType.Class.STATE)),
                    timeout=ACL_READ_TIMEOUT,
                )
                if hasattr(content, "serialize"):
                    content = content.serialize()
                policy = MatrixServerACL.from_content(content)
            except MNotFound:
                policy = MatrixServerACL.from_content({"allow": ["*"]})
            except Exception as exc:
                logger.debug("Matrix: could not resolve server ACL for %s: %s", room_id, exc)
                return routing.policy if self._client is client and self._rooms.get(room_id) is routing else None
            if self._client is not client or self._rooms.get(room_id) is not routing:
                return None
            if routing.revision != revision:
                return routing.policy
            if routing.policy is None:
                routing.policy = policy
            return routing.policy


def event_permalink(room_id: str, event_id: str, via: Sequence[str]) -> str | None:
    """matrix.to link to ``event_id``, with one ``via`` parameter per server."""
    if not event_id:
        return None
    permalink = f"https://matrix.to/#/{quote(room_id, safe='!$:@')}/{quote(event_id, safe='!$:@')}"
    if not via:
        return permalink
    return f"{permalink}?{urlencode([('via', server) for server in via])}"


def _server_of(user_id: str) -> str:
    return user_id.partition(":")[2]


def _is_ip_literal(server: str) -> bool:
    host = server[1:].partition("]")[0] if server.startswith("[") else server.partition(":")[0]
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def via_servers_from_members(member_levels: Mapping[str, int], policy: MatrixServerACL | None = None) -> list[str]:
    """Choose ``via`` servers from the joined members and their power levels.

    This follows the routing recommendation for room-ID permalinks in the Matrix
    specification. The first server is the server of the highest-power member, if that
    member has at least power level 50. The rest are the servers with the most joined
    members. IP literals and servers blocked by the supplied ACL are excluded.
    """
    population = Counter(
        server
        for user_id in member_levels
        if (server := _server_of(user_id)) and not _is_ip_literal(server) and (policy is None or policy.permits(server))
    )
    anchors = [
        user_id
        for user_id, level in member_levels.items()
        if level >= MIN_ANCHOR_POWER_LEVEL and _server_of(user_id) in population
    ]
    servers = []
    if anchors:
        servers.append(_server_of(min(anchors, key=lambda user_id: (-member_levels[user_id], user_id))))

    ranked = sorted(population, key=lambda server: (-population[server], server))
    servers.extend(server for server in ranked if server not in servers)
    return servers[:MAX_VIA_SERVERS]


async def room_via_servers(
    state_store: Any, room_id: str, fallback: Iterable[str | None], *, policy: MatrixServerACL | None,
) -> list[str]:
    """``via`` servers for a room, read from the mautrix state store without a request.

    Sync keeps the store's members, power levels and create event current. When it has
    no joined members for the room, the routable ``fallback`` servers are used in order.
    """
    if policy is None:
        return []
    if state_store is not None:
        from mautrix.types import Membership

        joined = await state_store.get_members(room_id, memberships=(Membership.JOIN,))
        levels = await state_store.get_power_levels(room_id)
        create = await state_store.get_create(room_id)
        servers = via_servers_from_members(
            {user_id: levels.get_user_level(user_id, create) if levels else 0 for user_id in joined}, policy,
        )
        if servers:
            return servers

    return [server for server in dict.fromkeys(fallback) if server and not _is_ip_literal(server) and policy.permits(server)]
