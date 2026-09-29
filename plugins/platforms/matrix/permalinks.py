"""matrix.to permalinks for the event that triggered a turn."""

from __future__ import annotations

import ipaddress
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlencode

MAX_VIA_SERVERS = 3
MIN_ANCHOR_POWER_LEVEL = 50


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


def via_servers_from_members(member_levels: Mapping[str, int]) -> list[str]:
    """Choose ``via`` servers from the joined members and their power levels.

    This follows the routing recommendation for room-ID permalinks in the Matrix
    specification. The first server is the server of the highest-power member, if that
    member has at least power level 50. The rest are the servers with the most joined
    members. IP literals are never chosen. Server ACLs are not checked, because the
    client's state store does not keep ``m.room.server_acl``.
    """
    population = Counter(
        server
        for user_id in member_levels
        if (server := _server_of(user_id)) and not _is_ip_literal(server)
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


async def room_via_servers(state_store: Any, room_id: str, fallback: Iterable[str | None]) -> list[str]:
    """``via`` servers for a room, read from the mautrix state store without a request.

    Sync keeps the store's members, power levels and create event current. When it has
    no joined members for the room, the routable ``fallback`` servers are used in order.
    """
    if state_store is not None:
        from mautrix.types import Membership

        joined = await state_store.get_members(room_id, memberships=(Membership.JOIN,))
        levels = await state_store.get_power_levels(room_id)
        create = await state_store.get_create(room_id)
        servers = via_servers_from_members(
            {user_id: levels.get_user_level(user_id, create) if levels else 0 for user_id in joined}
        )
        if servers:
            return servers

    return [server for server in dict.fromkeys(fallback) if server and not _is_ip_literal(server)]
