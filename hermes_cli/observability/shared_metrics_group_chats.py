"""Group chat counts for the daily install snapshot (``hermes.install.snapshot``).

A gateway keeps the group chats it hosts in its profile's ``state.db``; a standalone ``hermes serve``
keeps its own in the root ``shared-state.db`` until the root profile's gateway takes them over. Each
profile's snapshot counts its gateway's group chats and the root profile's also counts the standalone
ones not yet handed over, so each group chat is counted once. The stores are only read: a missing one
reads zero and is never created, and one that cannot be read leaves the fields out rather than
reporting a false zero. Only four bucketed counts leave the machine; never a room id, name, member,
link target or message text.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# The contract builds the closed dimension sets from these names (``SIZE_BUCKETS`` each).
GROUP_CHAT_SNAPSHOT_FIELDS = (
    "group_chat_count_bucket", "group_chat_active_7d_count_bucket",
    "group_chat_created_7d_count_bucket", "group_chat_peer_count_bucket",
)
_WINDOW_SECONDS = 7 * 24 * 3600

# One row per group chat this store drives: disbanded, then a flag for each count after the total: a
# message posted in the window (by a person or a scheduled post, not a bot reply), created in the
# window, a member served by another gateway. A standalone group chat handed to the gateway keeps a
# fenced row here under the ``handoff_gateway_id()`` authority and is the gateway's to count.
_ROOMS = (
    "SELECT r.room_id, r.disbanded_at IS NOT NULL,"
    " EXISTS (SELECT 1 FROM hosted_room_events e WHERE e.room_id = r.room_id"
    " AND e.kind = 'message.user' AND e.created_at >= :since),"
    " r.created_at >= :since,"
    " EXISTS (SELECT 1 FROM hosted_room_links l WHERE l.room_id = r.room_id)"
    " FROM hosted_rooms r WHERE r.authority_gateway_id NOT LIKE '%:gateway-store'"
)


def _read(database: Path, since: float) -> tuple[dict[str, list[int]], set[str]]:
    """The store's group chats by id, and the ids of disbanded ones already pruned."""
    if not database.is_file():
        return {}, set()
    # Without -wal/-shm no process has the store open; reading it immutable keeps SQLite from creating them.
    in_use = any(os.path.exists(f"{database}{suffix}") for suffix in ("-wal", "-shm"))
    uri = f"{database.absolute().as_uri()}?mode=ro{'' if in_use else '&immutable=1'}"
    with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=1)) as connection:
        tables = {name for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "hosted_rooms" not in tables:
            return {}, set()
        rooms = {room_id: flags for room_id, *flags in connection.execute(_ROOMS, {"since": since})}
        retired = set()
        if "hosted_room_retired_ids" in tables:
            retired = {room_id for (room_id,) in connection.execute("SELECT room_id FROM hosted_room_retired_ids")}
        return rooms, retired


def group_chat_snapshot_fields() -> dict[str, str]:
    """Bucketed group chat counts for this profile's snapshot; ``{}`` when a store cannot be read
    (the snapshot then keeps its pre-group-chat shape)."""
    try:
        from gateway.hosted_rooms import default_db_path
        from hermes_constants import get_hermes_home

        from .shared_metrics_contract import size_bucket

        home = get_hermes_home()
        since = time.time() - _WINDOW_SECONDS
        rooms, _ = _read(home / "state.db", since)
        standalone = default_db_path()
        if standalone.parent == home:  # the root profile also counts the standalone backend's
            # An upgraded install holds old rooms in both stores; the standalone copy (live, disbanded,
            # or pruned to a retired id) wins until the gateway takes it over (and fences it), so each
            # group chat is counted once.
            standalone_rooms, retired = _read(standalone, since)
            rooms.update(standalone_rooms)
            for room_id in retired:
                rooms.pop(room_id, None)
        live = [flags for disbanded, *flags in rooms.values() if not disbanded]
        counts = [len(live), *(sum(flags[i] for flags in live) for i in range(3))]
        return {field: size_bucket(n) for field, n in zip(GROUP_CHAT_SNAPSHOT_FIELDS, counts, strict=True)}
    except Exception:  # never cost the rest of the snapshot
        logger.debug("group chat counts unavailable for the install snapshot", exc_info=True)
        return {}
