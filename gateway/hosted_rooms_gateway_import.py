"""Hand the standalone backend's Group Chats to the gateway that owns the profile (WA-19).

A native Desktop on a standalone ``hermes serve``/``hermes dashboard`` (no session authority) keeps
its rooms in the root ``shared-state.db`` and drives them in that process (``tui_gateway``
route), while the gateway keeps its own in the profile's ``state.db`` (owner claim and epoch fence
in the room insert's transaction). Once a gateway serves the profile, the standalone backend
forwards ``groups.*`` to it (``tui_gateway/methods_groups_forward.py``); this module moves the rooms
that backend already hosts into the gateway's store so the one list shows them.

Each room moves in two crash-safe steps, so it is driven by exactly one process:

1. **Fence the source.** The room's authority in ``shared-state.db`` moves to
   :func:`handoff_gateway_id` at ``epoch + 1`` (``authority.lost``, the RoomLink demotion). The
   standalone driver binds only rooms of its own authority and every room write checks it, so that
   copy is frozen from here on. The source rows are never deleted.
2. **Copy and claim.** A fenced room the gateway store does not hold is copied with its child rows,
   then claimed for this install at ``epoch + 1`` (``authority.claimed``) and owned by the host
   owner, the subject native Desktop and a signed-in Webapp tab both act as.

Re-running is a no-op: step 1 only takes rooms still on the local authority, step 2 only copies
fenced rooms the target lacks. A crash between them is finished by the next start.

``shared-state.db`` is one root file with no profile column, so only the root home's gateway (the
default profile, whose ``state.db`` sits beside it) imports; a named profile's standalone rooms stay
where they are for gateway-less use.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SOURCE_NAME = "shared-state.db"
_OWNER = "gateway.hosted.owner.v1:"


def handoff_gateway_id() -> str:
    """The authority a standalone room is fenced to once the gateway store holds it."""
    from gateway.hosted_rooms import local_authority_gateway_id
    return f"{local_authority_gateway_id()}:gateway-store"


def source_for(target_db: Path) -> Path | None:
    """The standalone store this gateway store imports from, or ``None`` (named profile, tests)."""
    target_db = Path(target_db)
    source = target_db.with_name(SOURCE_NAME)
    return source if target_db.name == "state.db" and source.is_file() else None


def _fence_source(source: Path, local_id: str, handoff_id: str) -> int:
    from gateway.hosted_room_replicas import demote_room
    from gateway.hosted_rooms import list_rooms, MAX_ROOM_LIST_LIMIT
    fenced, offset = 0, 0
    while True:
        page = list_rooms(source, limit=MAX_ROOM_LIST_LIMIT, offset=offset)
        for room in page:
            if room["authority_gateway_id"] == local_id:
                demote_room(source, room_id=room["room_id"], observed_gateway_id=handoff_id,
                            observed_epoch=int(room["authority_epoch"]) + 1)
                fenced += 1
        if len(page) < MAX_ROOM_LIST_LIMIT:
            return fenced
        offset += len(page)


def _fenced_ids(source: Path, handoff_id: str) -> set[str]:
    from contextlib import closing
    from gateway.hosted_rooms import _read_connection
    with closing(_read_connection(source)) as conn:
        return {str(row[0]) for row in conn.execute(
            "SELECT room_id FROM hosted_rooms WHERE authority_gateway_id=? AND disbanded_at IS NULL",
            (handoff_id,))}


def _claim(conn, room_id: str, local_id: str, handoff_id: str, owner: str, now: float) -> None:
    from gateway.hosted_room_replicas import _append_control_event, _control_event
    row = conn.execute("SELECT authority_epoch, next_seq FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
    epoch = int(row["authority_epoch"]) + 1
    _append_control_event(conn, room_id, int(row["next_seq"]), epoch, _control_event("claimed", epoch, {
        "previous_gateway_id": handoff_id, "authority_gateway_id": local_id, "authority_epoch": epoch}), now)
    conn.execute("""UPDATE hosted_rooms SET authority_gateway_id=?, authority_epoch=?, next_seq=next_seq+1,
                 revision=revision+1, updated_at=? WHERE room_id=?""", (local_id, epoch, now, room_id))
    conn.execute("INSERT OR IGNORE INTO state_meta(key, value) VALUES (?, ?)", (_OWNER + room_id, owner))


def import_standalone_rooms(target_db: Path, *, runtime_epoch: int | None = None) -> int:
    """Move the standalone backend's live rooms into ``target_db`` once; returns rooms copied.

    ``runtime_epoch``: the authority's writer epoch, checked inside the copy's transaction so a
    superseded gateway process never writes the store. Never raises: a locked or unreadable source
    leaves both stores as they were and is retried on the next start.
    """
    import sqlite3
    from gateway.hosted_rooms import HostedRoomError, _transaction, local_authority_gateway_id
    from gateway.hosted_rooms_common import clock
    from gateway.hosted_rooms_legacy_import import _copy_rows
    from gateway.owner_identity import host_owner_subject
    from hermes_state_runtime import RuntimeStoreError, _epoch

    source = source_for(target_db)
    if source is None:
        return 0
    try:
        local_id, handoff_id = local_authority_gateway_id(), handoff_gateway_id()
        _fence_source(source, local_id, handoff_id)
        fenced = _fenced_ids(source, handoff_id)
        if not fenced:
            return 0
        owner, now = host_owner_subject(), clock(None)
        with _transaction(target_db, immediate=True) as conn:
            if runtime_epoch is not None:
                _epoch(conn, runtime_epoch)
            held = {str(row[0]) for row in conn.execute("SELECT room_id FROM hosted_rooms")}
            missing = frozenset(fenced - held)
            if not missing:
                return 0
            copied = _copy_rows(conn, source, only=missing)
            for room_id in sorted(missing):
                _claim(conn, room_id, local_id, handoff_id, owner, now)
    except (OSError, sqlite3.Error, HostedRoomError, RuntimeStoreError) as exc:
        logger.warning("hosted rooms: could not import the standalone Group Chats from %s (%s); "
                       "will retry on the next start", source, exc)
        return 0
    logger.info("hosted rooms: imported %d standalone Group Chat(s) from %s", copied, source)
    return copied
