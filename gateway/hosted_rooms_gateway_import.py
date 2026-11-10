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
fenced rooms the target has never known (neither held nor reserved in ``hosted_room_retired_ids``),
so a room the user disbanded in the gateway never comes back once its tombstone is pruned. After
the copy commits, each handed room is recorded in the source's hand-off record, which step 2 never
offers again. A crash between the steps is finished by the next start. Both stores are reached
through the storage interface: the gateway store's handle names the source as its shared file.

``shared-state.db`` is one root file with no profile column, so only the root home's gateway (the
default profile, whose ``state.db`` sits beside it) imports. A standalone room that was not imported
stays listed and driven by the standalone backend from every profile's view, and a named profile's
view reaches the imported ones on the root gateway (``tui_gateway/methods_groups_forward.py``), so
no view loses a room it showed before.

The import runs when the gateway builds its room service (before ``ready``) and once more right
after ``ready`` (:func:`import_after_ready`), covering a room written in between.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SOURCE_NAME = "shared-state.db"


def handoff_gateway_id() -> str:
    """The authority a standalone room is fenced to once the gateway store holds it."""
    from gateway.hosted_rooms import local_authority_gateway_id
    return f"{local_authority_gateway_id()}:gateway-store"


def source_for(target_db: Path) -> Path | None:
    """The standalone store this gateway store imports from, or ``None`` (named profile, tests)."""
    target_db = Path(target_db)
    source = target_db.with_name(SOURCE_NAME)
    return source if target_db.name == "state.db" and source.is_file() else None


def _store(db_path: Path):
    """A storage handle on the home holding ``db_path``: its ``state.db``, or its root ``shared-state.db``."""
    from persistence_assembly import open_persistence, profile_scope
    return open_persistence(profile_scope(home=Path(db_path).parent))


def _fence_source(store, local_id: str, handoff_id: str) -> int:
    from session_rules.hosted_rooms import MAX_ROOM_LIST_LIMIT, _actor_id
    fenced, offset = 0, 0
    while True:
        page = store.hosted_rooms.list_rooms(placement="shared", limit=MAX_ROOM_LIST_LIMIT, offset=offset)
        for room in page:
            if room["authority_gateway_id"] == local_id:
                store.hosted_room_replicas.demote_room(
                    placement="shared", room_id=room["room_id"],
                    observed_gateway_id=_actor_id(handoff_id, "observed_gateway_id"),
                    observed_epoch=int(room["authority_epoch"]) + 1, local_gateway_id=local_id)
                fenced += 1
        if len(page) < MAX_ROOM_LIST_LIMIT:
            return fenced
        offset += len(page)


def handed_room_ids(source: Path) -> set[str]:
    """Every room ``source`` handed to the gateway store (recorded or still only fenced)."""
    with _store(source) as store:
        return set(store.hosted_rooms.handed_room_ids(handoff_gateway_id=handoff_gateway_id()))


def import_standalone_rooms(target_db: Path | None, *, runtime_epoch: int | None = None) -> int:
    """Move the standalone backend's live rooms into ``target_db`` once; returns rooms copied. A store with no file
    (``target_db`` None) has no standalone file beside it, so there is nothing to move.

    ``runtime_epoch``: the authority's writer epoch, checked inside the copy's transaction so a
    superseded gateway process never writes the store. Never raises: a locked or unreadable source
    leaves both stores as they were and is retried on the next start.
    """
    from gateway.hosted_rooms import local_authority_gateway_id
    from persistence.gateway.hosted_rooms import HostedRoomError
    from gateway.hosted_rooms_common import clock
    from gateway.owner_identity import host_owner_subject
    from persistence import PersistenceError

    source = source_for(target_db) if target_db is not None else None
    if source is None:
        return 0
    try:
        with _store(target_db) as store:
            local_id, handoff_id = local_authority_gateway_id(), handoff_gateway_id()
            _fence_source(store, local_id, handoff_id)
            fenced = store.hosted_rooms.fenced_room_ids(handoff_gateway_id=handoff_id)
            if not fenced:
                return 0
            owner, now = host_owner_subject(), clock(None)
            copied = store.hosted_rooms.import_fenced_rooms(
                epoch=runtime_epoch, room_ids=fenced, owner=owner, local_gateway_id=local_id,
                handoff_gateway_id=handoff_id, now=now)
            store.hosted_rooms.record_room_handoff(room_ids=fenced, now=now)
    except (OSError, PersistenceError, HostedRoomError) as exc:
        logger.warning("hosted rooms: could not import the standalone Group Chats from %s (%s); "
                       "will retry on the next start", source, exc)
        return 0
    if copied:
        logger.info("hosted rooms: imported %d standalone Group Chat(s) from %s", copied, source)
    return copied


def import_after_ready(authorities) -> None:
    """Import once more after the gateway published ``ready``, off the event loop.

    The start-up import runs before readiness; a standalone backend that saw no gateway yet may
    still write a room in between. Once ready, that backend forwards every call (or waits while
    the gateway is starting), so this second pass leaves nothing stranded until the next restart.
    The service's driver re-reads its bindings each tick and is woken for the claimed rooms."""
    import threading
    for authority in authorities:
        service = getattr(authority, "hosted_room_service", None)
        if service is None or getattr(service, "_ready_import_started", False):
            continue
        service._ready_import_started = True
        from gateway.session_persistence import authority_persistence
        from persistence_assembly import store_location
        target, epoch = store_location(authority_persistence(authority).scope), authority.epoch

        def run(target=target, epoch=epoch, service=service, authority=authority):
            from gateway.session_authorities import owner_scope
            with owner_scope(authority):  # a new thread holds the launch profile's scope otherwise
                if import_standalone_rooms(target, runtime_epoch=epoch):
                    service.wakeup()
        threading.Thread(target=run, name="hosted-rooms-ready-import", daemon=True).start()
