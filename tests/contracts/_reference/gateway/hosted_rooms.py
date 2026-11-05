"""In-memory reference ``HostedRoomRepository``.

Owner keys, transport bindings and member homes are ``state_meta`` rows of the bound profile (the namespace the ``state_meta`` reference keeps), under the key prefixes below; the epoch fence reads the profile's :class:`AuthorityState`. ``label_room_session`` and ``ensure_room_member_session`` act on the profile's ``sessions`` rows as the conversation reference keeps them. The room verbs act on the placement's hosted-room tables (``hosted_room_tables``); the room and event verbs other than ``create_room`` live in ``hosted_room_log``, the hand-off verbs in ``hosted_room_handoff``.

:func:`hosted_member_check` is the hosted member check after the epoch fence; ``member_authorized`` runs it as a read and the admission reference runs it inside ``admit(hosted=)``.

Reference limits: the one-shot import of pre-isolation rooms and the schema migrations of a first open have no in-memory counterpart (a reference file starts empty and current). The work-record freeze the legacy adoption of ``create_room`` triggers is not modelled: no verb reads frozen records.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from persistence import AuthorityRefusal, JsonValue, SessionId
from persistence.authority.vocabulary import OwnerRule
from persistence.conversation.sessions import SessionLineageKind, TitleConflict, sanitize_title
from persistence.gateway.hosted_rooms import (
    HostedMemberGuard,
    HostedOwnerClaim,
    HostedPlacement,
    HostedRoomError,
    HostedRoomRefusal,
    RoomConflictError,
)
from session_rules.legacy_local import same_owner

from .._authority import authority_state
from ..backend import register_repository
from ..conversation.sessions import blank_session_row
from ..housekeeping.state_meta import state_meta_table
from . import hosted_room_rules as rules
from .hosted_room_handoff import HostedRoomHandoffMixin
from .hosted_room_log import HostedRoomLogMixin
from .hosted_room_tables import (
    MEMBER_HOMES_PREFIX,
    HostedTables,
    attempt_home_key,
    event_from_row,
    find_event,
    insert_event,
    member_attempts_prefix,
    member_homes_key,
    prepare_event,
    read_tables,
    room_from_row,
    task_view,
    write_tables,
)

__all__ = ["InMemoryHostedRoomRepository", "hosted_member_check"]

# ``state_meta`` key prefixes of a room's owner key and transport binding.
OWNER_PREFIX = "gateway.hosted.owner.v1:"
TRANSPORT_PREFIX = "gateway.hosted.transport.v1:"
#: Task statuses in which a recorded attempt home is still the attempt's (kept), and those of a Stop not yet confirmed.
_LIVE_ATTEMPT = frozenset({"running", "stopping", "deferred", "indeterminate"})
_STOPPING = frozenset({"stopping", "deferred"})

_ROOM_SESSION_CONFLICT = "room session identity conflicts with existing data"
_ROOM_TITLE_CONFLICT = (
    "Another group already uses this room title on the target gateway. Rename or migrate that group before retrying.")


def _seated(member: Any, guard: HostedMemberGuard) -> bool:
    target = member.get("target")
    return (member.get("member_id") == guard.member_id and member.get("profile") == guard.profile
            and (target is None or (isinstance(target, dict) and target.get("kind", "local") == "local")))


def _task_allowed(tables: HostedTables, guard: HostedMemberGuard, operation: str) -> bool:
    assert guard.task is not None
    row = tables["hosted_room_driver_tasks"].get(guard.room_id, {}).get(guard.task.task_id)
    if row is None:
        return False
    current = task_view(row)
    allowed = {"running"} if operation in {"submit", "execute"} else {"running", "stopping"}
    return (current["identity"] == guard.task and current["execution_generation"] == guard.generation
            and current["payload"].get("target_profile") == guard.profile
            and current["payload"].get("target_member_id", guard.profile) == guard.member_id
            and current["status"] in allowed)


def hosted_member_check(store: Any, guard: HostedMemberGuard, *, operation: str) -> bool:
    """The hosted member check after the epoch fence, in this order: the owner key equals ``guard.owner``; the room row is live with authority ``(guard.gateway_id, guard.authority_epoch)``; a local member with ``guard.member_id`` and ``guard.profile`` is seated; ``guard.home_current``; then, with ``guard.task``, the task row matches identity, generation, target profile and member with a status ``operation`` allows (``running``; ``stopping`` too for operations other than submit and execute). Reads the bound profile's committed rows."""
    if state_meta_table(store, store.scope).get(OWNER_PREFIX + guard.room_id) != guard.owner:
        return False
    tables = read_tables(store, "profile")
    stored = tables["hosted_rooms"].get(guard.room_id)
    if stored is None or stored["disbanded_at"] is not None:
        return False
    if (stored["authority_gateway_id"], int(stored["authority_epoch"])) != (guard.gateway_id, guard.authority_epoch):
        return False
    if not any(_seated(member, guard) for member in json.loads(stored["members_json"])):
        return False
    if not guard.home_current:
        return False
    return guard.task is None or _task_allowed(tables, guard, operation)


def _adopt_legacy_room(tables: HostedTables, existing: dict[str, Any], *, members_json: str, gateway_id: str, now: float) -> dict[str, Any]:
    """Claim a ``'legacy'`` room at the next epoch with an ``authority.claimed`` event."""
    room_id = existing["room_id"]
    target_epoch, seq = int(existing["authority_epoch"]) + 1, int(existing["next_seq"])
    actor_json = rules.system_actor_json("authority-control")
    payload_json = rules.payload_json({"previous_gateway_id": "legacy", "authority_gateway_id": gateway_id, "authority_epoch": target_epoch})
    claim_bytes = prepare_event(tables, existing, "system:authority-adopted", "authority.claimed", actor_json, payload_json, allow_control=True)
    insert_event(tables, room_id, seq, "system:authority-adopted", "authority.claimed", actor_json, target_epoch, payload_json, now)
    existing.update(
        members_json=members_json, authority_gateway_id=gateway_id, authority_epoch=target_epoch, next_seq=seq + 1,
        revision=int(existing["revision"]) + 1, event_bytes=int(existing["event_bytes"]) + claim_bytes, updated_at=now)
    claim = find_event(tables, room_id, "system:authority-adopted")
    assert claim is not None
    return {**room_from_row(existing, idempotent=True), "adopted": True, "claim_event": event_from_row(claim)}


def _existing_room(tables: HostedTables, existing: dict[str, Any], *, name: str, members: list[dict[str, Any]], members_json: str, gateway_id: str, now: float) -> dict[str, Any]:
    """An existing room: the idempotent replay, the legacy adoption or a conflict."""
    if existing["disbanded_at"] is not None:
        raise RoomConflictError("room_id belongs to a disbanded room")
    legacy = existing["authority_gateway_id"] == "legacy" and gateway_id != "legacy"
    members_match = existing["members_json"] == members_json or (legacy and rules.legacy_members_match(existing["members_json"], members))
    if existing["name"] != name or not members_match:
        raise RoomConflictError("room_id already exists with different state")
    if legacy:
        return _adopt_legacy_room(tables, existing, members_json=members_json, gateway_id=gateway_id, now=now)
    if existing["authority_gateway_id"] != gateway_id:
        raise RoomConflictError("room_id already belongs to a different authority")
    return room_from_row(existing, idempotent=True)


def _create(tables: HostedTables, *, room_id: str, name: str, members: list[dict[str, Any]], members_json: str, gateway_id: str, now: float) -> dict[str, Any]:
    """Create the room after the owner claim: a retired id refuses, an existing room goes through :func:`_existing_room`, the active-room cap applies to a new one."""
    if room_id in tables["hosted_room_retired_ids"]:
        raise RoomConflictError("room_id belongs to a disbanded room")
    existing = tables["hosted_rooms"].get(room_id)
    if existing is not None:
        return _existing_room(tables, existing, name=name, members=members, members_json=members_json, gateway_id=gateway_id, now=now)
    if sum(row["disbanded_at"] is None for row in tables["hosted_rooms"].values()) >= rules.MAX_ACTIVE_ROOMS:
        raise HostedRoomError("This host has too many active Group Chats. Delete one and try again.")
    row = tables["hosted_rooms"][room_id] = {
        "room_id": room_id, "name": name, "members_json": members_json, "authority_gateway_id": gateway_id,
        "authority_epoch": 1, "next_seq": 1, "event_bytes": 0, "revision": 1, "created_at": now, "updated_at": now,
        "disbanded_at": None}
    return {**room_from_row(row, latest_seq=False), "members": members}


class InMemoryHostedRoomRepository(HostedRoomLogMixin, HostedRoomHandoffMixin):
    """See ``persistence.gateway.hosted_rooms.HostedRoomRepository`` for the contract each method implements."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _meta(self) -> dict[str, str]:
        return state_meta_table(self._store, self._store.scope)

    def _require_epoch(self, epoch: int) -> None:
        state = authority_state(self._store, self._store.scope)
        state.require_epoch(state.snapshot, epoch)

    def _claim_for_create(self, tables: HostedTables, room_id: str, owner: HostedOwnerClaim) -> bool:
        """The owner claim of ``create_room``: True when the claim inserts the owner key, False when the stored owner is the actor's under the claim's rule, else ``permission_denied``."""
        self._require_epoch(owner.epoch)
        stored = self._meta().get(OWNER_PREFIX + room_id)
        if stored is None and room_id not in tables["hosted_rooms"] and room_id not in tables["hosted_room_retired_ids"]:
            return True
        if stored is None or not same_owner(stored, owner.actor, owner.rule):
            raise AuthorityRefusal("permission_denied")
        return False

    def claim_owner(self, *, epoch: int, room_id: str, actor: str, owner: OwnerRule | None) -> None:
        self._require_epoch(epoch)
        stored = self._meta().get(OWNER_PREFIX + room_id)
        if stored is None or not same_owner(stored, actor, owner):
            raise AuthorityRefusal("permission_denied")

    def create_room(self, *, placement: HostedPlacement, room_id: str, name: str, members: Sequence[Mapping[str, JsonValue]], authority_gateway_id: str, owner: HostedOwnerClaim | None = None, now: float | None = None) -> Mapping[str, JsonValue]:
        room_id = rules.room_id(room_id)
        name = rules.validate_room_name(name)
        normalized, members_json = rules.validate_members(members)
        gateway_id = rules.actor_id(authority_gateway_id, "authority_gateway_id")
        timestamp = rules.clock(now)
        if owner is not None and placement != "profile":
            raise ValueError("an owner claim requires placement='profile'")
        with write_tables(self._store, placement) as tables:
            claimed = owner is not None and self._claim_for_create(tables, room_id, owner)
            room = _create(tables, room_id=room_id, name=name, members=normalized, members_json=members_json, gateway_id=gateway_id, now=timestamp)
        if claimed:
            assert owner is not None
            self._meta()[OWNER_PREFIX + room_id] = owner.actor
        return room

    def bind_transport(self, *, epoch: int, session_id: str, binding: str) -> None:
        state = authority_state(self._store, self._store.scope)
        with state.transaction() as staged:
            state.require_epoch(staged.authority, epoch)
            meta = staged.scope.setdefault("state_meta", {})
            old = meta.get(TRANSPORT_PREFIX + session_id)
            if old is not None and old != binding:
                raise AuthorityRefusal("permission_denied")
            meta.setdefault(TRANSPORT_PREFIX + session_id, binding)

    def label_room_session(self, *, epoch: int, session_id: str, title: str) -> None:
        self._require_epoch(epoch)
        table = self._store.scope_store(self._store.scope).setdefault("sessions", {})
        row = table.get(session_id)
        if row is None:
            return
        holder = next((other for other, held in table.items() if other != session_id and held["title"] == title), None)
        if holder is not None:
            raise TitleConflict(title, SessionId(holder))
        row.update(title=title, hidden=True)

    def ensure_room_member_session(self, *, session_id: str, title: str) -> str:
        table = self._store.scope_store(self._store.scope).setdefault("sessions", {})
        row = table.get(session_id)
        if row is not None:
            if row["title"] != title or row["source"] != "bot_room":
                raise HostedRoomRefusal(_ROOM_SESSION_CONFLICT)
            return session_id
        clean = sanitize_title(title)
        if clean is not None and any(other != session_id and held["title"] == clean for other, held in table.items()):
            raise HostedRoomRefusal(_ROOM_TITLE_CONFLICT)
        created = blank_session_row(session_id, lineage_kind=SessionLineageKind.NONE, model_config=None)
        created.update(source="bot_room", title=clean, hidden=True)
        table[session_id] = created
        return session_id

    def claim_router_call(self, *, placement: HostedPlacement, room_id: str, discussion_event_id: str, owner: str, expires_at: float, now: float | None = None) -> bool:
        timestamp = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            claims = tables.rows("hosted_room_router_claims", room_id)
            held = claims.get(discussion_event_id)
            if held is None or float(held["expires_at"]) <= timestamp:
                claims[discussion_event_id] = {"room_id": room_id, "discussion_event_id": discussion_event_id, "owner": owner, "expires_at": float(expires_at)}
                return True
            return False

    def release_router_call(self, *, placement: HostedPlacement, room_id: str, discussion_event_id: str, owner: str) -> None:
        with write_tables(self._store, placement) as tables:
            claims = tables.rows("hosted_room_router_claims", room_id)
            if claims.get(discussion_event_id, {}).get("owner") == owner:
                del claims[discussion_event_id]

    def record_member_home(self, *, room_id: str, member_id: str, home: str) -> None:
        meta = self._meta()
        key = member_homes_key(room_id, member_id)
        homes = json.loads(meta[key]) if key in meta else []
        if home not in homes:
            meta[key] = json.dumps([*homes, home])

    def member_homes(self, *, room_id: str, member_id: str) -> tuple[str, ...]:
        stored = self._meta().get(member_homes_key(room_id, member_id))
        return tuple(json.loads(stored)) if stored is not None else ()

    def _attempt_homes(self, room_id: str, member_id: str, statuses: frozenset[str]) -> dict[str, tuple[str, bool]]:
        """``{key: (home, live)}`` for every attempt home recorded for this member; live while its profile task row is still that attempt in one of ``statuses``."""
        prefix, rows = member_attempts_prefix(room_id, member_id), read_tables(self._store, "profile")["hosted_room_driver_tasks"].get(room_id, {})
        found: dict[str, tuple[str, bool]] = {}
        for key, home in self._meta().items():
            if not key.startswith(prefix):
                continue
            task_id, generation = json.loads(key[len(MEMBER_HOMES_PREFIX):])[2:]
            row = rows.get(task_id)
            found[key] = (home, row is not None and row["status"] in statuses and int(row["execution_generation"]) == generation)
        return found

    def record_attempt_home(self, *, room_id: str, member_id: str, task_id: str, generation: int, home: str) -> None:
        meta, key = self._meta(), attempt_home_key(room_id, member_id, task_id, generation)
        for old, (_home, live) in self._attempt_homes(room_id, member_id, _LIVE_ATTEMPT).items():
            if not live and old != key:
                del meta[old]
        meta.setdefault(key, home)

    def attempt_home(self, *, room_id: str, member_id: str, task_id: str, generation: int) -> str | None:
        return self._meta().get(attempt_home_key(room_id, member_id, task_id, generation))

    def stopping_attempt_homes(self, *, room_id: str, member_id: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(home for home, live in self._attempt_homes(room_id, member_id, _STOPPING).values() if live))

    def owner(self, room_id: str) -> str | None:
        return self._meta().get(OWNER_PREFIX + room_id)

    def owned_rooms(self) -> frozenset[str]:
        return frozenset(key[len(OWNER_PREFIX):] for key in self._meta() if key.startswith(OWNER_PREFIX))

    def probe_room_owner(self, room_id: str) -> bool:
        return OWNER_PREFIX + room_id in self._meta()

    def transport_binding(self, session_id: str) -> str | None:
        return self._meta().get(TRANSPORT_PREFIX + session_id)

    def member_authorized(self, guard: HostedMemberGuard, *, epoch: int, operation: str) -> bool:
        state = authority_state(self._store, self._store.scope)
        state.require_epoch(state.snapshot, epoch)
        return hosted_member_check(self._store, guard, operation=operation)


register_repository("hosted_rooms", InMemoryHostedRoomRepository)
