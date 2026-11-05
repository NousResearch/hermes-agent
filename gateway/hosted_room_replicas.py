"""Replica store and takeover primitives for hosted Group Chat rooms.

Non-authority gateways keep a durable local copy of the room log (``ingest_page()``: idempotent,
gap- and epoch-regression-safe) plus fenced primitives to continue the room when the authority host
dies: ``promote_replica()`` resumes locally at ``epoch + 1`` with a lineage-proving ``authority.claimed``
event; ``demote_room()`` records ``authority.lost`` when a returning stale authority is shown a newer
epoch. Storage primitives only: the caller decides *when* takeover is safe.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from gateway.hosted_rooms import (
    MAX_ACTOR_ID_CHARS, _room_id, _validate_identifier, _validate_members, _validate_room_name,
    local_authority_gateway_id)
from gateway.hosted_rooms_common import bounded_int, clock
from persistence.gateway.hosted_rooms import HostedPlacement, ReplicaError, ReplicaGapError, ReplicaPage

if TYPE_CHECKING:
    from persistence_assembly import Persistence


_positive_int = partial(bounded_int, error=ReplicaError, low=1)


def _validate_page(page: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(page, dict):
        raise ReplicaError("page must be an object")
    events, authority = page.get("events"), page.get("authority")
    if not isinstance(events, list):
        raise ReplicaError("page.events must be a list")
    if not isinstance(authority, dict):
        raise ReplicaError("page.authority is required for replication")
    gateway_id = _validate_identifier(
        authority.get("gateway_id"), label="page.authority.gateway_id", max_chars=MAX_ACTOR_ID_CHARS)
    epoch = _positive_int(authority.get("epoch"), message="page.authority.epoch must be a positive integer")
    previous_seq: int | None = None
    for event in events:
        if not isinstance(event, dict):
            raise ReplicaError("page events must be objects")
        seq = _positive_int(event.get("seq"), message="event.seq must be a positive integer")
        if previous_seq is not None and seq != previous_seq + 1:
            raise ReplicaGapError("page events must be contiguous")
        previous_seq = seq
        for field in ("event_id", "kind"):
            if not isinstance(event.get(field), str) or not event[field]:
                raise ReplicaError(f"event.{field} must be a non-empty string")
        if not isinstance(event.get("actor"), dict):
            raise ReplicaError("event.actor must be an object")
        if "payload" not in event:
            raise ReplicaError("event.payload is required")
    return events, {"gateway_id": gateway_id, "epoch": epoch}


def ingest_page(
    store: Persistence, *, placement: HostedPlacement, room_id: Any, room_name: Any, members: Any, page: Any,
    now: float | None = None
) -> dict[str, Any]:
    """Persist one verbatim ``read_events()`` page idempotently; refuses seq gaps and epoch regressions."""
    room_id = _room_id(room_id)
    room_name = _validate_room_name(room_name)
    _, members_json = _validate_members(members)
    events, authority = _validate_page(page)
    latest_seq = page.get("latest_seq")
    replica_page = ReplicaPage(
        authority_gateway_id=authority["gateway_id"], authority_epoch=authority["epoch"], events=tuple(events),
        latest_seq=None if isinstance(latest_seq, bool) or not isinstance(latest_seq, int) else latest_seq)
    return dict(store.hosted_room_replicas.ingest_page(
        placement=placement, room_id=room_id, room_name=room_name, members_json=members_json, page=replica_page,
        now=clock(now)))


def replica_state(store: Persistence, *, placement: HostedPlacement, room_id: Any) -> dict[str, Any]:
    """Return the stored replica's coverage and authority lineage."""
    return dict(store.hosted_room_replicas.replica_state(placement=placement, room_id=_room_id(room_id)))


def promote_replica(
    store: Persistence, *, placement: HostedPlacement, room_id: Any, reason: Any = "authority-unreachable",
    now: float | None = None
) -> dict[str, Any]:
    """Continue a replicated room on THIS gateway at ``epoch + 1``.

    Copies the replica log into the authoritative store and appends a lineage-proving ``authority.claimed``
    event, so wherever the claim replicates the old epoch is stale and every fenced primitive rejects it.
    The caller decides takeover is safe; this makes it atomic and provable.
    """
    room_id = _room_id(room_id)
    if not isinstance(reason, str) or not reason or len(reason) > 200:
        raise ReplicaError("reason must be a non-empty string of at most 200 chars")
    now = clock(now)
    return dict(store.hosted_room_replicas.promote_replica(
        placement=placement, room_id=room_id, local_gateway_id=local_authority_gateway_id(), reason=reason, now=now))


def demote_room(
    store: Persistence, *, placement: HostedPlacement, room_id: Any, observed_gateway_id: Any, observed_epoch: Any,
    now: float | None = None
) -> dict[str, Any]:
    """Fence THIS gateway's stale room authority against a proven newer epoch.

    When a returning gateway observes (replicated ``authority.claimed`` or a transport rejection) that another
    gateway owns the room at a higher epoch, append ``authority.lost`` and adopt the observed lineage so no
    local send can commit at the stale epoch. Idempotent per lineage.
    """
    room_id = _room_id(room_id)
    observed_gateway_id = _validate_identifier(
        observed_gateway_id, label="observed_gateway_id", max_chars=MAX_ACTOR_ID_CHARS)
    observed_epoch = _positive_int(observed_epoch, message="observed_epoch must be a positive integer")
    now = clock(now)
    return dict(store.hosted_room_replicas.demote_room(
        placement=placement, room_id=room_id, observed_gateway_id=observed_gateway_id,
        observed_epoch=observed_epoch, local_gateway_id=local_authority_gateway_id(), now=now))
