"""Publish an exhausted native context without replacing its accepted FIFO owner."""

import asyncio
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from gateway.session import SessionEntry
from gateway.session_lifecycle import _new_session_id
from hermes_state_runtime import RuntimeStoreError


async def maybe_reset(runner, entry, route, source):
    from gateway.config import Platform
    from gateway.session_ingress import executing_claim
    from gateway.session_runtime_workers import track_mutation

    claim = executing_claim.get()
    if (
        not isinstance(claim, tuple)
        or "native_text_v1" not in claim[2]["payload"]
        or source.platform in {Platform.LOCAL, Platform.API_SERVER}
    ):
        return False, None
    authority, ref, row = claim
    if authority.runner is not runner or authority.profile_id != ref.profile_id:
        raise RuntimeStoreError("profile_mismatch")
    task = track_mutation(
        authority, _reset(authority, ref, row, replace(entry), route, source)
    )
    return True, await asyncio.shield(task)


def recovery_entry(store, previous_id, target_id, route):
    """Restore fresh fields after a committed reset outlives its routing-index save."""
    if not target_id or previous_id == target_id:
        return None
    from hermes_state_native_context_reset import reset_projection

    db = store._db_for_key(route)
    if db is None:
        return None
    saved = reset_projection(db, previous_id, target_id, route)
    return SessionEntry.from_dict(saved) if saved is not None else None


def _fresh_entry(original):
    now = datetime.now()
    return SessionEntry(
        session_key=original.session_key,
        session_id=_new_session_id(now),
        created_at=now,
        updated_at=now,
        origin=original.origin,
        platform=original.platform,
        chat_type=original.chat_type,
        display_name=original.display_name,
        transport_profile=original.transport_profile,
        is_fresh_reset=True,
    )


def automation_write_guard(authority, event, identity):
    """Fence a late completion whose snapshot raced the reset's SQLite commit."""
    from gateway.config import Platform

    guard = authority._admission_gate()
    expected = event.metadata["gateway_session_id"]
    pinned = event.source.platform != Platform.LOCAL and not _route_wake(identity)

    def check(conn):
        guard(conn)
        if pinned:
            from hermes_state_native_context_reset import require_context_target

            require_context_target(conn, expected)

    return check


def _route_wake(identity):
    import json
    from gateway.session_automation import _ROUTE_WAKE_PRODUCERS

    try:
        producer = json.loads(identity)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(producer, list)
        and bool(producer)
        and isinstance(producer[0], str)
        and producer[0] in _ROUTE_WAKE_PRODUCERS
    )


async def cancel_old_completion(authority, ref, row):
    """Retire a context-bound completion without stranding human followers behind it."""
    envelope = row["payload"].get("native_text_v1", {})
    if not envelope.get("automation") or _route_wake(row["request_id"]):
        return False
    from hermes_state_native_context_reset import predates_context_reset

    if not predates_context_reset(authority.db, ref.session_id, row["seq"]):
        return False
    from gateway.session_contract import Principal

    actor = Principal(
        "messaging-control",
        authority.profile_id,
        frozenset({"session:submit", "session:control"}),
        "reset",
    )
    await authority.cancel_queued(actor, ref, row["admission_id"])
    return True


async def _physical(authority, function, *args):
    from gateway.session_authorities import owner_scope

    def scoped():
        with owner_scope(authority):
            return function(*args)

    return await asyncio.to_thread(scoped)


def _commit(authority, ref, row, original, fresh):
    from hermes_state_native_context_reset import (
        ContextResetClaim,
        publish_context_reset,
    )

    store = authority.runner.session_store
    db = store._db_for_key(original.session_key)
    if db is None or Path(db.db_path).resolve() != Path(authority.db.db_path).resolve():
        raise RuntimeStoreError("profile_mismatch")
    claim = ContextResetClaim(
        epoch=authority.epoch,
        owner_id=ref.session_id,
        admission_id=row["admission_id"],
        generation=row["generation"],
        parent_id=original.session_id,
        profile_id=authority.profile_id,
    )

    def current():
        authority._require_admission_open()
        with store._lock:
            observed = store._entries.get(original.session_key)
            if observed is None or observed.session_id != claim.parent_id:
                raise RuntimeStoreError("admission_conflict")

    return publish_context_reset(db, claim, fresh.to_dict(), current)


def _publish(authority, live, original, fresh):
    runner, store = authority.runner, authority.runner.session_store
    authority._require_admission_open()
    if authority.sessions.get(live[0]) is not live[1]:
        raise RuntimeStoreError("admission_conflict")
    with store._lock:
        observed = store._entries.get(original.session_key)
        if observed is None or observed.session_id not in {
            original.session_id,
            fresh.session_id,
        }:
            raise RuntimeStoreError("admission_conflict")
        if observed.session_id == original.session_id:
            store._entries[original.session_key] = fresh
        else:
            fresh = observed  # a concurrent admitted capture already healed this exact child
    runner._evict_cached_agent(original.session_key)
    runner._clear_conversation_scope(
        original.session_key, reason="compression_exhausted_reset"
    )
    return fresh


async def _reset(authority, ref, row, original, route, source):
    live = authority.sessions.get(ref.session_id)
    if live is None or live.route != route or original.session_key != route:
        raise RuntimeStoreError("admission_conflict")
    async with live.mutation_lock:
        authority._require_admission_open()
        fresh = _fresh_entry(original)
        encoded = await _physical(
            authority, _commit, authority, ref, row, original, fresh
        )
        fresh = _publish(
            authority, (ref.session_id, live), original, SessionEntry.from_dict(encoded)
        )
        # Save the CURRENT projection. A later /new or capture must never be
        # overwritten by a delayed whole-index snapshot of this earlier reset.
        await _physical(authority, authority.runner.session_store._save_entries)
        await _physical(
            authority,
            _sync_topic_if_current,
            authority.runner,
            source,
            fresh,
        )
        return fresh


def _sync_topic_if_current(runner, source, entry):
    current = runner.session_store.lookup_by_session_key(entry.session_key)
    if current is not None and current.session_id == entry.session_id:
        runner._sync_telegram_topic_binding(
            source, entry, reason="compression-exhausted-reset"
        )
