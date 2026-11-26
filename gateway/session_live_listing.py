"""The live session feeds (``session.list`` live scope, ``session.active_list``) read off the loop.

Both feeds cover every session in ``authority.sessions``, which can hold thousands of rows (mostly
finished cron runs; boot restores only open sessions, an ended one comes back on first touch).
Walking it one row at a time on the loop cost three full hydrations plus a receipt JSON parse per
session: the loop stalled past 10 s, WS writes and the Telegram adapter's ``updater.stop()`` missed
their deadlines (#141236).

Here the loop does only in-memory work (authorization and liveness from ``LiveSession``); the store
facts for a batch of sessions are ONE worker-thread read: picker columns, runtime revision and
generation, pending admissions, and the local receipt target's ``ended_at``.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from gateway.session_contract import SessionHandle, SessionRef
from persistence.core.errors import AuthorityRefusal

# Sessions per worker-thread read; each batch is one await, so the loop serves other work between them.
LISTING_BATCH = 256

_SKIPPED_REASONS = frozenset({'permission_denied', 'profile_mismatch', 'not_found'})


@dataclass(frozen=True)
class LiveEntry:
    session_id: str
    live: object
    handle: SessionHandle
    row: dict
    pending: list
    closed: bool


def _store_facts(store, session_ids):
    """Worker thread: ``{sid: row}`` and ``{sid: [pending admission]}`` for *session_ids*, in one read.

    ``target_ended_at`` is the local receipt target's ``ended_at`` (``closed_and_unviewed``'s read). A
    malformed or non-local receipt reads as no target, as ``_local_target`` does."""
    facts = store.sessions.live_listing_facts(session_ids)
    pending = {sid: [{'target_session_id': sid, **asdict(ref)} for ref in refs] for sid, refs in facts.pending.items()}
    return {sid: asdict(row) for sid, row in facts.rows.items()}, pending


def _masked_pending(live, rows):
    """``SessionAuthority._pending_rows``: a settlement not yet published still reads as its started claim."""
    if not live.settling:
        return rows
    merged = {row['admission_id']: row for row in rows}
    merged.update(live.settling)
    return sorted(merged.values(), key=lambda row: row['seq'])


def _execution_state(pending) -> str:
    statuses = {row['status'] for row in pending}
    return 'unknown' if 'unknown' in statuses else ('running' if 'started' in statuses else 'idle')


def _readable(authority, actor, session_id) -> bool:
    if session_id not in authority.sessions:
        return False  # retired since the snapshot: a listing never restores a session
    from gateway.session_retirement import LISTING_READ
    token = LISTING_READ.set(True)
    try:
        authority.authorize(actor, SessionRef(actor.profile_id, session_id), 'session:read')
    except AuthorityRefusal as exc:
        if exc.reason not in _SKIPPED_REASONS:
            raise
        return False
    finally:
        LISTING_READ.reset(token)
    return True


async def live_entries(authority, actor) -> list[LiveEntry]:
    """Every live session *actor* may read, with its handle, picker row and closed-and-unviewed verdict."""
    from gateway.session_local_close import closed_idle_unviewed
    from gateway.session_persistence import authority_persistence
    entries = []
    snapshot = tuple(authority.sessions)
    for start in range(0, len(snapshot), LISTING_BATCH):
        batch = [sid for sid in snapshot[start:start + LISTING_BATCH] if _readable(authority, actor, sid)]
        if not batch:
            continue
        rows, pending = await asyncio.to_thread(_store_facts, authority_persistence(authority), batch)
        if authority.retiring or authority.retired:
            return []  # parked or unserved during the read: nothing here is this profile's to list any more
        for sid in batch:
            live, row = authority.sessions.get(sid), rows.get(sid)
            if live is None or row is None:
                continue  # retired during the read, or its row was deleted (``_handle``'s not_found)
            masked = _masked_pending(live, pending.get(sid, []))
            state = _execution_state(masked)
            handle = SessionHandle(SessionRef(authority.profile_id, sid), authority.instance_id, authority.epoch,
                                   row['runtime_revision'], row['runtime_generation'], state)
            entries.append(LiveEntry(sid, live, handle, row, masked,
                                     closed_idle_unviewed(live, state, row['target_ended_at'])))
    return entries
