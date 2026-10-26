"""The live session feeds (``session.list`` live scope, ``session.active_list``) read off the loop.

Both feeds cover every session in ``authority.sessions``, and a gateway restart refills that registry
from receipts, so it can hold thousands of rows (mostly finished cron runs). Walking it one row at a
time on the loop cost three full hydrations plus a receipt JSON parse per session: the loop stalled
past 10 s, WS writes and the Telegram adapter's ``updater.stop()`` missed their deadlines (#141236).

Here the loop does only in-memory work (authorization and liveness from ``LiveSession``); the store
facts for a batch of sessions are ONE worker-thread read: picker columns, runtime revision and
generation, pending admissions, and the local receipt target's ``ended_at``.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from gateway.session_contract import SessionHandle, SessionRef
from hermes_state_runtime import RuntimeStoreError

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


def _store_facts(db, session_ids):
    """Worker thread: ``{sid: row}`` and ``{sid: [pending admission]}`` for *session_ids*, in two reads.

    ``target_ended_at`` is the local receipt target's ``ended_at`` (``closed_and_unviewed``'s read):
    SQLite extracts the target from the receipt, so no receipt is parsed in Python. A malformed or
    non-local receipt reads as no target, as ``_local_target`` does."""
    from hermes_state_local import POLICY_PREFIX
    ids = json.dumps(list(session_ids))
    with db._read_ctx() as conn:
        rows = conn.execute(
            "SELECT s.id, s.source, s.title, s.started_at, s.ended_at, s.message_count, s.model, s.hidden, "
            "s.archived, s.runtime_revision, s.runtime_generation, "
            "(SELECT t.ended_at FROM state_meta m JOIN sessions t ON t.id = CASE WHEN json_valid(m.value) "
            "THEN json_extract(m.value, '$.entry.session_id') END WHERE m.key = ? || s.id) AS target_ended_at "
            "FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?))", (POLICY_PREFIX, ids)).fetchall()
        pending = conn.execute(
            "SELECT target_session_id, admission_id, seq, status FROM session_admissions "
            "WHERE target_session_id IN (SELECT value FROM json_each(?)) "
            "AND status IN ('queued','started','unknown') ORDER BY seq", (ids,)).fetchall()
    by_session: dict[str, list] = {}
    for row in pending:
        by_session.setdefault(row['target_session_id'], []).append(dict(row))
    return {row['id']: dict(row) for row in rows}, by_session


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
    try:
        authority.authorize(actor, SessionRef(actor.profile_id, session_id), 'session:read')
    except RuntimeStoreError as exc:
        if exc.reason not in _SKIPPED_REASONS:
            raise
        return False
    return True


async def live_entries(authority, actor) -> list[LiveEntry]:
    """Every live session *actor* may read, with its handle, picker row and closed-and-unviewed verdict."""
    from gateway.session_local_close import closed_idle_unviewed
    entries = []
    snapshot = tuple(authority.sessions)
    for start in range(0, len(snapshot), LISTING_BATCH):
        batch = [sid for sid in snapshot[start:start + LISTING_BATCH] if _readable(authority, actor, sid)]
        if not batch:
            continue
        rows, pending = await asyncio.to_thread(_store_facts, authority.db, batch)
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
