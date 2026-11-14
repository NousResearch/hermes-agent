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

    ``target_ended_at`` is the local receipt target's ``ended_at`` (``closed_and_unviewed``'s read) and
    ``receipt_target_id`` is that target's id (``live_entries``' moved-on read): SQLite extracts both from
    the receipt, so no receipt is parsed in Python. A malformed or non-local receipt reads as no target,
    as ``_local_target`` does. The second read follows the receipt's key — a rotation re-points it under
    the chat's logical id (``advance_local_target`` uses ``parent.chat_id``), which for a plain local
    session is its own id. ``reset_successor`` covers the one rotation that writes no receipt at all: a
    messaging ``/new`` stamps the child's ``_reset_from`` (or, pre-marker, reuses the parent's session
    key) while the parent ends at a reset boundary, so the successor is read from the child row."""
    from hermes_state_common import _RESET_CHILD_SQL, _RESET_END_REASONS_SQL
    from hermes_state_local import POLICY_PREFIX
    ids = json.dumps(list(session_ids))
    # A reset parent with a successor continues elsewhere even though no receipt says so. Guarded by the
    # reset end reasons so a session ended some other way (and merely holding an old reset child) stays
    # resumable; ``_RESET_CHILD_SQL`` is the picker's own definition of a reset child, legacy shape included.
    reset_successor = (
        "s.end_reason IN (" + _RESET_END_REASONS_SQL + ") AND EXISTS ("
        "SELECT 1 FROM sessions child WHERE child.parent_session_id = s.id AND ("
        + _RESET_CHILD_SQL.format(a='child') + "))")
    with db._read_ctx() as conn:
        rows = conn.execute(
            "SELECT s.id, s.source, s.title, s.started_at, s.ended_at, s.end_reason, s.message_count, s.model, s.hidden, "
            "s.archived, s.runtime_revision, s.runtime_generation, "
            "(SELECT t.ended_at FROM state_meta m JOIN sessions t ON t.id = CASE WHEN json_valid(m.value) "
            "THEN json_extract(m.value, '$.entry.session_id') END WHERE m.key = ? || s.id) AS target_ended_at, "
            "(SELECT CASE WHEN json_valid(m.value) THEN json_extract(m.value, '$.entry.session_id') END "
            "FROM state_meta m WHERE m.key = ? || COALESCE(NULLIF(s.chat_id, ''), s.id)) AS receipt_target_id, "
            f"({reset_successor}) AS reset_successor "
            "FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?))",
            (POLICY_PREFIX, POLICY_PREFIX, ids)).fetchall()
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
    from gateway.session_retirement import LISTING_READ
    token = LISTING_READ.set(True)
    try:
        authority.authorize(actor, SessionRef(actor.profile_id, session_id), 'session:read')
    except RuntimeStoreError as exc:
        if exc.reason not in _SKIPPED_REASONS:
            raise
        return False
    finally:
        LISTING_READ.reset(token)
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
            if row.get('ended_at') is not None and (
                    row.get('end_reason') == 'compression'
                    or row.get('receipt_target_id') not in (None, sid)
                    or row.get('reset_successor')):
                # The conversation continues in another session. A rotation — compression, or a
                # ``session_reset`` / ``/new`` boundary — closes the parent, re-points the receipt
                # (and the route) at the child and leaves this mount standing, so an unfiltered
                # feed keeps offering an id every mutate refuses (``session_superseded``, #143767).
                # Read the moved-on target rather than the reason string: the same shape comes out
                # of every rotation, while an end nothing continues stays listed (those are
                # resumable). A messaging ``/new`` writes no receipt at all, so its parent is read
                # from the reset child instead (``reset_successor``, #146626 review).
                continue
            masked = _masked_pending(live, pending.get(sid, []))
            state = _execution_state(masked)
            handle = SessionHandle(SessionRef(authority.profile_id, sid), authority.instance_id, authority.epoch,
                                   row['runtime_revision'], row['runtime_generation'], state)
            entries.append(LiveEntry(sid, live, handle, row, masked,
                                     closed_idle_unviewed(live, state, row['target_ended_at'])))
    return entries
