"""Read-only delivery status (``delivery.status``, Q5 puente B).

Answers "what deferred deliveries does this session still owe?" for the status bar, from the
two sources the core owns: the transcript's detached-result rows and the bot relay receipts.
The plugin outbox (``delivery_store.py`` / ``deliveries.db``) lives in the plugin's home and
the core never reads it, so it is outside this verb's contract by design.

"Pending" is a documented approximation, not a stored flag: a detached delivery recorded
after the session's last assistant row. The core keeps no "seen" column on ``messages``
(that would be a schema change of its own), so a delivery already seen through another
channel still counts until the next turn runs; ``last_delivery_at`` lets the client reason
for itself.

Read-only in the strong sense: two indexed ``SELECT``s per session over its resume lineage
(``session_id IN (...)`` under ``idx_messages_session``) plus one scan of the profile home's
relay mailbox, filtered by session. No writes, no admissions, no turn, lease or cache touch,
no result caching — every call reads the current state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from persistence.core.errors import AuthorityRefusal as RuntimeStoreError

_log = logging.getLogger(__name__)

# Detached API delegation deliveries, as ``hermes_state_detached_results`` records them.
_DELIVERY_KINDS = ("async_delegation_complete", "hidden")
_DEFAULT_WINDOW_SECONDS = (
    86400  # delivered_recent counts the last day unless asked otherwise
)
_MAX_WINDOW_SECONDS = 30 * 86400
_MAX_LIMIT = 200

_SOURCE_MAP = {
    "pending": "transcript",
    "delivered_recent": "transcript+relay",
    "dead_letter": "relay",
    "last_delivery_at": "transcript+relay",
}


def _lineage_map(store, session_ids):
    """``{session_id: its resume lineage}``: the tip a pre-rotation id resolves to, then every
    compression ancestor a resume loads. A delivery recorded before a compaction lives on the
    parent segment; without the lineage the verb would go blind exactly when a long session
    compacts.

    Both halves are the store's own lineage verbs: ``resolve_resume_target`` is the tip a
    resume loads and ``resume_lineage_ids`` the segments it reads (the same pair
    ``conversation.lineage`` serves every other reader)."""
    from persistence.core.identity import SessionId

    out = {}
    for session_id in session_ids:
        tip = store.lineage.resolve_resume_target(SessionId(session_id))
        out[session_id] = store.lineage.resume_lineage_ids(SessionId(tip))
    return out


def _transcript_statuses(store, db_path, lineages, window_seconds, now):
    """Per-session detached-delivery aggregates over the session's resume lineage.

    Returns ``{session_id: (pending, oldest_pending_ts, recent, last_ts, total)}``. The pending cut
    is the lineage's last assistant row: a delivery recorded after it has not been consumed
    by a turn yet (detached rows are only ever recorded between turns). The read mirrors the
    display set — ``(active = 1 OR compacted = 1)`` — and dedupes by ``message_uid``: a
    compaction that carries the protected tail writes the same logical row again under a new
    id, and the superseded original is the one the filter drops.
    """
    from contextlib import closing
    from hermes_cli.sqlite_safe_read import connect_tracked

    path = Path(db_path).resolve()
    out = {}
    with closing(connect_tracked(path.as_uri() + "?mode=ro", uri=True, tracking_path=path)) as conn:
        for session_id, lineage in lineages.items():
            marks = ",".join("?" * len(lineage))
            last_assistant = conn.execute(
                "SELECT MAX(timestamp) FROM messages "
                f"WHERE session_id IN ({marks}) AND role = 'assistant' AND (active = 1 OR compacted = 1)",
                lineage,
            ).fetchone()[0]
            last_assistant = last_assistant if last_assistant is not None else 0
            rows = conn.execute(
                "SELECT id, message_uid, timestamp FROM messages "
                f"WHERE session_id IN ({marks}) AND (active = 1 OR compacted = 1) "
                f"AND display_kind IN ({','.join('?' * len(_DELIVERY_KINDS))}) "
                "AND json_extract(display_metadata, '$.delegation_id') IS NOT NULL",
                (*lineage, *_DELIVERY_KINDS),
            ).fetchall()
            seen, pending, recent, last_ts, oldest_pending = set(), 0, 0, None, None
            for row_id, uid, timestamp in rows:
                key = uid or row_id
                if key in seen:
                    continue  # a carried copy of a row the lineage already holds
                seen.add(key)
                if timestamp >= now - window_seconds:
                    recent += 1
                if last_ts is None or timestamp > last_ts:
                    last_ts = timestamp
                if timestamp > last_assistant:
                    pending += 1
                    if oldest_pending is None or timestamp < oldest_pending:
                        oldest_pending = timestamp
            out[session_id] = (pending, oldest_pending, recent, last_ts, len(seen))
    return out


def _coerce_epoch(value):
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    return ts if ts > 0 else None


async def _relay_statuses(home, owner_of, window_seconds, now):
    """Per-session relay-receipt aggregates for the profile home's mailbox.

    ``owner_of`` maps every lineage member to the session it is counted under, so a receipt
    filed under a pre-rotation id belongs to the same conversation as the transcript rows
    beside it. A receipt counts only for its own session and its own home (``profile_home``
    must match this authority's home when present, so a stray file from another profile
    never leaks in). ``failed`` is the relay's dead letter; ``settled`` inside the window
    counts as a recent delivery. Malformed receipts are skipped by the mailbox scan itself.
    """
    from gateway.session_bot_mailbox import scan_receipts

    wanted = set(owner_of)
    home_str = str(home)
    settled_recent = {sid: 0 for sid in owner_of.values()}
    dead_letter = {sid: 0 for sid in owner_of.values()}
    last_settled = {sid: None for sid in owner_of.values()}
    from tools.bot_live_delivery import DELIVERY_DIR_NAME

    root = Path(home).resolve() / "runtime" / DELIVERY_DIR_NAME
    if not root.is_dir():
        # No mailbox, no receipts — and no dir to create: a read must not conjure one.
        return settled_recent, dead_letter, last_settled
    try:
        _root, records = await scan_receipts(home)
    except (OSError, ValueError) as exc:
        # A mailbox the owner cannot read (permissions, a torn dir) answers zeros, never a
        # refusal: the transcript numbers above are still the verb's answer. Loud, so a
        # missing dead-letter count is never silent.
        _log.warning(
            "delivery.status: relay mailbox unreadable at %s (%s); relay counts read as zero",
            root,
            exc,
        )
        return settled_recent, dead_letter, last_settled
    for _path, record in records:
        sid = record.get("session_id")
        if sid not in wanted:
            continue
        owner = owner_of[sid]
        owner_home = record.get("profile_home") or (record.get("owner") or {}).get(
            "profile_home"
        )
        if owner_home is not None and owner_home != home_str:
            continue
        status = record.get("status")
        if status == "failed":
            dead_letter[owner] += 1
        elif status == "settled":
            ts = _coerce_epoch(record.get("created_at"))
            if ts is not None:
                if ts >= now - window_seconds:
                    settled_recent[owner] += 1
                if last_settled[owner] is None or ts > last_settled[owner]:
                    last_settled[owner] = ts
    return settled_recent, dead_letter, last_settled


async def delivery_status(connection, ref, params):
    """``delivery.status`` handler: ``session:read`` + the owning profile, then read-only
    aggregation. Exactly one of ``session_id`` / ``scope="all"``; neither is fail-closed.
    """
    authority, actor = connection.authority, connection.actor
    if "session:read" not in actor.capabilities:
        raise RuntimeStoreError("permission_denied")
    session_id = params.get("session_id")
    scope = params.get("scope")
    # The contract's null is "not provided": an explicit null takes the default, as an
    # omitted key does; only a non-null out-of-range value is refused.
    limit = params.get("limit")
    limit = _MAX_LIMIT if limit is None else limit
    window = params.get("window_seconds")
    window = _DEFAULT_WINDOW_SECONDS if window is None else window
    if type(limit) is not int or not 1 <= limit <= _MAX_LIMIT:
        raise RuntimeStoreError("invalid_params", field="limit")
    if type(window) is not int or not 1 <= window <= _MAX_WINDOW_SECONDS:
        raise RuntimeStoreError("invalid_params", field="window_seconds")
    if scope is not None and scope != "all":
        raise RuntimeStoreError("invalid_params", field="scope")
    if session_id is not None and (not isinstance(session_id, str) or not session_id):
        raise RuntimeStoreError("invalid_params", field="session_id")
    if (session_id is None) == (scope is None):
        # Both set, or neither: the call names no single session and no list.
        raise RuntimeStoreError("invalid_params", field="session_id")
    now = time.time()
    if session_id is not None:
        # The same resolve ``session.info`` uses: profile + capability + existence, and a
        # local session answers only to its owner. It writes no delivery state.
        await authority.resolve(actor, ref)
        session_ids = [session_id]
        scope_name = "session"
    else:
        from gateway.session_live_listing import live_entries
        from persistence.conversation.sessions import INTERNAL_LISTING_SOURCES

        # The same visible feed ``session.list`` defaults to: live sessions this actor may
        # read, minus internal sources, closed-and-unviewed rows and hidden/archived ones.
        session_ids = []
        for entry in await live_entries(authority, actor):
            row = entry.row
            if (
                row.get("source") in INTERNAL_LISTING_SOURCES
                or entry.closed
                or row.get("hidden")
                or row.get("archived")
            ):
                continue
            session_ids.append(entry.session_id)
            if len(session_ids) >= limit:
                break
        scope_name = "all"
    store = authority.persistence
    from persistence_assembly import store_location

    home = Path(store.scope.profile_dir).resolve()
    db_path = store_location(store.scope)
    lineages = await asyncio.to_thread(_lineage_map, store, session_ids)
    # Lineages of live sessions are disjoint (a compression-ended parent is never live);
    # a member names the one session it is counted under.
    owner_of = {member: sid for sid, lineage in lineages.items() for member in lineage}
    transcript = await asyncio.to_thread(
        _transcript_statuses, store, db_path, lineages, window, now
    )
    settled_recent, dead_letter, last_settled = await _relay_statuses(
        home, owner_of, window, now
    )
    sessions = []
    for sid in session_ids:
        pending, oldest_pending, recent, last_ts, _total = transcript[sid]
        relay_last = last_settled[sid]
        last_delivery = max(
            (ts for ts in (last_ts, relay_last) if ts is not None), default=None
        )
        sessions.append(
            {
                "session_id": sid,
                "pending": pending,
                "oldest_pending_age_seconds": (now - oldest_pending)
                if pending and oldest_pending
                else None,
                "delivered_recent": recent + settled_recent[sid],
                "dead_letter": dead_letter[sid],
                "last_delivery_at": last_delivery,
                "source": dict(_SOURCE_MAP),
            }
        )
    return {"sessions": sessions, "scope": scope_name, "window_seconds": window}
