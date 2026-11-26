"""The hand-off of a local conversation whose current target a messaging lane took over, run by the routing verbs
inside their transactions: the receipts that moved, their release, and the move of the conversation's other segments
onto the lane. Each is a module function on the caller's open connection."""

from __future__ import annotations

import json

from .._schema import _title_rank
from ..conversation._store.local import POLICY_PREFIX
from ..conversation._store.local_migration import LEGACY_PREFIX


def moved_receipts(conn, target_ids=None):
    """``{owner id: (moved row id, its source)}`` of local receipts whose current physical target row
    now carries another lane's route. An explicit repoint (``/handoff``, a platform ``/resume``)
    rewrote that row's routing identity, so the receipt can never restore the conversation again
    (the stored identity check in ``gateway.session_local_recovery._checked_entry``). Only the
    current target counts: an earlier ``/new`` segment that moved leaves the conversation the
    receipt guards local, and releasing it would strand that transcript. *target_ids* narrows the
    scan to the rows one repoint rewrote."""
    rows = conn.execute(
        "SELECT substr(m.key, ?), s.id, s.source FROM state_meta m JOIN sessions s "
        "ON s.id = CASE WHEN json_valid(m.value) THEN json_extract(m.value, '$.entry.session_id') END "
        "WHERE m.key GLOB ? AND s.session_key IS NOT NULL "
        "AND s.session_key IS NOT CASE WHEN json_valid(m.value) THEN json_extract(m.value, '$.route') END",
        (len(POLICY_PREFIX) + 1, POLICY_PREFIX + '*')).fetchall()
    return {owner: (target, source) for owner, target, source in rows
            if target_ids is None or target in target_ids}


def release_receipts(conn, owners):
    """Retire *owners*' local receipts (creation policy and legacy binding): the row they guarded now
    belongs to a messaging lane, which restores it from its route like any other platform session."""
    conn.executemany('DELETE FROM state_meta WHERE key IN (?,?)',
                     [(POLICY_PREFIX + owner, LEGACY_PREFIX + owner) for owner in owners])


_LANE_IDENTITY = 'session_key, source, user_id, chat_id, chat_type, thread_id, display_name, origin_json, transport_profile'


def hand_off_conversations(conn, moved):
    """Retire the receipts in *moved* (``moved_receipts``: a messaging lane took over each one's
    current target) and move the rest of each conversation onto that lane with it. Its earlier
    ``/new`` segments take the lane's identity, so they open there as read-only history (a lane
    moved past them) instead of staying local rows no receipt restores; the conversation's name
    follows its head, which the lane continues as its own conversation. Returns the receipts it
    handed off: one whose lineage is malformed stays as it is (its own reads refuse it, logged)."""
    handed = {}
    for owner, (target, source) in moved.items():
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + owner,)).fetchone()
        lineage = (_receipt(saved[0]) or {}).get('lineage', [owner]) if saved else [owner]
        if not isinstance(lineage, list) or not all(isinstance(sid, str) for sid in lineage):
            continue
        segments = list(dict.fromkeys([owner, *lineage]))
        conn.execute(f'UPDATE sessions SET ({_LANE_IDENTITY}) = (SELECT {_LANE_IDENTITY} FROM sessions WHERE id=?) '
                     'WHERE id IN (SELECT value FROM json_each(?)) AND id != ?', (target, json.dumps(segments), target))
        _title_follows_head(conn, owner, _segment_root(conn, segments, target))
        handed[owner] = (target, source)
    release_receipts(conn, handed)
    return handed


def _receipt(raw):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _segment_root(conn, segments, target):
    """The first row of *target*'s ``/new`` segment: back through the compressions behind it."""
    if target not in segments:
        return target
    index = segments.index(target)
    while index > 0 and (row := conn.execute('SELECT end_reason FROM sessions WHERE id=?',
                                               (segments[index - 1],)).fetchone()) and row[0] == 'compression':
        index -= 1
    return segments[index]


def _title_follows_head(conn, owner, head):
    """The conversation's name (a Bot Chat's registry title included) moves to its head, over a
    title of lower provenance there (a generated one); a name of equal rank the head carries stays
    (``_title_rank``: a pre-provenance title ranks as the user's)."""
    row = conn.execute('SELECT title, title_source, hidden FROM sessions WHERE id=?', (owner,)).fetchone()
    if head == owner or row is None or not row[0]:
        return
    kept = conn.execute('SELECT title, title_source FROM sessions WHERE id=?', (head,)).fetchone()
    if kept is None or (kept[0] and _title_rank(kept[1]) >= _title_rank(row[1])):
        return
    conn.execute('UPDATE sessions SET title=NULL WHERE id IN (?,?)', (owner, head))
    conn.execute('UPDATE sessions SET title=?, title_source=?, hidden=? WHERE id=?', (*row, head))
