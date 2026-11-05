"""Local conversation lineages: owners, segments and receipts."""


import json
import threading
from collections import OrderedDict


POLICY_PREFIX = 'gateway.local_policy.v1:'


API_BINDING_PREFIX = 'gateway.api.binding.v1.'


API_DECLARED_PREFIX = 'gateway.api.conversation.v1.'


def _policy_target(raw):
    """Current physical target a creation receipt routes to; None for a malformed receipt, which
    then stays (its own reads already refuse it as ``storage_unavailable``) and cannot abort
    the delete of an unrelated session."""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    entry = value.get('entry') if isinstance(value, dict) else None
    return entry.get('session_id') if isinstance(entry, dict) else None


def retire_local_receipts(conn, session_ids):
    """Deletion fence half for owner receipts. A local creation policy carries the launch request
    (``request_json``: a cron fire's job and raw prompt), so it goes exactly when its route does:
    when the receipt's current physical target is deleted (a pruned reset ancestor whose logical
    session lives on in a child keeps it). An API binding and the declared conversation key that
    resolves to it go with their id, so that key starts a fresh session."""
    targets = set(session_ids)
    if not targets:
        return
    # Prefix scans (GLOB uses the key index), not one LIKE per id: a prune retires thousands.
    gone = [key for key, value in conn.execute('SELECT key,value FROM state_meta WHERE key GLOB ?',
                                               (POLICY_PREFIX + '*',))
            if _policy_target(value) in targets]
    gone += [key for key, value in conn.execute('SELECT key,value FROM state_meta WHERE key GLOB ?',
                                                (API_DECLARED_PREFIX + '*',)) if value in targets]
    gone += [API_BINDING_PREFIX + sid for sid in targets]
    conn.executemany('DELETE FROM state_meta WHERE key=?', [(key,) for key in gone])


def local_lineage_owner(conn, session_id):
    """The creation id whose receipt lineage holds physical segment *session_id* (a reset or
    compression child, which carries the owner as ``chat_id``); *session_id* itself otherwise."""
    row = conn.execute('SELECT chat_id FROM sessions WHERE id=?', (session_id,)).fetchone()
    owner = row[0] if row else None
    if not owner or owner == session_id:
        return session_id
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + owner,)).fetchone()
    try:
        lineage = json.loads(saved[0]).get('lineage') if saved else None
    except (TypeError, ValueError, AttributeError):
        return session_id
    return owner if isinstance(lineage, list) and session_id in lineage else session_id


def owned_lineage_ids(conn, session_id):
    """[session_id] for any row outside a local reset/compression lineage, else the whole
    conversation (*session_id* first). Any segment deletes as all of it: the owner's policy goes
    with its current target, and an earlier segment's gap breaks the lineage, so a lone segment
    delete strands the rest (and the owner's queued work) readable but never continuable."""
    return list(dict.fromkeys([session_id, *local_conversation_ids(conn, session_id)]))


def local_conversation_ids(conn, session_id):
    """Every row of the local conversation *session_id* belongs to: its creation id first, then
    each segment of the receipt lineage. A canonical reset never forks the logical session (one
    policy, FIFO and generation), so listing, delete and retirement treat these ids as one."""
    owner = local_lineage_owner(conn, session_id)
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + owner,)).fetchone()
    try:
        lineage = json.loads(saved[0]).get('lineage') if saved else None
    except (TypeError, ValueError, AttributeError):
        lineage = None
    if not isinstance(lineage, list):
        return [session_id]
    return list(dict.fromkeys([owner, *(sid for sid in lineage if isinstance(sid, str))]))


def local_lineage_index(conn):
    """``(superseded, heads)`` for listing a local conversation as one row, as
    ``acp_adapter.catalog`` does: a canonical reset keeps the creation id as the owner and moves the
    transcript to a child, so every segment is one conversation. ``superseded`` holds the listable
    roots of earlier reset segments (filtered in SQL, before paging); ``heads`` maps the latest
    reset segment to ``(owner, lineage)`` for the row that represents the conversation. A delete
    of that row removes the whole lineage (``hermes_state_mutation_guards.delete_targets``)."""
    superseded, heads = set(), {}
    rows = conn.execute("SELECT value FROM state_meta WHERE key GLOB ? AND json_valid(value) "
                        "AND json_array_length(value, '$.lineage') > 1", (POLICY_PREFIX + '*',)).fetchall()
    for (raw,) in rows:
        receipt = json.loads(raw)
        owner, lineage = receipt.get('session_id'), receipt.get('lineage')
        if not isinstance(owner, str) or not all(isinstance(sid, str) for sid in lineage):
            continue
        marks = ','.join('?' * len(lineage))
        ended = dict(conn.execute(f'SELECT id,end_reason FROM sessions WHERE id IN ({marks})', lineage).fetchall())
        starts = [sid for i, sid in enumerate(lineage) if i == 0 or ended.get(lineage[i - 1]) == 'session_reset']
        superseded.update(starts[:-1])
        heads[starts[-1]] = (owner, lineage)
    return superseded, heads


def _lineage_version(conn):
    # data_version sees OTHER connections' commits; total_changes sees this connection's writes.
    return conn.total_changes, conn.execute('PRAGMA main.data_version').fetchone()[0]


def _copy_lineage_index(value):
    superseded, heads = value
    return set(superseded), {key: (owner, list(lineage)) for key, (owner, lineage) in heads.items()}


def _cached_local_lineage_index(db, conn):
    # Owned by this SessionDB, not a process-global path/id cache. Bound retained connections
    # across read-pool churn and close/reopen; a closed/replaced connection cannot match a new one.
    cache, lock = db.__dict__.setdefault('_local_lineage_cache', (OrderedDict(), threading.Lock()))
    if conn.in_transaction:
        # A snapshot or uncommitted answer must never outlive its transaction (including rollback).
        with lock:
            cache.pop(conn, None)
        return local_lineage_index(conn)
    version = _lineage_version(conn)
    with lock:
        saved = cache.get(conn)
        if saved is not None and saved[0] == version:
            cache.move_to_end(conn)
            return _copy_lineage_index(saved[1])
    value = local_lineage_index(conn)
    # A concurrent writer may commit between the receipt scan and the end_reason reads. Never
    # publish that in-flight projection as a stable cache entry; the next call must rebuild it.
    if not conn.in_transaction and _lineage_version(conn) == version:
        with lock:
            cache[conn] = (version, value)
            cache.move_to_end(conn)
            if len(cache) > 8:
                cache.popitem(last=False)
    return _copy_lineage_index(value)


def exclude_superseded_segments(db, where_clauses, params):
    """Drop earlier reset segments from a listing or count query (one conversation, one row),
    in SQL before paging; returns ``heads`` for :func:`annotate_local_lineages`."""
    with db._read_ctx() as conn:
        superseded, heads = _cached_local_lineage_index(db, conn)
    if superseded:
        where_clauses.append("s.id NOT IN (SELECT value FROM json_each(?))")
        params.append(json.dumps(sorted(superseded)))
    return heads


def annotate_local_lineages(sessions, heads):
    """Name every segment on the conversation's representative row (``_lineage_ids``, which the
    Desktop matches deletes, tombstones and stored ids against). ``_lineage_root_id`` stays the
    compression root: pins and archive flags live on the listed segment's own chain."""
    for row in sessions:
        found = heads.get(row.get('_lineage_root_id') or row['id'])
        if found is not None:
            row['_lineage_ids'] = list(dict.fromkeys([*found[1], *(row.get('_lineage_ids') or ())]))
    return sessions
