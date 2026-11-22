"""The reclaim of aged-out deletion tombstones the prune runs inside its deleting transaction, a module function on the
open connection. The deletion fence itself is the conversation store's (``conversation._store.retirement``)."""

from __future__ import annotations

import json
import time

from ..conversation._store.retirement import RETIRED_PREFIX
from ..conversation._store.terminal import (
    ADMISSION_PREFIX, RESULT_PREFIX, SETTLED_PREFIX, WORKER_PREFIX, identity_key, settled_key,
)
from persistence.core.canonical import canonical_json
from ..gateway._routes import _EXPIRED, RETRY_HORIZON_S, ROUTE_FENCE_PREFIX, _reclaim_expired

__all__ = ["prune_retired_tombstones", "trim_settled_results"]

# Copies of the turn's transcript inside a stored result: dead once the exact-retry horizon passes,
# because the live transcript in ``messages`` already holds them.
_TRANSCRIPT_COPY_KEYS = frozenset({'_turn_user_row', 'last_reasoning', 'tools'})


def prune_retired_tombstones(conn, older_than=RETRY_HORIZON_S):
    """Reclaim deletion tombstones retired more than *older_than* seconds ago: the terminal admission
    with its identity pointer and compacted result, terminal workers and a sweep's (``expires``)
    retired-session marker or shared routing fence. A user delete's marker is the deletion fence itself and is never
    reclaimed: it holds only ``retired_at``, and without it a deleted id could be recreated,
    re-admitted or reissued as a Bot Chat generation.
    Tombstones written before ``retired_at`` existed are stamped now and age from this pass."""
    prefixes = (ADMISSION_PREFIX, WORKER_PREFIX)
    now = time.time()
    for prefix in prefixes:
        conn.execute("UPDATE state_meta SET value=json_set(value, '$.retired_at', ?) WHERE key GLOB ? "
                     "AND json_valid(value) AND json_extract(value, '$.retired_at') IS NULL", (now, prefix + '*'))
    cutoff = now - older_than
    for key, raw in conn.execute(f'SELECT key,value FROM state_meta WHERE key GLOB ? AND {_EXPIRED}',
                                 (ADMISSION_PREFIX + '*', cutoff)).fetchall():
        row = json.loads(raw)
        conn.execute('DELETE FROM state_meta WHERE key IN (?,?,?,?)', (key, RESULT_PREFIX + row['admission_id'],
                     settled_key(row['admission_id']),
                     identity_key(row['principal_id'], row['target_session_id'], row['request_id'])))
    conn.execute(f'DELETE FROM state_meta WHERE key GLOB ? AND {_EXPIRED}', (WORKER_PREFIX + '*', cutoff))
    for prefix in (RETIRED_PREFIX, ROUTE_FENCE_PREFIX):
        _reclaim_expired(conn, prefix, cutoff)


def trim_settled_results(conn, older_than=RETRY_HORIZON_S):
    """Trim settled admission results of LIVE sessions past the exact-retry horizon.

    The settlement records a ``gateway.admission.settled_at.v1.`` stamp beside its result. Once a
    result is older than *older_than* it can no longer answer an exact retry -- the same horizon that
    bounds a retired target's tombstone -- yet nothing collected it while its session stayed alive, so
    one long-lived chat kept one full result per turn forever (751 rows / 127 MB on the reference
    install, 96% of it the per-turn ``messages`` copies, and it grew with every turn).

    Trimmed, not deleted: the row keeps ``final_response``, the outcome flags, usage and accounting,
    and only the transcript copies go (``messages``, the turn's user row, reasoning, the tool schema
    list). That distinction is load-bearing -- ``finish_result_precheck`` and the hosted-task
    recovery path treat "terminal admission with no stored result" as a storage fault, and the bot
    receipt path degrades a settled turn to ``ambiguous`` without a ``final_response``. A trimmed row
    keeps every reader answering the same thing it answered before the horizon.

    A result stored before the stamp existed is stamped now and ages from that pass, exactly like an
    untimestamped tombstone. Returns the number of rows trimmed.
    """
    from ..conversation._store.terminal import encode_result, load_result
    now = time.time()
    # Backfill: every stored result without a stamp ages from this pass.
    conn.execute(
        "INSERT INTO state_meta(key,value) "
        "SELECT ? || substr(m.key, ?), ? FROM state_meta m WHERE m.key GLOB ? "
        "AND NOT EXISTS (SELECT 1 FROM state_meta s WHERE s.key = ? || substr(m.key, ?)) "
        "ON CONFLICT(key) DO NOTHING",
        (SETTLED_PREFIX, len(RESULT_PREFIX) + 1, str(int(now)), RESULT_PREFIX + '*',
         SETTLED_PREFIX, len(RESULT_PREFIX) + 1))
    cutoff = now - older_than
    aged = [key for (key,) in conn.execute(
        'SELECT key FROM state_meta WHERE key GLOB ? AND CAST(value AS INTEGER) < ?',
        (SETTLED_PREFIX + '*', cutoff)).fetchall()]
    trimmed = 0
    for stamp_key in aged:
        result_key = RESULT_PREFIX + stamp_key[len(SETTLED_PREFIX):]
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (result_key,)).fetchone()
        if row is None:
            continue
        try:
            value = load_result(row[0])
        except (ValueError, TypeError):
            continue  # one unreadable row must not abort the pass for the rest
        if not isinstance(value, dict):
            continue
        inner = value.get('result')
        if not isinstance(inner, dict):
            continue
        if not isinstance(inner.get('final_response'), str) or not inner['final_response'].strip():
            # A settled-but-empty reply is reconstructed from ``messages`` by the Bot delivery
            # fallback (``session_bot_outcome.forgotten_reply``); trimming it would erase the only
            # copy of what the bot said. Keep those rows whole.
            continue
        if not inner.get('messages') and not (_TRANSCRIPT_COPY_KEYS & inner.keys()):
            continue  # already trimmed: a repeated pass is a no-op
        value['result'] = {**{k: v for k, v in inner.items()
                              if k not in _TRANSCRIPT_COPY_KEYS and k != 'messages'}, 'messages': []}
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (canonical_json(encode_result(value)), result_key))
        trimmed += 1
    return trimmed
