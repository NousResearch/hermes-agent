"""The reclaim of aged-out deletion tombstones the prune runs inside its deleting transaction, a module function on the
open connection. The deletion fence itself is the conversation store's (``conversation._store.retirement``)."""

from __future__ import annotations

import json
import time

from ..conversation._store.retirement import RETIRED_PREFIX
from ..conversation._store.terminal import ADMISSION_PREFIX, RESULT_PREFIX, WORKER_PREFIX, identity_key
from ..gateway._routes import _EXPIRED, RETRY_HORIZON_S, ROUTE_FENCE_PREFIX, _reclaim_expired

__all__ = ["prune_retired_tombstones"]


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
        conn.execute('DELETE FROM state_meta WHERE key IN (?,?,?)', (key, RESULT_PREFIX + row['admission_id'],
                     identity_key(row['principal_id'], row['target_session_id'], row['request_id'])))
    conn.execute(f'DELETE FROM state_meta WHERE key GLOB ? AND {_EXPIRED}', (WORKER_PREFIX + '*', cutoff))
    for prefix in (RETIRED_PREFIX, ROUTE_FENCE_PREFIX):
        _reclaim_expired(conn, prefix, cutoff)
