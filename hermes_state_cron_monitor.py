"""Payload retention for sessionless cron observations, outside transcript rows."""
import json

from hermes_state_keys import admission_fingerprint
from hermes_state_runtime import _json

PREFIX = 'gateway.cron_monitor.v1:'


def fingerprint(session_id, params):
    return admission_fingerprint(canonical_target=session_id, payload=params)


def prune_monitor_results(conn, *, now, horizon):
    """Keep exact replies for the retry horizon, then only a non-reexecution fence.

    A compact fence never expires: forgetting an observed fire permits its source
    side effects again. Unknown/prepared observations are not terminal receipts.
    """
    terminal = "json_valid(value) AND json_extract(value, '$.status') = 'terminal'"
    conn.execute("UPDATE state_meta SET value=json_set(value, '$.settled_at', ?) "
                 f"WHERE key GLOB ? AND {terminal} "
                 "AND json_extract(value, '$.settled_at') IS NULL", (now, PREFIX + '*'))
    rows = conn.execute(f"SELECT key,value FROM state_meta WHERE key GLOB ? AND {terminal} "
                        "AND json_extract(value, '$.settled_at') < ?",
                        (PREFIX + '*', now - horizon)).fetchall()
    for key, raw in rows:
        record = json.loads(raw)
        fence = {'status': 'expired', 'settled_at': record['settled_at'],
                 'fingerprint': fingerprint(key[len(PREFIX):], record['params'])}
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(fence), key))
