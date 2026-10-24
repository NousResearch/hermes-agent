"""Receipt-owned UI orphan eligibility, fenced by the canonical owner's write transaction."""
import json
import time

from hermes_state_common import _sql_session_last_active
from hermes_state_local import POLICY_PREFIX, _end_idle_local_session
from hermes_state_local_lineage import validate_local_lineage
from hermes_state_maintenance import _not_pinned_sql
from hermes_state_mutation_guards import MUTATION_GUARD_REFUSALS, require_idle
from hermes_state_runtime import RuntimeStoreError, _epoch


def _expired_unprotected(conn, lineage, target, now, ttl):
    # Every segment matters: a new reset/compression target, activity on an ancestor, or
    # a pin anywhere in the conversation must keep the receipt's current target open.
    for sid in lineage:
        row = conn.execute(
            f'SELECT 1 FROM sessions s WHERE s.id=? AND s.started_at < ? '
            f'AND {_sql_session_last_active("s")} < ? AND {_not_pinned_sql("s")}',
            (sid, now - ttl, now - ttl)).fetchone()
        if row is None:
            return False
    # Same conservative sibling-backend ownership window as the legacy UI sweep.
    return conn.execute(
        'SELECT 1 FROM gateway_heartbeats h JOIN sessions s ON s.id=? '
        'WHERE h.last_heartbeat >= ? AND h.started_at <= s.started_at + ? LIMIT 1',
        (target, now - 2 * ttl, 2 * ttl)).fetchone() is None


def _reap_idle_local_session(db, *, epoch, receipt, ttl, stopped):
    """End only the unchanged, expired and unowned UI receipt; retain its route and history."""
    sid = receipt['session_id']

    def write(conn):
        _epoch(conn, epoch)
        if stopped():
            return 0
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchone()
        if saved is None or json.loads(saved[0]) != receipt:
            return 0  # a target/policy change since the gateway validated ownership
        target = validate_local_lineage(conn, receipt)
        lineage = list(dict.fromkeys([sid, *receipt.get('lineage', [target])]))
        if not _expired_unprotected(conn, lineage, target, time.time(), ttl):
            return 0
        try:
            require_idle(db, conn, lineage)
        except MUTATION_GUARD_REFUSALS:
            return 0
        except RuntimeStoreError as exc:
            if exc.reason not in ('session_busy', 'unknown_execution'):
                raise
            return 0
        if stopped():
            return 0
        return _end_idle_local_session(conn, db, epoch, sid, target, 'startup_orphan_reap')

    return db._execute_write(write)
