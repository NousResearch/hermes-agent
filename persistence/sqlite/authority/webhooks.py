"""SQLite's narrow webhook delivery/origin capability."""
from __future__ import annotations

import json

from persistence.authority.admissions import AdmissionGate, raise_gate
from persistence.authority.webhooks import StoreIncarnation, WebhookLedger, WebhookRecord
from persistence.core.errors import AuthorityRefusal
from .._backend import register_repository
from ._runtime import _epoch, refused
from . import _webhook_store as store


class SQLiteWebhookDeliveryRepository:
    def __init__(self, bound):
        self._store = bound

    def get(self, ledger: WebhookLedger, origin_id: str) -> WebhookRecord | None:
        with refused(), self._store.holder()._read_ctx() as conn:
            raw = conn.execute('SELECT value FROM state_meta WHERE key=?', (WebhookLedger(ledger).value + origin_id,)).fetchone()
            return store.record(ledger, origin_id, raw[0]) if raw else None

    def publish(self, *, epoch: int, record: WebhookRecord, native=None, claim=False,
                gate: AdmissionGate = None) -> WebhookRecord:
        with refused():
            return self._store.holder()._execute_write(lambda conn: store.publish(
                conn, epoch=epoch, change=record, native=native, claim=claim, gate=gate))

    def candidates(self, ledger: WebhookLedger, *, after='', limit=8) -> tuple[str, ...]:
        ledger = WebhookLedger(ledger)
        limit = max(1, min(limit, 50))
        prefix = ledger.value
        closed = store.CLOSED if ledger == WebhookLedger.FINAL else store.ORIGIN_CLOSED
        values = ','.join('?' for _ in closed)
        query = ("SELECT json_extract(value,'$.origin_id') AS origin_id FROM state_meta "
                 "WHERE key>=? AND key<? AND json_valid(value) "
                 f"AND json_extract(value,'$.state') NOT IN ({values})")
        params = [prefix, prefix + '~', *sorted(closed)]
        if ledger == WebhookLedger.FINAL:
            query += """ UNION SELECT a.admission_id AS origin_id FROM session_admissions a
                WHERE a.status IN ('terminal','unknown')
                AND json_extract(a.payload_json,'$.native_text_v1.source.platform')='webhook'
                AND (json_extract(a.payload_json,'$.native_text_v1.automation') IS NULL
                  OR json_extract(a.payload_json,'$.native_text_v1.webhook_root') IS NOT NULL)
                AND (json_extract(a.payload_json,'$.native_text_v1.webhook_delivery.deliver')='bot-chat'
                  OR json_extract(a.payload_json,'$.native_text_v1.webhook_delivery.deliver') LIKE 'bot-chat:%')
                AND NOT EXISTS(SELECT 1 FROM state_meta m WHERE m.key=? || a.admission_id)"""
            params.append(prefix)
        with refused(), self._store.holder()._read_ctx() as conn:
            return tuple(row[0] for row in conn.execute(
                'SELECT origin_id FROM (' + query + ') WHERE origin_id>? ORDER BY origin_id LIMIT ?',
                (*params, after, limit)))

    def inspection(self, *, after='', routes=None, limit=51):
        fields = ('origin_id', 'origin_kind', 'delivery_id', 'source_outcome', 'target', 'state',
                  'native_status', 'native_admission_id', 'source_admission_id', 'reason')
        columns = ','.join(f"json_extract(value,'$.{field}') AS {field}" for field in fields)
        clauses, params = [], []
        for ledger in WebhookLedger:
            prefix = ledger.value
            clauses.append('(key>=? AND key<? AND length(substr(key,?)) IN (32,64) '
                           "AND substr(key,?) NOT GLOB '*[^a-f0-9]*')")
            params.extend((prefix, prefix + '~', len(prefix) + 1, len(prefix) + 1))
        query = 'SELECT key,' + columns + ' FROM state_meta WHERE (' + ' OR '.join(clauses) + ')'
        query += ' AND key>? AND json_valid(value)'
        params.append(after)
        if routes is not None:
            query += " AND json_extract(value,'$.route') IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(routes))
        query += ' ORDER BY key LIMIT ?'
        params.append(max(1, min(limit, 51)))
        with refused(), self._store.holder()._read_ctx() as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_meta'").fetchone():
                return ()
            return tuple(dict(row) for row in conn.execute(query, params))

    def check_owner(self, *, epoch: int, gate: AdmissionGate = None) -> None:
        def check(conn):
            _epoch(conn, epoch)
            raise_gate(gate)
        with refused():
            self._store.holder()._execute_write(check)

    def capture_incarnation(self) -> StoreIncarnation:
        with refused():
            holder = self._store.holder()
            with holder._read_ctx() as conn:
                return store.capture(holder, conn)

    def require_incarnation(self, expected: StoreIncarnation) -> None:
        with refused():
            # The expected proof names an already-open writer. A registry sweep
            # must not reacquire/initialize a replacement before this check.
            if self._store._held is None:
                raise AuthorityRefusal('admission_conflict')
            holder = self._store.holder(reacquire=False)
            with holder._lock:
                if holder._conn is None:
                    raise AuthorityRefusal('admission_conflict')
                store.require_incarnation(holder, holder._conn, expected)


register_repository('webhook_deliveries', SQLiteWebhookDeliveryRepository, {'webhook_delivery_fences': True})
