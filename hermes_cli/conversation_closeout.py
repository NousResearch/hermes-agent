"""Durable closeout receipts; optional online preservation never owns a run permit."""

from __future__ import annotations

import json
import time
import uuid

from tui_gateway.contracts.closeout import SessionCloseoutResult

_PREFIX = 'conversation_closeout:v1:'
_PUBLIC = frozenset(SessionCloseoutResult.model_fields)


def public_receipt(record: dict) -> dict:
    return SessionCloseoutResult.model_validate(
        {key: value for key, value in record.items() if key in _PUBLIC}
    ).model_dump(exclude_none=True)


def _load(conn, key, *, required=False):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if row is None:
        if required:
            raise ValueError('closeout_receipt_unreadable')
        return None
    try:
        return json.loads(row[0])
    except (ValueError, TypeError) as exc:
        raise ValueError('closeout_receipt_unreadable') from exc


def _save(conn, key, value):
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                 'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                 (key, json.dumps(value, sort_keys=True, separators=(',', ':'))))


def _validate(record):
    try:
        public_receipt(record)
        if not isinstance(record['_cutoff'], str) or type(record['_review_started']) is not bool:
            raise ValueError('invalid closeout metadata')
        return record
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError('closeout_receipt_unreadable') from exc


def _same_target(left, right):
    return all(left[key] == right[key] for key in ('logical_session_id', 'profile'))


def _merge_destination(old, new, *, complete):
    if old == complete or (complete == 'accepted' and old == 'skipped'):
        return old
    if new == complete or new == 'unknown' or (complete == 'accepted' and new == 'skipped'):
        return new
    return old if old == 'unknown' else new


class CloseoutStore:
    """One existing profile database, with optional native authority fencing.

    The record references recoverable stored history instead of copying a transcript. Every
    callback that can transmit remotely runs only after ``claim_review`` has committed.
    """

    def __init__(self, db, *, epoch=None):
        self.db, self.epoch = db, epoch

    @staticmethod
    def operation_key(closeout_id):
        return _PREFIX + 'operation:' + closeout_id

    def _write(self, operation):
        def write(conn):
            if self.epoch is not None:
                from hermes_state_runtime import _epoch
                _epoch(conn, self.epoch)
            return operation(conn)
        return self.db._execute_write(write)

    def read(self, closeout_id):
        return self._write(lambda conn: _validate(_load(
            conn, self.operation_key(closeout_id), required=True)))

    def read_current(self, closeout_id, *, conn=None):
        """A publication may use only the current, unreleased operation's receipt."""
        def read(conn):
            record = _validate(_load(conn, self.operation_key(closeout_id), required=True))
            root = _load(conn, _PREFIX + 'root:' + record['old']['logical_session_id'], required=True)
            if record.get('_fence_released') or root.get('closeout_id') != closeout_id:
                return None
            return record
        return read(conn) if conn is not None else self._write(read)

    def for_request(self, request_id):
        def read(conn):
            binding = _load(conn, _PREFIX + 'request:' + request_id)
            return (_validate(_load(conn, self.operation_key(binding['closeout_id']), required=True))
                    if binding else None)
        return self._write(read)

    def prepare(self, old, request_id, intent, cutoff):
        if intent not in ('archive', 'new') or not request_id or not isinstance(cutoff, str):
            raise ValueError('invalid_closeout_request')
        root_key = _PREFIX + 'root:' + old['logical_session_id']
        request_key = _PREFIX + 'request:' + request_id

        def prepare(conn):
            binding = _load(conn, request_key)
            latest = binding or _load(conn, root_key)
            record = self._existing(conn, latest, old) if latest else None
            reusable = record and (binding or (not record.get('_fence_released') and (record['status'] == 'reviewing' or
                                   cutoff in (record['_cutoff'], record.get('_review_cutoff')))
                                   ))
            if reusable:
                if intent == 'new':
                    record['_intent'] = 'new'
                created = False
            else:
                record, created = self._new(old, intent, cutoff), True
            _save(conn, self.operation_key(record['closeout_id']), record)
            _save(conn, request_key, {'closeout_id': record['closeout_id']})
            if not binding or created:
                _save(conn, root_key, {'closeout_id': record['closeout_id']})
            return record, created
        return self._write(prepare)

    def _existing(self, conn, latest, old):
        try:
            record = _validate(_load(conn, self.operation_key(latest['closeout_id']), required=True))
        except (KeyError, TypeError) as exc:
            raise ValueError('closeout_receipt_unreadable') from exc
        if not _same_target(record['old'], old):
            raise ValueError('request_target_mismatch')
        return record

    @staticmethod
    def _new(old, intent, cutoff):
        operation = uuid.uuid4().hex
        return {'closeout_id': operation, 'status': 'reviewing', 'settled': False,
                'old': dict(old), 'receipt_id': 'closeout:' + operation,
                'destinations': {'memory': 'pending', 'tasks': 'pending'}, 'archived': False,
                'next_create_request_id': 'new-' + operation, 'task_ids': [],
                '_cutoff': cutoff, '_review_started': False, '_intent': intent,
                '_fence_until': time.time() + 30.0}

    def _update(self, closeout_id, change):
        def update(conn):
            record = _validate(_load(conn, self.operation_key(closeout_id), required=True))
            result = change(record)
            _validate(record)
            _save(conn, self.operation_key(closeout_id), record)
            return record if result is None else result
        return self._write(update)

    def claim_review(self, closeout_id):
        def claim(record):
            if record['_review_started'] or record['status'] != 'reviewing':
                return False
            record['_review_started'] = True
            return True
        return self._update(closeout_id, claim)

    def record_review(self, closeout_id, destinations, cutoff, *, memory_source_id=None, task_ids=()):
        if destinations.get('memory') == 'accepted' and not memory_source_id:
            raise ValueError('accepted_capture_requires_source_id')

        def finish(record):
            if record['status'] != 'reviewing':
                return
            merged = {name: _merge_destination(record['destinations'][name], state,
                                              complete='accepted' if name == 'memory' else 'reconciled')
                      for name, state in destinations.items()}
            record.update(status='preserved', destinations=merged,
                          _review_cutoff=cutoff, _review_started=True,
                          task_ids=sorted(set(record['task_ids']) | set(task_ids)))
            if memory_source_id:
                record['memory_source_id'] = memory_source_id
        return self._update(closeout_id, finish)

    def record_capture(self, closeout_id, native_session_id, *, memory, source_id=None,
                       tasks=None, task_ids=()):
        """Record the existing provider's outcome; this method never creates remote memory.

        Late confirmed acceptance may resolve an uncertain receipt after archival. It cannot
        retarget that receipt, downgrade a confirmed save, or change the conversation route.
        """
        before = self.read(closeout_id)
        lineage = self.db.get_compression_lineage(before['old']['stored_session_id'])
        if native_session_id not in lineage:
            raise ValueError('capture_target_mismatch')
        if memory == 'accepted' and not source_id:
            raise ValueError('accepted_capture_requires_source_id')
        from hermes_cli.conversation_closeout_jobs import history_cutoff
        cutoff = history_cutoff(self.db, before['old']['logical_session_id'])

        def capture(record):
            record['destinations']['memory'] = _merge_destination(
                record['destinations']['memory'], memory, complete='accepted')
            if source_id:
                record['memory_source_id'] = source_id
            if tasks is not None:
                record['destinations']['tasks'] = _merge_destination(
                    record['destinations']['tasks'], tasks, complete='reconciled')
                record['task_ids'] = sorted(set(record['task_ids']) | set(task_ids))
            record['_capture_received'] = True
            if not record.get('_fence_released'):
                # The primary review may finish after the bounded worker archived its pending
                # destinations. Its own final rows are still this batch, never new user work.
                record['_review_cutoff'] = cutoff
        return self._update(closeout_id, capture)

    def claim_capture(self, closeout_id, native_session_id):
        """Durably claim the existing writer before it can transmit this batch once."""
        record = self.read(closeout_id)
        if native_session_id not in self.db.get_compression_lineage(record['old']['stored_session_id']):
            raise ValueError('capture_target_mismatch')

        def claim(current):
            if not current['_review_started']:
                raise ValueError('closeout_review_not_started')
            if current.get('_capture_started') or current.get('_capture_received'):
                return False
            current['_capture_started'] = True
            return True
        return self._update(closeout_id, claim)

    def recover_uncertain(self, closeout_id):
        def recover(record):
            if record['status'] == 'reviewing':
                state = 'unknown' if record['_review_started'] else 'pending'
                record.update(status='preserved', _review_started=True,
                              destinations={'memory': state, 'tasks': state})
        return self._update(closeout_id, recover)

    def archive(self, closeout_id, *, settled):
        """Archive flags and receipt commit together, behind the current operation/release fence."""
        def write(conn):
            record = _validate(_load(conn, self.operation_key(closeout_id), required=True))
            root = _load(conn, _PREFIX + 'root:' + record['old']['logical_session_id'], required=True)
            if record.get('_fence_released') or root.get('closeout_id') != closeout_id:
                return record, []
            affected = []
            if record['status'] == 'preserved':
                if not record['archived']:
                    from hermes_cli.conversation_closeout_jobs import archive_conversation_in_transaction
                    affected = archive_conversation_in_transaction(self.db, conn, record['old']['logical_session_id'])
                    if not affected:
                        raise ValueError('closeout_archive_unconfirmed')
                record.update(settled=record['settled'] or settled, archived=True)
            else:
                record['settled'] = False
            _validate(record)
            _save(conn, self.operation_key(closeout_id), record)
            return record, affected
        record, affected = self._write(write)
        if affected:
            from hermes_state_common import emit_session_deleted
            emit_session_deleted(self.db, affected, reason='archive', primary=record['old']['logical_session_id'])
        return record
