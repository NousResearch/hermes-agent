"""Owner-transactional retry of retained Output, never execution admission.

Adapted from David Dudok de Wit's O4fb3f28 artifact retry/completion contract.
Historical table names preserve the driver's bounded retry-retention guard.
Legacy rows without exact commitments are not upgraded by guessing authority.
"""
import base64
from contextlib import contextmanager
import hashlib
import math
import json
import sqlite3
import threading
import time

from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactScope
from gateway.hosted_room_output_fence import require_output_task, require_peer_output_receipt
from hermes_state_runtime import RuntimeStoreError, _epoch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def retryable(error):
    # Retain historical classification. SQLite BUSY/LOCKED is transient, but
    # arbitrary SQL/corruption must never become a network retry classification.
    from gateway.hosted_rooms import EventCursorConflictError
    if isinstance(error, EventCursorConflictError):
        return True
    if isinstance(error, sqlite3.Error):
        return getattr(error, 'sqlite_errorcode', None) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
    return (getattr(error, 'retryable', False) is True
            or isinstance(error, (ConnectionError, OSError, TimeoutError)))


class CanonicalOutputRetry:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._artifact_clock = time.time
        self._output_publish_locks = {}
        self._output_publish_locks_guard = threading.Lock()
        from gateway.session_group_state import GroupStateOwner
        self._output_status_owner = GroupStateOwner(
            self.authority, self.authority.db, self.authority.epoch,
            self.authority.instance_id, self.authority.profile_id, self, self.runtime)
        self._output_db = self.authority.db
        self._output_epoch = self.authority.epoch
        self._output_instance = self.authority.instance_id
        self._prepare_artifact_retry_store()

    @contextmanager
    def _output_status_read(self, room_id, *, state_read=None):
        # Informational obligations are not NEW work. Keep _output_owner and
        # _output_policy_read strict for publication and held eligibility.
        owner = self._output_status_owner
        if not self._output_retry_ready:
            raise RuntimeStoreError('output_owner_unavailable')
        if state_read is not None:
            if (state_read.owner != owner or state_read.service is not self
                    or state_read.room['room_id'] != room_id):
                raise RuntimeStoreError('output_owner_unavailable')
            state_read.authorize()
            yield state_read.conn
            state_read.authorize()
            return
        def current(conn):
            owner.current(conn)
            if (self.authority is not owner.authority
                    or getattr(owner.authority, 'hosted_room_service', None) is not self):
                raise RuntimeStoreError('output_owner_unavailable')
        with owner.read() as conn:
            current(conn)
            conn.execute('BEGIN')
            try:
                yield conn
            finally:
                conn.rollback()
            current(conn)

    @contextmanager
    def _output_policy_read(self):
        # Policy lock precedes this non-reentrant owner lifetime lock. No pooled
        # reads, writes, pathname opens or recovery inside the selection fence.
        db = self._output_db
        with db.live_read_connection() as conn:
            def current():
                if (conn is None or db._read_conns_closed or conn is not db._conn
                        or self.authority.db is not db or db._db_replaced
                        or db._db_wal_generation_lost or db._wal_generation_was_lost()):
                    raise RuntimeStoreError('output_owner_unavailable')
                db._raise_if_db_corrupt()
                self._output_owner(conn)
            current()
            yield conn
            # Policy ends its read transaction before this check, so a changed
            # epoch is read fresh rather than from the selected historical view.
            current()

    def _policy_snapshot(self, room):
        # Refuse an already unavailable owner before checkpoint sync opens its
        # separate writer. Sync finishes before the read-only selection begins.
        with self._output_policy_read():
            pass
        room_id = str(room['room_id'])
        def held_threads(conn):
            self._output_owner(conn)
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_driver_tasks'").fetchone() is None:
                return frozenset()
            rows = conn.execute("""SELECT task.thread_id, retry.* FROM hosted_room_artifact_retries retry
                JOIN hosted_room_driver_tasks task ON task.room_id=retry.room_id AND task.task_id=retry.task_id
                  AND task.execution_generation=retry.execution_generation
                WHERE retry.room_id=? AND task.status='settled'""", (room_id,)).fetchall()
            held = set()
            for row in rows:
                key = (room_id, row['task_id'], row['execution_generation'])
                current = self._output_metadata(conn, key)
                # The retry is not settlement authority: the exact settled task,
                # frozen context/recipients, owner, receipt and result must match.
                old = json.loads(row['metadata_json'])
                if old['work'] == current['work'] and old['member_id'] == current['member_id']:
                    held.add(row['thread_id'])
            return frozenset(held)
        return self.policy_checkpoint.snapshot(room_id=room_id, latest_seq=int(room['latest_seq']),
                                               held_output_threads=held_threads,
                                               read_connection=self._output_policy_read)

    def _output_room_lock(self, room_id):
        with self._output_publish_locks_guard:
            return self._output_publish_locks.setdefault(room_id, threading.RLock())

    def _prepare_artifact_retry_store(self):
        def prepare(conn):
            _epoch(conn, self._output_epoch)
            from gateway.hosted_room_task_scan import initialize
            initialize(conn)
            conn.execute('''CREATE TABLE IF NOT EXISTS hosted_room_artifact_retries (
                room_id TEXT NOT NULL, task_id TEXT NOT NULL, execution_generation INTEGER NOT NULL,
                member_id TEXT NOT NULL, attempts INTEGER NOT NULL, next_attempt_at REAL NOT NULL,
                blocked INTEGER NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                metadata_json TEXT NOT NULL, operation TEXT NOT NULL, reason_code TEXT NOT NULL,
                PRIMARY KEY(room_id, task_id, execution_generation))''')
            conn.execute('''CREATE TABLE IF NOT EXISTS hosted_room_artifact_completions (
                room_id TEXT NOT NULL, task_id TEXT NOT NULL, execution_generation INTEGER NOT NULL,
                completed_at REAL NOT NULL, valid_until REAL NOT NULL, metadata_json TEXT NOT NULL,
                event_digest TEXT NOT NULL, operation TEXT NOT NULL,
                PRIMARY KEY(room_id, task_id, execution_generation))''')
            return all('metadata_json' in {r['name'] for r in conn.execute('PRAGMA table_info(' + table + ')')}
                       for table in ('hosted_room_artifact_retries', 'hosted_room_artifact_completions'))
        self._output_retry_ready = self.authority.db._execute_write(prepare)

    def _prune_output_retry_metadata(self, room_id):
        if not self._output_retry_ready:
            return  # legacy rows need a separately authorized exact migration

        def prune(conn):
            self._output_owner(conn)
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_driver_tasks'").fetchone() is None:
                return
            for table, age, params in (
                ('hosted_room_artifact_completions', '', (room_id,)),

            ):
                conn.execute(f"""DELETE FROM {table} WHERE rowid IN (
                    SELECT metadata.rowid FROM {table} metadata WHERE metadata.room_id=? {age}
                      AND NOT EXISTS (SELECT 1 FROM hosted_room_driver_tasks task
                        WHERE task.room_id=metadata.room_id AND task.task_id=metadata.task_id
                          AND task.execution_generation=metadata.execution_generation)
                    ORDER BY metadata.rowid LIMIT 256)""", params)
        self.authority.db._execute_write(prune)

    def _output_owner(self, conn):
        a = self.authority
        if (a.hosted_room_service is not self or a.runner.session_authority is not a
                or (a.epoch, a.instance_id) != (self._output_epoch, self._output_instance)
                or a.db._db_file_was_replaced()):
            raise RoomArtifactError('Group Chat output owner changed')
        a._require_admission_open()
        _epoch(conn, self._output_epoch)
        if not self._output_retry_ready:
            raise RoomArtifactError('legacy output retries require exact authority migration')

    @staticmethod
    def _output_key(task):
        return task['identity'].room_id, task['identity'].task_id, task['execution_generation']

    def _output_metadata(self, conn, key):
        self._output_owner(conn)
        row = conn.execute('SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?', key[:2]).fetchone()
        if row is None or row['execution_generation'] != key[2]:
            raise RoomArtifactError('Group Chat output attempt changed')
        result, payload = json.loads(row['result_json']), json.loads(row['payload_json'])
        scope = RoomArtifactScope.from_mapping(result.get('artifact_scope') or {})
        task = require_output_task(conn, scope, row['cancel_generation'], cleanup=True)
        if (scope.room_id, scope.task_id, scope.execution_generation) != key or not payload.get('recipient_member_ids'):
            raise RoomArtifactError('Group Chat output receipt changed')
        owner = conn.execute('SELECT value FROM state_meta WHERE key=?', ('gateway.hosted.owner.v1:' + key[0],)).fetchone()
        room = conn.execute('SELECT * FROM hosted_rooms WHERE room_id=?', (key[0],)).fetchone()
        if owner is None:
            raise RoomArtifactError('Group Chat output owner changed')
        work = dict(scope=scope.as_mapping(), payload=payload, result=result,
                    cancel_generation=task['cancel_generation'], status=task['status'], owner=owner[0],
                    roster=json.loads(room['members_json']), epoch=self._output_epoch, instance=self._output_instance)
        route_hash, lineage, until = '', '', float(self._artifact_clock()) + 86400
        if scope.target_install_id == scope.home_install_id and result.get('owner_output_receipt') is not None:
            from pathlib import Path
            from gateway.session_hosted_output_rpc import validate_owner_output_receipt
            from gateway.session_authorities import authority_for_home
            from hermes_constants import hermes_home_key
            receipt = validate_owner_output_receipt(result['owner_output_receipt'])
            target_home = Path(self.profile_homes().get(scope.target_profile, ''))
            target = authority_for_home(self.authority.runner, target_home)
            if (not target_home.is_absolute() or target_home == Path(self.authority.profile_id)
                    or target is None or target.profile_id != str(target_home)
                    or getattr(target, 'hosted_room_service', None) is None):
                raise RoomArtifactError('Group Chat output owner route unavailable')
            route_hash = digest(dict(
                source_home_key=hermes_home_key(self.authority.profile_id),
                target_home_key=hermes_home_key(target_home), room_id=scope.room_id,
                member_id=scope.member_id, target_profile=scope.target_profile,
                consent_digest=receipt['consent_digest'], receipt_id=receipt['receipt_id']))
            until = receipt['expires_at']
        elif scope.target_install_id != scope.home_install_id:
            from gateway.session_authorities import authority_for_home
            from gateway.runtime_ownership import process_ownership
            from gateway import hosted_rooms
            if (authority_for_home(self.authority.runner, self.authority.profile_id) is not self.authority
                    or not process_ownership.owns(self.root)
                    or scope.home_install_id != hosted_rooms.local_authority_gateway_id()):
                raise RoomArtifactError('Group Chat output owner changed')
            work['receipt'] = require_peer_output_receipt(conn, scope, result)
            link = conn.execute('SELECT * FROM hosted_room_links WHERE room_id=? AND member_id=?',
                                (key[0], scope.member_id)).fetchone()
            if link is None:
                raise RoomArtifactError('Group Chat output route unavailable')
            link = dict(link)
            from gateway.hosted_room_links import route_security_digest
            route_hash = route_security_digest(link)
            # Metadata is not authorization. Decode only the signed token's
            # lifetime/scope commitment; the actual target verifies its signature
            # and operation rights. Do not require NEW/input permissions here.
            try:
                encoded, signature = link['grant'].split('.')
                claims = json.loads(base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)))
                expiry = claims['expires_at']
                if not signature or type(expiry) not in {int, float} or not math.isfinite(expiry):
                    raise ValueError('invalid horizon')
            except (ValueError, TypeError, KeyError) as exc:
                raise RoomArtifactError('Group Chat output grant commitment unavailable') from exc
            until = claims['expires_at']  # artifact rights, NOT the longer status horizon
            # Only authenticated same-scope renewal can replace these three fields.
            stable_claims = {k: v for k, v in claims.items() if k not in {'grant_id', 'issued_at', 'expires_at'}}
            lineage = digest(dict(target_url=link['target_url'], target_profile=link['target_profile'],
                catalog=link['catalog_json'], cancellation_scope_id=link['cancellation_scope_id'],
                trace_id=link['trace_id'], claims=stable_claims))
        return dict(work=digest(work), route=route_hash, lineage=lineage,
                    member_id=scope.member_id, valid_until=until)

    @staticmethod
    def _output_events_digest(conn, key):
        stem = key[1].removeprefix('dtask:')
        rows = conn.execute('SELECT event_id,kind,actor_json,payload_json,authority_epoch FROM hosted_room_events '
                            'WHERE room_id=? AND event_id IN (?,?) ORDER BY event_id',
                            (key[0], 'dmessage:' + stem, 'dterminal:' + stem)).fetchall()
        return digest([dict(r) for r in rows]) if rows else ''

    def _has_output_obligation(self, task):
        with self.authority.db._read_ctx() as conn:
            self._output_owner(conn)
            return any(conn.execute(f'SELECT 1 FROM {table} WHERE room_id=? AND task_id=?',
                       self._output_key(task)[:2]).fetchone() is not None for table in
                       ('hosted_room_artifact_retries', 'hosted_room_artifact_completions'))

    def _publication_operation(self, conn, key):
        stem = key[1].removeprefix('dtask:')
        terminal = conn.execute('SELECT kind,payload_json FROM hosted_room_events WHERE room_id=? AND event_id=?',
                                (key[0], 'dterminal:' + stem)).fetchone()
        message = conn.execute('SELECT kind FROM hosted_room_events WHERE room_id=? AND event_id=?',
                               (key[0], 'dmessage:' + stem)).fetchone()
        if terminal is None:
            raise RoomArtifactError('Group Chat output terminal evidence missing')
        promised = json.loads(terminal['payload_json']).get('message_event_id')
        if promised:
            if (promised != 'dmessage:' + stem or message is None or message['kind'] != 'message.member'
                    or terminal['kind'] != 'turn.settled'):
                raise RoomArtifactError('Group Chat output promised message missing or changed')
            return 'ack'
        if message is not None or terminal['kind'] not in {'turn.settled', 'turn.failed', 'turn.cancelled'}:
            raise RoomArtifactError('Group Chat output silent terminal changed')
        return 'discard'

    def _record_output_disposition(self, task, operation):
        """Commit disposition before target I/O, including interrupted/lost replies."""
        key = self._output_key(task)
        def record(conn):
            current = self._output_metadata(conn, key)
            row = conn.execute('SELECT * FROM hosted_room_artifact_retries WHERE room_id=? AND task_id=? '
                               'AND execution_generation=?', key).fetchone()
            if row is None or row['blocked']:
                raise RoomArtifactError('Group Chat output reservation missing')
            old = json.loads(row['metadata_json'])
            if (any(old[k] != current[k] for k in ('work', 'route', 'lineage', 'member_id'))
                    or row['operation'] not in {'publish', operation}):
                raise RoomArtifactError('Group Chat output disposition changed')
            if operation == 'ack' and self._publication_operation(conn, key) != 'ack':
                raise RoomArtifactError('Group Chat output ACK publication missing')
            events = self._output_events_digest(conn, key)
            if old.get('publication') and old['publication'] != events:
                raise RoomArtifactError('Group Chat output publication changed')
            conn.execute('UPDATE hosted_room_artifact_retries SET operation=?,metadata_json=? '
                         'WHERE room_id=? AND task_id=? AND execution_generation=?',
                         (operation, json.dumps(dict(old, publication=events), sort_keys=True), *key))
        self.authority.db._execute_write(record)

    def _begin_output_retry(self, task):
        key, now = self._output_key(task), float(self._artifact_clock())
        def begin(conn):
            self._output_owner(conn)
            row = conn.execute('SELECT * FROM hosted_room_artifact_retries WHERE room_id=? AND task_id=? '
                               'AND execution_generation=?', key).fetchone()
            try:
                metadata = self._output_metadata(conn, key)
            except (RoomArtifactError, RuntimeStoreError):
                if row is None:
                    raise
                conn.execute('UPDATE hosted_room_artifact_retries SET blocked=1,reason_code=? '
                             'WHERE room_id=? AND task_id=? AND execution_generation=?', ('stale_binding', *key))
                return None
            current = conn.execute('SELECT payload_json,result_json,cancel_generation FROM hosted_room_driver_tasks '
                                   'WHERE room_id=? AND task_id=?', key[:2]).fetchone()
            if (json.loads(current['payload_json']) != task['payload'] or json.loads(current['result_json']) != task['result']
                    or current['cancel_generation'] != task['cancel_generation']):
                raise RoomArtifactError('Group Chat output snapshot changed')
            done = conn.execute('SELECT * FROM hosted_room_artifact_completions WHERE room_id=? AND task_id=? '
                                'AND execution_generation=?', key).fetchone()
            same = lambda old: all(old[k] == metadata[k] for k in ('work', 'route', 'lineage', 'member_id'))
            if (done and same(json.loads(done['metadata_json'])) and now < done['valid_until']
                    and done['event_digest'] == self._output_events_digest(conn, key)):
                return dict(metadata, completed_operation=done['operation'])
            prior = json.loads(row['metadata_json']) if row else {}
            stale = bool(row and (not same(prior) or
                         (prior.get('publication') and prior['publication'] != self._output_events_digest(conn, key))))
            if row and row['operation'] == 'ack':
                try:
                    stale = stale or self._publication_operation(conn, key) != 'ack'
                except RoomArtifactError:
                    stale = True
            if row and stale:
                conn.execute('UPDATE hosted_room_artifact_retries SET blocked=1,reason_code=? '
                             'WHERE room_id=? AND task_id=? AND execution_generation=?', ('stale_binding', *key))
                return None
            if row and (row['blocked'] or now < row['next_attempt_at']):
                return None
            # Completion never grants a new route/result the old success.
            stale = stale or bool(done and not row and not same(json.loads(done['metadata_json'])))
            stale = stale or bool(done and done['event_digest'] != self._output_events_digest(conn, key))
            expired = now >= metadata['valid_until']
            attempts = min(2147483647, row['attempts'] + 1) if row else 1
            delay = min(60.0, 2.0 ** min(attempts - 1, 16))
            encoded = done['metadata_json'] if done and not row else json.dumps(metadata, sort_keys=True)
            conn.execute('''INSERT INTO hosted_room_artifact_retries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(room_id,task_id,execution_generation) DO UPDATE SET
                attempts=excluded.attempts,next_attempt_at=excluded.next_attempt_at,blocked=excluded.blocked,
                updated_at=excluded.updated_at,reason_code=excluded.reason_code''',
                (*key, metadata['member_id'], attempts, now + delay, int(stale or expired),
                 row['created_at'] if row else now, now, encoded, 'publish',
                 'stale_binding' if stale else 'expired_grant' if expired else 'pending'))
            return None if stale or expired else dict(metadata, attempt=attempts)
        return self.authority.db._execute_write(begin)

    def _finish_output_retry(self, task, metadata, *, operation, error=None):
        key, now = self._output_key(task), float(self._artifact_clock())
        def finish(conn):
            self._output_owner(conn)
            row = conn.execute('SELECT * FROM hosted_room_artifact_retries WHERE room_id=? AND task_id=? '
                               'AND execution_generation=?', key).fetchone()
            if row is None or row['attempts'] != metadata['attempt']:
                raise RoomArtifactError('Group Chat output attempt changed')
            retained = json.loads(row['metadata_json'])
            try:
                current = self._output_metadata(conn, key)
                exact = all(current[k] == metadata[k] for k in ('work', 'route', 'lineage', 'member_id'))
            except (RoomArtifactError, RuntimeStoreError):
                exact = False
            if error is not None or not exact:
                delay = min(60.0, 2.0 ** min(row['attempts'] - 1, 16))
                conn.execute('UPDATE hosted_room_artifact_retries SET blocked=?,reason_code=?,updated_at=?,next_attempt_at=? '
                             'WHERE room_id=? AND task_id=? AND execution_generation=?',
                             (int(not exact or not retryable(error)),
                              'stale_binding' if not exact else 'transient' if retryable(error) else 'authorization_or_verification',
                              now, now + delay, *key))
                return
            event_digest = self._output_events_digest(conn, key)
            if (row['operation'] != operation or self._publication_operation(conn, key) != operation
                    or (retained.get('publication') and retained['publication'] != event_digest)):
                raise RoomArtifactError('Group Chat output completion disposition changed')
            if not event_digest or now >= metadata['valid_until']:
                raise RoomArtifactError('Group Chat output completion authority expired')
            conn.execute('''INSERT INTO hosted_room_artifact_completions VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(room_id,task_id,execution_generation) DO UPDATE SET
                completed_at=excluded.completed_at,valid_until=excluded.valid_until,metadata_json=excluded.metadata_json,
                event_digest=excluded.event_digest,operation=excluded.operation''',
                (*key, now, metadata['valid_until'], json.dumps(metadata, sort_keys=True), event_digest, operation))
            conn.execute('DELETE FROM hosted_room_artifact_retries WHERE room_id=? AND task_id=? AND execution_generation=?', key)
            self._complete_stopped_output_from_receipt(conn, task, metadata, operation)
        self.authority.db._execute_write(finish)

    def _unblock_authenticated_output_routes(self, room_id, *, tasks=None):
        """Consume Route's exact authenticated CAS notification on normal ticks."""
        if not self._output_retry_ready:
            return
        def unblock(conn):
            self._output_owner(conn)
            if tasks is None:
                rows = conn.execute('SELECT * FROM hosted_room_artifact_retries WHERE room_id=? AND blocked=1',
                                    (room_id,)).fetchall()
            else:
                rows = [row for task in tasks for row in conn.execute(
                    'SELECT * FROM hosted_room_artifact_retries WHERE room_id=? AND task_id=? '
                    'AND execution_generation=? AND blocked=1',
                    (room_id, task['identity'].task_id, task['execution_generation']))]
            for row in rows:
                member_id = row['member_id']
                link = conn.execute('SELECT * FROM hosted_room_links WHERE room_id=? AND member_id=?',
                                    (room_id, member_id)).fetchone()
                marker = 'gateway.hosted.route.recovered.v1:' + json.dumps([room_id, member_id], separators=(',', ':'))
                notification = conn.execute('SELECT value FROM state_meta WHERE key=?', (marker,)).fetchone()
                if link is None or notification is None or link['status'] != 'ready':
                    continue
                from gateway.hosted_room_links import route_security_digest
                try:
                    transition = json.loads(notification[0])
                except (ValueError, TypeError):
                    continue  # old whole-record hashes have no transition authority
                if not isinstance(transition, dict) or set(transition) != {'old', 'new'}:
                    continue
                key = (room_id, row['task_id'], row['execution_generation'])
                old = json.loads(row['metadata_json'])
                if (transition.get('old') != old['route'] or transition.get('new') != route_security_digest(dict(link))
                        or (old.get('publication') and old['publication'] != self._output_events_digest(conn, key))):
                    continue
                done = conn.execute('SELECT event_digest FROM hosted_room_artifact_completions WHERE room_id=? AND task_id=? AND execution_generation=?', key).fetchone()
                if done and done['event_digest'] != self._output_events_digest(conn, key):
                    continue
                try:
                    current = self._output_metadata(conn, key)
                except (RoomArtifactError, RuntimeStoreError):
                    continue
                if (old['work'] != current['work'] or old['lineage'] != current['lineage']
                        or old['route'] == current['route'] or float(self._artifact_clock()) >= current['valid_until']):
                    continue
                conn.execute('UPDATE hosted_room_artifact_retries SET metadata_json=?,blocked=0,next_attempt_at=0,'
                             'reason_code=? WHERE room_id=? AND task_id=? AND execution_generation=?',
                             (json.dumps({**old, **current}, sort_keys=True), 'route_recovered', *key))
        self.authority.db._execute_write(unblock)

    def status(self, room_id=None, *, state_read=None):
        result = (super().status(room_id, state_read=state_read) if state_read is not None
                  else super().status(room_id))
        if room_id is None:
            return result
        obligations = self.output_retry_status(room_id, state_read=state_read)
        cleanup = self.output_cleanup_status(room_id, state_read=state_read)
        # Informational only: these are retained output operations, never the
        # driver's NEW-execution Retry action.
        return {**result, 'pending_actions': [*result['pending_actions'], *cleanup, *[
            dict(row, kind='output_retry', blocked=bool(row['blocked'])) for row in obligations]]}

    def output_retry_status(self, room_id, *, state_read=None):
        with self._output_status_read(room_id, state_read=state_read) as conn:
            return [dict(r) for r in conn.execute('SELECT task_id,execution_generation,member_id,attempts,next_attempt_at,'
                'blocked,operation,reason_code FROM hosted_room_artifact_retries WHERE room_id=?', (room_id,))]
