"""Same-owner stopped Output: retained admission, exact intent, confirmed cleanup.

No result is fabricated and no peer authority is inferred. Records deliberately
survive failed physical cleanup and retain the original owner, not restart adoption.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactOutbox, RoomArtifactScope
from gateway.hosted_room_output_discard import OutputCleanupUnavailable, retire_exact, cleanup_exact
from gateway.hosted_room_output_fence import require_output_task
from gateway.session_hosted_output_retry import retryable

PREFIX = 'gateway.hosted.output_cleanup.v1:'
BATCH = 64


def key_for(task):
    return PREFIX + hashlib.sha256(json.dumps(
        [asdict(task['identity']), task['execution_generation']], sort_keys=True).encode()).hexdigest()


def records(conn, room_id):
    return [(r['key'], json.loads(r['value'])) for r in conn.execute(
        "SELECT key,value FROM state_meta WHERE key LIKE ? AND json_extract(value,'$.room_id')=?",
        (PREFIX + '%', room_id))]


class CanonicalOutputLifecycle:
    def stop_room(self, *args, **kwargs):
        with self._policy_lock, self._output_policy_read():
            pass
        return super().stop_room(*args, **kwargs)

    def _capture_room_cancel(self, task, cancel_id):
        captured = super()._capture_room_cancel(task, cancel_id)
        if captured['execution_generation'] > 0:
            self._reconcile_stopped_output(captured, capture_only=True)
        return captured

    def _cleanup_snapshot(self, conn, task):
        if self.authority.db is not self._output_db:
            raise RoomArtifactError('Group Chat cleanup owner replaced')
        self._output_owner(conn)
        identity = task['identity']
        row = conn.execute('SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?',
                           (identity.room_id, identity.task_id)).fetchone()
        if (row is None or row['execution_generation'] != task['execution_generation']
                or row['cancel_generation'] != task['cancel_generation']
                or row['cancel_id'] != task.get('cancel_id')):
            raise RoomArtifactError('Group Chat cleanup attempt changed')
        payload = json.loads(row['payload_json'])
        room = conn.execute('SELECT * FROM hosted_rooms WHERE room_id=?', (identity.room_id,)).fetchone()
        owner = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             ('gateway.hosted.owner.v1:' + identity.room_id,)).fetchone()
        if room is None or owner is None or room['disbanded_at'] is not None:
            raise RoomArtifactError('Group Chat cleanup owner changed')
        member = payload.get('target_member_id', payload.get('target_profile'))
        roster = json.loads(room['members_json'])
        participant = next((m for m in roster if m['member_id'] == member and m['profile'] == payload['target_profile']), None)
        if participant is None:
            raise RoomArtifactError('Group Chat cleanup participant changed')
        base = dict(version=1, room_id=identity.room_id, task_id=identity.task_id,
                    member_id=member, execution_generation=row['execution_generation'],
                    cancel_generation=row['cancel_generation'], cancel_id=row['cancel_id'],
                    identity=asdict(identity), payload=payload, owner=owner[0], roster=roster,
                    epoch=self._output_epoch, instance=self._output_instance)
        stop = conn.execute("SELECT event_id,seq,payload_json FROM hosted_room_events WHERE room_id=? "
            "AND kind='room.stop_requested' AND seq>? ORDER BY seq LIMIT 1",
            (identity.room_id, payload['source_event_seq'])).fetchone()
        base['stop_event'] = dict(stop) if stop else None
        if (participant.get('target', {}).get('kind', 'local') != 'local'
                or self.profile_homes().get(payload['target_profile']) != self.root
                or self.root.parent.name == 'profiles'):
            return dict(base, unavailable='unsupported_output_route'), None
        request = 'hosted:' + json.dumps([asdict(identity), row['execution_generation']], sort_keys=True, separators=(',', ':'))
        admissions = conn.execute('SELECT * FROM session_admissions WHERE request_id=? AND principal_id=?',
                                  (request, owner[0])).fetchall()
        if len(admissions) != 1:
            return dict(base, unavailable='admission_unavailable'), None
        admission = dict(admissions[0])
        from gateway.session_local_recovery import local_identity
        binding = json.dumps([identity.room_id, member, payload['target_profile']], separators=(',', ':'))
        sid = local_identity(self.authority.profile_id, owner[0], 'hosted:' + hashlib.sha256(binding.encode()).hexdigest())
        if (admission['target_session_id'] != sid or admission['owner_epoch'] != self._output_epoch
                or type(admission['generation']) is not int or admission['generation'] < 1):
            raise RoomArtifactError('Group Chat cleanup admission changed')
        admitted_payload = json.loads(admission['payload_json'])
        from hermes_state_runtime import admission_fingerprint
        if admission_fingerprint(canonical_target=sid, payload={
                'input': admitted_payload, 'intent': admission['intent']}) != admission['payload_digest']:
            raise RoomArtifactError('Group Chat cleanup admitted payload changed')
        if not payload.get('attachments') and admitted_payload.get('text') != payload['prompt']:
            raise RoomArtifactError('Group Chat cleanup task input changed')
        scope = RoomArtifactScope.from_mapping(dict(room_id=identity.room_id, task_id=identity.task_id,
            execution_generation=row['execution_generation'], member_id=member, target_profile=payload['target_profile'],
            home_install_id=room['authority_gateway_id'], target_install_id=room['authority_gateway_id'],
            authority_gateway_id=room['authority_gateway_id'], authority_epoch=room['authority_epoch']))
        require_output_task(conn, scope, row['cancel_generation'], status=row['status'])
        base.update(scope=scope.as_mapping(), scope_key=scope.key, lineage_identity=scope.lineage_json,
                    admission={k: admission[k] for k in (
            'admission_id', 'request_id', 'principal_id', 'target_session_id', 'owner_epoch', 'generation', 'payload_json')})
        if payload.get('attachments'):
            # Resultless cleanup cannot reconstruct input custody by reading or
            # recapturing aged files. Keep this case blocked until its exact
            # retained preparation binding is available at this writer seam.
            base['unavailable'] = 'input_binding_unavailable'
        return base, admission

    def _reconcile_stopped_output(self, task, *, capture_only=False):
        key = key_for(task)
        now = float(self._artifact_clock())
        def stage(conn):
            snapshot, admission = self._cleanup_snapshot(conn, task)
            saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
            old = json.loads(saved[0]) if saved else None
            if old is not None and old['binding'] != snapshot:
                # Existing exact explicit discard may resolve unknown. It does
                # not readmit the input or transfer the original owner binding.
                prior = old['binding']
                resolved = (old['state'] == 'waiting' and admission is not None
                    and admission['status'] == 'terminal' and admission['outcome'] == 'interrupted'
                    and snapshot['cancel_id'] == f"discard:{snapshot['execution_generation']}"
                    and snapshot['cancel_generation'] == prior['cancel_generation'] + 1
                    and {k: v for k, v in prior.items() if k not in {'cancel_generation', 'cancel_id'}}
                        == {k: v for k, v in snapshot.items() if k not in {'cancel_generation', 'cancel_id'}})
                if not resolved:
                    raise RoomArtifactError('Group Chat cleanup commitment changed')
                old = dict(old, binding=snapshot, original_binding=prior)
            if old and old['state'] == 'completed':
                return old
            record = old or dict(version=1, room_id=task['identity'].room_id,
                task_id=task['identity'].task_id, member_id=snapshot['member_id'],
                execution_generation=task['execution_generation'], binding=snapshot,
                state='waiting', reason_code='waiting_for_terminal', attempts=0,
                next_attempt_at=0, items=[], blobs=[], removed=0)
            if snapshot.get('unavailable'):
                record['reason_code'] = snapshot['unavailable']
            elif admission['status'] != 'terminal':
                record['reason_code'] = 'unknown_execution' if admission['status'] == 'unknown' else 'waiting_for_terminal'
            elif record['state'] == 'waiting' and not capture_only:
                if admission['outcome'] not in {'interrupted', 'cancelled', 'failed', 'completed'}:
                    raise RoomArtifactError('Group Chat cleanup terminal unavailable')
                try:
                    record = self._inventory_cleanup(conn, task, snapshot, record)
                except OutputCleanupUnavailable:
                    record['reason_code'] = 'inventory_unavailable'
            conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)', (key, json.dumps(record, sort_keys=True)))
            return record
        record = self.authority.db._execute_write(stage)
        if capture_only or record['state'] != 'pending' or record['next_attempt_at'] > now:
            return record['state'] == 'completed'
        def complete(conn):
            self._require_cleanup_binding(conn, task, record['binding'], terminal=True)
            saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
            if saved is None or json.loads(saved[0]) != record:
                raise RoomArtifactError('Group Chat cleanup reservation changed')
            outbox = RoomArtifactOutbox.borrow_existing(self.authority.db, conn)
            cleanup_exact(outbox, conn, RoomArtifactScope.from_mapping(record['binding']['scope']),
                record['items'], record['blobs'], authorize=lambda c: self._require_cleanup_binding(c, task, record['binding'], terminal=True))
            done = dict(record, state='completed', blobs=[], reason_code='completed', next_attempt_at=0)
            conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(done, sort_keys=True), key))
        try:
            self.authority.db._execute_write(complete)
        except Exception as exc:
            if not (retryable(exc) or isinstance(exc, OutputCleanupUnavailable)):
                raise
            def pending(conn):
                self._require_cleanup_binding(conn, task, record['binding'], terminal=True)
                attempts = min(record['attempts'] + 1, 30)
                failed = dict(record, attempts=attempts, reason_code='cleanup_unavailable',
                              next_attempt_at=float(self._artifact_clock()) + min(300, 2 ** min(attempts, 8)))
                conn.execute('UPDATE state_meta SET value=? WHERE key=? AND value=?',
                             (json.dumps(failed, sort_keys=True), key, json.dumps(record, sort_keys=True)))
            self.authority.db._execute_write(pending)
            return False
        return True

    def _inventory_cleanup(self, conn, task, snapshot, record):
        outbox = RoomArtifactOutbox.borrow_existing(self.authority.db, conn)
        scope = RoomArtifactScope.from_mapping(snapshot['scope'])
        rows = conn.execute('SELECT * FROM hosted_room_output_artifacts WHERE scope_key=? '
            'ORDER BY created_at,artifact_id LIMIT ?', (scope.key, BATCH + 1)).fetchall()
        if len(rows) > BATCH:
            return dict(record, reason_code='inventory_limit')
        items = [outbox._manifest(r) for r in rows]
        if items:
            blobs = retire_exact(outbox, conn, scope, items,
                authorize=lambda c: self._require_cleanup_binding(c, task, snapshot, terminal=True))
            return dict(record, state='pending', items=items, blobs=blobs,
                        removed=len(items), reason_code='cleanup_pending')
        # Exhaustive scope query on initialized schema, never schema absence.
        outbox._retire_generation(conn, scope)
        return dict(record, state='completed', reason_code='completed')

    def _complete_stopped_output_from_receipt(self, conn, task, metadata, operation):
        """A racing completed Run uses existing settled custody, never local unlink."""
        key = key_for(task)
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        if saved is None:
            return
        record = json.loads(saved[0])
        if record['state'] == 'completed':
            return
        current, _ = self._cleanup_snapshot(conn, task)
        if record['binding'] != current or record['state'] != 'waiting':
            raise RoomArtifactError('Group Chat cleanup disposition changed')
        done = dict(record, state='completed', reason_code='completed',
                    completion=dict(metadata=metadata, operation=operation), next_attempt_at=0)
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(done, sort_keys=True), key))

    def _require_cleanup_binding(self, conn, task, expected, *, terminal=False):
        current, admission = self._cleanup_snapshot(conn, task)
        if current != expected or (terminal and (admission is None or admission['status'] != 'terminal')):
            raise RoomArtifactError('Group Chat cleanup authority changed')

    def output_cleanup_status(self, room_id):
        with self.authority.db._read_ctx() as conn:
            self._output_owner(conn)
            return [dict(kind='output_cleanup', **{k: r[k] for k in (
                'task_id', 'member_id', 'execution_generation', 'state', 'reason_code', 'attempts', 'next_attempt_at')})
                for _, r in records(conn, room_id) if r['state'] != 'completed']
