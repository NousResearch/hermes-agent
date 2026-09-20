"""Exact operator controls for canonical hosted attempts, never legacy replay."""
from dataclasses import asdict
import hashlib
import json

from gateway import hosted_room_driver as tasks
from hermes_state_runtime import RuntimeStoreError, _epoch
from tui_gateway.hosted_room_driver import HostedRoomBinding


_DISCARD_RESERVATION = 'gateway.hosted.discard_reservation.v1:'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _discard_key(task):
    return _DISCARD_RESERVATION + _digest([
        asdict(task['identity']), task['execution_generation'],
    ])


class HostedControls:
    def _capture_room_cancel(self, task, cancel_id):
        # Canonical unknown is not legacy inactivity. Preserve the exact
        # explicit-discard control rather than converting it into a cancelled row.
        if task['status'] in {'indeterminate', 'running', 'stopping'}:
            room = self._room(task['identity'].room_id)
            binding = HostedRoomBinding(room['room_id'], room['authority_gateway_id'], room['authority_epoch'])
            member = task['payload'].get('target_member_id', task['payload']['target_profile'])
            if not self._member_is_peer(room['room_id'], member):
                rpc = self._resolve_member_transport(binding, task)
                from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
                if (type(rpc) is HostedRoomAuthorityRPC
                    and self.authority.db.get_session(rpc.ref.session_id) is not None and any(
                        row['status'] == 'unknown' and identity == task['identity']
                        and generation == task['execution_generation'] for row, identity, generation in rpc._rows())):
                    return task
        return super()._capture_room_cancel(task, cancel_id)

    def _control_task(self, room_id, member_id, task_id, execution_generation, *, proven_peer_retry=False):
        if (type(execution_generation) is not int or execution_generation < 1
                or not isinstance(member_id, str) or not member_id
                or not isinstance(task_id, str) or not task_id):
            raise RuntimeStoreError('invalid_params')
        gateway, epoch = self._owned_authority(room_id)
        task = next((t for t in tasks.list_tasks(self.db_path, room_id=room_id)
                     if t['identity'].task_id == task_id), None)
        if (task is None or task['execution_generation'] != execution_generation
                or (task['payload'].get('target_member_id') or task['payload'].get('target_profile')) != member_id):
            raise RuntimeStoreError('stale_generation')
        if not any(m.get('member_id') == member_id
                   and m.get('profile') == task['payload']['target_profile']
                   for m in self._room(room_id)['members']):
            raise RuntimeStoreError('permission_denied')
        if self._member_is_peer(room_id, member_id):
            if not proven_peer_retry or not tasks.is_proven_nonadmission(task):
                raise RuntimeStoreError('unsupported_operation')
        return task, HostedRoomBinding(room_id, gateway, epoch)

    def discard_room_task(self, room_id, *, member_id, task_id, execution_generation):
        with self._policy_lock:
            task, binding = self._control_task(room_id, member_id, task_id, execution_generation)
            cancel_id = f'discard:{execution_generation}'
            if task['status'] == 'cancelled' and task.get('cancel_id') == cancel_id:
                self._complete_discard_reservation(task)
                return task
            if task['status'] != 'indeterminate':
                raise RuntimeStoreError('stale_generation')
            rpc = self._resolve_member_transport(binding, task)
            reservation = self._reserve_discard(task, binding, rpc.ref.session_id)
        # The durable source reservation is the revocation linearization point.
        # Already-authorized target work may finish, but no policy or DB writer is
        # held over this reverse-attested owner action.
        target_result = rpc.discard(
            profile=task['payload']['target_profile'], source='bot_room',
            session_id=rpc.ref.session_id, expected_task_id=task_id,
            execution_generation=execution_generation,
            _source_discard_digest=reservation['reservation_digest'],
        )
        expected_result = {
            'discarded': True, 'status': 'cancelled', 'task_id': task_id,
            'execution_generation': execution_generation,
        }
        if type(target_result) is not dict or target_result != expected_result:
            raise RuntimeStoreError('storage_unavailable')
        with self._policy_lock:
            current, current_binding = self._control_task(
                room_id, member_id, task_id, execution_generation)
            if current['status'] != 'indeterminate' or current_binding != binding:
                raise RuntimeStoreError('stale_generation')
            self._require_discard_reservation(current, reservation)
            lease = self.runtime._ensure_lease(binding)
            result = self.runtime._fenced(
                tasks.resolve_indeterminate_cancellation, binding, current, lease,
                cancel_id=cancel_id, publish=False,
            )
            self.runtime._set_blocked(room_id, False)
            self._complete_discard_reservation(result, target_result=target_result)
        # The fenced control commits under the lock; Output I/O must not inherit it.
        self.publish_terminal(binding, result)
        return result

    def _discard_reservation_record(self, task, binding, target_session_id):
        value = {
            'version': 1,
            'state': 'reserved',
            'task': asdict(task['identity']),
            'execution_generation': task['execution_generation'],
            'cancel_generation': task['cancel_generation'],
            'cancel_id': f"discard:{task['execution_generation']}",
            'member_id': task['payload'].get('target_member_id', task['payload']['target_profile']),
            'target_profile': task['payload']['target_profile'],
            'target_session_id': target_session_id,
            'payload_digest': _digest(task['payload']),
            'authority_gateway_id': binding.gateway_id,
            'authority_epoch': binding.authority_epoch,
        }
        return {**value, 'reservation_digest': _digest(value)}

    def _reserve_discard(self, task, binding, target_session_id):
        key = _discard_key(task)
        expected = self._discard_reservation_record(task, binding, target_session_id)
        def reserve(conn):
            _epoch(conn, self.authority.epoch)
            row = conn.execute(
                'SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?',
                (task['identity'].room_id, task['identity'].task_id),
            ).fetchone()
            if (row is None or row['status'] != 'indeterminate'
                    or row['execution_generation'] != task['execution_generation']
                    or row['cancel_generation'] != task['cancel_generation']
                    or _digest(json.loads(row['payload_json'])) != expected['payload_digest']):
                raise RuntimeStoreError('stale_generation')
            old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
            if old is not None:
                saved = json.loads(old[0])
                if saved != expected:
                    raise RuntimeStoreError('admission_conflict')
                return saved
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, _canonical(expected)))
            return expected
        return self.authority.db._execute_write(reserve)

    def _require_discard_reservation(self, task, expected):
        with self.authority.db._read_ctx() as conn:
            _epoch(conn, self.authority.epoch)
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_discard_key(task),)).fetchone()
        if row is None or json.loads(row[0]) != expected or expected['state'] != 'reserved':
            raise RuntimeStoreError('stale_generation')

    def _complete_discard_reservation(self, task, *, target_result=None):
        key = _discard_key(task)
        expected_result = target_result or {
            'discarded': True, 'status': 'cancelled',
            'task_id': task['identity'].task_id,
            'execution_generation': task['execution_generation'],
        }
        def complete(conn):
            _epoch(conn, self.authority.epoch)
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
            if row is None:
                return
            saved = json.loads(row[0])
            current = conn.execute(
                'SELECT status,cancel_id,execution_generation,cancel_generation '
                'FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?',
                (task['identity'].room_id, task['identity'].task_id),
            ).fetchone()
            if (current is None or current['status'] != 'cancelled'
                    or current['cancel_id'] != saved['cancel_id']
                    or current['execution_generation'] != saved['execution_generation']):
                raise RuntimeStoreError('stale_generation')
            done = {**saved, 'state': 'completed',
                    'target_result_digest': _digest(expected_result)}
            conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_canonical(done), key))
        self.authority.db._execute_write(complete)

    def attest_discard_reservation(self, selector, params):
        if set(params) != {
                'profile', 'source', 'session_id', 'expected_task_id',
                'execution_generation', '_source_discard_digest', '_target_home'}:
            raise RuntimeStoreError('invalid_params')
        if (params['profile'] != selector['profile'] or params['source'] != 'bot_room'
                or type(params['execution_generation']) is not int
                or params['execution_generation'] < 1):
            raise RuntimeStoreError('invalid_params')
        matches = [
            task for task in tasks.list_tasks(self.db_path, room_id=selector['room_id'])
            if task['identity'].task_id == params['expected_task_id']
            and task['execution_generation'] == params['execution_generation']
        ]
        if len(matches) != 1:
            raise RuntimeStoreError('permission_denied')
        task = matches[0]
        with self.authority.db._read_ctx() as conn:
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_discard_key(task),)).fetchone()
        saved = json.loads(row[0]) if row is not None else None
        if (not isinstance(saved, dict) or saved.get('state') != 'reserved'
                or saved.get('reservation_digest') != params['_source_discard_digest']
                or saved.get('target_session_id') != params['session_id']
                or saved.get('member_id') != selector['member_id']
                or saved.get('target_profile') != selector['profile']
                or task['status'] != 'indeterminate'):
            raise RuntimeStoreError('permission_denied')
        return {'source_discard_digest': saved['reservation_digest']}

    def retry_room_task(self, room_id, *, member_id, task_id, execution_generation):
        with self._policy_lock:
            task, binding = self._control_task(room_id, member_id, task_id, execution_generation,
                                               proven_peer_retry=True)
            if not self._member_is_peer(room_id, member_id):
                # Unknown is not non-admission. Never advance its hosted generation
                # while leaving the canonical unknown head behind it.
                if task['status'] == 'indeterminate':
                    raise RuntimeStoreError('unknown_execution')
                if task['status'] != 'deferred':
                    raise RuntimeStoreError('stale_generation')
                rpc = self._resolve_member_transport(binding, task)
                info = rpc.info(profile=task['payload']['target_profile'], source='bot_room',
                                session_id=rpc.ref.session_id)
                if info.get('status') == 'unknown':
                    raise RuntimeStoreError('unknown_execution')
                if info.get('active'):
                    raise RuntimeStoreError('session_busy')
                lease = self.runtime._ensure_lease(binding)
                return self.runtime._requeue(tasks.requeue_deferred_task, task, lease, room_id)
        # The peer path owns its short capture/final locks, not the network wait.
        from gateway.session_hosted_peer_retry import retry_peer
        return retry_peer(self, task, binding)

    def status(self, room_id=None, *, state_read=None):
        if state_read is not None:
            from gateway.session_group_state import driver_status
            return driver_status(self, room_id, state_read)
        result = super().status(room_id)
        if room_id is None:
            return result
        actions = [a for a in result['pending_actions'] if a['kind'] != 'retry']
        for task in tasks.list_tasks(self.db_path, room_id=room_id):
            if task['status'] not in {'indeterminate', 'deferred'}:
                continue
            member = task['payload'].get('target_member_id') or task['payload']['target_profile']
            if self._member_is_peer(room_id, member):
                from gateway.session_hosted_peer_retry import retry_available
                gateway, epoch = self._owned_authority(room_id)
                if not retry_available(self, task, HostedRoomBinding(room_id, gateway, epoch)):
                    continue
            actions.append({'kind': 'discard' if task['status'] == 'indeterminate' else 'retry',
                            'member_id': member, 'task_id': task['identity'].task_id,
                            'execution_generation': task['execution_generation']})
        return {**result, 'pending_actions': actions}
