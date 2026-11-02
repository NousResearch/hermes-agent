"""Exact operator controls for canonical hosted attempts, never legacy replay."""
from gateway import hosted_room_driver as tasks
from hermes_state_runtime import RuntimeStoreError
from tui_gateway.hosted_room_driver import HostedRoomBinding
from tui_gateway.hosted_room_prepare_stall import RETRY_TASK_PREFIX
from tui_gateway.hosted_room_service import attempt_action, task_member_id


class HostedControls:
    def _control_task(self, room_id, member_id, task_id, execution_generation):
        if (type(execution_generation) is not int or execution_generation < 1
                or not isinstance(member_id, str) or not member_id
                or not isinstance(task_id, str) or not task_id):
            raise RuntimeStoreError('invalid_params')
        gateway, epoch = self._owned_authority(room_id)
        task = next((t for t in tasks.list_tasks(self.db_path, room_id=room_id)
                     if t['identity'].task_id == task_id), None)
        if (task is None or task['execution_generation'] != execution_generation
                or task_member_id(task) != member_id):
            raise RuntimeStoreError('stale_generation')
        if not any(m.get('member_id') == member_id
                   and m.get('profile') == task['payload']['target_profile']
                   for m in self._room(room_id)['members']):
            raise RuntimeStoreError('permission_denied')
        if self._member_is_peer(room_id, member_id):
            raise RuntimeStoreError('unsupported_operation')
        return task, HostedRoomBinding(room_id, gateway, epoch)

    def discard_room_task(self, room_id, *, member_id, task_id, execution_generation):
        with self._policy_lock:
            task, binding = self._control_task(room_id, member_id, task_id, execution_generation)
            cancel_id = f'discard:{execution_generation}'
            if task['status'] == 'cancelled' and task.get('cancel_id') == cancel_id:
                return task
            if task['status'] != 'indeterminate':
                raise RuntimeStoreError('stale_generation')
            rpc = self._resolve_member_transport(binding, task)
            # Canonical receipt commits first. If the driver write fails, exact
            # RPC replay must recover it rather than invent another execution.
            rpc.discard(profile=task['payload']['target_profile'], source='bot_room',
                        session_id=rpc.ref.session_id, expected_task_id=task_id,
                        execution_generation=execution_generation)
            lease = self.runtime._ensure_lease(binding)
            result = self.runtime._fenced(tasks.resolve_indeterminate_cancellation,
                binding, task, lease, cancel_id=cancel_id)
            self.runtime._set_blocked(room_id, False)
            self.runtime.wakeup()
            return result

    def retry_room_task(self, room_id, *, member_id, task_id, execution_generation):
        if isinstance(task_id, str) and task_id.startswith(RETRY_TASK_PREFIX):
            # A stalled room preparation, not an attempt: the shared service retries it.
            return super().retry_room_task(
                room_id, task_id=task_id, member_id=member_id, execution_generation=execution_generation)
        with self._policy_lock:
            task, binding = self._control_task(room_id, member_id, task_id, execution_generation)
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

    def status(self, room_id=None):
        result = super().status(room_id)
        if room_id is None:
            return result
        actions = [a for a in result['pending_actions']
                   if a['kind'] != 'retry' or str(a.get('task_id', '')).startswith(RETRY_TASK_PREFIX)]
        for task in tasks.list_tasks(self.db_path, room_id=room_id):
            if task['status'] not in {'indeterminate', 'deferred'}:
                continue
            if self._member_is_peer(room_id, task_member_id(task)):
                continue
            actions.append(attempt_action(
                'discard' if task['status'] == 'indeterminate' else 'retry', task))
        return {**result, 'pending_actions': actions}
