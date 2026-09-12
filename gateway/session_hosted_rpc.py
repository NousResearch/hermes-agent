"""Private hosted-room producer over the gateway's canonical local-session FIFO.

One instance belongs to a durable room/member/profile binding. The service supplies
its principal and current-membership check; neither is taken from RPC payloads.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import hashlib
import json

from gateway.hosted_room_driver import TaskIdentity
from gateway.session_contract import SessionRef, Submission
from hermes_state_runtime import RuntimeStoreError, list_session_admissions


class HostedRoomAuthorityRPC:
    def __init__(self, authority, loop, *, room_id, member_id, profile, principal,
                 authorize, timeout=30):
        self.authority, self.loop = authority, loop
        self.room_id, self.member_id, self.profile = room_id, member_id, profile
        self.principal, self.authorizer, self.timeout = principal, authorize, timeout
        self.callbacks = {}
        binding = json.dumps([room_id, member_id, profile], separators=(',', ':'))
        self.creation_id = 'hosted:' + hashlib.sha256(binding.encode()).hexdigest()
        from gateway.session_local_recovery import local_identity
        self.ref = SessionRef(authority.profile_id, local_identity(
            authority.profile_id, principal.subject, self.creation_id))

    def _call(self, operation, **params):
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            raise RuntimeStoreError('invalid_params')
        if not self.loop.is_running():
            raise RuntimeStoreError('runtime_draining')
        future = asyncio.run_coroutine_threadsafe(self._dispatch(operation, params), self.loop)
        # A timeout is ambiguous: never cancel an admission or claim not_admitted.
        return future.result(self.timeout)

    async def _dispatch(self, operation, params):
        from gateway.session_authorities import owner_scope
        with owner_scope(self.authority):
            return await self._dispatch_owned(operation, params)

    async def _dispatch_owned(self, operation, params):
        if (params.get('profile', self.profile) != self.profile
                or self.principal.profile_id != self.authority.profile_id):
            raise RuntimeStoreError('profile_mismatch')
        if (params.get('source', 'bot_room') != 'bot_room'
                or params.get('session_id', self.ref.session_id) != self.ref.session_id
                or params.get('title', f'Group: {self.room_id}') != f'Group: {self.room_id}'):
            raise RuntimeStoreError('permission_denied')
        task, generation = params.get('task'), params.get('execution_generation')
        if operation == 'submit' and (not isinstance(task, TaskIdentity)
                or task.room_id != self.room_id or type(generation) is not int or generation < 1):
            raise RuntimeStoreError('invalid_params')
        if self.authorizer(operation, task, generation) is not True:
            raise RuntimeStoreError('permission_denied')
        return await getattr(self, '_' + operation)(params)

    async def _resolve_exact(self, params):
        if self.authority.db.get_session(self.ref.session_id) is None:
            return None
        return await self._resume(params)

    async def _create(self, params):
        if self.authority.db.get_session(self.ref.session_id) is not None:
            return await self._resume(params)
        from gateway.run import _load_gateway_config, _resolve_gateway_model
        from gateway.session_policy import build_policy
        from gateway.session_local import create_local_session
        private = {}
        policy = build_policy({'source': 'gui'}, _load_gateway_config(), private_secrets=private)
        policy = replace(policy, source='bot_room', platform='bot_room',
                         model=policy.model or _resolve_gateway_model(policy.config()),
                         toolsets=tuple(sorted(set(policy.toolsets) | {'bot_room'})))
        create_local_session(self.authority, self.principal, {'request_id': self.creation_id},
                             trusted_policy=policy, trusted_secrets=private)
        from hermes_state_runtime import _epoch
        def label(conn):
            _epoch(conn, self.authority.epoch)
            conn.execute('UPDATE sessions SET title=?, hidden=1 WHERE id=?',
                         (f'Group: {self.room_id}', self.ref.session_id))
        self.authority.db._execute_write(label)
        return await self._resume(params)

    async def _resume(self, params):
        await self.authority.resolve(self.principal, self.ref)
        return {'session_id': self.ref.session_id, 'title': f'Group: {self.room_id}'}

    def _rows(self):
        self.authority.authorize(self.principal, self.ref, 'session:read')
        rows = list_session_admissions(self.authority.db, session_id=self.ref.session_id, pending_only=False)
        result = []
        for row in rows:
            if row['principal_id'] != self.principal.subject or not row['request_id'].startswith('hosted:'):
                continue
            try:
                task, generation = json.loads(row['request_id'][7:])
                identity = TaskIdentity(**task)
                if identity.room_id != self.room_id or type(generation) is not int:
                    raise ValueError('invalid hosted identity')
            except (TypeError, ValueError, KeyError) as exc:
                raise RuntimeStoreError('storage_unavailable') from exc
            result.append((row, identity, generation))
        return result

    def _terminal(self, row, task, generation):
        from gateway.session_results import admission_result
        saved = admission_result(self.authority.db, row['admission_id'])
        if saved is None and row['outcome'] != 'interrupted':
            raise RuntimeStoreError('storage_unavailable')
        # Explicit unknown discard has no execution result to recover.
        value = saved['result'] if saved is not None else {}
        status = {'completed': 'settled', 'interrupted': 'cancelled'}.get(row['outcome'], 'failed')
        receipt = {'status': status, 'text': value.get('final_response', ''),
                   'message_id': row['admission_id'], 'settlement_id': row['admission_id'],
                   'task_id': task.task_id, 'execution_generation': generation}
        callback = self.callbacks.pop(row['admission_id'], None)
        if callback is not None:
            callback(receipt)
        return receipt

    async def _submit(self, params):
        task, generation = params['task'], params['execution_generation']
        request_id = 'hosted:' + json.dumps([asdict(task), generation], sort_keys=True, separators=(',', ':'))
        # Refuse unknown before submit: submit itself schedules the queue on retries.
        rows = self._rows()
        if any(row['status'] == 'unknown' for row, _, _ in rows):
            raise RuntimeStoreError('unknown_execution')
        from gateway.session_hosted_attachments import submission_payload
        payload = await asyncio.to_thread(
            submission_payload, self, params['prompt'], params.get('attachments'))
        receipt = await self.authority.submit(self.principal, Submission(
            request_id, self.ref, payload, 'queue'))
        self.callbacks[receipt.admission_id] = params['on_terminal']
        if receipt.status in {'queued', 'started'}:
            waiter = self.authority.waiters.get(receipt.admission_id)
            if waiter is None:
                waiter = self.loop.create_future()
                self.authority.waiters[receipt.admission_id] = waiter
            waiter.add_done_callback(lambda done: self._completed(receipt.admission_id, done))
        if receipt.status == 'terminal':
            for row, identity, gen in self._rows():
                if row['admission_id'] == receipt.admission_id:
                    self._terminal(row, identity, gen)
        return asdict(receipt)

    def _completed(self, admission_id, future):
        if future.cancelled() or admission_id not in self.callbacks:
            return
        if self.authorizer('terminal', None, None) is not True:
            self.callbacks.pop(admission_id, None)
            return
        for row, task, generation in self._rows():
            if row['admission_id'] == admission_id and row['status'] == 'terminal':
                self._terminal(row, task, generation)

    async def _history(self, params):
        from gateway.session_local_recovery import local_history
        rows = self._rows()
        history = list(local_history(self.authority, self.ref))
        for row, task, generation in rows:
            if row['status'] == 'terminal':
                receipt = self._terminal(row, task, generation)
                history.append({**receipt, 'role': 'assistant', 'content': receipt['text']})
        return history

    async def _info(self, params):
        rows = self._rows()
        for row, task, generation in rows:
            if row['status'] == 'terminal':
                self._terminal(row, task, generation)
        current = next(((row, task, generation) for row, task, generation in rows if row['status'] in {'started', 'unknown', 'queued'}), None)
        result = {'active': current is not None and current[0]['status'] != 'unknown',
                  'task_id': current[1].task_id if current else None,
                  'execution_generation': current[2] if current else None,
                  'status': current[0]['status'] if current else 'idle'}
        snapshot = await self.authority.attach(self.principal, self.ref)
        if snapshot.prompts:
            prompt = next((p for p in snapshot.prompts if p.get('kind') == 'approval'), None)
            if prompt:
                result.update(status='waiting_for_approval', pending_approval={
                    **prompt, 'request_id': prompt['prompt_id'],
                    'choices': [c for c in prompt['choices'] if c in {'once', 'deny'}]})
        return result

    async def _interrupt(self, params):
        rows = self._rows()
        current = next(((row, task) for row, task, _ in rows if row['status'] in {'started', 'unknown', 'queued'}), None)
        if current is None or current[1].task_id != params['expected_task_id']:
            raise RuntimeStoreError('stale_generation')
        row, _ = current
        if row['status'] == 'unknown':
            raise RuntimeStoreError('unknown_execution')
        if row['status'] == 'queued':
            await self.authority.cancel_queued(self.principal, self.ref, row['admission_id'])
        else:
            await self.authority.interrupt(self.principal, self.ref, row['generation'])
        return {'interrupted': True, 'status': 'interrupted'}

    async def _discard(self, params):
        generation = params['execution_generation']
        if type(generation) is not int or generation < 1:
            raise RuntimeStoreError('invalid_params')
        matches = [(row, task) for row, task, hosted_generation in self._rows()
                   if (row['status'] == 'unknown' or (row['status'] == 'terminal' and row['outcome'] == 'interrupted'))
                   and task.task_id == params['expected_task_id']
                   and hosted_generation == generation]
        if len(matches) != 1:
            raise RuntimeStoreError('stale_generation')
        row, task = matches[0]
        # The public fence is hosted; the canonical CAS uses its own generation.
        if row['status'] == 'unknown':
            await self.authority.resolve_unknown(
                self.principal, self.ref, row['admission_id'], row['generation'])
        return {'discarded': True, 'status': 'cancelled', 'task_id': task.task_id,
                'execution_generation': generation}

    async def _approve(self, params):
        if params['choice'] not in {'once', 'deny'}:
            raise RuntimeStoreError('invalid_params')
        snapshot = await self.authority.attach(self.principal, self.ref)
        prompt = next((p for p in snapshot.prompts if p.get('prompt_id') == params['request_id']
                       and p.get('kind') == 'approval'), None)
        if prompt is None:
            raise RuntimeStoreError('stale_generation')
        return await self.authority.respond(self.principal, self.ref,
            snapshot.handle.execution_generation, params['request_id'], {'choice': params['choice']})

    def resolve_exact(self, *, profile, title, source):
        return self._call('resolve_exact', profile=profile, title=title, source=source)

    def create(self, *, profile, title, source):
        return self._call('create', profile=profile, title=title, source=source)

    def resume(self, *, profile, session_id, source):
        return self._call('resume', profile=profile, session_id=session_id, source=source)

    def submit(self, *, profile, session_id, prompt, source, task, execution_generation, on_terminal, attachments=None):
        return self._call('submit', profile=profile, session_id=session_id, source=source,
                          prompt=prompt, task=task, execution_generation=execution_generation, on_terminal=on_terminal,
                          attachments=attachments)

    def history(self, *, profile, session_id, source):
        return self._call('history', profile=profile, session_id=session_id, source=source)

    def info(self, *, profile, session_id, source):
        return self._call('info', profile=profile, session_id=session_id, source=source)

    def interrupt(self, *, profile, session_id, source, expected_task_id):
        return self._call('interrupt', profile=profile, session_id=session_id, source=source, expected_task_id=expected_task_id)

    def discard(self, *, profile, session_id, source, expected_task_id, execution_generation):
        return self._call('discard', profile=profile, session_id=session_id, source=source,
                          expected_task_id=expected_task_id, execution_generation=execution_generation)

    def approve(self, *, session_id, request_id, choice):
        return self._call('approve', session_id=session_id, request_id=request_id, choice=choice)
