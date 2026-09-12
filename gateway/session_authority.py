"""Gateway-owned live sessions and canonical durable FIFO scheduling.

Only the runtime bootstrap holding profile ownership may initialize this service.
Transport attachment never constructs an agent or takes a turn lease.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import uuid

from gateway.session_contract import (
    CANONICAL_GATEWAY_PROTOCOL, AdmissionReceipt, PendingAdmission, Principal, SessionHandle, SessionRef, Submission,
    SubscriptionSnapshot,
)
from gateway.session_events import SessionEvents
from gateway.session_pending_controls import PendingControls
from hermes_state_runtime import (
    RuntimeStoreError, admit_session_input, begin_runtime_epoch,
    cancel_session_input, claim_session_input, get_session_admission,
    list_session_admissions, recover_session_inputs, resolve_unknown_session_input,
)


@dataclass
class LiveSession:
    source: object
    route: str
    task: asyncio.Task | None = None
    subscribers: dict = field(default_factory=dict)
    event_stream: SessionEvents = field(default_factory=SessionEvents)
    controls: PendingControls = field(init=False)

    def __post_init__(self):
        self.controls = PendingControls(self.event_stream)


class SessionAuthority:
    def __init__(self, runner, *, profile_id, instance_id, db, epoch):
        self.runner = runner
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.db = db
        self.epoch = epoch
        self.sessions = {}
        self.waiters = {}
        self.events = {}
        self.native_waiters = set()
        self.pending_results = {}

    def authorize(self, actor, ref, capability):
        if actor.profile_id != self.profile_id or ref.profile_id != self.profile_id:
            raise RuntimeStoreError('profile_mismatch')
        if capability not in actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        if ref.session_id not in self.sessions:
            row = self.db.get_session(ref.session_id)
            if row is None:
                raise RuntimeStoreError('not_found')
            from hermes_state_local import POLICY_PREFIX
            with self.db._read_ctx() as conn:
                from hermes_state_local_migration import LEGACY_PREFIX
                local = conn.execute('SELECT 1 FROM state_meta WHERE key IN (?,?)',
                    (POLICY_PREFIX + ref.session_id, LEGACY_PREFIX + ref.session_id)).fetchone()
            if local or str(row.get('chat_id') or '').startswith('local-'):
                from gateway.session_local_recovery import restore_local_session
                restore_local_session(self, ref.session_id)
            else:
                from gateway.session_api import restore_api_session
                restore_api_session(self, ref.session_id)
        from gateway.config import Platform
        source = self.sessions[ref.session_id].source
        if (source is not None and source.platform == Platform.LOCAL
                and source.user_id != actor.subject and 'session:operator' not in actor.capabilities):
            raise RuntimeStoreError('permission_denied')

    def _require_admission_open(self):
        if self.runner._draining:
            raise RuntimeStoreError('runtime_draining')

    def register(self, source):
        self._require_admission_open()
        entry = self.runner.session_store.get_or_create_session(source)
        sid = entry.session_id
        self.sessions.setdefault(sid, LiveSession(source, entry.session_key))
        # SessionStore reserves routing metadata before the first AIAgent exists.
        if self.db.get_session(sid) is None:
            self.db.create_session(sid, source=source.platform.value)
        return SessionRef(self.profile_id, sid)

    def agent(self, ref):
        return self.runner._cached_agent_for(self.sessions[ref.session_id].route)

    def _handle(self, ref):
        row = self.db.get_session(ref.session_id)
        pending = list_session_admissions(self.db, session_id=ref.session_id)
        state = 'unknown' if any(r['status'] == 'unknown' for r in pending) else (
            'running' if any(r['status'] == 'started' for r in pending) else 'idle')
        return SessionHandle(ref, self.instance_id, self.epoch, row['runtime_revision'],
                             row['runtime_generation'], state)

    async def resolve(self, actor, ref):
        self.authorize(actor, ref, 'session:read')
        return self._handle(ref)

    async def attach(self, actor, ref):
        self.authorize(actor, ref, 'session:read')
        live = self.sessions[ref.session_id]
        from gateway.session_local_recovery import local_history
        with live.event_stream.lock:
            subscription = next((key for key, member in live.subscribers.items()
                                 if member == actor), None) or uuid.uuid4().hex
            live.subscribers[subscription] = actor
            transport = self.events.get(actor.transport_id)
            if transport is not None:
                live.event_stream.fanout.attach(transport)
            handle = self._handle(ref)
            active_generation = handle.execution_generation if handle.execution_state == "running" else None
            prompts = live.controls.snapshot(ref.session_id, active_generation)
            epoch, sequence = live.event_stream.watermark()
            return SubscriptionSnapshot(subscription, handle, epoch,
                                        sequence, tuple(local_history(self, ref)),
                                        tuple(self._pending_receipt(r) for r in list_session_admissions(
                                            self.db, session_id=ref.session_id)), prompts)

    async def detach(self, actor, subscription_id):
        for live in self.sessions.values():
            if subscription_id in live.subscribers:
                if live.subscribers[subscription_id] != actor:
                    raise RuntimeStoreError('permission_denied')
                del live.subscribers[subscription_id]
                transport = self.events.get(actor.transport_id)
                if transport is not None:
                    live.event_stream.fanout.detach(transport)
                return
        raise RuntimeStoreError('not_found')

    def _receipt(self, row):
        return AdmissionReceipt(row['admission_id'], SessionRef(self.profile_id, row['target_session_id']),
                                row['seq'], row['status'], row['outcome'],
                                row['owner_epoch'] or self.epoch, row['generation'])

    def _pending_receipt(self, row):
        # Only public input text crosses the viewer boundary, never the
        # private native envelope's routing or authorization provenance.
        payload = row['payload']
        text = payload.get('text')
        if text is None:
            text = payload.get('native_text_v1', {}).get('event', {}).get('text', '')
        return PendingAdmission(**vars(self._receipt(row)), input_id=row['request_id'], text=text)

    def _publish_pending(self, ref):
        live = self.sessions[ref.session_id]
        with live.event_stream.lock:
            handle = self._handle(ref)
            pending = [asdict(self._pending_receipt(row)) for row in
                       list_session_admissions(self.db, session_id=ref.session_id)]
            # Turns bump runtime_revision without any session.updated event, so
            # this is the only place a viewer learns the CAS revision a later
            # prepared mutation must present.
            live.event_stream.publish(ref.session_id, {
                'stored_session_id': ref.session_id, 'pending': pending,
                'desktop_protocol': CANONICAL_GATEWAY_PROTOCOL,
                'running': handle.execution_state == 'running',
                'execution_generation': handle.execution_generation,
                'revision': handle.revision,
            }, event_type='session.info')

    def _schedule(self, ref):
        live = self.sessions[ref.session_id]
        if live.task is None or live.task.done():
            live.task = asyncio.create_task(self._drain(ref))

    async def admit_automation(self, adapter, event, identity):
        from gateway.session_automation import admit_automation
        return await admit_automation(self, adapter, event, identity)

    async def admit_native(self, event):
        """Await current connector policy, then commit before ACK or scheduling execution."""
        import json
        from gateway.session_envelope import prepare_native, restore_native
        self._require_admission_open()
        payload = await prepare_native(self.runner, event)
        self._require_admission_open()
        source = restore_native(payload).source
        ref = self.register(source)
        identity = json.dumps([source.profile, source.platform.value, source.chat_id,
                               source.thread_id, source.user_id], separators=(',', ':'))
        row = admit_session_input(self.db, epoch=self.epoch, principal_id='messaging:' + identity,
                                  session_id=ref.session_id,
                                  request_id=str(payload['native_text_v1']['event']['message_id'] or uuid.uuid4().hex),
                                  payload=payload)
        event._gateway_accepted = True
        self._publish_pending(ref)
        self._schedule(ref)
        return self._receipt(row)

    async def recover_native_sessions(self, bindings):
        """Bind only server-observed native routes; unknown work stays paused."""
        from collections import Counter
        from gateway.session_envelope import check_native_route
        bindings = list(bindings)
        counts = Counter(sid for sid, _, _ in bindings)
        results = {}
        for sid, available_source, adapter in bindings:
            try:
                if counts[sid] != 1:
                    raise RuntimeStoreError('admission_conflict')
                rows = list_session_admissions(self.db, session_id=sid, pending_only=False)
                native = [row for row in rows if 'native_text_v1' in row['payload']]
                if not native:
                    raise RuntimeStoreError('not_found')
                source, route = await check_native_route(self.runner, native[-1]['payload'], sid,
                                                    available_source, adapter)
                for row in rows:
                    if row['status'] == 'queued':
                        if 'native_text_v1' not in row['payload']:
                            raise RuntimeStoreError('invalid_params')
                        await check_native_route(self.runner, row['payload'], sid, available_source, adapter)
                self._require_admission_open()
                self.sessions.setdefault(sid, LiveSession(source, route))
                if any(row['status'] == 'unknown' for row in rows):
                    raise RuntimeStoreError('unknown_execution')
                self._schedule(SessionRef(self.profile_id, sid))
                results[sid] = 'ready'
            except RuntimeStoreError as exc:
                results[sid] = exc.reason
        return results

    async def submit(self, actor: Principal, request: Submission):
        self.authorize(actor, request.ref, 'session:submit')
        self._require_admission_open()
        if (request.intent != 'queue' or not {'text'} <= set(request.payload) <= {'text', 'attachments', 'finite'}
                or not isinstance(request.payload['text'], str)):
            raise RuntimeStoreError('invalid_params')
        from gateway.session_ingress_media import admit_attachments
        from gateway.session_finite import admit_finite
        finite = admit_finite(request.payload)
        payload = {'text': request.payload['text'], **finite,
                   **admit_attachments(request.payload.get('attachments'))}
        from gateway.config import Platform
        source = self.sessions[request.ref.session_id].source
        if source is not None and source.platform == Platform.LOCAL and source.user_id != actor.subject:
            # Durable server authorization, not a client payload field. The original
            # principal remains the admission/retry identity across owner restarts.
            payload['local_operator_v1'] = {
                'profile_id': self.profile_id, 'session_id': request.ref.session_id,
                'principal_id': actor.subject}
        row = admit_session_input(self.db, epoch=self.epoch, principal_id=actor.subject,
                                  session_id=request.ref.session_id, request_id=request.request_id,
                                  payload=payload, intent=request.intent)
        self._publish_pending(request.ref)
        self._schedule(request.ref)
        return self._receipt(row)

    async def receipt(self, actor, ref, admission_id):
        self.authorize(actor, ref, 'session:submit')
        row = get_session_admission(self.db, admission_id=admission_id)
        if row is None or row['target_session_id'] != ref.session_id:
            raise RuntimeStoreError('not_found')
        if row['principal_id'] != actor.subject and 'session:control' not in actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        return self._receipt(row)

    async def cancel_queued(self, actor, ref, admission_id):
        await self.receipt(actor, ref, admission_id)
        row = cancel_session_input(self.db, epoch=self.epoch, admission_id=admission_id)
        self._publish_pending(ref)
        return self._receipt(row)

    async def resolve_unknown(self, actor, ref, admission_id, generation):
        """Operator acknowledgement that a turn lost across an owner restart will not
        finish; the paused FIFO behind it resumes. Never requeues the lost input."""
        self.authorize(actor, ref, 'session:control')
        await self.receipt(actor, ref, admission_id)
        row = resolve_unknown_session_input(self.db, epoch=self.epoch, admission_id=admission_id,
                                            generation=generation)
        self._publish_pending(ref)
        self._schedule(ref)
        return self._receipt(row)

    async def interrupt(self, actor, ref, generation):
        self.authorize(actor, ref, 'session:control')
        handle = self._handle(ref)
        if handle.execution_generation != generation:
            raise RuntimeStoreError('stale_generation')
        agent = self.agent(ref)
        if agent is not None and handle.execution_state == 'running':
            agent.interrupt()
        return self._handle(ref)

    def check_approval_generation(self, session_id, generation):
        handle = self._handle(SessionRef(self.profile_id, session_id))
        if handle.execution_generation != generation or handle.execution_state != "running":
            raise RuntimeStoreError("stale_generation")

    def publish_execution(self, session_id, generation, event_type, payload):
        """Worker callbacks never outlive their exact running claim."""
        live = self.sessions[session_id]
        with live.event_stream.lock:
            try:
                self.check_approval_generation(session_id, generation)
            except RuntimeStoreError:
                return False
            live.event_stream.publish(session_id, payload, event_type=event_type)
            from gateway.session_api_turn import publish_api_event
            publish_api_event(self, session_id, event_type, payload)
            return True

    def register_approval(self, session_id, generation, route, data):
        live = self.sessions[session_id]
        with live.event_stream.lock:
            self.check_approval_generation(session_id, generation)
            live.controls.register(session_id, route, generation, data)

    def register_clarify(self, session_id, generation, entry):
        live = self.sessions[session_id]
        with live.event_stream.lock:
            self.check_approval_generation(session_id, generation)
            live.controls.register_clarify(session_id, generation, entry)

    async def respond(self, actor, ref, generation, prompt_id, response, *, kind="approval"):
        capability = {"approval": "session:approve", "clarify": "session:respond"}.get(kind)
        if capability is None:
            raise RuntimeStoreError("invalid_params")
        self.authorize(actor, ref, capability)
        live = self.sessions[ref.session_id]
        with live.event_stream.lock:
            if actor not in live.subscribers.values():
                raise RuntimeStoreError("permission_denied")
            if type(generation) is not int:
                raise RuntimeStoreError("stale_generation")
            self.check_approval_generation(ref.session_id, generation)
            if not isinstance(prompt_id, str) or not prompt_id:
                raise RuntimeStoreError("invalid_params")
            return live.controls.respond(ref.session_id, generation, prompt_id, response, kind=kind)

    async def _drain(self, ref):
        from gateway.session_finite import execute_finite_admission
        live = self.sessions[ref.session_id]
        while True:
            try:
                pending = list_session_admissions(self.db, session_id=ref.session_id)
                if any(row['status'] == 'unknown' for row in pending):
                    return
                first = next((row for row in pending if row['status'] == 'queued'), None)
                from gateway.config import Platform
                if first is not None and live.source.platform == Platform.LOCAL:
                    from gateway.session_local_recovery import restore_local_session
                    restore_local_session(self, ref.session_id)
                    if first['request_id'].startswith('hosted:'):
                        from gateway.session_hosted_transport import check_remote_hosted_admission
                        if not await asyncio.to_thread(check_remote_hosted_admission, self, ref, first):
                            service = getattr(self, 'hosted_room_service', None)
                            if service is None:
                                raise RuntimeStoreError('permission_denied')
                            await asyncio.to_thread(service.check_admission, ref, first)
                    if 'local_automation_v1' in first['payload']:
                        from gateway.session_automation import check_local_automation
                        check_local_automation(self, ref, first)
                    else:
                        from gateway.session_operator import check_local_input
                        check_local_input(self, ref, first)
                if first is not None and 'native_text_v1' in first['payload']:
                    from gateway.session_envelope import check_native_route
                    await check_native_route(self.runner, first['payload'], ref.session_id, live.source,
                                       self.runner._adapter_for_source(live.source))
                    # Cancellation may advance FIFO while the connector is awaited.
                    # Never let the successor inherit this row's fresh verdict.
                    current = get_session_admission(self.db, admission_id=first['admission_id'])
                    if current is None or current['status'] != 'queued':
                        continue
                if first is not None and live.source.platform == Platform.API_SERVER:
                    from gateway.session_api_turn import check_api_turn
                    check_api_turn(self, ref, first['payload'])
                self._require_admission_open()
                row = claim_session_input(self.db, epoch=self.epoch, session_id=ref.session_id)
            except RuntimeStoreError as exc:
                import logging
                logging.getLogger(__name__).warning('Session %s paused: %s', ref.session_id, exc.reason)
                return
            if row is None:
                return
            admission_id = row['admission_id']
            with live.event_stream.lock:
                live.event_stream.execution = {
                    'authority_epoch': self.epoch, 'execution_generation': row['generation'],
                    'admission_id': admission_id}
                live.event_stream.publish(ref.session_id, {}, event_type='message.start')
                self._publish_pending(ref)
            try:
                response = await execute_finite_admission(self, ref, row)
                outcome = 'completed'
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Admitted turn %s failed', admission_id)
                response = 'The admitted turn failed.'
                outcome = 'failed'
            with live.event_stream.lock:
                from gateway.session_results import finish_result
                settled, response = finish_result(self.db, epoch=self.epoch, row=row,
                    response=response, outcome=outcome,
                    result=self.pending_results.pop(admission_id, None))
                live.controls.snapshot(ref.session_id, None)
                self._publish_pending(ref)
                live.event_stream.publish(ref.session_id, {
                    'text': response, 'content': response, 'admission_id': admission_id,
                    'outcome': 'cancelled' if settled['outcome'] == 'interrupted' else settled['outcome']})
            waiter = self.waiters.pop(admission_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(response)


async def initialize_session_authority(runner, *, profile_id, instance_id, db=None, register=True):
    """Call after exclusive profile ownership, before connecting adapters/API.

    ``register=False`` builds a served secondary's authority without making it the runner's
    launch authority (``runner.session_authority``); the per-home registry owns the lookup.
    """
    if db is None:
        db = getattr(runner._session_db, '_db', runner._session_db)
    epoch = begin_runtime_epoch(db, instance_id=instance_id)
    recover_session_inputs(db, epoch=epoch)
    authority = SessionAuthority(runner, profile_id=profile_id, instance_id=instance_id, db=db, epoch=epoch)
    if register:
        runner.session_authority = authority
    from gateway.session_cron import bind_owner
    bind_owner(authority)
    store = runner.session_store
    if register:
        store._local_authority_epoch = epoch
    # Local resets write the owning profile's store; the epoch fence must be that store's.
    epochs = getattr(store, '_local_authority_epochs', None)
    if epochs is None:
        epochs = store._local_authority_epochs = {}
    from pathlib import Path
    epochs[Path(db.db_path).resolve()] = epoch
    from gateway.session_local_recovery import recover_local_sessions
    recover_local_sessions(authority)
    return authority
