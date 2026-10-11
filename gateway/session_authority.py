"""Gateway-owned live sessions and canonical durable FIFO scheduling.

Only the runtime bootstrap holding profile ownership may initialize this service.
Transport attachment never constructs an agent or takes a turn lease.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from functools import partial
import sqlite3
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

# Capped backoff for a claim refused by transient storage (writer lock, full disk).
_CLAIM_RETRY_MIN_S, _CLAIM_RETRY_MAX_S = 0.25, 5.0


@dataclass
class LiveSession:
    source: object
    route: str
    task: asyncio.Task | None = None
    subscribers: dict = field(default_factory=dict)
    event_stream: SessionEvents = field(default_factory=SessionEvents)
    controls: PendingControls = field(init=False)
    # The messaging ingress tells the platform user once per pause episode (unknown head,
    # preflight refusal), not per message; the drain clears it when the FIFO moves again.
    pause_notified: bool = False
    mutation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Held by the FIFO claim and by compress across its summarization (which runs outside
    # ``mutation_lock``, so admissions are not parked behind an LLM call): no turn is claimed
    # under a compression that is still preparing its replacement transcript.
    claim_gate: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Set by ``_schedule`` while a drain is alive. The claim commits off-loop, so input admitted
    # after its transaction read the FIFO would otherwise be stranded by a drain that exits idle.
    rescan: bool = False
    # admission id -> its started claim, from the moment the settlement worker starts the terminal
    # write until the completion frame is published. The write runs outside ``event_stream.lock``
    # (a contended SQLite writer must not stall loop-side lock takers); readers holding that lock
    # still see the claim as started, so no snapshot shows a terminal row without its completion.
    settling: dict = field(default_factory=dict)

    def __post_init__(self):
        self.controls = PendingControls(self.event_stream)


def _log_drain_failure(task):
    """A dead pump is the one failure this module must never swallow."""
    if task.cancelled() or task.exception() is None:
        return
    import logging
    logging.getLogger(__name__).error('Session drain task died: %r', task.exception())


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
        # Replies of recovered no-waiter turns, sent only once their terminal outcome commits.
        self.pending_deliveries = {}
        # Stops accepted for a running generation whose agent does not exist yet
        # (first-turn construction); consumed by adopt_agent, keyed session -> generation.
        self.pending_stops = {}
        # session -> generation of the last Stop delivered to a live agent. A repeated
        # Stop can land on the session's reusable cached agent after that turn's finalizer
        # cleared it; the next generation's adopt_agent drops it instead of starting interrupted.
        self.delivered_stops = {}
        # session -> (generation, agent) its turn adopted. Until then the resident agent is the
        # previous turn's cached one, which pre-turn compression or a changed agent config
        # replaces, so a Stop that only reached it would let the rebuilt agent run the turn. After
        # it, a Stop goes to this agent, never through a cache that may have evicted it.
        self.adopted = {}
        # Queued admissions whose cancellation may have committed before its observers settled.
        self.cancel_obligations = set()
        # Set by profile retirement (unserve): the drain claims no successor after its running turn.
        self.retiring = False

    def authorize(self, actor, ref, capability):
        """Every handler calls this first, so a later ``self.sessions[ref.session_id]`` is
        safe: a deleted/evicted live entry surfaces here as ``not_found``, not as a KeyError
        deeper in the handler."""
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
        if self.runner._draining or self.retiring:
            raise RuntimeStoreError('runtime_draining')

    def _admission_gate(self, authorize=None):
        """The admission gate re-checked inside an off-loop write transaction (with the caller's
        own guard), so a drain that began while the write waited on the writer still refuses it."""
        def guard(conn):
            self._require_admission_open()
            if authorize is not None:
                authorize(conn)
        return guard

    def logical_owner(self, session_id):
        """The FIFO/admission identity of a route: the root of its compression lineage.
        Compression advances the physical transcript, never the admission identity. A session this
        store does not hold (another served profile's, under multiplex) keeps its own id."""
        if not session_id:
            return session_id
        lineage = self.db.get_compression_lineage(session_id)
        return lineage[0] if lineage else session_id

    def physical_target(self, ref):
        return self.db.get_compression_tip(ref.session_id) or ref.session_id

    def register(self, source):
        self._require_admission_open()
        entry = self.runner.session_store.get_or_create_session(source)
        sid = self.logical_owner(entry.session_id)
        self.sessions.setdefault(sid, LiveSession(source, entry.session_key))
        # SessionStore reserves routing metadata before the first AIAgent exists.
        if self.db.get_session(sid) is None:
            self.db.create_session(sid, source=source.platform.value)
        return SessionRef(self.profile_id, sid)

    def agent(self, ref):
        lookup = getattr(self.runner, '_resident_agent_for', None) or self.runner._cached_agent_for
        return lookup(self.sessions[ref.session_id].route)

    def _pending_rows(self, ref):
        """Pending admissions as the event stream presents them: a settlement committed but not yet
        published (``LiveSession.settling``) still reads as its started claim. Callers that pair the
        result with stream state hold ``event_stream.lock``, under which ``settling`` changes."""
        rows = list_session_admissions(self.db, session_id=ref.session_id)
        live = self.sessions.get(ref.session_id)
        masked = dict(live.settling) if live is not None else {}
        if not masked:
            return rows
        rows = {row['admission_id']: row for row in rows}
        rows.update(masked)
        return sorted(rows.values(), key=lambda row: row['seq'])

    def _handle(self, ref):
        row = self.db.get_session(ref.session_id)
        pending = self._pending_rows(ref)
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
                live.event_stream.on_overflow = partial(self._retire_overflowed, ref.session_id)
            handle = self._handle(ref)
            active_generation = handle.execution_generation if handle.execution_state == "running" else None
            prompts = live.controls.snapshot(ref.session_id, active_generation)
            epoch, sequence = live.event_stream.watermark()
            return SubscriptionSnapshot(subscription, handle, epoch,
                                        sequence, tuple(local_history(self, ref)),
                                        tuple(self._pending_receipt(r) for r in self._pending_rows(ref)), prompts)

    def _retire_overflowed(self, session_id, transport):
        """The fanout dropped this peer's backlog: its subscription is over even though the
        socket still answers RPCs. A later resume re-attaches it with a fresh snapshot."""
        live = self.sessions[session_id]
        for subscription, member in list(live.subscribers.items()):
            if self.events.get(member.transport_id) is transport:
                del live.subscribers[subscription]

    async def detach(self, actor, subscription_id):
        for session_id, live in self.sessions.items():
            if subscription_id in live.subscribers:
                if live.subscribers[subscription_id] != actor:
                    raise RuntimeStoreError('permission_denied')
                del live.subscribers[subscription_id]
                transport = self.events.get(actor.transport_id)
                if transport is not None:
                    live.event_stream.fanout.detach(transport)
                if not live.subscribers:
                    from gateway.session_acp_lifecycle import end_idle_acp_session
                    end_idle_acp_session(self, session_id)
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
            pending = [asdict(self._pending_receipt(row)) for row in self._pending_rows(ref)]
            # Turns bump runtime_revision without any session.updated event, so
            # this is the only place a viewer learns the CAS revision a later
            # prepared mutation must present.
            live.event_stream.publish(ref.session_id, {
                'stored_session_id': ref.session_id, 'pending': pending, 'authority_epoch': self.epoch,
                'desktop_protocol': CANONICAL_GATEWAY_PROTOCOL,
                'running': handle.execution_state == 'running',
                'execution_generation': handle.execution_generation,
                'revision': handle.revision,
            }, event_type='session.info')
        from gateway.session_bot_mailbox import wake_bot_receipts
        wake_bot_receipts(self, ref.session_id)

    def _schedule(self, ref):
        live = self.sessions[ref.session_id]
        live.rescan = True
        if live.task is None or live.task.done():
            live.task = asyncio.create_task(self._drain(ref))
            live.task.add_done_callback(_log_drain_failure)
            from gateway.session_runtime_workers import persist_work_count
            live.task.add_done_callback(lambda task: persist_work_count(self, task))

    def wake_after_worker(self, session_id):
        """A worker execution on ``session_id`` (a logical owner or its compression continuation)
        ended: a drain that parked behind it re-runs. A no-op while a drain is running."""
        for sid in dict.fromkeys((session_id, self.logical_owner(session_id))):
            if sid in self.sessions:
                self._schedule(SessionRef(self.profile_id, sid))

    def _pause(self, ref, reason):
        """The FIFO stopped without claiming its head. Committed rows stay queued for a later
        drain; process-local delivery/HTTP/producer waiters on this session are released,
        with the reason instead of a reply, so an adapter loop is never parked on a turn that
        will not run. The ingress turns that refusal into one user-facing notice per episode.
        ``session_busy`` is transient (a registered worker finishes and wakes the FIFO), so only
        the messaging delivery waiters are released for it; an API/webhook/Bot observer keeps
        waiting for the answer instead of reporting accepted work as a conflict."""
        for row in list_session_admissions(self.db, session_id=ref.session_id):
            admission_id = row['admission_id']
            if reason == 'session_busy' and admission_id not in self.native_waiters:
                continue
            self.native_waiters.discard(admission_id)
            waiter = self.waiters.pop(admission_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_exception(RuntimeStoreError(reason))

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
        from gateway.session_runtime_workers import tracked_write
        request_id = str(payload['native_text_v1']['event']['message_id'] or uuid.uuid4().hex)
        return await tracked_write(self, partial(
            self._admit_native_write, principal_id='messaging:' + identity, session_id=ref.session_id,
            request_id=request_id, payload=payload, authorize=self._admission_gate()),
            then=partial(self._admitted, ref, event), ordered=True,
            after=partial(self._schedule_admitted, ref))

    def _admit_native_write(self, *, principal_id, session_id, request_id, payload, authorize):
        # Off the loop and in admission order: the redelivery reconcile reads the ledger it decides
        # against, so a concurrent first delivery of the same message cannot slip between them.
        from gateway.session_ingress_media import reconcile_native_retry
        payload = reconcile_native_retry(self.db, principal_id=principal_id, session_id=session_id,
                                         request_id=request_id, payload=payload)
        return admit_session_input(self.db, epoch=self.epoch, principal_id=principal_id, session_id=session_id,
                                   request_id=request_id, payload=payload, _authorize_write=authorize)

    def _admitted(self, ref, event, row):
        if event is not None:
            event._gateway_accepted = True
        self._publish_pending(ref)
        return self._receipt(row)

    def _schedule_admitted(self, ref, _receipt=None):
        # tracked_write's ``after``: in the admitting caller's own step (see tracked_write).
        self._schedule(ref)

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
                target = self.physical_target(SessionRef(self.profile_id, sid))
                source, route = await check_native_route(self.runner, native[-1]['payload'], target,
                                                    available_source, adapter)
                for row in rows:
                    if row['status'] != 'queued':
                        continue
                    if 'native_text_v1' in row['payload']:
                        await check_native_route(self.runner, row['payload'], target, available_source, adapter)
                    elif 'local_automation_v1' in row['payload']:
                        from gateway.session_automation import check_local_automation
                        check_local_automation(
                            self, SessionRef(self.profile_id, sid), row, route=route)
                    else:
                        from gateway.session_operator import check_local_input
                        check_local_input(
                            self, SessionRef(self.profile_id, sid), row, source=source)
                self._require_admission_open()
                self.sessions.setdefault(sid, LiveSession(source, route))
                if any(row['status'] == 'unknown' for row in rows):
                    raise RuntimeStoreError('unknown_execution')
                self._schedule(SessionRef(self.profile_id, sid))
                results[sid] = 'ready'
            except RuntimeStoreError as exc:
                results[sid] = exc.reason
        return results

    async def submit(self, actor: Principal, request: Submission, *, _authorize_write=None):
        self.authorize(actor, request.ref, 'session:submit')
        async with self.sessions[request.ref.session_id].mutation_lock:
            # Deletion can retire this session while the caller waits on its writer.
            self.authorize(actor, request.ref, 'session:submit')
            return await self._submit(actor, request, _authorize_write=_authorize_write)

    async def _submit(self, actor, request, *, _authorize_write=None):
        self._require_admission_open()
        if (request.intent != 'queue' or not {'text'} <= set(request.payload) <= {
                'text', 'attachments', 'finite', 'unattended', 'surface', 'voice_context', 'interrupted',
                'voice_turn', 'display_kind', 'title_preview'}
                or not isinstance(request.payload['text'], str)):
            raise RuntimeStoreError('invalid_params')
        from gateway.session_ingress_media import admit_attachments
        from gateway.session_finite import admit_finite
        from gateway.session_surface import admit_surface
        from gateway.session_display import admit_display
        finite = admit_finite(request.payload)
        def admitted():
            # The exact durable identity (authenticated principal + target + request id); the
            # attachment fields it yields still pass the full payload-digest comparison below.
            with self.db._read_ctx() as conn:
                row = conn.execute('SELECT * FROM session_admissions WHERE principal_id=? AND '
                                   'target_session_id=? AND request_id=?', (actor.subject,
                                   request.ref.session_id, request.request_id)).fetchone()
            from hermes_state_runtime import _row
            return _row(row) if row is not None else None
        payload = {'text': request.payload['text'], **finite, **admit_surface(request.payload),
                   **admit_display(request.payload),
                   **admit_attachments(request.payload.get('attachments'), admitted=admitted)}
        source = self.sessions[request.ref.session_id].source
        if source is not None and source.user_id != actor.subject:
            # Durable server authorization, not a client payload field. The original
            # principal remains the admission/retry identity across owner restarts.
            # Operator provenance is about who was authorized at this boundary, not
            # whether the bound session happens to use the LOCAL transport.
            payload['local_operator_v1'] = {
                'profile_id': self.profile_id, 'session_id': request.ref.session_id,
                'principal_id': actor.subject}
        def admit():
            try:
                return admit_session_input(self.db, epoch=self.epoch, principal_id=actor.subject,
                                           session_id=request.ref.session_id, request_id=request.request_id,
                                           payload=payload, intent=request.intent,
                                           _authorize_write=self._admission_gate(_authorize_write))
            except Exception:
                self._release_refused_capture(request, payload)
                raise
        from gateway.session_runtime_workers import tracked_write
        return await tracked_write(self, admit, then=partial(self._admitted, request.ref, None), ordered=True,
                                   after=partial(self._schedule_admitted, request.ref))

    def _release_refused_capture(self, request, payload):
        """A refused submission's captured bytes have no owner unless another admission holds the
        same bytes. They become durable retirement candidates (collected again by delete/prune if
        this pass fails), then a holder-aware release runs. Its failure is logged and never replaces
        the refusal the client must see (e.g. ``admission_conflict``)."""
        media = payload.get('attachments_v1', {}).get('media', [])
        if not media:
            return
        from hermes_state_media import collect_retired_media, retire_media
        try:
            self.db._execute_write(lambda conn: retire_media(conn, {'attachments_v1': {'media': media}}))
            collect_retired_media(self.db)
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                'Releasing refused attachments of request %s failed', request.request_id)

    async def receipt(self, actor, ref, admission_id):
        self.authorize(actor, ref, 'session:submit')
        row = get_session_admission(self.db, admission_id=admission_id)
        if row is None or row['target_session_id'] != ref.session_id:
            raise RuntimeStoreError('not_found')
        if row['principal_id'] != actor.subject and 'session:control' not in actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        live = self.sessions[ref.session_id]
        with live.event_stream.lock:
            # Read after the row: a terminal row whose completion is not yet published reads as
            # its claim, so a receipt poller never replays events that lack the completion frame.
            row = live.settling.get(admission_id, row)
        return self._receipt(row)

    async def cancel_queued(self, actor, ref, admission_id):
        before = await self.receipt(actor, ref, admission_id)
        from gateway.session_runtime_workers import track_mutation
        # Write off the shared loop; commit and observer settlement are one tracked task the
        # caller cannot cancel between (an exact retry still pays a left-over obligation).
        row = await asyncio.shield(track_mutation(self, self._commit_cancel(ref, admission_id, before)))
        # Media collection is its own write txn after the committed, settled cancellation: its
        # failure surfaces to the caller, and an exact retry collects again without re-settling.
        from gateway.session_ingress_media import release_admission_media
        await asyncio.to_thread(release_admission_media, self.db, admission_id)
        return self._receipt(row)

    async def _commit_cancel(self, ref, admission_id, before):
        if before.status == 'queued':
            # Owed from before the write: a commit that reports failure, or anything raising
            # between the commit and observer settlement, leaves it for an exact retry to pay.
            self.cancel_obligations.add(admission_id)
        try:
            row = await asyncio.to_thread(cancel_session_input, self.db, epoch=self.epoch, admission_id=admission_id)
        except RuntimeStoreError:
            self.cancel_obligations.discard(admission_id)  # a definite refusal: nothing committed
            raise
        if (row['status'], row['outcome']) == ('terminal', 'cancelled') and admission_id in self.cancel_obligations:
            self._settle_cancelled(ref, admission_id)
        else:
            if row['status'] != 'queued':
                self.cancel_obligations.discard(admission_id)
            self._publish_pending(ref)
        return row

    def _settle_cancelled(self, ref, admission_id):
        # The only place a queued row becomes terminal: every observer kind that waits on
        # the admission (native delivery, API/webhook/hosted waiters, ACP and viewer streams)
        # settles here, or a cancelled row that never reaches _drain blocks them forever.
        live = self.sessions[ref.session_id]
        with live.event_stream.lock:
            self._publish_pending(ref)
            running = live.event_stream.execution
            # A queued row owns no execution generation; stamp its own identity so the
            # completion is not attributed to the turn currently running ahead of it.
            live.event_stream.execution = {'authority_epoch': self.epoch, 'admission_id': admission_id}
            try:
                live.event_stream.publish(ref.session_id, {
                    'text': '', 'content': '', 'admission_id': admission_id, 'outcome': 'cancelled'})
            finally:
                live.event_stream.execution = running
        # Published once: a later exact retry re-collects media but never repeats the completion.
        self.cancel_obligations.discard(admission_id)
        self.native_waiters.discard(admission_id)
        waiter = self.waiters.pop(admission_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(None)
        # A paused drain (preclaim refusal on this head) ended its task; the successors
        # need a fresh drain that revalidates them on their own merits.
        self._schedule(ref)

    async def resolve_unknown(self, actor, ref, admission_id, generation):
        """Operator acknowledgement that an ``unknown`` turn will not finish; the paused FIFO behind
        it resumes. Never requeues the input. A turn that finished in THIS owner but whose
        settlement write failed (its exact result is still in ``pending_results``) commits that
        result instead of discarding it: the answer was produced, only its receipt was lost."""
        self.authorize(actor, ref, 'session:control')
        before = await self.receipt(actor, ref, admission_id)
        captured = self.pending_results.get(admission_id)
        prepared = None
        if captured is not None and before.status == 'unknown':
            from gateway.session_results import prepare_result
            row = get_session_admission(self.db, admission_id=admission_id)
            prepared = prepare_result(row, captured['result'].get('final_response') or '', 'completed', captured)
        # The transcript boundary commits WITH the terminal transition: a follower can never be
        # claimed with the discarded text left open to be merged into its request. With a captured
        # result the answer is normally already the transcript tail, so the closer is a no-op.
        from gateway.session_results import close_discarded_turn
        from gateway.session_runtime_workers import tracked_write
        return await tracked_write(self, partial(
            resolve_unknown_session_input, self.db, epoch=self.epoch, admission_id=admission_id,
            generation=generation, _captured=prepared,
            _terminal_write=lambda conn, lost: close_discarded_turn(self.db, conn, lost)),
            then=partial(self._resolved_unknown, ref, admission_id, prepared, captured))

    async def _resolved_unknown(self, ref, admission_id, prepared, captured, row):
        self.pending_results.pop(admission_id, None)
        if prepared is not None and row['owner_epoch'] == self.epoch:
            # The captured result committed: viewers told the turn is unknown get its completion,
            # unless the drain still owns the claim (resolution raced its recovery stamp) and
            # publishes the committed result itself.
            from gateway.session_settlement_recovery import _publish_completion
            live = self.sessions[ref.session_id]
            with live.event_stream.lock:
                if live.event_stream.execution.get('admission_id') != admission_id:
                    _publish_completion(self, live, ref, row, row,
                                        captured['result'].get('final_response') or '', captured)
        # Its own write txn + unlink, so it runs after the resolution commits; a discarded image
        # would otherwise stay on disk forever (the drain releases only settled turns).
        from gateway.session_ingress_media import release_admission_media
        await asyncio.to_thread(release_admission_media, self.db, admission_id)
        # Before the successor can run: a committed captured answer reaches the chat it was
        # owed to (a turn fenced unknown already released its delivery waiter), exactly once.
        from gateway.session_ingress import deliver_resolved
        await deliver_resolved(self, admission_id, owed=prepared is not None and row['owner_epoch'] == self.epoch)
        self._publish_pending(ref)
        self._schedule(ref)
        return self._receipt(row)

    async def interrupt(self, actor, ref, generation):
        self.authorize(actor, ref, 'session:control')
        handle = self._handle(ref)
        if handle.execution_generation != generation:
            raise RuntimeStoreError('stale_generation')
        if handle.execution_state == 'running':
            adopted_generation, agent = self.adopted.get(ref.session_id, (None, None))
            if adopted_generation != generation:
                agent = self.agent(ref)
            if agent is not None:
                agent.interrupt()
                self.delivered_stops[ref.session_id] = generation
            if adopted_generation != generation and not self._runs_own_agent(ref):
                # Accepted for this exact claim before its turn adopted an agent: whatever agent
                # the turn adopts for this generation must not run the work as if no Stop arrived.
                self.pending_stops[ref.session_id] = generation
        return self._handle(ref)

    def _runs_own_agent(self, ref):
        """The route's running slot holds a real agent (not the claim's pending sentinel), so the
        Stop above reached the turn's own agent, not a cached one it may still replace."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        running = getattr(self.runner, '_running_agents', None) or {}
        agent = running.get(self.sessions[ref.session_id].route)
        return agent is not None and agent is not _AGENT_PENDING_SENTINEL

    def adopt_agent(self, session_id, generation, agent):
        """The turn installs its agent for the running claim; a Stop latched while there
        was no agent to deliver it to fires now, never against a later generation. A Stop
        delivered to the cached agent for an earlier generation may have landed after that
        turn's finalizer cleared it, so it is dropped rather than cancelling this turn. A managed
        worker is a fresh process per turn and carries no earlier flag to drop."""
        self.adopted[session_id] = (generation, agent)
        clear = getattr(agent, 'clear_interrupt', None)
        if self.delivered_stops.pop(session_id, generation) != generation and clear is not None:
            clear()
        if self.pending_stops.get(session_id) == generation:
            del self.pending_stops[session_id]
            agent.interrupt()

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
            delivered = live.event_stream.publish(session_id, payload, event_type=event_type)
            from gateway.session_api_turn import publish_api_event
            observed = publish_api_event(self, session_id, event_type, payload)
            return bool(delivered or observed)

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

    async def _claim_next(self, ref, live):
        # False asks the pump to re-read after a queued head changed during preflight.
        async with live.claim_gate, live.mutation_lock:
            pending = list_session_admissions(self.db, session_id=ref.session_id)
            if any(row['status'] == 'unknown' for row in pending):
                raise RuntimeStoreError('unknown_execution')
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
                    # Cancellation may advance the FIFO while the source owner is awaited;
                    # the successor must earn its own reauthorization, not inherit this one.
                    current = get_session_admission(self.db, admission_id=first['admission_id'])
                    if current is None or current['status'] != 'queued':
                        return False, first
                if 'local_automation_v1' in first['payload']:
                    from gateway.session_automation import check_local_automation
                    check_local_automation(self, ref, first)
                else:
                    from gateway.session_operator import check_local_input
                    check_local_input(self, ref, first)
                # The first real turn reopens a finalized row (#85303): mounts are reads, and
                # SessionStore would route a stamped row as stale onto a FRESH session. A write: off-loop.
                from gateway.session_local_recovery import reopen_local_session
                await asyncio.to_thread(reopen_local_session, self, ref)
            if first is not None and 'native_text_v1' in first['payload']:
                from gateway.session_envelope import check_native_route
                await check_native_route(self.runner, first['payload'], self.physical_target(ref), live.source,
                                   self.runner._adapter_for_source(live.source))
                # Cancellation may advance FIFO while the connector is awaited.
                # Never let the successor inherit this row's fresh verdict.
                current = get_session_admission(self.db, admission_id=first['admission_id'])
                if current is None or current['status'] != 'queued':
                    return False, first
            if first is not None and live.source.platform == Platform.API_SERVER:
                from gateway.session_api_turn import check_api_turn
                check_api_turn(self, ref, first['payload'])
            self._require_admission_open()
            from gateway.session_runtime_workers import tracked_write
            live.rescan = False  # a _schedule from here on may not be visible to this claim's read
            # The row these checks validated, never whichever is first when the write runs: a
            # cancellation commits from a worker thread and can advance the FIFO at any point.
            row = await tracked_write(self, partial(claim_session_input, self.db, epoch=self.epoch,
                                                    session_id=ref.session_id, _guard=self._admission_gate(),
                                                    head=first['admission_id'] if first is not None else None),
                                      then=partial(self._stamp_claim, ref, live))
            # Input scheduled while this claim's transaction ran may postdate its read: re-read
            # instead of letting the drain exit idle over a committed admission.
            return (False if row is None and live.rescan else row), first

    def _stamp_claim(self, ref, live, row):
        """The claim's execution stamp, in the same tracked task as its commit: no observer (or
        retirement's join) ever sees a committed claim that no execution stamp names."""
        if row:  # None or False (the head changed under the write) claimed nothing
            with live.event_stream.lock:
                live.event_stream.execution = {
                    'authority_epoch': self.epoch, 'execution_generation': row['generation'],
                    'admission_id': row['admission_id']}
                live.event_stream.publish(ref.session_id, {}, event_type='message.start')
                self._publish_pending(ref)
        return row

    async def _drain(self, ref):
        from gateway.session_finite import execute_finite_admission
        from gateway.session_managed_worker import ManagedExecutionUnknown
        live = self.sessions[ref.session_id]
        backoff = 0.0
        while True:
            try:
                row, first = await self._claim_next(ref, live)
            except RuntimeStoreError as exc:
                import logging
                logging.getLogger(__name__).warning('Session %s paused: %s', ref.session_id, exc.reason)
                self._pause(ref, exc.reason)
                return
            except (OSError, sqlite3.OperationalError) as exc:
                # A writer held past its patience or a full disk is transient, and every _schedule
                # caller is event-driven: a dead drain would strand the committed head and hang its
                # waiters until the next submit. Retry in place (like recover_failed_settlement),
                # keeping waiters: releasing them fails requests that would run a moment later.
                if self.runner._draining or self.retiring or self.sessions.get(ref.session_id) is not live:
                    self._pause(ref, 'runtime_draining')
                    return
                backoff = min(max(backoff * 2, _CLAIM_RETRY_MIN_S), _CLAIM_RETRY_MAX_S)
                import logging
                logging.getLogger(__name__).warning(
                    'Session %s claim deferred by storage (%r); retrying in %.1fs', ref.session_id, exc, backoff)
                await asyncio.sleep(backoff)
                continue
            backoff = 0.0
            if row is False:
                continue
            if row is None and first is not None:
                # A live worker (or a claim this drain does not own) blocks the head: the same
                # pause episode, so its notice is not repeated for every message queued behind it.
                self._pause(ref, 'session_busy')
                return
            # The FIFO is moving again (or empty): the next pause is a new episode.
            live.pause_notified = False
            if row is None:
                from gateway.session_acp_lifecycle import end_idle_acp_session
                end_idle_acp_session(self, ref.session_id)
                return
            admission_id = row['admission_id']
            try:
                response = await execute_finite_admission(self, ref, row)
                outcome = 'completed'
            except ManagedExecutionUnknown:
                with live.event_stream.lock:
                    live.event_stream.execution = {}
                self.pending_stops.pop(ref.session_id, None)
                self.adopted.pop(ref.session_id, None)
                self._pause(ref, 'unknown_execution')
                return
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Admitted turn %s failed', admission_id)
                response = 'The admitted turn failed.'
                outcome = 'failed'
            settled = None
            settlement = {}
            try:
                # Redaction, encoding and the SQLite write run off-loop: one turn's settlement must
                # not stall every other session. Commit and completion share one stream-lock hold
                # there; a tracked writer outlives a cancelled drain like the recovery stamp does.
                from gateway.session_settlement_recovery import settle_with_retry
                captured = self.pending_results.get(admission_id)
                try:
                    await settle_with_retry(self, live, ref, row, response, outcome, captured, settlement)
                finally:
                    if 'settled' in settlement:
                        settled, response = settlement['settled'], settlement['response']
                        self.pending_results.pop(admission_id, None)
                # Idle follows the completion; its bot receipt wakeups are loop tasks. A delete
                # that committed while the write yielded already retired this live entry.
                if self.sessions.get(ref.session_id) is live:
                    self._publish_pending(ref)
            except Exception:
                import logging
                logging.getLogger(__name__).exception(
                    'Settlement or publication of admission %s failed', admission_id)
                if settled is None:
                    self.pending_results.setdefault(admission_id, {
                        'result': {'final_response': response, 'failed': outcome == 'failed'}, 'usage': {}})
                    from gateway.session_settlement_recovery import recover_failed_settlement
                    try:
                        status = await recover_failed_settlement(self, row)
                    except RuntimeStoreError:
                        # Ownership moved; this owner may only release its process-local observers.
                        self._pause(ref, 'unknown_execution')
                        return
                    # Discard can race the off-loop recovery stamp; its terminal decision wins.
                    current = get_session_admission(self.db, admission_id=admission_id)
                    status = current['status']
                    if status == 'unknown':
                        from gateway.session_ingress import retain_unsettled_delivery
                        retain_unsettled_delivery(self, admission_id)
                        self._publish_pending(ref)
                        self._pause(ref, 'unknown_execution')
                        return
                    # A write can commit before reporting failure. Re-read its exact result,
                    # then publish the terminal outcome rather than inventing uncertainty.
                    from gateway.session_results import admission_result
                    from gateway.session_settlement_recovery import publish_terminal
                    with live.event_stream.lock:
                        captured = admission_result(self.db, admission_id)
                        self.pending_results.pop(admission_id, None)
                        if captured is None:
                            # An operator discarded uncertainty while recovery yielded. There is
                            # no committed answer to replay; its queued successor may now run.
                            waiter = self.waiters.pop(admission_id, None)
                            if waiter is not None and not waiter.done():
                                waiter.set_exception(RuntimeStoreError('unknown_execution'))
                            continue
                        settled = current
                        response = captured['result'].get('final_response') or ''
                        publish_terminal(self, ref, row, settled, response, captured)
            finally:
                # The stamp names a claimed, unsettled execution. Left in place, idle
                # mutations (session.updated) would carry a terminal generation and
                # a versioned viewer fence would discard them as late frames.
                with live.event_stream.lock:
                    live.event_stream.execution = {}
                self.pending_stops.pop(ref.session_id, None)
                self.adopted.pop(ref.session_id, None)
                if settled is None:
                    # Uncommitted (unknown or discarded): the user is not told an answer the FIFO lost.
                    self.pending_deliveries.pop(admission_id, None)
                from gateway.session_ingress import waiter_replies
                waiter_replies(self).pop(admission_id, None)
            from gateway.session_ingress import deliver_settled
            await deliver_settled(self, admission_id)
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
    # Resolved off-loop before any state is published: the rest of the bring-up runs without
    # yielding, so no other task observes a half-registered authority.
    from pathlib import Path
    db_key = await asyncio.to_thread(Path(db.db_path).resolve)
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
    epochs[db_key] = epoch
    from gateway.session_local_recovery import recover_local_sessions
    recover_local_sessions(authority)
    return authority
