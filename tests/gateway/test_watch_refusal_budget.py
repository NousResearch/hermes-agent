"""A proven-permanent watch refusal spends ONE producer-stable budget, not one per dict copy.

Two liveness defects in the notification drain, and the safety controls that must survive their fix:

* an owned notice whose payload digest conflicts with the immutable admission already stored at its
  exact ``(principal, target, request)`` identity was refused ``admission_conflict`` and requeued
  with no terminal bound while that admission/route stayed stable;
* the permanent count lived on the event dictionary, so a re-emitted copy of one logical producer
  identity started again at attempt 1.

Both are now bounded by the store's own closed proof (``AdmissionConflict.IMMUTABLE_PAYLOAD``) and
by ``run_notifications._watch_refusal_key``'s shared budget. These tests are the regressions for
those two invariants plus the controls that must not move: transient/unproven refusals, unowned
notices, foreign live pins, ``queue_full``, a missing pin, route mismatch, an unavailable store and
FIFO/restart refusals — and the durable async lane's spend/refund and lost-ACK shortcut.

The consumer path under test is the real one: the real ``admit_session_input`` store over a real
``SessionDB``, the real ``snapshot_automation``/``admit_process_notice`` ownership checks, and the
real drain/requeue. Boundary fakes only where a live gateway runtime is required: connector
envelope restore, transport resolution, the authority's worker-tracking write wrapper, the event's
source resolver, and (async lane) the durable claim preflight. Nothing here sleeps on the clock:
backoff is exercised as arithmetic, and the drain's pacing is observed by its own dict fields.
"""
import asyncio
import contextlib
import json
import queue
import sqlite3
import sys
import threading
import types
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from gateway import run_notifications as rn
from gateway import session_automation as sa
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.wake import WakeNotAccepted
from hermes_state import SessionDB
from hermes_state_keys import admission_fingerprint
from hermes_state_runtime import (
    RuntimeStoreError, admit_session_input, begin_runtime_epoch, list_session_admissions,
)

M = rn.GatewayNotificationsMixin
CAP = rn._MAX_COMPLETION_INJECTION_ATTEMPTS
ROUTE = 'agent:fixture:telegram:dm:fixture-chat'


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Every test gets its own HERMES_HOME: nothing here reads or writes the live profile."""
    home = tmp_path_factory.mktemp('hermes-home')
    monkeypatch.setenv('HERMES_HOME', str(home))
    yield home


@pytest.fixture(autouse=True)
def _gateway_run_seams(monkeypatch):
    """The two ``gateway.run`` seams this path imports, faked at the module boundary.

    ``gateway.run`` cannot be imported here: its bootstrap reads the install root's
    ``manifest.json``, which a source checkout under test does not own. The queue filter is
    reproduced verbatim (it is the seam under test's caller) and the formatter is a fake because
    real formatting needs the process registry's live output buffers.
    """
    module = sys.modules.get('gateway.run')
    if module is None:
        module = types.ModuleType('gateway.run')
        monkeypatch.setitem(sys.modules, 'gateway.run', module)

    def _drain_gateway_watch_events(completion_queue):
        watch_events, requeue = [], []
        while not completion_queue.empty():
            try:
                evt = completion_queue.get_nowait()
            except Exception:
                break
            evt_type = evt.get("type", "completion")
            if evt_type in {"watch_match", "watch_disabled", "watch_overflow_tripped",
                            "watch_overflow_released", "heartbeat"}:
                watch_events.append(evt)
            elif evt_type == "async_delegation":
                requeue.append(evt)
        for evt in requeue:
            completion_queue.put(evt)
        return watch_events

    monkeypatch.setattr(module, '_USER_BOUNDARY_END_REASONS',
                        ('session_reset', 'user_exit', 'session_switch', 'new_session'), raising=False)
    monkeypatch.setattr(module, '_format_gateway_process_notification',
                        lambda evt: evt.get('_fixture_text', 'changed'), raising=False)
    monkeypatch.setattr(module, '_drain_gateway_watch_events',
                        _drain_gateway_watch_events, raising=False)


@pytest.fixture(autouse=True)
def _clean_ledger():
    """Each test starts from an empty shared budget, and cannot leak one into the next."""
    lock = getattr(rn, '_WATCH_REFUSAL_LOCK', None)
    ledger = getattr(rn, '_WATCH_REFUSAL_LEDGER', None)
    if lock is None or ledger is None:
        yield  # no shared budget in this build: the per-test outcome is the point
        return
    with lock:
        ledger.clear()
    yield
    with lock:
        ledger.clear()


class _AsyncDB:
    """The real profile store behind the async dialect the ownership checks await."""

    def __init__(self, db):
        self._db = db

    async def get_session(self, session_id):
        return self._db.get_session(session_id)

    async def get_compression_tip(self, session_id):
        return self._db.get_compression_tip(session_id)


class _Runner(M):
    """GatewayNotificationsMixin bound to a real profile store and an in-process route."""

    def __init__(self, tmp_path):
        self.db = SessionDB(tmp_path / 'state.db')
        self.epoch = begin_runtime_epoch(self.db, instance_id='fixture')
        for session_id in ('owner', 'foreign'):
            self.db.create_session(session_id, source='telegram')
        self.source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture-chat', chat_type='dm')
        # What ``restore_native`` hands back for the committed human envelope: normally the same
        # route, and a different one to stage a route mismatch.
        self.native_source = self.source
        self.entry = SimpleNamespace(session_id='owner', session_key=ROUTE, suspended=False,
                                     platform=Platform.TELEGRAM, origin=self.source)
        self.session_store = SimpleNamespace(
            _entries={ROUTE: self.entry}, _ensure_loaded=lambda: None,
            lookup_by_session_key=lambda key: self.entry if key == ROUTE else None,
            _generate_session_key=lambda source: ROUTE,
            peek_session_id=lambda key: self.entry.session_id)
        self._session_db = _AsyncDB(self.db)
        self.adapter = SimpleNamespace(supports_async_delivery=True)
        self.authority = SimpleNamespace(db=self.db, runner=self, logical_owner=lambda sid: sid,
                                         admit_automation=self._admit_automation)
        self.session_authority = self.authority
        self._completion_delivery_lock = threading.Lock()
        self._completion_deliveries_inflight = set()
        self._completion_deliveries_delivered = OrderedDict()
        self._completion_delivery_retention = 100
        self._build_process_event_source = lambda evt: self.source
        self._resolve_injection_adapter = lambda *args, **kwargs: self.adapter
        self._adapter_for_source = lambda source: self.adapter
        self._settled = []
        self.admissions = 0
        self.forced_reason = None
        self.forced_exc = None
        self._commit_human_envelope('owner', 'human-owner')

    # -- the authority write the real ``admit_automation`` performs, minus worker tracking --------
    async def _admit_automation(self, adapter, event, identity, **kwargs):
        if self.forced_exc is not None:
            raise self.forced_exc
        if self.forced_reason:
            raise RuntimeStoreError(self.forced_reason)
        payload, entry = sa.snapshot_automation(self.authority, adapter, event, identity)
        row = admit_session_input(self.db, epoch=self.epoch, principal_id='automation:' + entry.session_key,
                                  session_id=entry.session_id, request_id=identity, payload=payload)
        self.admissions += 1
        return row

    def _commit_human_envelope(self, session_id, request_id):
        """A committed human route message: the provenance a producer turn borrows."""
        admit_session_input(self.db, epoch=self.epoch, principal_id='human', session_id=session_id,
                            request_id=request_id,
                            payload={'text': 'fixture human', 'native_text_v1': {
                                'source': {'platform': 'telegram', 'chat_id': 'fixture-chat',
                                           'chat_type': 'dm', 'profile': None},
                                'provenance': {'fixture': True}}})

    # -- the drain's runtime seams ---------------------------------------------------------------
    async def _preflight_completion_delivery(self, evt):
        return M._CompletionClaim(delegation_id='delegation', claim_id='fixture-claim')

    def _settle_durable_claim(self, kind, delegation_id, claim_id):
        self._settled.append(kind)

    async def _closed_local_target(self, *args):
        return ''

    def _load_background_notifications_mode(self):
        return 'on'

    def _completion_event_scope(self, evt):
        return contextlib.nullcontext()

    # -- events ----------------------------------------------------------------------------------
    def event(self, *, pinned='owner', kind='watch_match', identity='same-event'):
        return {'type': kind, 'session_id': 'process-fixture', 'event_id': identity, 'pattern': 'READY',
                'session_key': ROUTE, 'parent_session_id': pinned}

    def seed(self, evt, text='original'):
        assert asyncio.run(self._inject_watch_notification(text, evt)) is True

    def offer(self, evt, text='changed'):
        """One drain pass: admit, then requeue exactly as the drain does on refusal."""
        q = queue.Queue()
        result = asyncio.run(self._inject_watch_notification(text, evt))
        # The requeue consumes the verdict, so record it first.
        self.last_permanent = evt.get('_refusal_permanent')
        if result is False:
            self._requeue_refused_watch_event(q, evt)
        return result, q

    def drain(self, evt):
        """The real drain over one queued event."""
        q = queue.Queue()
        q.put(evt)
        asyncio.run(self._drain_watch_notifications(q))
        return q


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """A runner whose envelope restore, transport routing and authority lookup are boundary fakes."""
    def restore_native(payload, runner_):
        return SimpleNamespace(source=runner_.native_source)

    monkeypatch.setattr('gateway.session_envelope.restore_native', restore_native)
    monkeypatch.setattr('gateway.session_authorities.active_authority', lambda runner_: runner_.authority)
    monkeypatch.setattr('tools.process_registry_notifications.should_surface_notification', lambda evt: True)
    r = _Runner(tmp_path)
    yield r
    r.db.close()


# ---------------------------------------------------------------------------------------------
# The two invariants
# ---------------------------------------------------------------------------------------------
def test_owned_immutable_payload_conflict_reaches_a_terminal_state(runner):
    """RED before: a changed payload at an owned identity was requeued with no permanent attempt."""
    from hermes_state_runtime import AdmissionConflict

    runner.seed(runner.event())
    evt = runner.event()
    for _ in range(3 * CAP):
        result, q = runner.offer(evt, 'changed')
        assert result is False
        assert runner.last_permanent is True
        if q.empty():
            break
        assert q.get_nowait() is evt
    assert q.empty(), 'an owned immutable payload conflict must stop being requeued'
    assert evt['_watch_injection_attempts'] == CAP
    # The changed payload was never admitted: the immutable original still stands, once.
    rows = list_session_admissions(runner.db, session_id='owner', pending_only=False)
    automation = [r for r in rows if r['principal_id'].startswith('automation:')]
    assert len(automation) == 1
    stored = automation[0]['payload']['text']
    assert stored.startswith('original') and 'changed' not in stored
    assert runner.admissions == 1
    assert AdmissionConflict.IMMUTABLE_PAYLOAD.value == 'immutable_payload'


def test_owned_immutable_payload_conflict_is_bounded_across_fresh_copies(runner):
    """The same conflict re-emitted as fresh dictionaries is one candidate, so one budget."""
    runner.seed(runner.event())
    for copy in range(3 * CAP):
        evt = runner.event()
        result, q = runner.offer(evt, 'changed')
        assert result is False
        assert evt['_watch_injection_attempts'] == min(copy + 1, CAP)
        if q.empty():
            break
        assert q.get_nowait() is evt
    assert q.empty(), 'a re-emitted immutable conflict must share its predecessor budget'
    assert copy == CAP - 1


def test_reemitted_proven_permanent_identity_shares_the_cap(runner):
    """RED before: every fresh dictionary with one producer identity got attempt 1, forever."""
    runner.forced_reason = 'invalid_params'
    for copy in range(3 * CAP):
        evt = runner.event()
        _, q = runner.offer(evt, 'changed')
        assert evt['_watch_injection_attempts'] == min(copy + 1, CAP)
        if q.empty():
            break
        assert q.get_nowait() is evt
    assert q.empty(), 'a proven-permanent identity must not reset its cap on a fresh dictionary'
    assert copy == CAP - 1


# ---------------------------------------------------------------------------------------------
# The budget's key: corrected candidate, later route, other store namespace
# ---------------------------------------------------------------------------------------------
def test_a_corrected_candidate_starts_with_its_own_budget(runner):
    runner.forced_reason = 'invalid_params'
    for _ in range(CAP):
        _, q = runner.offer(runner.event(), 'first candidate')
    assert q.empty()
    evt = runner.event()
    _, q = runner.offer(evt, 'corrected candidate')
    assert not q.empty()
    assert evt['_watch_injection_attempts'] == 1


def test_a_rightful_later_route_does_not_inherit_an_exhausted_budget(runner):
    runner.forced_reason = 'invalid_params'
    for _ in range(CAP):
        _, q = runner.offer(runner.event(pinned='owner'), 'changed')
    assert q.empty()
    evt = runner.event(pinned='foreign')
    _, q = runner.offer(evt, 'changed')
    assert not q.empty()
    assert evt['_watch_injection_attempts'] == 1


def test_the_store_namespace_separates_two_profiles(runner):
    """Identical ids admitted under different profile homes never share one budget."""
    for _ in range(CAP - 1):
        evt = runner.event()
        evt['_refusal_permanent'] = True
        evt['_watch_refusal_candidate'] = rn._watch_candidate_digest('text')
        q = queue.Queue()
        runner._requeue_refused_watch_event(q, evt, scope='profiles/a')
    assert evt['_watch_injection_attempts'] == CAP - 1
    other = runner.event()
    other['_refusal_permanent'] = True
    other['_watch_refusal_candidate'] = rn._watch_candidate_digest('text')
    q = queue.Queue()
    runner._requeue_refused_watch_event(q, other, scope='profiles/b')
    assert other['_watch_injection_attempts'] == 1


def test_the_budget_cache_is_bounded_and_never_cleared_wholesale(runner):
    for n in range(rn._MAX_WATCH_REFUSAL_TRACKED + 10):
        rn._spend_watch_refusal(('', ('route',), '["identity"]', f'candidate-{n}'), n + 1)
    assert len(rn._WATCH_REFUSAL_LEDGER) == rn._MAX_WATCH_REFUSAL_TRACKED
    assert ('', ('route',), '["identity"]', 'candidate-0') not in rn._WATCH_REFUSAL_LEDGER
    assert ('', ('route',), '["identity"]',
            f'candidate-{rn._MAX_WATCH_REFUSAL_TRACKED + 9}') in rn._WATCH_REFUSAL_LEDGER
    # A surviving identity still carries its count, and the shared pacing is not reset.
    attempts, next_at = rn._spend_watch_refusal(('', ('route',), '["identity"]', 'candidate-200'), 1)
    assert attempts == 2 and next_at > 0


def test_a_notice_without_a_producer_identity_keeps_the_event_local_count(runner):
    """Behaviour for events nothing producer-stable can be shared with."""
    evt = {'type': 'process_watch', 'event_id': 'fixture'}
    q = queue.Queue()
    for attempt in range(1, CAP + 1):
        evt['_refusal_permanent'] = True
        runner._requeue_refused_watch_event(q, evt)
        assert evt['_watch_injection_attempts'] == attempt
        if attempt < CAP:
            assert q.get_nowait() is evt
    assert q.empty()


# ---------------------------------------------------------------------------------------------
# Controls that must not move: unproven and transient refusals stay deliverable
# ---------------------------------------------------------------------------------------------
def test_transient_refusals_survive_beyond_the_cap_then_deliver(runner):
    runner.forced_reason = 'queue_full'
    evt = runner.event()
    for _ in range(3 * CAP):
        _, q = runner.offer(evt, 'changed')
        assert q.get_nowait() is evt
    assert evt.get('_watch_injection_attempts', 0) == 0
    runner.forced_reason = None
    assert asyncio.run(runner._inject_watch_notification('changed', evt)) is True


def test_a_real_full_fifo_is_retryable_then_delivers(runner):
    """``queue_full`` from the real store, cleared the way a draining turn clears it."""
    from hermes_state_runtime import ADMISSION_QUEUE_MAX
    queued = runner.db._read_ctx
    with queued() as conn:
        already = conn.execute("SELECT COUNT(*) FROM session_admissions WHERE target_session_id='owner' "
                               "AND status='queued'").fetchone()[0]
    for n in range(ADMISSION_QUEUE_MAX - already):
        admit_session_input(runner.db, epoch=runner.epoch, principal_id='human', session_id='owner',
                            request_id=f'fill-{n}', payload={'text': f'fill-{n}'})
    evt = runner.event()
    result, q = runner.offer(evt, 'changed')
    assert result is False
    assert runner.last_permanent is False
    assert q.get_nowait() is evt
    runner.db._execute_write(lambda conn: conn.execute(
        "UPDATE session_admissions SET status='terminal' WHERE request_id LIKE 'fill-%'"))
    assert asyncio.run(runner._inject_watch_notification('changed', evt)) is True


def test_an_unowned_notice_is_not_discarded_and_delivers_when_pinned(runner):
    evt = runner.event(pinned='')
    for _ in range(3 * CAP):
        _, q = runner.offer(evt, 'changed')
        assert q.get_nowait() is evt
    assert evt.get('_watch_injection_attempts', 0) == 0
    evt['parent_session_id'] = 'owner'
    assert asyncio.run(runner._inject_watch_notification('changed', evt)) is True


def test_a_foreign_live_pin_is_not_discarded_and_delivers_on_its_own_route(runner):
    assert asyncio.run(runner._classify_completion_owner('foreign', ROUTE))[0] == 'deliver'
    evt = runner.event(pinned='foreign')
    for _ in range(3 * CAP):
        _, q = runner.offer(evt, 'changed')
        assert q.get_nowait() is evt
    assert evt.get('_watch_injection_attempts', 0) == 0
    runner.entry.session_id = 'foreign'
    runner._commit_human_envelope('foreign', 'human-foreign')
    assert asyncio.run(runner._inject_watch_notification('changed', evt)) is True


def test_a_missing_pin_is_terminal(runner):
    evt = runner.event(pinned='deleted')
    result, q = runner.offer(evt, 'changed')
    assert result is None
    assert q.empty()


def test_a_route_mismatch_refusal_stays_unproven_and_retryable(runner):
    """The store's OTHER ``admission_conflict`` (route/owner identity, not a payload digest)."""
    evt = runner.event()
    runner.native_source = SessionSource(platform=Platform.TELEGRAM, chat_id='other-chat', chat_type='dm')
    result, q = runner.offer(evt, 'changed')
    assert result is False
    assert runner.last_permanent is False
    assert q.get_nowait() is evt
    assert evt.get('_watch_injection_attempts', 0) == 0
    runner.native_source = runner.source
    assert asyncio.run(runner._inject_watch_notification('changed', evt)) is True


def test_a_restart_draining_refusal_is_retryable(runner):
    runner.forced_reason = 'runtime_draining'
    evt = runner.event()
    for _ in range(3 * CAP):
        _, q = runner.offer(evt, 'changed')
        assert q.get_nowait() is evt
    assert evt.get('_watch_injection_attempts', 0) == 0


def test_an_unavailable_store_is_retryable_and_delivers_when_it_returns(runner):
    runner.forced_exc = sqlite3.OperationalError('database is locked')
    evt = runner.event()
    for _ in range(3 * CAP):
        result, q = runner.offer(evt, 'changed')
        assert result is False
        assert q.get_nowait() is evt
    assert evt.get('_watch_injection_attempts', 0) == 0
    runner.forced_exc = None
    assert asyncio.run(runner._inject_watch_notification('changed', evt)) is True


def test_requeue_pacing_is_shared_and_capped(runner):
    """Copying a dictionary cannot reset the pacing either: the delay follows the shared count."""
    runner.forced_reason = 'invalid_params'
    delays = []
    for _ in range(CAP):
        before = rn.time.monotonic()
        evt = runner.event()
        runner.offer(evt, 'changed')
        delays.append(round(evt['_watch_next_retry_at'] - before, 3))
    base, ceiling = rn._WATCH_INJECTION_RETRY_BASE_DELAY, rn._COMPLETION_INJECTION_RETRY_MAX_DELAY
    shared = [min(base * 2 ** step, ceiling) for step in range(CAP)]
    # Tolerance only for the few milliseconds between reading the clock and spending the budget.
    assert all(abs(got - want) <= 0.2 for got, want in zip(delays, shared)), (delays, shared)
    assert delays[-1] <= ceiling + 1


# ---------------------------------------------------------------------------------------------
# Durable async lane: unchanged spending/refund and the lost-ACK shortcut
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('permanent, expected', [(True, 'release'), (False, 'defer')])
def test_completion_permanent_refusal_spends_attempt_transient_refunds(runner, permanent, expected):
    runner._completion_delivery_identity = lambda evt: None
    async def inject(*args, **kwargs):
        exc = WakeNotAccepted('fixture refusal')
        exc.permanent = permanent
        raise exc
    runner._inject_watch_notification = inject
    assert asyncio.run(runner._deliver_completion_notification_scoped('fixture', {})) is False
    assert runner._settled == [expected]


def test_the_async_lane_still_acknowledges_a_lost_ack_before_reinjection(runner):
    evt = runner.event(kind='async_delegation')
    evt['delegation_id'] = 'delegation'
    runner.seed(evt, 'original')
    # A CHANGED text would conflict at the store; the shortcut acknowledges the committed identity.
    assert asyncio.run(runner._deliver_completion_notification_scoped('changed', dict(evt))) is True
    assert runner._settled == ['complete']


def test_the_typed_conflict_is_permanent_in_the_async_lane_without_touching_the_event(runner):
    from hermes_state_runtime import AdmissionConflict

    runner.seed(runner.event(kind='async_delegation') | {'delegation_id': 'delegation'}, 'original')
    evt = runner.event(kind='async_delegation') | {'delegation_id': 'delegation'}
    with pytest.raises(WakeNotAccepted) as caught:
        asyncio.run(runner._inject_watch_notification('changed', evt, raise_not_accepted=True))
    assert caught.value.reason == 'admission_conflict'
    assert caught.value.conflict is AdmissionConflict.IMMUTABLE_PAYLOAD
    # Proven permanent, so the durable caller spends its own attempt (release), never a refund.
    assert caught.value.permanent is True
    assert '_refusal_permanent' not in evt and '_watch_refusal_candidate' not in evt


def test_async_group_refusals_back_off(runner, monkeypatch):
    monkeypatch.setattr(rn.time, 'monotonic', lambda: 100.0)
    group = [{'delegation_id': 'a'}, {'delegation_id': 'b'}]
    M._back_off_refused_completions(group)
    assert all(evt['_watch_retry_step'] == 1 and 100 < evt['_watch_next_retry_at'] <= 160
               for evt in group)


# ---------------------------------------------------------------------------------------------
# The discriminator itself: a closed enum, never text
# ---------------------------------------------------------------------------------------------
def test_the_store_tags_only_the_immutable_digest_mismatch(runner):
    from hermes_state_runtime import AdmissionConflict

    payload = {'text': 'original'}
    admit_session_input(runner.db, epoch=runner.epoch, principal_id='human', session_id='owner',
                        request_id='same', payload=payload)
    # An exact replay is still a replay, not a refusal.
    assert admit_session_input(runner.db, epoch=runner.epoch, principal_id='human', session_id='owner',
                               request_id='same', payload=payload)['status'] == 'queued'
    with pytest.raises(RuntimeStoreError) as caught:
        admit_session_input(runner.db, epoch=runner.epoch, principal_id='human', session_id='owner',
                            request_id='same', payload={'text': 'changed'})
    assert caught.value.reason == 'admission_conflict'          # public wire code unchanged
    assert caught.value.conflict is AdmissionConflict.IMMUTABLE_PAYLOAD
    assert caught.value.field == 'payload_digest'
    # Every other store refusal keeps the bare code: nothing to infer permanence from.
    with pytest.raises(RuntimeStoreError) as stale:
        admit_session_input(runner.db, epoch=runner.epoch + 1, principal_id='human', session_id='owner',
                            request_id='fresh', payload=payload)
    assert stale.value.reason == 'stale_epoch' and stale.value.conflict is None


def test_a_string_that_spells_the_enum_is_not_proof():
    assert RuntimeStoreError('admission_conflict', conflict='immutable_payload').conflict is None
    assert RuntimeStoreError('admission_conflict').conflict is None


def test_an_uncanonicalizable_identity_falls_back_instead_of_losing_the_event(runner):
    """A key that cannot be built must cost the budget, never the requeue."""
    evt = {'type': 'completion', 'session_id': 'producer', 'started_at': object()}
    evt['_watch_refusal_candidate'] = rn._watch_candidate_digest('text')
    assert rn._watch_refusal_key(evt, '') is None
    evt['_refusal_permanent'] = True
    q = queue.Queue()
    runner._requeue_refused_watch_event(q, evt)
    assert q.get_nowait() is evt  # requeued on the event-local count
    assert evt['_watch_injection_attempts'] == 1


def test_the_retired_payload_mismatch_carries_the_same_proof(runner):
    """A retired admission's frozen payload refuses a differing digest with the same closed proof."""
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    from hermes_state_terminal import ADMISSION_PREFIX, identity_key
    from hermes_state_runtime import AdmissionConflict

    payload = {'text': 'retired'}
    digest = admission_fingerprint(canonical_target='deleted-session',
                                   payload={'input': payload, 'intent': 'queue'})
    retired_row = {'admission_id': 'adm-retired', 'request_id': 'r', 'principal_id': 'human',
                   'target_session_id': 'deleted-session', 'lineage_json': '["deleted-session"]',
                   'payload_json': json.dumps(payload), 'payload_digest': digest,
                   'intent': 'queue', 'status': 'terminal', 'owner_epoch': runner.epoch}
    runner.db._execute_write(lambda conn: [
        conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)',
                     (RETIRED_PREFIX + 'deleted-session', '{}')),
        conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)',
                     (identity_key('human', 'deleted-session', 'r'), json.dumps('adm-retired'))),
        conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)',
                     (ADMISSION_PREFIX + 'adm-retired', json.dumps(retired_row))),
    ])
    from hermes_state_terminal import retry_terminal_admission
    assert retry_terminal_admission(runner.db, epoch=runner.epoch, principal_id='human',
                                    session_id='deleted-session', request_id='r',
                                    payload=payload)['admission_id'] == 'adm-retired'
    with pytest.raises(RuntimeStoreError) as caught:
        retry_terminal_admission(runner.db, epoch=runner.epoch, principal_id='human',
                                 session_id='deleted-session', request_id='r',
                                 payload={'text': 'changed'})
    assert caught.value.reason == 'admission_conflict'
    assert caught.value.conflict is AdmissionConflict.IMMUTABLE_PAYLOAD


# ---------------------------------------------------------------------------------------------
# The real drain, not just the seams
# ---------------------------------------------------------------------------------------------
def test_the_real_drain_settles_an_owned_immutable_conflict(runner, monkeypatch):
    runner.seed(runner.event(), 'original')
    evt = runner.event()
    evt['_fixture_text'] = 'changed'
    clock = [1000.0]
    monkeypatch.setattr(rn.time, 'monotonic', lambda: clock[0])  # pacing is simulated, never slept
    q = queue.Queue()
    q.put(evt)
    for _ in range(3 * CAP):
        asyncio.run(runner._drain_watch_notifications(q))
        if q.empty():
            break
        evt = q.get_nowait()
        q.put(evt)
        clock[0] += 61.0  # past the capped backoff, so the next tick really offers it
    assert q.empty(), 'the drain must settle an owned immutable conflict, not spin at the ceiling'
    assert evt['_watch_injection_attempts'] == CAP
