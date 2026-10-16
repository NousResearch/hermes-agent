"""Canonical queued-source identity (#63298) on the ``hermes-gateway-v1`` wire.

The legacy Desktop API registers ``message_id``/``submitted_at`` on ``PromptSubmitParams`` (the
busy/queue path's stable client source identity). This module pins the same registration and the
same semantics on the canonical contract: the submit is accepted (never ``invalid_params``), one
durable row is ever minted per client identity, the dedup answer is the canonical receipt shape
(``admission_id`` echo + ``terminal`` + ``outcome: 'duplicate'``) with the source identity echoed,
and a bare ``submission_id`` submit keeps the admission idempotency upstream pins untouched.
"""
import asyncio
from datetime import datetime, UTC
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_contract import SessionRef
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tui_gateway.contracts.registry import CANONICAL_METHODS

SOURCE = 'queued-1700000000000-source'
SUBMITTED_AT = 1_700_000_000.0
REF = SessionRef('owned', 's')


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A connected canonical viewer over a real authority + state.db, its drain executing through a
    stub that records each admitted payload (the accept-time durable stamp the drain persists from)."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='current')
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='current', db=db, epoch=epoch)
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
    viewer = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    admitted = []
    from gateway import session_finite

    async def execute(authority, ref, row):
        admitted.append(dict(row['payload']))
        authority.pending_results[row['admission_id']] = {
            'result': {'final_response': 'ok', 'messages': []}, 'usage': {}}
        return 'ok'

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    yield viewer, db, admitted, authority
    awaitable = viewer.close()
    asyncio.run(awaitable) if asyncio.iscoroutine(awaitable) else None
    db.close()


async def _connected(harness):
    viewer, db, admitted, _authority = harness
    resumed = await viewer.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
    assert resumed.get('error') is None, resumed
    return viewer, db, admitted


def _submit_params(**extra):
    return {'session_id': 's', 'text': 'queued message', 'queued': True, **extra}


async def _drain(authority):
    task = authority.sessions['s'].task
    await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_canonical_submit_accepts_the_source_identity_and_mints_one_durable_row_per_client_id(harness):
    """The gap's own symptom as the red pin: a canonical submit carrying ``message_id``/``submitted_at``
    answered ``4001 invalid_params`` at the dispatcher's closed key set. After the port it succeeds and
    a resubmit of the same client identity mints no second durable row."""
    viewer, db, admitted = await _connected(harness)
    _authority = harness[3]
    params = _submit_params(submission_id='sub-1700000000000-source',
                            message_id=SOURCE, submitted_at=SUBMITTED_AT)
    first = await viewer.dispatch({'id': 2, 'method': 'prompt.submit', 'params': dict(params)})
    assert first.get('error') is None, first
    await _drain(_authority)
    # The accept-time durable stamp the drained turn's persist reads: the same identity semantics
    # the legacy wire's ``_persist_queued_user_row`` writes on its accept-time row.
    assert admitted[0]['message_id'] == SOURCE
    assert admitted[0]['submitted_at'] == SUBMITTED_AT
    # A reconnect resend of the same turn under a fresh request id: acknowledged, never a second turn.
    retry = await viewer.dispatch({'id': 3, 'method': 'prompt.submit', 'params': {
        **params, 'submission_id': 'sub-retry-1700000000000'}})
    assert retry.get('error') is None, retry
    await _drain(_authority)
    rows = db._read_all('SELECT * FROM session_admissions')
    assert len(rows) == 1, rows
    assert len(admitted) == 1, admitted


@pytest.mark.asyncio
async def test_canonical_dedup_receipt_echoes_the_source_identity(harness):
    """The canonical dedup receipt keeps upstream's shape (``admission_id`` echo + ``terminal`` +
    ``outcome: 'duplicate'``) and adds the source identity echo — the legacy
    ``PromptSubmitStatus.duplicate`` equivalent on this wire."""
    viewer, _db, _admitted = await _connected(harness)
    _authority = harness[3]
    params = _submit_params(submission_id='sub-1700000000000-source',
                            message_id=SOURCE, submitted_at=SUBMITTED_AT)
    first = await viewer.dispatch({'id': 2, 'method': 'prompt.submit', 'params': dict(params)})
    assert first.get('error') is None, first
    await _drain(_authority)
    retry = await viewer.dispatch({'id': 3, 'method': 'prompt.submit', 'params': {
        **params, 'submission_id': 'sub-retry-1700000000000'}})
    result = retry['result']
    CANONICAL_METHODS['prompt.submit'].result.model_validate(result)
    assert result['status'] == 'terminal'
    assert result['outcome'] == 'duplicate'
    # The already-owned turn's admission, echoed so the retry binds its receipt to that turn.
    assert result['admission_id'] == first['result']['admission_id']
    # The added source identity echo (legacy parity: the ``duplicate`` reply names the source id).
    assert result['message_id'] == SOURCE
    assert result['submitted_at'] == SUBMITTED_AT


@pytest.mark.asyncio
async def test_submission_id_only_submits_keep_upstream_admission_idempotency(harness):
    """``submission_id`` coexistence, untouched: an identityless exact retry replays its own
    admission (never a ``duplicate`` receipt) and a fresh id is a fresh turn."""
    viewer, db, admitted = await _connected(harness)
    _authority = harness[3]
    params = _submit_params(submission_id='sub-x')
    first = await viewer.dispatch({'id': 2, 'method': 'prompt.submit', 'params': dict(params)})
    assert first.get('error') is None, first
    await _drain(_authority)
    assert first['result'].get('outcome') != 'duplicate'
    retry = await viewer.dispatch({'id': 3, 'method': 'prompt.submit', 'params': dict(params)})
    assert retry.get('error') is None, retry
    assert retry['result']['admission_id'] == first['result']['admission_id']
    assert retry['result'].get('outcome') != 'duplicate'
    fresh = await viewer.dispatch({'id': 4, 'method': 'prompt.submit', 'params': {
        **params, 'submission_id': 'sub-y'}})
    assert fresh.get('error') is None, fresh
    await _drain(_authority)
    assert fresh['result']['admission_id'] != first['result']['admission_id']
    rows = db._read_all('SELECT * FROM session_admissions')
    assert len(rows) == 2, rows
    assert len(admitted) == 2, admitted


@pytest.mark.asyncio
async def test_one_durable_row_per_client_identity_survives_a_reminted_target(harness):
    """A reconnect retry re-homed onto a re-minted target (compression tip rotation / resume
    recovery) is the SAME client identity: acknowledged against its already-owned admission, never
    a second durable row — the fixture contract this PR stamped on the seam (``canonicalRows 1``
    across three wire sends of one source identity)."""
    viewer, db, _admitted = await _connected(harness)
    _authority = harness[3]
    db.create_session('s2', source='test')
    _authority.sessions['s2'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
    resumed = await viewer.dispatch({'id': 2, 'method': 'session.resume', 'params': {'session_id': 's2'}})
    assert resumed.get('error') is None, resumed
    params = _submit_params(submission_id='sub-1700000000000-source',
                            message_id=SOURCE, submitted_at=SUBMITTED_AT)
    first = await viewer.dispatch({'id': 3, 'method': 'prompt.submit', 'params': dict(params)})
    assert first.get('error') is None, first
    await _drain(_authority)
    retry = await viewer.dispatch({'id': 4, 'method': 'prompt.submit', 'params': {
        'session_id': 's2', 'text': 'queued message', 'queued': True,
        'submission_id': 'sub-1700000000000-source', 'message_id': SOURCE, 'submitted_at': SUBMITTED_AT}})
    assert retry.get('error') is None, retry
    result = retry['result']
    CANONICAL_METHODS['prompt.submit'].result.model_validate(result)
    assert result['outcome'] == 'duplicate'
    assert result['admission_id'] == first['result']['admission_id']
    assert result['message_id'] == SOURCE
    rows = db._read_all('SELECT * FROM session_admissions')
    assert len(rows) == 1, rows


def test_user_row_persist_stamps_the_source_identity_like_the_legacy_accept_row():
    """The drained turn's persist carries the source identity onto its ONE user row
    (``platform_message_id``/row timestamp): attribution and ``has_platform_message_id`` retry dedup
    ride the row exactly as the legacy ``_persist_queued_user_row`` accept-time row documents."""
    from gateway.platforms.event import MessageEvent
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin

    event = MessageEvent(text='queued message', message_id='admission-minted-id',
                         source_message_id=SOURCE)
    prepared = SimpleNamespace(persist_user_message=None, message_text='queued message',
                               persist_user_timestamp=SUBMITTED_AT, persist_user_display_kind=None,
                               persistence_owner=None)
    entry = GatewayTurnPersistenceMixin._hmwa_user_transcript_entry(event, prepared, 0.0)
    assert entry['message_id'] == SOURCE
    assert entry['timestamp'] == SUBMITTED_AT
    # An identityless submit keeps the existing behaviour byte-identical.
    plain = GatewayTurnPersistenceMixin._hmwa_user_transcript_entry(
        MessageEvent(text='plain', message_id='admission-minted-id'),
        SimpleNamespace(persist_user_message=None, message_text='plain', persist_user_timestamp=None,
                        persist_user_display_kind=None, persistence_owner=None), 5.0)
    assert plain['message_id'] == 'admission-minted-id'
    assert plain['timestamp'] == 5.0


@pytest.mark.asyncio
async def test_drained_turn_event_carries_the_source_identity_to_the_persist(tmp_path, monkeypatch):
    """``execute_admission`` hands the turn an event stamped with the client source identity (and the
    client clock) while ``message_id`` stays the admission id — the ``display_v1`` binding upstream's
    ``admission_display`` pins."""
    from gateway.session_ingress import execute_admission
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='current')
    captured = {}

    async def handle(event):
        captured['event'] = event
        return 'ok'

    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None, _handle_message=handle)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='current', db=db, epoch=epoch)
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
    row = {'admission_id': 'admission-minted-id', 'status': 'started', 'generation': 3,
           'payload': {'text': 'queued message', 'message_id': SOURCE, 'submitted_at': SUBMITTED_AT}}
    try:
        response = await execute_admission(authority, REF, row)
    finally:
        db.close()
    assert response == 'ok'
    event = captured['event']
    assert event.message_id == 'admission-minted-id'
    assert event.source_message_id == SOURCE
    assert event.timestamp == datetime.fromtimestamp(SUBMITTED_AT, UTC)
