"""External maintenance drain must fence the real native durable FIFO."""
import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import RuntimeStoreError, get_session_admission, list_session_admissions
from plugins.platforms.slack.adapter import SlackAdapter


@pytest_asyncio.fixture
async def native_lane(tmp_path, monkeypatch, private_db_probe_cleanup):
    monkeypatch.setenv('SLACK_ALLOWED_USERS', 'U_TEST')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    runner._running = True
    adapter = SlackAdapter(PlatformConfig(enabled=True, token='fixture-token'))
    runner.adapters = {Platform.SLACK: adapter}
    adapter._resolve_user_name = AsyncMock(return_value='Fixture')
    adapter._resolve_channel_name = AsyncMock(return_value='fixture')
    adapter._humanize_user_mentions = AsyncMock(side_effect=lambda text, **kw: text)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='drain-fixture')
    runner._wire_adapter_handlers(adapter)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)  # hold the FIFO, never construct an agent
    event = await adapter._build_message_event(
        {'type': 'message', 'user': 'U_TEST', 'channel': 'C_TEST', 'team': 'T_TEST',
         'text': 'fixture', 'ts': '1792485702.680459'},
        text='fixture', original_text='fixture', command_probe_text='fixture', is_command_text=False,
        channel_id='C_TEST', team_id='T_TEST', ts='1792485702.680459', user_id='U_TEST',
        thread_ts='1792485702.680459', is_dm=False, media_urls=[], media_types=[],
        media_text_inlined=[], channel_context=None)
    try:
        yield runner, authority, adapter, event
    finally:
        authority.db.close()


async def _admit(runner, authority, event):
    with native_callback(runner, event, get_hermes_home()):
        return await authority.admit_native(event)


@pytest.mark.asyncio
async def test_external_drain_refuses_native_admission_without_changing_fifo(native_lane):
    runner, authority, _adapter, event = native_lane
    first = await _admit(runner, authority, event)
    before = list_session_admissions(authority.db, session_id=first.ref.session_id, pending_only=False)
    runner._enter_external_drain()
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        await _admit(runner, authority, replace(event, message_id='refused'))
    assert list_session_admissions(authority.db, session_id=first.ref.session_id, pending_only=False) == before
    runner._exit_external_drain()
    admitted = await _admit(runner, authority, replace(event, message_id='after-release'))
    assert admitted.status == 'queued'


@pytest.mark.asyncio
async def test_external_drain_refuses_real_claim_and_preserves_generation(native_lane):
    runner, authority, _adapter, event = native_lane
    receipt = await _admit(runner, authority, event)
    before = get_session_admission(authority.db, admission_id=receipt.admission_id)
    runner._enter_external_drain()
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        await authority._claim_next(receipt.ref, authority.sessions[receipt.ref.session_id])
    assert get_session_admission(authority.db, admission_id=receipt.admission_id) == before
    assert authority.executing == {}
    runner._exit_external_drain()
    claim, _ = await authority._claim_next(receipt.ref, authority.sessions[receipt.ref.session_id])
    assert claim['status'] == 'started' and claim['generation'] > 0


@pytest.mark.asyncio
async def test_external_drain_rechecks_actual_native_write_guard(native_lane, monkeypatch):
    import threading
    runner, authority, _adapter, event = native_lane
    first = await _admit(runner, authority, event)
    before = list_session_admissions(authority.db, session_id=first.ref.session_id, pending_only=False)
    reached = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    real_write = authority._admit_native_write

    def held_write(**kwargs):
        loop.call_soon_threadsafe(reached.set)
        if not release.wait(5):
            raise TimeoutError('fixture release missing')
        return real_write(**kwargs)

    monkeypatch.setattr(authority, '_admit_native_write', held_write)
    task = asyncio.create_task(_admit(runner, authority, replace(event, message_id='late-write')))
    try:
        await asyncio.wait_for(reached.wait(), 5)
        runner._enter_external_drain()
        release.set()
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            await asyncio.wait_for(task, 5)
        assert list_session_admissions(authority.db, session_id=first.ref.session_id, pending_only=False) == before
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_external_drain_rechecks_actual_claim_transaction_guard(native_lane, monkeypatch):
    import threading
    from gateway import session_authority as module
    runner, authority, _adapter, event = native_lane
    receipt = await _admit(runner, authority, event)
    before = get_session_admission(authority.db, admission_id=receipt.admission_id)
    reached = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    real_claim = module.claim_session_input

    def held_claim(*args, **kwargs):
        loop.call_soon_threadsafe(reached.set)
        if not release.wait(5):
            raise TimeoutError('fixture release missing')
        return real_claim(*args, **kwargs)

    monkeypatch.setattr(module, 'claim_session_input', held_claim)
    task = asyncio.create_task(authority._claim_next(receipt.ref, authority.sessions[receipt.ref.session_id]))
    try:
        await asyncio.wait_for(reached.wait(), 5)
        runner._enter_external_drain()
        release.set()
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            await asyncio.wait_for(task, 5)
        assert get_session_admission(authority.db, admission_id=receipt.admission_id) == before
        assert authority.executing == {}
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_external_drain_allows_started_turn_settlement_but_not_follower(native_lane, monkeypatch):
    runner, authority, _adapter, event = native_lane
    first = await _admit(runner, authority, event)
    follower = await _admit(runner, authority, replace(event, message_id='follower'))
    entered, release = asyncio.Event(), asyncio.Event()
    executions = []

    async def inert_execution(owner, ref, row):
        executions.append(row['admission_id'])
        entered.set()
        await asyncio.wait_for(release.wait(), 5)
        return 'fixture complete'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', inert_execution)
    task = asyncio.create_task(authority._drain(first.ref))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        runner._enter_external_drain()
        assert not task.done()
        assert get_session_admission(authority.db, admission_id=first.admission_id)['status'] == 'started'
        release.set()
        await asyncio.wait_for(task, 5)
        assert executions == [first.admission_id]
        assert get_session_admission(authority.db, admission_id=first.admission_id)['outcome'] == 'completed'
        assert get_session_admission(authority.db, admission_id=follower.admission_id)['status'] == 'queued'
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['retiring', 'shutdown', 'restart-wait', 'external-restart-wait'])
async def test_admission_preserves_shutdown_retirement_and_queue_only_contract(native_lane, mode):
    from gateway.drain_control import open_restart_wait
    from gateway.session_contract import Principal, Submission
    runner, authority, _adapter, event = native_lane
    first = await _admit(runner, authority, event)
    actor = Principal(event.source.user_id, authority.profile_id, frozenset({'session:submit'}), 'cli')
    runner.session_runtime_descriptor = {'state': 'ready'}
    runner._running = True
    authority.retiring = mode == 'retiring'
    runner._draining = mode in {'shutdown', 'restart-wait', 'external-restart-wait'}
    if 'restart-wait' in mode:
        open_restart_wait(runner)
    runner._external_drain_active = mode == 'external-restart-wait'
    request = Submission(ref=first.ref, request_id=mode, intent='queue', payload={'text': 'fixture'})
    if mode == 'restart-wait':
        queued = await authority.submit(actor, request, queue_while_restarting=True)
        assert queued.status == 'queued'
    else:
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            await authority.submit(actor, request, queue_while_restarting=True)
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        await authority._claim_next(first.ref, authority.sessions[first.ref.session_id])


@pytest.mark.asyncio
async def test_cancel_external_drain_wakes_the_existing_native_follower(native_lane, monkeypatch):
    runner, authority, _adapter, event = native_lane
    first = await _admit(runner, authority, event)
    follower = await _admit(runner, authority, replace(event, message_id='follower'))
    entered, release = asyncio.Event(), asyncio.Event()
    executed = []

    async def execute(owner, ref, row):
        executed.append(row['admission_id'])
        if row['admission_id'] == first.admission_id:
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
        return 'fixture complete'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    task = asyncio.create_task(authority._drain(first.ref))
    authority.sessions[first.ref.session_id].task = task
    try:
        await asyncio.wait_for(entered.wait(), 5)
        runner._enter_external_drain()
        release.set()
        await asyncio.wait_for(task, 5)
        assert executed == [first.admission_id]
        assert get_session_admission(authority.db, admission_id=follower.admission_id)['status'] == 'queued'
        with monkeypatch.context() as waking:
            waking.setattr(authority, '_schedule', authority.__class__._schedule.__get__(authority))
            runner._exit_external_drain()
            resumed = authority.sessions[first.ref.session_id].task
            assert resumed is not task, 'cancel left the accepted follower stranded until another message'
            await asyncio.wait_for(resumed, 5)
        assert executed == [first.admission_id, follower.admission_id]
        assert get_session_admission(authority.db, admission_id=follower.admission_id)['outcome'] == 'completed'
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_release_wakes_only_external_pauses_in_each_served_owner_scope(native_lane, tmp_path, monkeypatch):
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session import SessionSource
    from gateway.session_contract import SessionRef
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch
    runner, _launch, _adapter, _event = native_lane
    homes = [tmp_path / name for name in ('first', 'second', 'retiring', 'retired', 'other-pause')]
    registry = SessionAuthorities(homes[0], multiplexed=True)
    runner.session_authorities = registry
    runner.config.multiplex_profiles = True
    authorities, observed = [], []
    try:
        for home in homes:
            home.mkdir()
            db = SessionDB(home / 'state.db')
            db.create_session('s', source='telegram')
            owner = SessionAuthority(runner, profile_id=str(home), instance_id='fixture', db=db,
                epoch=begin_runtime_epoch(db, instance_id='fixture'))
            authorities.append(owner)
            registry.add(home, owner, name=home.name)
            owner.sessions['s'] = LiveSession(SessionSource(Platform.TELEGRAM, 'chat', user_id='human'), 'route')
            admit_session_input(db, epoch=owner.epoch, principal_id='human', session_id='s',
                                request_id='queued', payload={'text': 'fixture'})
            runner._external_drain_active = True
            owner._pause(SessionRef(owner.profile_id, 's'), 'runtime_draining')
            monkeypatch.setattr(owner, '_schedule', lambda ref: observed.append((ref.profile_id, str(get_hermes_home()))))
        authorities[2].retiring = True
        authorities[3].retired = True
        authorities[4].sessions['s'].external_drain_paused = False
        runner._exit_external_drain()
        assert observed == [(str(home), str(home)) for home in homes[:2]]
    finally:
        for owner in authorities:
            owner.db.close()
        del runner.session_authorities


@pytest.mark.asyncio
async def test_external_drain_settles_pump_waiting_on_classic_slot(native_lane, monkeypatch):
    runner, authority, _adapter, event = native_lane
    receipt = await _admit(runner, authority, event)
    checks = []

    def classic_busy(route):
        checks.append(route)
        # Bound the predecessor: its second check sees a free slot and reaches
        # admission refusal, instead of leaving this negative fixture spinning.
        return len(checks) == 1

    monkeypatch.setattr(authority, '_classic_turn_holds', classic_busy)
    runner._enter_external_drain()
    assert await asyncio.wait_for(authority._claim_with_retry(
        receipt.ref, authority.sessions[receipt.ref.session_id]), 5) is None
    assert len(checks) == 1, 'external drain left the pump waiting on a classic busy slot'
    assert get_session_admission(authority.db, admission_id=receipt.admission_id)['status'] == 'queued'


@pytest.mark.asyncio
async def test_external_drain_does_not_resurrect_a_removed_live_session(native_lane, monkeypatch):
    runner, authority, _adapter, event = native_lane
    receipt = await _admit(runner, authority, event)
    live = authority.sessions[receipt.ref.session_id]

    def removed_while_classic_busy(route):
        authority.sessions.pop(receipt.ref.session_id, None)
        return True

    monkeypatch.setattr(authority, '_classic_turn_holds', removed_while_classic_busy)
    runner._enter_external_drain()
    assert await authority._claim_with_retry(receipt.ref, live) is None
    assert receipt.ref.session_id not in authority.sessions
    assert get_session_admission(authority.db, admission_id=receipt.admission_id)['status'] == 'queued'


@pytest.mark.asyncio
async def test_retired_pump_does_not_mark_a_replacement_fifo_for_release(native_lane, monkeypatch):
    from gateway.session_authority import LiveSession
    runner, authority, _adapter, event = native_lane
    receipt = await _admit(runner, authority, event)
    previous = authority.sessions[receipt.ref.session_id]
    replacement = LiveSession(previous.source, previous.route)

    def swap_while_busy(route):
        authority.sessions[receipt.ref.session_id] = replacement
        return True

    monkeypatch.setattr(authority, '_classic_turn_holds', swap_while_busy)
    runner._enter_external_drain()
    assert await authority._claim_with_retry(receipt.ref, previous) is None
    assert replacement.external_drain_paused is False
    assert get_session_admission(authority.db, admission_id=receipt.admission_id)['status'] == 'queued'
