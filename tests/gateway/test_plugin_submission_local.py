"""Local plugin receipts follow the creation owner while pinning the current transcript."""
import asyncio
import threading
from contextlib import asynccontextmanager

import hermes_yaml as yaml
import pytest
import pytest_asyncio

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages

pytestmark = pytest.mark.usefixtures('authority_cwd_in_tmp')


@pytest_asyncio.fixture
async def local_submission(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    runner = run.GatewayRunner(GatewayConfig())
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='local-receipt')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    actor = SessionActor('owner', 'default', frozenset({'session:create', 'session:read', 'session:submit',
                                                     'session:control'}), 'native')
    ref = create_local_session(authority, actor, dict(request_id='create', source='gui', cwd=str(tmp_path),
                                                    model='fixture', toolsets=[]))
    seed_messages(authority.persistence, ref.session_id, {'role': 'user', 'content': 'before reset'})
    home = get_hermes_home()
    (home / 'config.yaml').write_text(yaml.safe_dump({'plugins': {'entries': {
        'receipt-probe': {'allow_gateway_injection': True}}}}), encoding='utf-8')
    manager = PluginManager(scope_key=str(home))
    ctx = PluginContext(PluginManifest(name='receipt-probe', key='receipt-probe', source='user'), manager)
    runner._gateway_loop, runner._running = asyncio.get_running_loop(), True
    manager.set_gateway_message_injector(runner, runner._schedule_plugin_message_injection)
    identity = dict(session_key=authority.sessions[ref.session_id].route,
                    expected_session_id=ref.session_id, request_id='work')
    yield ctx, runner, authority, actor, ref, identity
    runner._running = False
    manager.clear_gateway_message_injector(runner)


def reset_params(authority, ref):
    row = session_row(authority.persistence, ref.session_id)
    return dict(session_id=ref.session_id, request_id='reset', expected_revision=row['runtime_revision'],
                expected_generation=row['runtime_generation'], operation='reset', payload={})


@pytest.mark.asyncio
async def test_fresh_local_binding_after_reset_can_recover_and_observe_its_receipt(local_submission):
    from gateway.session_mutations import mutate_session

    ctx, _, authority, actor, ref, identity = local_submission
    reset = await mutate_session(authority, actor, ref, reset_params(authority, ref))
    fresh = {**identity, 'expected_session_id': reset['target_session_id']}
    first = await asyncio.to_thread(ctx.submit_message, 'after reset', **fresh)
    assert first['ok'], first
    repeated = await asyncio.to_thread(ctx.submit_message, 'after reset', **fresh)
    observed = await asyncio.to_thread(ctx.submission_status, **fresh)
    assert repeated['admission_id'] == observed['admission_id'] == first['admission_id']
    assert observed['ok'] and observed['status'] == 'queued', observed
    stale = await asyncio.to_thread(ctx.submission_status, **identity)
    assert stale['error'] == 'admission_conflict', stale
    ledger = authority.persistence.admissions
    work = ledger.claim(epoch=authority.epoch, session_id=ref.session_id)
    assert work.admission_id == first['admission_id']
    ledger.settle(epoch=authority.epoch, admission_id=work.admission_id, generation=work.generation,
                  outcome='completed', result={'result': {'final_response': 'done'}})
    final = await asyncio.to_thread(ctx.submission_status, **fresh)
    assert final['status'] == 'terminal' and final['result']['final_response'] == 'done'


@pytest.mark.asyncio
async def test_local_reset_winning_the_mutation_gate_refuses_stale_admission(local_submission, monkeypatch):
    from gateway.session_mutations import write_mutation

    ctx, _, authority, actor, ref, identity = local_submission
    live = authority.sessions[ref.session_id]
    gate = live.mutation_lock
    waiting, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    @asynccontextmanager
    async def observed_gate():
        waiting.set()
        async with gate:
            yield

    monkeypatch.setattr(live, 'mutation_lock', observed_gate())
    ledger = authority.persistence.admissions
    original = type(ledger).admit

    def held_admit(self, **kwargs):
        # On the old code this marks the unprotected snapshot-to-commit gap.
        loop.call_soon_threadsafe(waiting.set)
        assert release.wait(10), 'the test did not release admission'
        return original(self, **kwargs)

    monkeypatch.setattr(type(ledger), 'admit', held_admit)
    async with gate:
        pending = asyncio.create_task(asyncio.to_thread(ctx.submit_message, 'stale work', **identity))
        try:
            await asyncio.wait_for(waiting.wait(), 10)
            # The actual local mutation writer runs under the same owner lock as mutate_session.
            reset = await write_mutation(authority, ref, reset_params(authority, ref), None,
                                         principal_id=actor.subject)
            assert reset['target_session_id'] != ref.session_id
        finally:
            release.set()
    result = await asyncio.wait_for(pending, 10)
    assert result['status'] == 'refused' and result['error'] == 'admission_conflict', result
    assert not any(row.payload.get('text') == 'stale work'
                   for row in ledger.list(session_id=ref.session_id, pending_only=False))


@pytest.mark.asyncio
async def test_cancelled_observer_keeps_local_gate_until_admission_commits(local_submission, monkeypatch):
    from gateway.plugin_submission.service import Request, dispatch
    from tests.waiting import async_wait_until

    ctx, runner, authority, _, ref, identity = local_submission
    entered, release = threading.Event(), threading.Event()
    ledger = authority.persistence.admissions
    original = type(ledger).admit

    def held_admit(self, **kwargs):
        entered.set()
        assert release.wait(10), 'the test did not release admission'
        return original(self, **kwargs)

    monkeypatch.setattr(type(ledger), 'admit', held_admit)
    request = Request(**identity, plugin_id=ctx.plugin_id, plugin_home=ctx._manager.home_path,
                      content='committing work')
    pending = asyncio.create_task(dispatch(runner, request))
    gate = authority.sessions[ref.session_id].mutation_lock
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert gate.locked(), 'a physical writer lost the reset gate with its observer'
    finally:
        release.set()
    await async_wait_until(lambda: not gate.locked(), timeout=10)
    receipt = await asyncio.to_thread(ctx.submission_status, **identity)
    assert receipt['ok'] and receipt['status'] == 'queued'
