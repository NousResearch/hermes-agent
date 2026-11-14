"""Real two-home restart and hot re-serve through the production local recovery callers."""
import asyncio
import importlib
import threading

import pytest

from tests.gateway.test_local_session_orphan_end import _attach, _create
from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner
from tests.waiting import async_wait_until


def _observe(monkeypatch, records):
    from gateway.session_authority import SessionAuthority
    from gateway.session_local import LocalSessionAdapter
    from hermes_constants import get_hermes_home
    targets = [
        ('gateway.session_local_adoption', '_due'),
        ('gateway.session_local_recovery', 'local_receipt_ids'),
        ('gateway.session_local_startup', '_prepare'),
        ('gateway.session_local_startup', 'list_session_admissions'),
        ('gateway.session_viewer_close', 'persisted_closes'),
        ('gateway.session_viewer_close', 'orphan_close_grace'),
        ('gateway.session_viewer_close', 'orphan_candidate'),
        ('gateway.session_viewer_close', '_idle_s'),
        ('gateway.session_local_recovery', 'publish_local_session'),
        ('gateway.session_viewer_close', '_call_later'),
    ]

    def observe(target, name):
        original = getattr(target, name)

        def call(*args, **kwargs):
            records.append((name, threading.get_ident(), str(get_hermes_home())))
            return original(*args, **kwargs)

        monkeypatch.setattr(target, name, call)

    for module, name in targets:
        observe(importlib.import_module(module), name)
    observe(SessionAuthority, '_schedule')
    observe(LocalSessionAdapter, 'register_source')


@pytest.mark.asyncio
async def test_two_home_restart_and_hot_reserve_keep_recovery_contract(tmp_path, monkeypatch):
    from gateway import session_finite
    from gateway.config import Platform
    from gateway.run_runtime import (
        _retire_profile_authority, initialize_gateway_runtime, recover_gateway_native_sessions, serve_profile_runtime,
    )
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import owner_scope
    from gateway.session_viewer_close import CLOSE_PREFIX, _PendingClose, _persist_close
    from hermes_state_local import end_idle_local_session, local_receipt
    from hermes_state_runtime import admit_session_input, claim_session_input, get_session_admission
    from tests.gateway.test_local_startup_off_loop import _empty_cron_receipts

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('beta',))
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    process_ownership.reserve([home for _, home in homes])
    runner, saved = _runner(root, homes), {}
    viewers = []
    executed, records = [], []
    entered, release = asyncio.Event(), asyncio.Event()
    owner_thread = threading.get_ident()

    async def execute(authority, ref, row):
        executed.append((authority.profile_id, ref.session_id, row['payload']['text']))
        entered.set()
        await release.wait()
        return 'done'

    async def no_external(*args, **kwargs):
        return {}

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr('gateway.session_hosted_service.ensure_hosted_service', no_external)
    monkeypatch.setattr('gateway.session_hosted_service._ensure_hosted_service', no_external)
    monkeypatch.setattr('gateway.platforms.webhook_ingress.recover_webhook_finalizations', no_external)
    # Native and webhook recovery are orthogonal; all local authority/FIFO/close paths stay real.
    monkeypatch.setattr('gateway.run_runtime.recover_profile_native_sessions', no_external)
    try:
        await initialize_gateway_runtime(runner)
        for name, home in homes:
            authority = runner.session_authorities.for_home(home)
            with owner_scope(authority):
                ids = {kind: _create(authority, kind).session_id
                       for kind in ('queued', 'unknown', 'close', 'ended', 'compressed')}
                ids['orphans'] = _empty_cron_receipts(authority, monkeypatch, count=4)
                for kind in ('queued', 'unknown', 'close'):
                    sid = ids[kind]
                    row = admit_session_input(authority.db, epoch=authority.epoch,
                        principal_id=authority.sessions[sid].source.user_id,
                        session_id=sid, request_id=kind, payload={'text': kind})
                    ids[kind + '_admission'] = row['admission_id']
                    if kind != 'queued':
                        claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
                _persist_close(authority, ids['close'], _PendingClose('cli_close').record)
                assert end_idle_local_session(authority.db, epoch=authority.epoch,
                    session_id=ids['ended'], target_id=ids['ended'], reason='cli_close')
                child = ids['compressed'] + '-child'
                authority.db.publish_compression_child(parent_session_id=ids['compressed'], child_session_id=child,
                    source='gui', messages=[{'role': 'user', 'content': 'summary'}], require_compression_lease=False)
                assert runner.session_store.advance_compression_session(
                    authority.sessions[ids['compressed']].route, ids['compressed'], child)
                ids['child'] = child
                saved[name] = ids
            (home / 'config.yaml').write_text(f'sessions:\n  orphan_end_grace_s: {7 if name == "beta" else 11}\n')
        runner.session_store.close_all_db_handles()
        runner = _runner(root, homes)
        _observe(monkeypatch, records)
        await initialize_gateway_runtime(runner)
        for name, home in homes:
            authority, ids = runner.session_authorities.for_home(home), saved[name]
            assert ids['ended'] not in authority.sessions
            assert set(ids['orphans']) <= authority.sessions.keys()
            assert local_receipt(authority.db, ids['compressed'])['entry']['session_id'] == ids['child']
            assert get_session_admission(authority.db, admission_id=ids['unknown_admission'])['status'] == 'unknown'
        await recover_gateway_native_sessions(runner)
        await asyncio.wait_for(entered.wait(), 10)
        await async_wait_until(lambda: len(executed) == 2)
        timers = {a.profile_id: dict(a.orphan_closes) for a in runner.session_authorities}
        tasks = {a.profile_id: a.sessions[saved[name]['queued']].task
                 for name, home in homes for a in [runner.session_authorities.for_home(home)]}
        await recover_gateway_native_sessions(runner)
        for name, home in homes:
            authority, ids = runner.session_authorities.for_home(home), saved[name]
            assert authority.sessions[ids['queued']].task is tasks[authority.profile_id]
            assert authority.orphan_closes == timers[authority.profile_id]
            assert ids['unknown'] not in authority.orphan_closes
            assert authority.sessions[ids['unknown']].task is None
            assert local_receipt(authority.db, ids['ended'])  # kept, not deleted or eagerly mounted
        release.set()
        await async_wait_until(lambda: all(
            runner.session_authorities.for_home(home).db.get_session(saved[name]['close'])['end_reason'] == 'cli_close'
            for name, home in homes))
        for name, home in homes:
            authority, ids = runner.session_authorities.for_home(home), saved[name]
            assert authority.db.get_meta(CLOSE_PREFIX + ids['close']) is None
            assert get_session_admission(authority.db, admission_id=ids['unknown_admission'])['status'] == 'unknown'
            closed = get_session_admission(authority.db, admission_id=ids['close_admission'])
            assert (closed['status'], closed['outcome']) == ('terminal', 'interrupted')
            with owner_scope(authority):
                viewer, resumed = await _attach(authority, ids['ended'])
                viewers.append(viewer)
                assert resumed['messages'][0]['content'] == 'ended'
        assert sorted(text for _, _, text in executed) == ['queued', 'queued']
        # Retire and re-serve beta through the actual production entry point.
        beta = homes[1][1]
        authority = runner.session_authorities.for_home(beta)
        for viewer in viewers:
            await viewer.close()
        viewers.clear()
        assert await _retire_profile_authority(authority)
        runner.session_authorities.remove(beta)
        replacement = await serve_profile_runtime(runner, 'beta', beta)
        assert replacement is not authority
        assert runner._profile_adapters['beta'][Platform.LOCAL].authority is replacement
        assert saved['beta']['unknown'] in replacement.sessions
        assert replacement.sessions[saved['beta']['unknown']].task is None
        assert sorted(text for _, _, text in executed) == ['queued', 'queued']
        reads = {'_due', 'local_receipt_ids', '_prepare', 'list_session_admissions', 'persisted_closes',
                 'orphan_close_grace', 'orphan_candidate', '_idle_s'}
        # Publication also happens for lazy resume on-loop; the recovery-only reads never do.
        assert reads <= {name for name, _, _ in records}
        assert all(thread != owner_thread for name, thread, _ in records if name in reads)
        assert all(thread == owner_thread for name, thread, _ in records if name not in reads)
        assert {home for name, _, home in records if name == '_prepare'} == {str(h) for _, h in homes}
    finally:
        release.set()
        for viewer in viewers:
            await viewer.close()
        for authority in list(runner.session_authorities):
            await _retire_profile_authority(authority)
        runner.session_store.close_all_db_handles()
        for _, home in homes:
            process_ownership.release(home)
