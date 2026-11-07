"""Canonical-owner orphan recovery after a real creator crash (#142592)."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest


async def _crashed_sessions(home, scenario='idle'):
    child = await asyncio.to_thread(subprocess.run, [
        sys.executable, '-m', 'tests.gateway.fixtures.local_orphan_probe', str(home), scenario,
    ], cwd=Path(__file__).resolve().parents[2], env={**os.environ, 'HERMES_HOME': str(home)},
        capture_output=True, text=True, timeout=20)
    assert child.returncode == 23, (child.stdout, child.stderr)
    evidence = json.loads((home / 'creator-evidence.json').read_text())
    assert evidence['network_attempts'] == []
    return evidence['records']


def _runner(home):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_state import SessionDB
    store = SessionStore(home / 'sessions', GatewayConfig())
    # The suite pins argless DB defaults per test; this owner must reopen the child's exact file.
    store._db = SessionDB(home / 'state.db')
    return SimpleNamespace(session_store=store, _session_db=store._db, adapters={},
                           _draining=False, _cached_agent_for=lambda route: None,
                           _adapter_for_source=lambda source: None)


@pytest.mark.asyncio
async def test_cold_crash_recovery_finalizes_only_idle_ui_rows(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    records = await _crashed_sessions(tmp_path)
    from gateway import run
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    runner = _runner(tmp_path)
    authority = await initialize_session_authority(
        runner, profile_id=str(tmp_path), instance_id='restarted-owner')
    connection = AuthorityConnection(authority, object(), {'user_id': 'fixture-owner'})
    try:
        for name in ('tui_local', 'desktop_local', 'unrefreshed_tui'):
            sid = records[name]['id']
            row = authority.db.get_session(sid)
            assert row['end_reason'] == 'startup_orphan_reap', row
            assert row['message_count'] == 1
            assert sid in authority.sessions
        for name in ('tool_control', 'cli_control', 'acp_control'):
            assert authority.db.get_session(records[name]['id'])['ended_at'] is None
        reply = await connection.dispatch({'id': 1, 'method': 'session.active_list', 'params': {}})
        assert not {records[name]['id'] for name in ('tui_local', 'desktop_local', 'unrefreshed_tui')} & {
            row['id'] for row in reply['result']['sessions']}, reply
        sid = records['tui_local']['id']
        resumed = await connection.dispatch({'id': 2, 'method': 'session.resume',
                                              'params': {'session_id': sid}})
        assert resumed['result']['session_id'] == sid, resumed
        assert resumed['result']['messages'][0]['content'] == 'fixture history for tui_local'
        # Mounting history is read-only; a real admitted turn owns reopening.
        assert authority.db.get_session(sid)['ended_at'] is not None
        viewed = await connection.dispatch({'id': 3, 'method': 'session.active_list', 'params': {}})
        assert sid in {row['id'] for row in viewed['result']['sessions']}, viewed
        ended_at = authority.db.get_session(sid)['ended_at']
        await connection.close()
        again = await initialize_session_authority(
            _runner(tmp_path), profile_id=str(tmp_path), instance_id='third-owner')
        try:
            assert again.db.get_session(sid)['ended_at'] == ended_at
        finally:
            again.db.close()
    finally:
        await connection.close()
        authority.db.close()


def _protect(db, sid, protection):
    if protection.startswith('compressed_'):
        from tests.gateway.test_local_resume_compression_child import _compress
        child = sid + '-tip'
        _compress(db, sid, child, 'local')
        if protection != 'compressed_fresh':
            old = time.time() - 100 * 86400
            db._write_sql('UPDATE sessions SET started_at=?,last_activity_at=NULL WHERE id=?', (old, child))
            db._write_sql('UPDATE messages SET timestamp=? WHERE session_id=?', (old, child))
        if protection == 'compressed_pin':
            db.set_session_pinned(sid, True)
        return child
    actions = {
        'pin': lambda: db.set_session_pinned(sid, True),
        'recent_start': lambda: db._write_sql('UPDATE sessions SET started_at=? WHERE id=?', (time.time(), sid)),
        'recent_message': lambda: db.append_message(sid, 'user', 'recent activity'),
        'recent_activity': lambda: db._write_sql('UPDATE sessions SET last_activity_at=? WHERE id=?', (time.time(), sid)),
        'turn_lease': lambda: db.try_acquire_session_turn_lease(sid, f'fixture:{os.getpid()}'),
        'compression_lock': lambda: db.try_acquire_compression_lock(sid, f'fixture:{os.getpid()}'),
        'heartbeat': lambda: db.register_backend_heartbeat(backend_id='fixture', pid=os.getpid(),
                                                         started_at=time.time() - 101 * 86400),
    }
    if action := actions.get(protection):
        result = action()
        if protection in ('turn_lease', 'compression_lock'):
            assert result
    return sid


@pytest.mark.asyncio
@pytest.mark.parametrize('protection', [
    'recent_start', 'recent_message', 'recent_activity', 'pin', 'turn_lease',
    'compression_lock', 'heartbeat', 'queued', 'started', 'worker',
    'compressed_old', 'compressed_fresh', 'compressed_pin',
])
async def test_cold_recovery_preserves_protected_conversations(tmp_path, monkeypatch, protection):
    from gateway import run
    from gateway.session_authority import initialize_session_authority
    from hermes_state_local import local_receipt
    from hermes_state_runtime import list_session_admissions
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    records = await _crashed_sessions(tmp_path, protection)
    monkeypatch.setattr(run, '_load_gateway_config', dict)
    runner = _runner(tmp_path)
    sid = records['tui_local']['id']
    target = _protect(runner._session_db, sid, protection)
    authority = await initialize_session_authority(
        runner, profile_id=str(tmp_path), instance_id='restarted-owner')
    try:
        row = authority.db.get_session(target)
        assert (row['ended_at'] is not None) == (protection == 'compressed_old'), row
        assert local_receipt(authority.db, sid)['entry']['session_id'] == target
        assert sid in authority.sessions
        if protection != 'heartbeat':
            assert authority.db.get_session(records['desktop_local']['id'])['end_reason'] == 'startup_orphan_reap'
        if protection in ('queued', 'started'):
            assert [item['status'] for item in list_session_admissions(authority.db, session_id=sid)] == [
                'unknown' if protection == 'started' else 'queued']
        if protection == 'worker':
            with authority.db._read_ctx() as conn:
                assert conn.execute('SELECT status FROM worker_executions').fetchone()[0] == 'unknown'
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('disabled', ['flag', 'ttl', 'grace_config', 'grace_env'])
async def test_existing_opt_outs_preserve_rows(tmp_path, monkeypatch, disabled):
    from gateway import run
    from gateway.session_authority import initialize_session_authority
    records = await _crashed_sessions(tmp_path)
    config = {'dashboard': {}}
    if disabled == 'flag':
        config['dashboard']['startup_orphan_sweep'] = False
    elif disabled == 'grace_config':
        config['dashboard']['ws_orphan_reap_grace_s'] = 0
    else:
        monkeypatch.setenv('HERMES_TUI_SESSION_TTL_S' if disabled == 'ttl' else
                           'HERMES_TUI_WS_ORPHAN_REAP_GRACE_S', '0')
    monkeypatch.setattr(run, '_load_gateway_config', lambda: config)
    authority = await initialize_session_authority(_runner(tmp_path), profile_id=str(tmp_path), instance_id='restart')
    try:
        assert authority.db.get_session(records['tui_local']['id'])['ended_at'] is None
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_cleanup_reads_owning_home_config_not_ambient_profile(tmp_path, monkeypatch):
    from gateway.session_authority import initialize_session_authority
    enabled, disabled = tmp_path / 'enabled', tmp_path / 'disabled'
    for home in (enabled, disabled):
        home.mkdir()
    records_a, records_b = await _crashed_sessions(enabled), await _crashed_sessions(disabled)
    (enabled / 'config.yaml').write_text('dashboard:\n  startup_orphan_sweep: true\n')
    (disabled / 'config.yaml').write_text('dashboard:\n  startup_orphan_sweep: false\n')
    monkeypatch.setenv('HERMES_HOME', str(disabled))
    authority_a = await initialize_session_authority(_runner(enabled), profile_id=str(enabled), instance_id='a')
    monkeypatch.setenv('HERMES_HOME', str(enabled))
    authority_b = await initialize_session_authority(_runner(disabled), profile_id=str(disabled), instance_id='b')
    try:
        assert authority_a.db.get_session(records_a['tui_local']['id'])['end_reason'] == 'startup_orphan_reap'
        assert authority_b.db.get_session(records_b['tui_local']['id'])['ended_at'] is None
        assert authority_a.db.get_session(records_b['tui_local']['id']) is None
        assert authority_b.db.get_session(records_a['tui_local']['id']) is None
    finally:
        authority_a.db.close()
        authority_b.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('interrupt', ['cancel', 'cancel_twice', 'epoch'])
async def test_unpublished_cleanup_is_joined_and_epoch_fenced(tmp_path, monkeypatch, interrupt):
    from gateway import run
    from gateway.session_authority import initialize_session_authority
    import hermes_state_local_reap as reaper
    from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch
    records = await _crashed_sessions(tmp_path)
    monkeypatch.setattr(run, '_load_gateway_config', dict)
    runner = _runner(tmp_path)
    entered, release = threading.Event(), threading.Event()
    real_reap = reaper._reap_idle_local_session

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(10), 'test did not release bootstrap writer'
        return real_reap(*args, **kwargs)

    monkeypatch.setattr(reaper, '_reap_idle_local_session', blocked)
    task = asyncio.create_task(initialize_session_authority(
        runner, profile_id=str(tmp_path), instance_id='interrupted-owner'))
    try:
        assert await asyncio.to_thread(entered.wait, 10), 'bootstrap did not reach cleanup'
        if interrupt.startswith('cancel'):
            task.cancel()
            # Queue a loop barrier after cancellation delivery, without a timed sleep.
            barrier = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(barrier.set_result, None)
            await barrier
            assert not task.done(), 'cancelled bootstrap must join its writer'
            if interrupt == 'cancel_twice':
                task.cancel()
        else:
            await asyncio.to_thread(begin_runtime_epoch, runner._session_db, instance_id='newer-owner')
        release.set()
        with pytest.raises(asyncio.CancelledError if interrupt.startswith('cancel') else RuntimeStoreError) as caught:
            await asyncio.wait_for(task, 10)
        if interrupt == 'epoch':
            assert caught.value.reason == 'stale_epoch'
        assert not hasattr(runner, 'session_authority')
        assert runner._session_db.get_session(records['tui_local']['id'])['ended_at'] is None
        monkeypatch.setattr(reaper, '_reap_idle_local_session', real_reap)
        # The joined worker cannot stamp rows after cancellation; a new owner can recover normally.
        authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='final-owner')
        assert authority.db.get_session(records['tui_local']['id'])['end_reason'] == 'startup_orphan_reap'
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        runner._session_db.close()
