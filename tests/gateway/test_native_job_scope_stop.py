"""Native local job scope survives wrapper retirement and fences durable FIFO Stop."""
import asyncio
import json
import shlex
import sys

import psutil
import pytest

from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from tests.waiting import wait_until


def live(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


async def spawn(registry, root, command, *, parent='s', route='route', pty=False):
    from gateway.session_context import clear_session_vars, set_session_vars
    tokens = set_session_vars(session_id=parent, session_key=route)
    try:
        return await asyncio.to_thread(registry.spawn_local, command, cwd=str(root), task_id='worker-uuid',
            owner_task_id='worker-uuid', session_key=route, persist_on_release=True, use_pty=pty)
    finally:
        clear_session_vars(tokens)


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('reload_registry', [False, True], ids=['live-owner', 'registry-reload'])
@pytest.mark.parametrize('primary_exits', [False, True], ids=['active-primary', 'exited-primary'])
@pytest.mark.parametrize('pty', [False, True], ids=['file-output', 'pty-output'])
async def test_native_job_scope_quiesces_detached_children_before_fifo(tmp_path, monkeypatch, reload_registry, primary_exits, pty):
    from gateway.session_ancillary import process_stop_pending
    from tools import process_registry as module
    from tools.process_registry_job import job_quiescent, read_job_receipt, request_job_stop
    db, authority = _authority(tmp_path, monkeypatch)
    original = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', original)
    monkeypatch.setattr(module, '_systemd_run_user_scope_available', lambda: False)
    child_ready, release, effect = tmp_path / 'child', tmp_path / 'release', tmp_path / 'old-effect'
    daemon = ('import os,signal,time; from pathlib import Path\n'
              'if os.fork(): os._exit(0)\n'
              'os.setsid(); signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
              f'Path({str(child_ready)!r}).write_text(str(os.getpid())); p=Path({str(release)!r}); end=time.monotonic()+60\n'
              'while not p.exists() and time.monotonic()<end: time.sleep(.01)\n'
              f'if p.exists(): Path({str(effect)!r}).write_text("old effect")')
    wrapper = ('import subprocess,sys,time; '
               f'p=subprocess.Popen([sys.executable,"-c",{daemon!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
               'p.wait(timeout=10); print("original primary output",flush=True); '
               + ('' if primary_exits else 'time.sleep(60)'))
    command = 'exec ' + shlex.join([sys.executable, '-c', wrapper])
    own = await spawn(original, tmp_path, command, pty=pty)
    sleeper = 'exec ' + shlex.join([sys.executable, '-c', 'import threading; threading.Event().wait(60)'])
    neighbors = [await spawn(original, tmp_path, sleeper, parent='foreign-parent'),
                 await spawn(original, tmp_path, sleeper, route='foreign-route')]
    child = None
    executed = []

    async def execute(owner, ref, row):
        assert not live(child), 'successor admitted with old detached child alive'
        assert job_quiescent(own)
        executed.append(row['request_id'])
        release.touch()
        return 'finite fixture answer'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        await asyncio.to_thread(wait_until, child_ready.exists, timeout=15)
        child = psutil.Process(int(child_ready.read_text()))
        assert own.native_job_scope and not own.systemd_unit and live(child)
        await asyncio.to_thread(wait_until, lambda: read_job_receipt(own) is not None, timeout=10)
        if primary_exits:
            await asyncio.to_thread(wait_until, lambda: read_job_receipt(own)['exit_code'] == 0, timeout=10)
        assert read_job_receipt(own)['quiescent'] is False and not own.exited
        await asyncio.to_thread(wait_until, lambda: 'original primary output' in own.output_buffer, timeout=10)
        if reload_registry:
            current = module.ProcessRegistry()
            monkeypatch.setattr(module, 'process_registry', current)
            assert current.recover_from_checkpoint() == 3
            assert current.get(own.id).native_job_scope
        else:
            current = original
        await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        assert process_stop_pending(db, 's') is None
        assert job_quiescent(own) and not live(child)
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert 'original primary output' in own.output_buffer
        if primary_exits:
            assert own.exit_code == 0
        assert all(not neighbor.exited for neighbor in neighbors)
        assert own.id not in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        await _submit(authority, 'verified human successor')
        await asyncio.wait_for(authority.sessions['s'].task, 15)
        assert executed == ['verified human successor'] and not effect.exists()
        assert all(not neighbor.exited for neighbor in neighbors)
    finally:
        for session in [own, *neighbors]:
            await asyncio.to_thread(request_job_stop, original, session)
            await asyncio.to_thread(session._completion_event.wait, 10)
        if child is not None and live(child):
            child.kill()
        db.close()


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass  # Old fixture deliberately orphans its exact captured finite child.
@pytest.mark.parametrize('primary_exits', [False, True], ids=['active-old-wrapper', 'completed-old-wrapper'])
@pytest.mark.parametrize('reload_registry', [False, True], ids=['old-live-cache', 'old-reload'])
async def test_old_unscoped_local_job_remains_unknown(tmp_path, monkeypatch, primary_exits, reload_registry):
    from gateway.session_ancillary import process_stop_pending
    from tools import process_registry as module
    db, authority = _authority(tmp_path, monkeypatch)
    original = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', original)
    monkeypatch.setattr(original, '_scope_argv', lambda session, command, suffix, label:
                        [module._find_shell(), '-lic', 'set +m; ' + command])
    ready, release = tmp_path / 'old-child', tmp_path / 'release'
    code = (f'from pathlib import Path; import os,time; Path({str(ready)!r}).write_text(str(os.getpid())); '
            f'p=Path({str(release)!r}); end=time.monotonic()+60\n'
            'while not p.exists() and time.monotonic()<end: time.sleep(.01)')
    wrapper = ('import subprocess,sys,time; '
               f'subprocess.Popen([sys.executable,"-c",{code!r}],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
               'print("old primary output",flush=True); ' + ('' if primary_exits else 'time.sleep(60)'))
    own = await spawn(original, tmp_path, 'exec ' + shlex.join([sys.executable, '-c', wrapper]))
    child, executed = None, []

    async def execute(owner, ref, row):
        executed.append(row['request_id'])
        return 'finite fixture answer'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        await asyncio.to_thread(wait_until, ready.exists, timeout=10)
        child = psutil.Process(int(ready.read_text()))
        if primary_exits:
            assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert not own.native_job_scope and not own.systemd_unit and live(child)
        if reload_registry:
            monkeypatch.setattr(module, 'process_registry', module.ProcessRegistry())
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        await _submit(authority, 'held human')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert not executed and live(child) and process_stop_pending(db, 's')
        assert own.id in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        assert list_session_admissions(db, session_id='s')[0]['status'] == 'queued'
    finally:
        release.touch()
        if child is not None and live(child):
            child.kill()
        if own.process is not None and own.process.poll() is None:
            own.process.terminate()
            await asyncio.to_thread(own.process.wait, 10)
        await asyncio.to_thread(own._completion_event.wait, 10)
        db.close()
