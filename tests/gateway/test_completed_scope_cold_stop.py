"""A completed wrapper retains native scope authority until an explicit, verified Stop."""
import asyncio
import json
import shlex
import sys

import psutil
import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority


@pytest.mark.asyncio
@pytest.mark.platforms('linux')
@pytest.mark.live_system_guard_bypass  # The fixture's birth-verified detached child outlives its wrapper.
@pytest.mark.parametrize('refuse', [False, True], ids=['stopped', 'refused-then-retry'])
@pytest.mark.parametrize('expired_result', [False, True], ids=['retained-output', 'result-expired'])
async def test_completed_scope_survives_normal_finish_and_cold_stop(tmp_path, monkeypatch, refuse, expired_result):
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import process_registry as module
    from tools.process_registry_results import load_completed_results

    db, authority = _authority(tmp_path, monkeypatch)
    original = module.ProcessRegistry()
    def unit_launch(session, command, suffix, label):
        # Select this test's native-unit transport before launch, rather than
        # layering a fake unit on the independently contained Linux helper.
        session.systemd_unit = 'hermes-worker-' + suffix + '.scope'
        return [module._find_shell(), '-lic', 'set +m; ' + command]
    monkeypatch.setattr(original, '_scope_argv', unit_launch)
    monkeypatch.setattr(module, 'process_registry', original)
    release, effects, pidfile = tmp_path / 'release', tmp_path / 'old-effect', tmp_path / 'child-pid'
    finish = tmp_path / 'wrapper-finish'
    child_code = (f'from pathlib import Path; import time,sys; p=Path({str(release)!r}); end=time.monotonic()+60\n'
                  'while not p.exists() and time.monotonic()<end: time.sleep(.01)\n'
                  'if not p.exists(): sys.exit(0)\n'
                  f'Path({str(effects)!r}).write_text("old effect")')
    wrapper_code = ('from pathlib import Path; import subprocess,sys,time; '
                    f'child=subprocess.Popen([sys.executable,"-c",{child_code!r}], '
                    'start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
                    f'Path({str(pidfile)!r}).write_text(str(child.pid)); p=Path({str(finish)!r})\n'
                    'while not p.exists(): time.sleep(.01)\nprint("completed output")')
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        own = original.spawn_local(shlex.join([sys.executable, '-c', wrapper_code]), cwd=str(tmp_path),
            task_id='s', owner_task_id='s', session_key='route', persist_on_release=True)
    finally:
        clear_session_vars(tokens)
    own.systemd_unit = unit = 'hermes-worker-completed-owned.scope'
    original._write_checkpoint()
    neighbors = []
    child = None
    attempts = []
    try:
        db.create_session('foreign-parent', source='cli')
        for parent, route in [('foreign-parent', 'route'), ('s', 'foreign-route')]:
            tokens = set_session_vars(session_id=parent, session_key=route)
            try:
                neighbor = original.spawn_local(shlex.join([sys.executable, '-c',
                    'import threading; threading.Event().wait(60)']), cwd=str(tmp_path),
                    task_id='s', owner_task_id='s', session_key=route, persist_on_release=True)
            finally:
                clear_session_vars(tokens)
            neighbor.systemd_unit = 'hermes-worker-foreign-' + neighbor.id + '.scope'
            neighbors.append(neighbor)
        finish.touch()
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        child = psutil.Process(int(pidfile.read_text()))
        child_birth = child.create_time()
        history = own.exit_code, own.completion_reason, own.output_buffer
        assert history[0] == 0 and 'completed output' in history[2]
        assert not own._scope_stop_pending
        # Native pruning and result expiry may not discard unresolved unit authority.
        own.started_at -= module.FINISHED_TTL_SECONDS + 1
        original._prune_if_needed()
        original._write_checkpoint()
        assert own.id in original._finished
        entry = next(row for row in json.loads(module._checkpoint_path().read_text()) if row['session_id'] == own.id)
        assert entry['systemd_unit'] == unit and not entry['_scope_stop_pending']
        if expired_result:
            from hermes_constants import get_hermes_home
            (get_hermes_home() / 'logs' / 'process-results' / (own.id + '.json')).unlink()

        def alive():
            try:
                return child.create_time() == child_birth and child.status() not in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}
            except psutil.NoSuchProcess:
                return False

        def stop_unit(name):
            assert name == unit, 'Stop crossed the native parent or route boundary'
            attempts.append(name)
            if refuse:
                return False
            if alive():
                child.terminate()
                try:
                    child.wait(5)
                except psutil.TimeoutExpired:
                    child.kill()
                    child.wait(5)
            return not alive()

        monkeypatch.setattr(module, '_stop_systemd_unit', stop_unit)
        registry = module.ProcessRegistry()
        monkeypatch.setattr(module, 'process_registry', registry)
        registry.recover_from_checkpoint()  # Ordinary startup, no Stop requested.
        assert attempts == [] and alive()
        assert own.id in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        epoch = begin_runtime_epoch(db, instance_id='cold-scope-owner')
        source = authority.sessions['s'].source
        authority = SessionAuthority(authority.runner, profile_id='owned', instance_id='cold-scope-owner', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(source, 'route')
        if refuse:
            with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
                await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
            assert process_stop_pending(db, 's') is not None and alive()
            assert own.id in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
            refuse = False
        await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        assert attempts and set(attempts) == {unit}
        assert process_stop_pending(db, 's') is None and not alive()
        assert all(not neighbor.exited and psutil.Process(neighbor.pid).is_running() for neighbor in neighbors)
        assert own.id not in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        assert history == (own.exit_code, own.completion_reason, own.output_buffer)
        if not expired_result:
            restored = load_completed_results(owns_entry=lambda row: row['id'] == own.id, strict=True)[own.id]
            assert history == (restored.exit_code, restored.completion_reason, restored.output_buffer)
        release.touch()
        assert not effects.exists()
    finally:
        if child is not None:
            try:
                child.kill()
                child.wait(5)
            except psutil.NoSuchProcess:
                pass
        monkeypatch.setattr(module, '_stop_systemd_unit', lambda unit: True)
        original.kill_process(own.id)
        for neighbor in neighbors:
            original.kill_process(neighbor.id)
        db.close()
