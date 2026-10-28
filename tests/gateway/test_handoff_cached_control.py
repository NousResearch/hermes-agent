"""Cached native controls revalidate committed process handoff before signalling."""
from types import SimpleNamespace
import pytest
from tests.gateway.test_process_handoff_stop import Parent, _child, _spawn

@pytest.mark.parametrize('control', ['old-child-control', 'new-parent-control', 'old-child-cleanup', 'new-parent-cleanup', 'selection-race', 'read-refused', 'birth-uncertain', 'pre-stopped-child'])
def test_stale_native_control_cannot_use_pre_handoff_owner(tmp_path, monkeypatch, control):
    from tools import process_registry as module
    from tools.delegate_tool import _unregister_subagent
    from tools.process_registry_handoff import handoff_process
    from agent.managed_worker import WorkerControls
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    parent = Parent()
    sid, _child_agent = _child(parent)
    release = tmp_path / 'finish'
    process = _spawn(registry, tmp_path, sid, 'native-child-session', 'route', logged=True, release=release)
    stale = module.ProcessRegistry()
    stale.recover_from_checkpoint()
    cached = stale._running[process.id]
    try:
        assert cached.parent_session_id == 'native-child-session'
        if control == 'pre-stopped-child':
            _child_agent._hard_interrupt_requested.set()
            assert 'error' in handoff_process(process.id, {'data': 'parent owns stopped job'}, sid)
            assert process.parent_session_id == 'native-child-session' and process.ownership_generation == 0
            assert process.owner_task_id == sid
            assert registry.kill_all(sid, source='session.interrupt') == 1
            return
        if control != 'selection-race':
            assert handoff_process(process.id, {'data':'parent owns work'}, sid)['status'] == 'handed_off'
            assert process.parent_session_id == 's'
        monkeypatch.setattr(module, 'process_registry', stale)
        if control == 'birth-uncertain':
            from tools.process_registry_handoff import ownership_identity
            assert stale.refresh_ownership()
            expected = (*ownership_identity(cached)[:-2], None, None)
            result = stale.kill_process(process.id, source='session.interrupt', expected_ownership=expected)
            assert result['status'] == 'error' and result['error'] == 'process_stop_unknown'
            assert not cached.exited and not stale.is_completion_consumed(process.id)
        elif control in {'old-child-cleanup', 'new-parent-cleanup'}:
            owner = sid if control == 'old-child-cleanup' else 'parent-turn'
            result = stale.kill_all(owner, source='session.interrupt')
            assert result == int(control == 'new-parent-cleanup'), 'cleanup used stale native ownership'
        else:
            controls = WorkerControls.__new__(WorkerControls)
            old_child = control in {'old-child-control', 'selection-race'}
            native_parent = 'native-child-session' if old_child else 's'
            controls.route, controls.session_id = 'route', native_parent
            controls.agent = SimpleNamespace(session_id=native_parent, _process_owner_task_ids={sid if old_child else 'parent-turn'})
            changed = []
            if control == 'selection-race':
                kill = stale.kill_process
                def handed_between_selection_and_signal(pid, **kwargs):
                    with monkeypatch.context() as origin:
                        origin.setattr(module, 'process_registry', registry)
                        assert handoff_process(process.id, {'data': 'parent owns selected job'}, sid)['status'] == 'handed_off'
                    result = kill(pid, **kwargs)
                    changed.append(result['status'])
                    return result
                monkeypatch.setattr(stale, 'kill_process', handed_between_selection_and_signal)
            if control == 'read-refused':
                from pathlib import Path
                read = Path.read_text
                def refuse_read(path, *args, **kwargs):
                    if path == module._checkpoint_path():
                        raise PermissionError('fixture native ownership cannot be read')
                    return read(path, *args, **kwargs)
                with monkeypatch.context() as failure:
                    failure.setattr(Path, 'read_text', refuse_read)
                    with pytest.raises(RuntimeError, match='process_stop_unknown'):
                        controls.stop_processes()
                assert not cached.exited
            else:
                controls.stop_processes()
            if control == 'selection-race':
                assert changed == ['ownership_changed']
                assert process.parent_session_id == 's'
                assert not stale.is_completion_consumed(process.id)
            if old_child:
                assert not cached.exited, 'old child native control killed handed parent job'
            elif control != 'read-refused':
                assert cached.exited, 'parent native control missed handed job'
    finally:
        _unregister_subagent(sid)
        release.touch()
        registry.kill_process(process.id)
