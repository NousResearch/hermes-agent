"""Persistent cd stays usable inside the pinned checkout without moving its binding."""
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from tests.gateway.test_coding_workspace_review_terminal_identity import compress, terminal
from tests.gateway.test_session_coding_workspace import create_params, rpc, workspace as _workspace

workspace = _workspace


@pytest.mark.asyncio
@pytest.mark.parametrize('relative', ['.', 'sub', ' white\tspace/line\nbreak/leaf', 'safe-link'])
@pytest.mark.parametrize('boundary', ['verify', 'submit', 'turn', 'managed', 'sequence'])
async def test_contained_persistent_cd_survives_each_execution_boundary(workspace, monkeypatch, relative, boundary):
    from gateway import session_managed_worker as managed
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_authorities import owner_scope
    from gateway.session_contract import SessionRef
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    from tools.terminal_tool import clear_session_cwd, clear_task_env_overrides, get_session_cwd, resolve_task_overrides
    home, _source, owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    child, live = compress(owner, sid)
    receipt = local_receipt(owner.db, sid)
    policy = restore_policy(receipt['policy'])
    root = Path(binding['cwd'])
    target = root / relative
    if relative == 'safe-link':
        (root / 'real-directory').mkdir()
        target.symlink_to(root / 'real-directory', target_is_directory=True)
        target = target.resolve(strict=True)  # the real terminal records its physical cwd
    before_config = (home / 'config.yaml').read_bytes()
    monkeypatch.setattr(owner, '_schedule_admitted', lambda ref, row: None)
    try:
        quoted = shlex.quote(relative)
        observed = terminal(policy, owner, child, live.route, f'mkdir -p -- {quoted} && cd -- {quoted} && pwd')
        assert observed['exit_code'] == 0, observed
        with owner_scope(owner):
            assert get_session_cwd(live.route) == str(target)
            before_overrides = resolve_task_overrides(child).copy()
        if boundary in ('verify', 'sequence'):
            reply = await rpc(connection, 'session.workspace.verify', session_id=sid, cwd=binding['cwd'])
            assert reply.get('result') == {'cwd': binding['cwd'], 'gatewayCwd': binding['cwd']}, reply
        if boundary in ('submit', 'sequence'):
            reply = await rpc(connection, 'prompt.submit', session_id=sid, text='continue in this checkout',
                              submission_id='contained')
            assert reply.get('result', {}).get('status') == 'queued', reply
        if boundary in ('turn', 'sequence'):
            turn = TurnRunner(owner.runner, SimpleNamespace(source=live.source, session_id=child, session_key=live.route))
            monkeypatch.setattr(turn, '_run_sync_scoped', lambda: {'ran_model': True})
            with owner_scope(owner):
                assert turn.run_sync() == {'ran_model': True}
        if boundary in ('managed', 'sequence'):
            async def would_spawn(*args):
                return 'worker would execute'
            monkeypatch.setattr(managed, 'execute_managed', would_spawn)
            with owner_scope(owner):
                result = await managed.admit_managed(owner, SessionRef(owner.profile_id, sid),
                                                     {'admission_id': 'contained-managed'}, policy)
            assert result == 'worker would execute', result
        # A probe or turn must not silently reset the user's persistent cd or retarget the root.
        with owner_scope(owner):
            assert get_session_cwd(live.route) == str(target)
            assert resolve_task_overrides(child) == before_overrides
        again = terminal(policy, owner, child, live.route, 'pwd')
        assert again['exit_code'] == 0 and again['output'] == str(target), again
        assert local_receipt(owner.db, sid)['policy'] == receipt['policy']
        assert owner.db.get_session(sid)['cwd'] == binding['cwd']
        assert policy.cwd == json.loads(policy.terminal_json)['TERMINAL_CWD'] == binding['cwd']
        assert (home / 'config.yaml').read_bytes() == before_config
    finally:
        with owner_scope(owner):
            for key in (child, sid, live.route):
                clear_task_env_overrides(key)
                clear_session_cwd(key)


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['missing', 'outside-link', 'symlink-loop', 'not-directory', 'contained-task-override'])
@pytest.mark.parametrize('boundary', ['verify', 'submit', 'turn', 'managed'])
async def test_containment_never_adopts_invalid_cd_or_configured_task_retarget(workspace, monkeypatch, fault, boundary):
    from gateway import session_managed_worker as managed
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_authorities import owner_scope
    from gateway.session_contract import SessionRef
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    from hermes_state_runtime import list_session_admissions
    from tools.terminal_tool import (
        clear_session_cwd, clear_task_env_overrides, get_session_cwd, register_task_env_overrides,
    )
    _home, source, owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    child, live = compress(owner, sid)
    policy = restore_policy(local_receipt(owner.db, sid)['policy'])
    target = Path(binding['cwd']) / 'sub'
    monkeypatch.setattr(owner, '_schedule_admitted', lambda ref, row: None)
    try:
        observed = terminal(policy, owner, child, live.route, 'mkdir -p sub && cd sub && pwd')
        assert observed['exit_code'] == 0 and observed['output'] == str(target), observed
        with owner_scope(owner):
            assert get_session_cwd(live.route) == str(target)
        if fault == 'contained-task-override':
            # A configured override is identity, not observed cd state, even inside the checkout.
            with owner_scope(owner):
                register_task_env_overrides(child, {'cwd': str(target)})
        else:
            target.rmdir()
            if fault == 'outside-link':
                target.symlink_to(source, target_is_directory=True)
            elif fault == 'symlink-loop':
                target.symlink_to(target, target_is_directory=True)
            elif fault == 'not-directory':
                target.touch()
        if boundary == 'verify':
            reply = await rpc(connection, 'session.workspace.verify', session_id=sid)
            assert reply.get('error', {}).get('message') == 'workspace_unavailable', reply
        elif boundary == 'submit':
            reply = await rpc(connection, 'prompt.submit', session_id=sid, text='must not run', submission_id='invalid-cd')
            assert reply.get('error', {}).get('message') == 'workspace_unavailable', reply
            assert list_session_admissions(owner.db, session_id=sid) == []
        elif boundary == 'turn':
            turn = TurnRunner(owner.runner, SimpleNamespace(source=live.source, session_id=child, session_key=live.route))
            monkeypatch.setattr(turn, '_run_sync_scoped', lambda: {'ran_model': True})
            with owner_scope(owner):
                result = turn.run_sync()
            assert result.get('failed') is True and result.get('ran_model') is not True, result
        else:
            async def would_spawn(*args):
                return 'worker would execute'
            monkeypatch.setattr(managed, 'execute_managed', would_spawn)
            with owner_scope(owner):
                result = await managed.admit_managed(owner, SessionRef(owner.profile_id, sid),
                                                     {'admission_id': 'invalid-cd-managed'}, policy)
            assert result != 'worker would execute', result
            assert owner.pending_results['invalid-cd-managed']['result']['failed'] is True
        with owner_scope(owner):
            assert get_session_cwd(live.route) == str(target)  # refusal never silently rebinds the route
        assert local_receipt(owner.db, sid)['policy']['cwd'] == binding['cwd']
    finally:
        with owner_scope(owner):
            for key in (child, sid, live.route):
                clear_task_env_overrides(key)
                clear_session_cwd(key)
