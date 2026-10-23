"""Exercise live route/physical terminal identity across real compression."""
import json
import shlex
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_coding_workspace import create_params, rpc, workspace as _workspace

workspace = _workspace


def compress(owner, sid):
    from tests.gateway.test_local_resume_compression_child import _compress
    live = owner.sessions[sid]
    child = sid + '-compressed'
    owner.runner.session_store._record_gateway_session_peer(sid, live.route, live.source)
    _compress(owner.db, sid, child, owner.db.get_session(sid)['source'])
    assert owner.runner.session_store.advance_compression_session(live.route, sid, child) is not None
    from gateway.session_contract import SessionRef
    assert owner.physical_target(SessionRef(owner.profile_id, sid)) == child
    return child, live


def terminal(policy, owner, task_id, route, command, **kwargs):
    from gateway.session_authorities import owner_scope
    from gateway.session_policy import policy_scope
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.terminal_tool import terminal_tool
    with owner_scope(owner), policy_scope(policy, authority=owner):
        token = set_current_session_key(route)
        try:
            return json.loads(terminal_tool(command, task_id=task_id, timeout=10, **kwargs))
        finally:
            reset_current_session_key(token)


@pytest.mark.asyncio
@pytest.mark.parametrize('retarget', ['route-cd', 'physical-override'])
@pytest.mark.parametrize('boundary', ['verify', 'submit', 'turn', 'managed'])
async def test_live_terminal_retarget_after_compression_is_refused(workspace, monkeypatch, retarget, boundary):
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_authorities import owner_scope
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    from hermes_state_runtime import list_session_admissions
    from tools.terminal_tool import clear_session_cwd, clear_task_env_overrides, get_session_cwd, register_task_env_overrides
    _home, source, owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    child, live = compress(owner, sid)
    policy = restore_policy(local_receipt(owner.db, sid)['policy'])
    monkeypatch.setattr(owner, '_schedule_admitted', lambda ref, row: None)
    try:
        if retarget == 'physical-override':
            with owner_scope(owner):
                register_task_env_overrides(child, {'cwd': str(source)})
            observed = terminal(policy, owner, child, live.route, 'pwd')
        else:
            observed = terminal(policy, owner, child, live.route, 'cd -- ' + shlex.quote(str(source)) + '; pwd')
        assert observed['exit_code'] == 0 and observed['output'].removesuffix('\n') == str(source), observed
        with owner_scope(owner):
            assert get_session_cwd(live.route) == str(source)
            assert get_session_cwd(sid) is None  # the rejected implementation checks only this unused key
            if retarget == 'physical-override':
                # Keep physical-key coverage independent from the separate route-cd regression.
                clear_session_cwd(live.route)
                assert get_session_cwd(child) == str(source)
        if boundary == 'verify':
            reply = await rpc(connection, 'session.workspace.verify', session_id=sid)
            assert reply.get('error', {}).get('message') == 'workspace_unavailable', reply
        elif boundary == 'submit':
            reply = await rpc(connection, 'prompt.submit', session_id=sid, text='plain prompt', submission_id='retarget')
            assert reply.get('error', {}).get('message') == 'workspace_unavailable', reply
            assert list_session_admissions(owner.db, session_id=sid) == []
        elif boundary == 'turn':
            turn = TurnRunner(owner.runner, SimpleNamespace(source=live.source, session_id=child, session_key=live.route))
            monkeypatch.setattr(turn, '_run_sync_scoped', lambda: {'ran_model': True})
            with owner_scope(owner):
                result = turn.run_sync()
            assert result.get('ran_model') is not True, result
            assert result.get('failed') is True, result
        else:
            from gateway import session_managed_worker as managed
            from gateway.session_contract import SessionRef
            async def would_spawn(*args):
                return 'worker would execute'
            monkeypatch.setattr(managed, 'execute_managed', would_spawn)
            with owner_scope(owner):
                result = await managed.admit_managed(owner, SessionRef(owner.profile_id, sid),
                                                     {'admission_id': 'managed-retarget'}, policy)
            assert result != 'worker would execute', result
            assert owner.pending_results['managed-retarget']['result']['failed'] is True
    finally:
        with owner_scope(owner):
            for key in (child, sid, live.route):
                clear_task_env_overrides(key)
                clear_session_cwd(key)


@pytest.mark.asyncio
async def test_normal_turn_clears_its_historical_mount_override_before_local_execution(workspace, monkeypatch):
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_authorities import owner_scope
    from tools.terminal_tool import clear_session_cwd, clear_task_env_overrides, register_task_env_overrides, terminal_tool
    _home, source, owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    child, live = compress(owner, sid)
    turn = TurnRunner(owner.runner, SimpleNamespace(source=live.source, session_id=child, session_key=live.route))
    # Real run_sync calls bind_turn_workspace, whose local unbind intentionally retires an old owned mount.
    monkeypatch.setattr(turn, '_run_sync_scoped', lambda: json.loads(terminal_tool('pwd', task_id=child, timeout=10)))
    try:
        with owner_scope(owner):
            register_task_env_overrides(child, {'cwd': str(source), 'workspace_key': 'historical-owned-mount'})
            result = turn.run_sync()
        assert result.get('output', '').removesuffix('\n') == binding['cwd'], result
    finally:
        with owner_scope(owner):
            for key in (child, sid, live.route):
                clear_task_env_overrides(key)
                clear_session_cwd(key)


@pytest.mark.asyncio
async def test_command_workdir_and_branch_switch_do_not_retarget_live_checkout(workspace, monkeypatch):
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_authorities import owner_scope
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    from tests.gateway.test_session_coding_workspace import git
    from tools.terminal_tool import clear_session_cwd, get_session_cwd
    _home, source, owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    child, live = compress(owner, sid)
    policy = restore_policy(local_receipt(owner.db, sid)['policy'])
    try:
        result = terminal(policy, owner, child, live.route, 'pwd', workdir=str(source))
        assert result['exit_code'] == 0 and result['output'].removesuffix('\n') == str(source), result
        with owner_scope(owner):
            assert get_session_cwd(live.route) is None
        git(binding['cwd'], 'switch', '-c', 'permitted-review-branch')
        verified = await rpc(connection, 'session.workspace.verify', session_id=sid)
        assert verified.get('result', {}).get('cwd') == binding['cwd'], verified
        with owner_scope(owner):
            assert get_session_cwd(live.route) == binding['cwd']
            assert get_session_cwd(sid) is None  # a probe now follows the actual turn, not another task
        turn = TurnRunner(owner.runner, SimpleNamespace(source=live.source, session_id=child, session_key=live.route))
        monkeypatch.setattr(turn, '_run_sync_scoped', lambda: {'ran_model': True})
        with owner_scope(owner):
            assert turn.run_sync() == {'ran_model': True}
        monkeypatch.setattr(owner, '_schedule_admitted', lambda ref, row: None)
        submitted = await rpc(connection, 'prompt.submit', session_id=sid, text='plain prompt', submission_id='permitted')
        assert submitted.get('result', {}).get('status') == 'queued', submitted
    finally:
        with owner_scope(owner):
            for key in (child, sid, live.route):
                clear_session_cwd(key)
