"""A persistent cd that leaves the checkout never locks a coding chat.

Agents are told to keep evidence in their session-artifacts dir and often ``cd`` into it, or into a
sibling repo. Refusing admission on that observed cwd locked the chat for good: only a turn could cd
back, and no turn (human prompt or subagent result delivery) was admitted. The next turn now starts
at the checkout root instead, while probes and admissions leave a possibly running turn's shell alone.
"""
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from tests.gateway.test_coding_workspace_review_terminal_identity import compress, terminal
from tests.gateway.test_session_coding_workspace import create_params, rpc, workspace as _workspace

workspace = _workspace

OUTSIDE = ('artifacts-dir', 'sibling-repo')
INVALID = ('missing', 'outside-link', 'symlink-loop', 'not-directory')


def _leave_checkout(home, source, root, sid, fault, run):
    """cd the live shell out of the checkout (or into a folder that then stops being one)."""
    if fault in OUTSIDE:
        target = home / 'cache' / 'session-artifacts' / sid if fault == 'artifacts-dir' else source
        target.mkdir(parents=True, exist_ok=True)
        observed = run(f'cd -- {shlex.quote(str(target))} && pwd')
    else:
        target = root / 'sub'
        observed = run('mkdir -p sub && cd sub && pwd')
    assert observed['exit_code'] == 0 and observed['output'] == str(target), observed
    if fault in INVALID:
        target.rmdir()
        if fault == 'outside-link':
            target.symlink_to(source, target_is_directory=True)
        elif fault == 'symlink-loop':
            target.symlink_to(target, target_is_directory=True)
        elif fault == 'not-directory':
            target.touch()
    return target


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', OUTSIDE + INVALID)
@pytest.mark.parametrize('boundary', ['verify', 'submit', 'managed', 'turn', 'sequence'])
async def test_cd_out_of_checkout_is_admitted_and_next_turn_starts_at_root(workspace, monkeypatch, fault, boundary):
    from gateway import session_managed_worker as managed
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_authorities import owner_scope
    from gateway.session_contract import SessionRef
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    from tools.terminal_tool import clear_session_cwd, clear_task_env_overrides, get_session_cwd
    home, source, owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    child, live = compress(owner, sid)
    receipt = local_receipt(owner.db, sid)
    policy = restore_policy(receipt['policy'])
    root = binding['cwd']
    monkeypatch.setattr(owner, '_schedule_admitted', lambda ref, row: None)

    def recorded():
        with owner_scope(owner):
            return get_session_cwd(live.route)

    try:
        target = _leave_checkout(home, source, Path(root), sid, fault,
                                 lambda command: terminal(policy, owner, child, live.route, command))
        assert recorded() == str(target)
        if boundary in ('verify', 'sequence'):
            reply = await rpc(connection, 'session.workspace.verify', session_id=sid, cwd=root)
            assert reply.get('result') == {'cwd': root, 'gatewayCwd': root}, reply
            assert recorded() == str(target)  # a probe never yanks a running turn's shell
        if boundary in ('submit', 'sequence'):
            reply = await rpc(connection, 'prompt.submit', session_id=sid, text='hows it going?',
                              submission_id='after-cd')
            assert reply.get('result', {}).get('status') == 'queued', reply
            assert recorded() == str(target)  # queued behind a running turn: still not yanked
        if boundary in ('managed', 'sequence'):
            async def would_spawn(*args):
                return 'worker would execute'
            monkeypatch.setattr(managed, 'execute_managed', would_spawn)
            with owner_scope(owner):
                result = await managed.admit_managed(owner, SessionRef(owner.profile_id, sid),
                                                     {'admission_id': 'after-cd-managed'}, policy)
            assert result == 'worker would execute', result
        if boundary in ('turn', 'sequence'):
            turn = TurnRunner(owner.runner, SimpleNamespace(source=live.source, session_id=child, session_key=live.route))
            monkeypatch.setattr(turn, '_run_sync_scoped', lambda: {'ran_model': True})
            with owner_scope(owner):
                assert turn.run_sync() == {'ran_model': True}
            assert recorded() == root
            again = terminal(policy, owner, child, live.route, 'pwd')
            assert again['exit_code'] == 0 and again['output'] == root, again
        # Recovery never rebinds the chat: the binding and its receipt stay the checkout.
        assert local_receipt(owner.db, sid)['policy'] == receipt['policy']
        assert owner.db.get_session(sid)['cwd'] == root
    finally:
        with owner_scope(owner):
            for key in (child, sid, live.route):
                clear_task_env_overrides(key)
                clear_session_cwd(key)
