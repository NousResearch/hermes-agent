"""A removed workspace keeps a local chat readable and movable; only a turn refuses (N87)."""
import pytest


@pytest.mark.asyncio
async def test_removed_workspace_reads_and_moves_but_refuses_a_turn(tmp_path, monkeypatch):
    from gateway.session_policy import policy_for_source, policy_scope
    from hermes_state_runtime import RuntimeStoreError
    from tests.gateway.test_session_policy_verbs import _mutation, _owner

    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, None)
    target = tmp_path / 'project-b'
    target.mkdir()
    try:
        (tmp_path / 'project-a').rmdir()  # git worktree remove / deleted scratch folder
        resumed = await owner.dispatch({'id': 1, 'method': 'session.resume',
                                        'params': {'session_id': ref.session_id}})
        assert 'result' in resumed, resumed
        policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
        with pytest.raises(RuntimeStoreError, match='workspace_unavailable'):
            with policy_scope(policy, authority=authority):
                pass
        moved = await owner.dispatch(_mutation(db, ref, 'mv', 'workspace', {'cwd': str(target)}))
        assert moved['result']['cwd'] == str(target), moved
    finally:
        await owner.close()
        store.close_all_db_handles()
