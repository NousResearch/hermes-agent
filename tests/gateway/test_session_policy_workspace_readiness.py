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


@pytest.mark.asyncio
async def test_turn_in_removed_workspace_names_the_folder_instead_of_failing_generically(tmp_path, monkeypatch):
    """A turn in a session whose folder is gone settles as a failed, never-run turn that names the
    folder, with its own reason code; it used to escape as an exception every client showed as a
    generic "model request failed" / "something went wrong"."""
    from types import SimpleNamespace

    from agent.error_surface import build_error_surface_from_result
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_results import execution_result
    from tests.gateway.test_session_policy_verbs import _owner

    store, _db, authority, owner, ref = _owner(tmp_path, monkeypatch, None)
    authority.runner.session_authority = authority
    folder = tmp_path / 'project-a'
    try:
        folder.rmdir()
        source = authority.sessions[ref.session_id].source
        ctx = SimpleNamespace(source=source, session_id=ref.session_id, session_key=None)
        token = execution_result.set({})
        try:
            result = TurnRunner(authority.runner, ctx).run_sync()
            captured = execution_result.get()
        finally:
            execution_result.reset(token)
        assert result['failed'] and not result['completed'] and result['api_calls'] == 0, result
        assert result['failure_reason'] == 'workspace_unavailable' and result['failure_retryable'] is False
        assert str(folder) in result['final_response'], result['final_response']
        assert captured['result'] == result  # what the classic CLI viewer reads off the receipt
        surface = build_error_surface_from_result(result)
        assert (surface['layer'], surface['code'], surface['retryable']) == ('gateway', 'workspace_unavailable', False)
    finally:
        await owner.close()
        store.close_all_db_handles()
