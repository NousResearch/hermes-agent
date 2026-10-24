"""A workspace that is going away takes no new work (#134889 TUI-27).

A launcher that removes a session's worktree tells the owner first: the tree is marked retiring,
so a submit that lands while ``git worktree remove`` runs is refused by name instead of being
admitted into a directory that is about to disappear (the prompt is taken and refused after the
fact). The mark expires: a launcher that dies mid-removal never wedges the session.
"""
import time
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_a_retiring_workspace_refuses_a_new_turn(tmp_path, monkeypatch):
    from gateway.session_contract import Submission
    from hermes_state_runtime import RuntimeStoreError
    from tests.gateway.test_session_policy_verbs import _owner

    async def handle(event):
        return 'ok'

    store, _db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    workspace = str(tmp_path / 'project-a')
    try:
        marked = await owner.dispatch({'id': 1, 'method': 'session.workspace.retire', 'params': {
            'session_id': ref.session_id, 'path': workspace, 'retiring': True}})
        assert marked['result'] == {'session_id': ref.session_id, 'retiring': True}, marked
        with pytest.raises(RuntimeStoreError, match='workspace_retiring'):
            await authority.submit(owner.actor, Submission('turn-1', ref, {'text': 'one'}, 'queue'))
        # The mark comes down once the removal finished: the session takes work again.
        cleared = await owner.dispatch({'id': 2, 'method': 'session.workspace.retire', 'params': {
            'session_id': ref.session_id, 'path': workspace, 'retiring': False}})
        assert cleared['result'] == {'session_id': ref.session_id, 'retiring': False}, cleared
        queued = await authority.submit(owner.actor, Submission('turn-2', ref, {'text': 'two'}, 'queue'))
        assert queued.status == 'queued'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_only_the_sessions_own_workspace_is_marked(tmp_path, monkeypatch):
    from tests.gateway.test_session_policy_verbs import _owner

    store, _db, authority, owner, ref = _owner(tmp_path, monkeypatch, None)
    try:
        other = await owner.dispatch({'id': 1, 'method': 'session.workspace.retire', 'params': {
            'session_id': ref.session_id, 'path': str(tmp_path / 'somewhere-else'), 'retiring': True}})
        assert other['error']['data']['reason'] == 'workspace_mismatch', other
        assert not authority.retiring_workspaces
    finally:
        await owner.close()
        store.close_all_db_handles()


def test_an_expired_mark_never_wedges_a_session():
    from gateway import session_workspace_retiring as retiring

    authority = SimpleNamespace(retiring_workspaces={})
    retiring.mark(authority, 's1', retiring=True)
    assert retiring.retiring(authority, 's1') is True
    authority.retiring_workspaces['s1'] = time.monotonic() - 1  # a launcher that died mid-removal
    assert retiring.retiring(authority, 's1') is False
    assert 's1' not in authority.retiring_workspaces
