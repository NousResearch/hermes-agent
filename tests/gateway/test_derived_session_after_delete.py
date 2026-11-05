"""A hosted-room member or an A2A peer context whose session was deleted continues on a new one.

Both ids are derived (``room_<digest>`` of the room identity, the A2A context's local identity) and
a user-deleted id stays retired for good, so the next dispatch or message moves to the next
generation of that id instead of being refused forever. The deleted id is never recreated, and an
exact retry of an input the deleted session already took is not run again.
"""
from types import SimpleNamespace

import pytest
import pytest_asyncio
from tests.fixtures.storage_holder import raw_holder, reclaim_tombstones


def _delete(authority, sid):
    """Delete *sid* as the delete door does; returns the removed ids."""
    from gateway.session_ingress_media import native_inputs_root
    from session_retention import after_sessions_removed
    store = authority.persistence
    removed = store.sessions.delete(sid, media_root=str(native_inputs_root()))
    after_sessions_removed(store, store.scope.profile_dir, removed, reason='delete', primary=sid)
    return removed


def _delete_past_the_horizon(authority, sid):
    assert _delete(authority, sid)
    authority.sessions.pop(sid, None)
    reclaim_tombstones(authority.persistence)


@pytest.mark.asyncio
async def test_a_hosted_room_member_rebinds_on_a_new_session_after_a_delete(api, owner):
    from gateway.hosted_room_peer import HostedMemberDispatch
    from gateway.session_api import restore_api_session
    from tests.gateway.test_hosted_room_execution_policy import _dispatch, _policy
    dispatch = _dispatch(_policy())
    deleted = await api._ensure_hosted_member_session(dispatch)
    _delete_past_the_horizon(owner, deleted)
    # The room's next task to the same member.
    later = HostedMemberDispatch.from_mapping(dict(dispatch.as_mapping(), task_id='task-2', execution_generation=2))
    rebound = await api._ensure_hosted_member_session(later)
    assert rebound != deleted
    assert owner.persistence.sessions.get(deleted) is None, 'the deleted member session was recreated'
    assert owner.persistence.sessions.get(rebound).title == f'Group: {dispatch.room_id}'
    assert await api._ensure_hosted_member_session(later) == rebound
    owner.sessions.pop(rebound, None)  # a restart restores the binding from its durable receipt
    assert restore_api_session(owner, rebound).session_id == rebound


@pytest_asyncio.fixture
async def a2a(tmp_path, monkeypatch):
    """``(authority, forward)``: a real authority and its ``a2a.forward`` over a producer connection."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config',
                        lambda: {'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: 'fixture')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = run.GatewayRunner.__new__(run.GatewayRunner)
    runner.adapters, runner.session_store, runner._session_db, runner._draining = {}, store, store.persistence, False
    authority = await initialize_session_authority(runner, profile_id='target', instance_id='test')
    authority._schedule = lambda ref: None

    async def forward(input_id, text):
        conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'producer'})
        try:
            return await conn.dispatch({'id': 1, 'method': 'a2a.forward', 'params': dict(
                agent='dev', tenant='team', peer='alice', context_id='ctx-a', input_id=input_id, text=text)})
        finally:
            await conn.close()
    yield authority, forward
    store.close_all_db_handles()


def _settle(authority, sid):
    """The context's queued input ran to completion (this authority schedules nothing)."""
    authority.persistence.sessions.end(sid, 'test')
    raw_holder(authority.persistence)._execute_write(lambda c: c.execute(
        "UPDATE session_admissions SET status='terminal', outcome='completed' WHERE target_session_id=?", (sid,)))


@pytest.mark.asyncio
async def test_an_a2a_peer_context_continues_on_a_new_session_after_a_delete(a2a):
    authority, forward = a2a
    first = await forward('first', 'hello')
    assert 'result' in first, first
    deleted = first['result']['session_id']
    _settle(authority, deleted)
    _delete_past_the_horizon(authority, deleted)
    later = await forward('second', 'are you there?')
    assert 'result' in later, later
    moved = later['result']['session_id']
    assert moved != deleted
    assert authority.persistence.sessions.get(deleted) is None, 'the deleted context session was recreated'
    again = await forward('third', 'still there?')
    assert again['result']['session_id'] == moved


@pytest.mark.asyncio
async def test_an_a2a_exact_retry_after_a_delete_is_not_run_again(a2a):
    authority, forward = a2a
    first = await forward('first', 'hello')
    deleted = first['result']['session_id']
    _settle(authority, deleted)
    assert _delete(authority, deleted)  # within the retry horizon
    authority.sessions.pop(deleted, None)
    retry = await forward('first', 'hello')
    assert 'result' not in retry, retry
    rows = raw_holder(authority.persistence)._read_all("SELECT target_session_id FROM session_admissions")
    assert rows == [], f'the retried input was admitted again: {rows}'
