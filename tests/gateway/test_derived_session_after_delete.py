"""A hosted-room member or an A2A peer context whose session was deleted continues on a new one.

Both ids are derived (``room_<digest>`` of the room identity, the A2A context's local identity) and
a user-deleted id stays retired for good, so the next dispatch or message moves to the next
generation of that id instead of being refused forever. The deleted id is never recreated, and an
exact retry of an input the deleted session already took is not run again.
"""
from types import SimpleNamespace

import pytest
import pytest_asyncio


def _delete_past_the_horizon(authority, sid):
    from hermes_state_mutation_retirement import prune_retired_tombstones
    assert authority.db.delete_session(sid)
    authority.sessions.pop(sid, None)
    authority.db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))


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
    assert owner.db.get_session(deleted) is None, 'the deleted member session was recreated'
    assert owner.db.get_session(rebound)['title'] == f'Group: {dispatch.room_id}'
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
    runner.adapters, runner.session_store, runner._session_db, runner._draining = {}, store, store._db, False
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
    authority.db.close()


def _settle(authority, sid):
    """The context's queued input ran to completion (this authority schedules nothing)."""
    authority.db.end_session(sid, 'test')
    authority.db._execute_write(lambda c: c.execute(
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
    assert authority.db.get_session(deleted) is None, 'the deleted context session was recreated'
    again = await forward('third', 'still there?')
    assert again['result']['session_id'] == moved


@pytest.mark.asyncio
async def test_an_a2a_exact_retry_after_a_delete_is_not_run_again(a2a):
    authority, forward = a2a
    first = await forward('first', 'hello')
    deleted = first['result']['session_id']
    _settle(authority, deleted)
    assert authority.db.delete_session(deleted)  # within the retry horizon
    authority.sessions.pop(deleted, None)
    retry = await forward('first', 'hello')
    assert 'result' not in retry, retry
    rows = authority.db._read_all("SELECT target_session_id FROM session_admissions")
    assert rows == [], f'the retried input was admitted again: {rows}'
