"""A derived conversation resumed after a housekeeping sweep stays on its new session.

A sweep (``prune_sessions``) retires a derived id: a header-less chat, a hosted-room member, an A2A
context. The next turn moves to the next generation of that id, and the conversation goes on
there. Once the swept id's retry horizon passes, the next turn must still land on that new
generation. If the swept id's fence aged out, the generation walk would fall back to generation 0
and split the conversation across two sessions.
"""
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_implicit_session_after_delete import (
    _OPENING, _app, _content, _post, served)
from tests.gateway.test_derived_session_after_delete import _settle, a2a


def _sweep(authority, sid):
    """A housekeeping prune retires the idle, ended session *sid*."""
    authority.db.end_session(sid, 'test')
    authority.db.prune_sessions(older_than_days=None)
    assert authority.db.get_session(sid) is None, 'the sweep kept the session'
    authority.sessions.pop(sid, None)


def _past_the_horizon(authority):
    from hermes_state_mutation_retirement import prune_retired_tombstones
    authority.db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))


@pytest.mark.asyncio
async def test_a_header_less_chat_resumed_after_a_sweep_keeps_its_new_session(api, owner, served):
    async with TestClient(TestServer(_app(api))) as client:
        status, first, swept = await _post(client, _OPENING)
        assert status == 200, first
        _sweep(owner, swept)
        status, opened, resumed = await _post(client, _OPENING)
        assert status == 200, opened
        assert resumed != swept
        _past_the_horizon(owner)
        follow_up = [*_OPENING, {'role': 'assistant', 'content': _content(opened)}, {'role': 'user', 'content': 'next'}]
        status, second, landed = await _post(client, follow_up)
        assert status == 200, second
        assert landed == resumed, 'the next turn left the conversation for the swept generation'


@pytest.mark.asyncio
async def test_a_hosted_room_member_resumed_after_a_sweep_keeps_its_new_session(api, owner):
    from gateway.hosted_room_peer import HostedMemberDispatch
    from tests.gateway.test_hosted_room_execution_policy import _dispatch, _policy
    dispatch = _dispatch(_policy())
    swept = await api._ensure_hosted_member_session(dispatch)
    _sweep(owner, swept)

    def task(n):
        return HostedMemberDispatch.from_mapping(dict(dispatch.as_mapping(), task_id=f'task-{n}', execution_generation=n))
    resumed = await api._ensure_hosted_member_session(task(2))
    assert resumed != swept
    _past_the_horizon(owner)
    assert await api._ensure_hosted_member_session(task(3)) == resumed, 'the room member fell back to the swept session'


@pytest.mark.asyncio
async def test_an_a2a_context_resumed_after_a_sweep_keeps_its_new_session(a2a):
    authority, forward = a2a
    first = await forward('first', 'hello')
    swept = first['result']['session_id']
    _settle(authority, swept)
    _sweep(authority, swept)
    later = await forward('second', 'are you there?')
    assert 'result' in later, later
    resumed = later['result']['session_id']
    assert resumed != swept
    _past_the_horizon(authority)
    again = await forward('third', 'still there?')
    assert 'result' in again, again
    assert again['result']['session_id'] == resumed, 'the A2A context fell back to the swept session'


def test_an_ordinary_swept_session_still_ages_out(owner):
    """Housekeeping on a cron-heavy store: an id no door derives keeps an expiring marker."""
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    owner.db.create_session('cron_job_run_1', source='cron')
    _sweep(SimpleNamespace(db=owner.db, sessions={}), 'cron_job_run_1')
    assert owner.db.get_meta(RETIRED_PREFIX + 'cron_job_run_1') is not None
    _past_the_horizon(owner)
    assert owner.db.get_meta(RETIRED_PREFIX + 'cron_job_run_1') is None
