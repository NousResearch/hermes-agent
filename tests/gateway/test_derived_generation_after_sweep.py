"""A derived conversation resumed after a housekeeping sweep stays on its new session.

A sweep (``prune_sessions``) retires a derived id: a header-less chat, a hosted-room member, an A2A
context. The next turn moves to the next generation of that id, and the conversation goes on
there. Once the swept id's retry horizon passes, the next turn must still land on that new
generation. If the swept id's fence aged out, the generation walk would fall back to generation 0
and split the conversation across two sessions.
"""
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway import test_api_implicit_session_after_delete as chat
from tests.gateway import test_derived_session_after_delete as derived
from tests.gateway.test_api_implicit_session_after_delete import _OPENING, _app, _content, _post
from tests.gateway.test_derived_session_after_delete import _settle

served = chat.served  # the header-less chat fixture of the delete tests
a2a = derived.a2a  # the A2A forward fixture of the delete tests


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


@pytest.mark.asyncio
async def test_a_swept_caller_named_api_session_id_is_usable_again_past_the_horizon(api, owner, served):
    """``POST /api/sessions`` and ``X-Hermes-Session-Id`` name the id: no door derives it or walks its
    generations, so its sweep marker ages out like any other, and the caller can use the id again."""
    app = _app(api)
    app.router.add_post('/api/sessions', api._handle_create_session)
    auth = {'Authorization': 'Bearer ' + chat._API_KEY}
    async with TestClient(TestServer(app)) as client:
        created = await client.post('/api/sessions', json={'id': 'user-42'}, headers=auth)
        assert created.status == 201, await created.text()
        status, first, _ = await _post(client, _OPENING, session_id='user-42')
        assert status == 200, first
        _sweep(owner, 'user-42')
        _past_the_horizon(owner)
        again = await client.post('/api/sessions', json={'id': 'user-42'}, headers=auth)
        assert again.status == 201, await again.text()
        status, second, landed = await _post(client, _OPENING, session_id='user-42')
        assert status == 200, second
        assert landed == 'user-42'


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['/v1/responses', '/v1/runs'])
async def test_a_chain_resumed_after_a_sweep_never_returns_to_its_swept_root(api, owner, served, path):
    """Any id can root a Responses/Runs chain (here a server-minted one), so its sweep marker is not
    permanent by itself. Once a continuation moved past the swept root to its next generation, a
    later continuation of a response stored on the root goes there too, past the horizon. Both
    continuations use one door, so each door's bind is the one that keeps the fence."""
    from gateway.session_api import implicit_api_session
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_post('/v1/runs', api._handle_runs)
    auth = {'Authorization': 'Bearer ' + chat._API_KEY}
    store = api._current_response_store()
    async with TestClient(TestServer(app)) as client:
        opened = await client.post('/v1/responses', json={'input': 'remember 42'}, headers=auth)
        assert opened.status == 200, await opened.text()
        first = (await opened.json())['id']
        root = store.get(first)['session_id']
        _sweep(owner, root)
        resumed = await client.post(path, json={'input': 'and then?', 'previous_response_id': first}, headers=auth)
        assert resumed.status in (200, 202), await resumed.text()
        successor = implicit_api_session(owner.db, root)
        assert successor != root and owner.db.get_session(successor) is not None, 'no next generation was opened'
        _past_the_horizon(owner)
        forked = await client.post(path, json={'input': 'what number?', 'previous_response_id': first}, headers=auth)
        assert forked.status in (200, 202), await forked.text()
    assert owner.db.get_session(root) is None, 'the continuation went back to the swept root'


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['/v1/responses', '/v1/runs'])
async def test_a_fence_reclaimed_between_chain_lookup_and_bind_never_splits_the_chain(
        api, owner, served, path, monkeypatch):
    """The lookup walks past the swept root to its next generation, then housekeeping reclaims the
    root's sweep marker before that successor's bind. Binding it would leave the walk unpinned: the
    next continuation would find the root free and land there (root, successor, root). The bind
    re-checks the walk in its own write and refuses with a retryable ``chain_moved``; the chain then
    stays on its root."""
    from gateway.platforms import api_server_authority_runs as runs
    from gateway.session_api import api_generation_id
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_post('/v1/runs', api._handle_runs)
    auth = {'Authorization': 'Bearer ' + chat._API_KEY}
    store = api._current_response_store()
    async with TestClient(TestServer(app)) as client:
        opened = await client.post('/v1/responses', json={'input': 'remember 42'}, headers=auth)
        assert opened.status == 200, await opened.text()
        first = (await opened.json())['id']
        root = store.get(first)['session_id']
        successor = api_generation_id(root, 1)
        _sweep(owner, root)
        lookup = runs.continued_session

        async def reclaimed_after_lookup(*args, **kwargs):
            found = await lookup(*args, **kwargs)
            _past_the_horizon(owner)
            return found
        monkeypatch.setattr(runs, 'continued_session', reclaimed_after_lookup)
        raced = await client.post(path, json={'input': 'and then?', 'previous_response_id': first}, headers=auth)
        assert raced.status == 503, await raced.text()
        assert (await raced.json())['error']['code'] == 'chain_moved'
        assert raced.headers.get('Retry-After') == '1'
        assert owner.db.get_session(successor) is None, 'the successor was bound without pinning its walk'
        monkeypatch.setattr(runs, 'continued_session', lookup)
        for turn in ('what number?', 'still there?'):
            again = await client.post(path, json={'input': turn, 'previous_response_id': first}, headers=auth)
            assert again.status in (200, 202), await again.text()
    assert owner.db.get_session(root) is not None, 'the retried continuation left the root'
    assert owner.db.get_session(successor) is None, 'the chain split across the root and its successor'


def test_keep_passed_fences_reports_a_walk_it_could_not_pin(owner):
    """``keep_passed_fences`` pins every fence a walk passed and says whether the walk still holds:
    a passed generation without a marker means the walk would now stop there."""
    import json
    from functools import partial
    from gateway.session_api import api_generation_id
    from hermes_state_mutation_retirement import RETIRED_PREFIX, keep_passed_fences, retire_routes
    derive = partial(api_generation_id, 'root-x')
    passed = ['root-x', derive(1)]
    owner.db._execute_write(lambda conn: retire_routes(conn, passed, expiring=passed))
    assert owner.db._execute_write(lambda conn: keep_passed_fences(conn, derive, derive(2))) is True
    assert all('expires' not in json.loads(owner.db.get_meta(RETIRED_PREFIX + sid)) for sid in passed)
    owner.db._execute_write(lambda conn: conn.execute('DELETE FROM state_meta WHERE key=?', (RETIRED_PREFIX + passed[1],)))
    assert owner.db._execute_write(lambda conn: keep_passed_fences(conn, derive, derive(2))) is False
