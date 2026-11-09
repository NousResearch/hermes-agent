"""A canonical model pick switches FROM the route the session runs now, as one tuple.

Main's switch read provider, model, endpoint and key together from the live runtime. A following
session's route is the profile's current one, so pairing its frozen creation endpoint with the
current key sends that key to a host the profile left. Real SessionStore/SessionAuthority, a real
config.yaml and loopback catalog endpoints; only models.dev metadata is held offline.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from tests.gateway.test_session_follow_profile_config import _create, _owner, _route

MODELS = ('original', 'current', 'picked')


class _Catalog(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.received.append(self.headers.get('Authorization'))
        raw = json.dumps({'data': [{'id': name} for name in MODELS]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def endpoints():
    servers = [ThreadingHTTPServer(('127.0.0.1', 0), _Catalog) for _ in range(2)]
    for server in servers:
        server.received = []
        threading.Thread(target=server.serve_forever, daemon=True).start()
    yield [(server, f'http://127.0.0.1:{server.server_port}/v1') for server in servers]
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture
def home(monkeypatch):
    from agent import models_dev
    from gateway import run
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    # gateway.run binds its home at first import; the loader itself is real.
    monkeypatch.setattr(run, '_hermes_home', home)
    catalog = {'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'tool_call': True, 'cost': {'input': 0, 'output': 0},
               'limit': {'context': 100000, 'output': 4000}} for name in MODELS}}}
    monkeypatch.setattr(models_dev, 'fetch_models_dev', lambda *args, **kwargs: catalog)
    return home


def _configure(home, model, url, key):
    (home / 'config.yaml').write_text(json.dumps({'model': {
        'provider': 'custom', 'default': model, 'base_url': url, 'api_key': key}}), encoding='utf-8')


async def _pick(owner, authority, sid, payload, request_id='model-pick'):
    row = authority.db.get_session(sid)
    return await owner.dispatch({'id': 'pick', 'method': 'session.mutate', 'params': {
        'session_id': sid, 'request_id': request_id, 'expected_revision': row['runtime_revision'],
        'expected_generation': row['runtime_generation'], 'operation': 'model', 'payload': payload}})


def _bind_resolver(runner):
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    runner._peek_session_state = lambda key: None
    runner._resolve_session_agent_runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime.__get__(runner)


@pytest.mark.asyncio
@pytest.mark.parametrize('rotate', [False, True])
async def test_following_pick_validates_on_the_current_endpoint_with_its_own_key(home, endpoints, rotate):
    (a, url_a), (b, url_b) = endpoints
    _configure(home, 'original', url_a, 'key-a')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        live = await _create(owner, authority, home, 'follow')
        sid = next(sid for sid, value in authority.sessions.items() if value is live)
        url, key = (url_b, 'key-b') if rotate else (url_a, 'key-a')
        _configure(home, 'current', url, key)
        assert _route(runner, live) == ('current', url, key)
        a.received.clear()
        b.received.clear()

        response = await _pick(owner, authority, sid, {'model': 'picked', 'provider': 'custom'})

        assert 'result' in response, response
        if rotate:
            # The retired endpoint never sees the current endpoint's credential.
            assert a.received == []
            assert b.received and set(b.received) == {'Bearer key-b'}
        assert _route(runner, live) == ('picked', url, key)
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_superseded_pick_takes_its_launch_key_with_it(home):
    """The key a keyed alias pick bound authenticates the alias's endpoint. Once a profile change
    supersedes the pick and the followed route moves to another endpoint, that endpoint gets its own
    configured key, never the alias's."""
    def profile(model, port, key):
        (home / 'config.yaml').write_text(json.dumps({
            'model': {'provider': 'custom', 'default': model, 'base_url': f'http://127.0.0.1:{port}/v1',
                      'api_key': key},
            'model_aliases': {'picked': {'model': 'alias-model', 'provider': 'custom',
                                         'base_url': 'http://127.0.0.1:2/v1', 'api_key': 'key-alias'}},
        }), encoding='utf-8')
    profile('original', 1, 'key-a')
    store, runner, authority, owner = _owner(home)
    try:
        live = await _create(owner, authority, home, 'picked', model='picked')
        assert _route(runner, live) == ('alias-model', 'http://127.0.0.1:2/v1', 'key-alias')

        profile('current', 3, 'key-b')

        assert _route(runner, live) == ('current', 'http://127.0.0.1:3/v1', 'key-b')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_bare_pick_on_a_named_provider_keeps_its_launch_key(home, endpoints):
    """A named ``providers:`` entry resolves to the bare ``custom`` class; switching from that class
    read a same-endpoint pick as a route change and dropped the key the session was launched with."""
    (_, url), _ = endpoints
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'loopback', 'default': 'original'},
        'providers': {'loopback': {'name': 'loopback', 'base_url': url}},
    }), encoding='utf-8')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        created = await owner.dispatch({'id': 'create', 'method': 'session.create', 'params': {
            'request_id': 'launch', 'source': 'cli', 'cwd': str(home), 'toolsets': [],
            'provider': 'loopback', 'api_key': 'key-launch'}})
        assert 'result' in created, created
        sid = created['result']['session_id']
        live = authority.sessions[sid]
        assert _route(runner, live) == ('original', url, 'key-launch')

        response = await _pick(owner, authority, sid, {'model': 'picked'})

        assert 'result' in response, response
        assert _route(runner, live) == ('picked', url, 'key-launch')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_pick_never_switches_from_the_fallback_route(home, endpoints):
    """While the session's own route cannot authenticate, a turn lands on its fallback; a pick still
    switches from the session's own route, so it never commits the fallback's endpoint."""
    _, (fallback, url) = endpoints
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'default': 'original'},
        'fallback_providers': [{'provider': 'custom', 'model': 'current', 'base_url': url, 'api_key': 'key-fallback'}],
    }), encoding='utf-8')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        live = await _create(owner, authority, home, 'bare-with-fallback')
        sid = next(sid for sid, value in authority.sessions.items() if value is live)
        assert _route(runner, live) == ('current', url, 'key-fallback')
        fallback.received.clear()

        response = await _pick(owner, authority, sid, {'model': 'picked', 'provider': 'custom'})

        assert response.get('error', {}).get('message') == 'model_resolution_failed', response
        assert fallback.received == []
        assert _route(runner, live) == ('current', url, 'key-fallback')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_route_that_cannot_authenticate_does_not_block_switching_away(home, endpoints):
    """#138879: resolving the route being replaced raised AuthError out of ``session.mutate``, so the
    client got ``internal error`` and could never leave a route it cannot use."""
    (_, url), _ = endpoints
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'default': 'original'},
        'providers': {'loopback': {'name': 'loopback', 'base_url': url, 'api_key': 'key-loopback'}},
    }), encoding='utf-8')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        live = await _create(owner, authority, home, 'bare')
        sid = next(sid for sid, value in authority.sessions.items() if value is live)
        from hermes_cli.auth_constants import AuthError
        with pytest.raises(AuthError):
            _route(runner, live)

        response = await _pick(owner, authority, sid, {'model': 'picked', 'provider': 'loopback'})

        assert 'result' in response, response
        model, base_url, key = _route(runner, live)
        assert (model, base_url, key) == ('picked', url, 'key-loopback')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_following_pick_keeps_the_current_endpoints_key_env(home, endpoints):
    """The profile names its key by ``key_env`` and moved from A to B before the pick. The pick lands
    on B, so B's own key keeps authenticating the next turn: the pointer must not be dropped as a
    route change measured from A, the endpoint the session was created on."""
    (a, url_a), (b, url_b) = endpoints
    (home / '.env').write_text('KEY_A=key-a\nKEY_B=key-b\n', encoding='utf-8')

    def profile(url, var):
        (home / 'config.yaml').write_text(json.dumps({'model': {
            'provider': 'custom', 'default': 'original', 'base_url': url, 'key_env': var}}), encoding='utf-8')
    profile(url_a, 'KEY_A')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        live = await _create(owner, authority, home, 'follow-key-env')
        sid = next(sid for sid, value in authority.sessions.items() if value is live)
        profile(url_b, 'KEY_B')
        assert _route(runner, live) == ('original', url_b, 'key-b')
        a.received.clear()
        b.received.clear()

        response = await _pick(owner, authority, sid, {'model': 'picked', 'provider': 'custom'})

        assert 'result' in response, response
        assert a.received == [] and set(b.received) == {'Bearer key-b'}
        assert _route(runner, live) == ('picked', url_b, 'key-b')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_superseded_alias_key_never_reaches_the_next_picks_endpoint(home, endpoints):
    """A following session picks a literal-key alias on A; the profile then moves to B and the user
    picks another model there. The alias's key pointer is bound to A's route: it must not survive
    the pick and authenticate B's next turn."""
    (_, url_a), (b, url_b) = endpoints

    def profile(model, url, key):
        (home / 'config.yaml').write_text(json.dumps({
            'model': {'provider': 'custom', 'default': model, 'base_url': url, 'api_key': key},
            'model_aliases': {'keyed': {'model': 'current', 'provider': 'custom',
                                        'base_url': url_a, 'api_key': 'key-alias'}},
        }), encoding='utf-8')
    profile('original', url_a, 'key-a')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        live = await _create(owner, authority, home, 'follow-alias')
        sid = next(sid for sid, value in authority.sessions.items() if value is live)
        assert 'result' in await _pick(owner, authority, sid, {'model': 'keyed'}, 'alias-pick')
        assert _route(runner, live) == ('current', url_a, 'key-alias')
        profile('current', url_b, 'key-b')  # a new profile target supersedes the alias pick
        assert _route(runner, live) == ('current', url_b, 'key-b')
        b.received.clear()

        response = await _pick(owner, authority, sid, {'model': 'picked', 'provider': 'custom'})

        assert 'result' in response, response
        assert _route(runner, live) == ('picked', url_b, 'key-b')
        assert 'Bearer key-alias' not in b.received
    finally:
        await owner.close()
        store.close_all_db_handles()


async def _launch(owner, home, provider, key):
    created = await owner.dispatch({'id': 'create', 'method': 'session.create', 'params': {
        'request_id': 'launch', 'source': 'cli', 'cwd': str(home), 'toolsets': [],
        'provider': provider, 'api_key': key, 'follow_profile_config': True}})
    assert 'result' in created, created
    return created['result']['session_id']


@pytest.mark.asyncio
@pytest.mark.parametrize('before,after', [('custom', 'vllm'), ('team', 'custom:team')])
async def test_a_respelled_provider_on_the_same_endpoint_keeps_the_launch_key(home, endpoints, before, after):
    """The profile respells its provider (``custom`` -> ``vllm``, ``team`` -> ``custom:team``) without
    moving the endpoint. The route is the same, so the key the session was launched with still
    authenticates it; the profile's own key must not replace it."""
    (a, url), (b, _) = endpoints

    def profile(provider):
        model = {'provider': provider, 'default': 'original'}
        if before == 'custom':
            model.update(base_url=url, api_key='key-profile')
        (home / 'config.yaml').write_text(json.dumps({
            'model': model, 'providers': {'team': {'name': 'team', 'base_url': url, 'api_key': 'key-profile'}},
        }), encoding='utf-8')
    profile(before)
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        sid = await _launch(owner, home, before, 'key-launch')
        live = authority.sessions[sid]
        profile(after)
        assert _route(runner, live) == ('original', url, 'key-launch')
        a.received.clear()

        response = await _pick(owner, authority, sid, {'model': 'picked'})

        assert 'result' in response, response
        # The pick's catalog probe uses a credential of this endpoint; the other server gets nothing.
        assert a.received and set(a.received) <= {'Bearer key-launch', 'Bearer key-profile'}
        assert b.received == []
        assert _route(runner, live) == ('picked', url, 'key-launch')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_named_entry_that_moves_never_sends_its_old_launch_key_to_the_new_endpoint(home, endpoints):
    """``model.provider: team`` with no ``model.base_url``: the endpoint is ``providers.team.base_url``.
    When that entry moves from A to B the route moved, so A's launch key stays behind, before and
    after a pick."""
    (a, url_a), (b, url_b) = endpoints

    def profile(url):
        (home / 'config.yaml').write_text(json.dumps({
            'model': {'provider': 'team', 'default': 'original'},
            'providers': {'team': {'name': 'team', 'base_url': url, 'api_key': 'key-profile'}},
        }), encoding='utf-8')
    profile(url_a)
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        sid = await _launch(owner, home, 'team', 'key-launch')
        live = authority.sessions[sid]
        assert _route(runner, live) == ('original', url_a, 'key-launch')
        profile(url_b)
        assert _route(runner, live) == ('original', url_b, 'key-profile')
        a.received.clear()
        b.received.clear()

        response = await _pick(owner, authority, sid, {'model': 'picked'})

        assert 'result' in response, response
        assert a.received == [] and set(b.received) == {'Bearer key-profile'}
        assert _route(runner, live) == ('picked', url_b, 'key-profile')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_level_picked_with_the_model_survives_a_profile_edit_and_a_resume_refresh(home, endpoints):
    """Desktop's model picker sends the reasoning level with the model. On a following session that level
    is the next turn's, after the profile's default moves and after a resume re-reads the profile."""
    from gateway.session_contract import SessionRef
    from gateway.session_policy import policy_for_source
    from gateway.session_policy_refresh import commit_refresh
    (_, url), _ = endpoints

    def profile(effort):
        (home / 'config.yaml').write_text(json.dumps({
            'model': {'provider': 'custom', 'default': 'original', 'base_url': url, 'api_key': 'key-a'},
            'agent': {'reasoning_effort': effort}}), encoding='utf-8')
    profile('low')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    runner._set_session_reasoning_override = lambda key, value: None
    try:
        live = await _create(owner, authority, home, 'follow')
        sid = next(sid for sid, value in authority.sessions.items() if value is live)
        response = await _pick(owner, authority, sid, {'model': 'picked', 'provider': 'custom', 'reasoning': 'high'})
        assert 'result' in response, response
        profile('minimal')
        assert policy_for_source(runner, live.source).current_reasoning_config(runner)['effort'] == 'high'
        await commit_refresh(authority, SessionRef(authority.profile_id, sid))
        assert policy_for_source(runner, live.source).current_reasoning_config(runner)['effort'] == 'high'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_model_committed_on_a_frozen_bot_chat_holds_when_an_open_adopts_it(home, endpoints):
    """A ``/model`` the user committed while the Bot Chat was frozen is the session's pick: the open that
    adopts the chat keeps it (like a composer pick) until the profile itself moves."""
    from tests.gateway.test_bot_chat_follows_profile import _follows, _frozen_bot_chat, _open
    (_, url), _ = endpoints
    _configure(home, 'original', url, 'key-a')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        ref, _params = _frozen_bot_chat(authority, owner.actor, home)
        live = authority.sessions[ref.session_id]
        response = await _pick(owner, authority, ref.session_id, {'model': 'picked', 'provider': 'custom'})
        assert 'result' in response, response
        assert not _follows(runner, live) and _route(runner, live)[0] == 'picked'
        await _open(authority, ref)
        assert _follows(runner, live)
        assert _route(runner, live) == ('picked', url, 'key-a')
        _configure(home, 'current', url, 'key-a')  # the profile moves: the pick ends
        assert _route(runner, live) == ('current', url, 'key-a')
    finally:
        await owner.close()
        store.close_all_db_handles()
