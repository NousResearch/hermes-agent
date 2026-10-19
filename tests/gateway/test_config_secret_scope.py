"""CFG-10 (#140290): a frozen config-secret reference is bound to the secrets it carries and the endpoints
they authenticate, not to every byte of the policy.

A reference stamped by an older gateway hashes the policy fields it knew then; once a later release added a
field to the policy (``skills_prompt``, ``kanban_json``, ``safe_mode``/``ignore_user_config``) that hash no
longer matched, and every turn of the session failed with a bare ``launch_credentials_unavailable``. ``/model``,
the one edit that would re-stamp it, recovered the old secrets first and failed the same way. Real
SessionStore/SessionAuthority, a real config.yaml, and a receipt holding the reference an older gateway wrote.
"""
import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_resume_config_refresh import _call, _connect, _owner_authority

CONFIG = ('model: {default: m1, provider: custom, base_url: "http://127.0.0.1:9/v1", api_key: route-token}\n'
          'agent: {reasoning_effort: high}\nplatform_toolsets: {cli: [terminal]}\nterminal: {backend: local}\n'
          'mcp_servers: {docs: {command: docs-mcp, env: {DOCS_TOKEN: mcp-token}}}\n')
# The policy fields an older gateway's reference did not hash (they did not exist when it was stamped).
_LATER_FIELDS = ('kanban_json', 'safe_mode', 'ignore_user_config', 'skills_prompt', 'editor_mcp_json',
                 'coding_posture', 'client_env_json', 'client_env_remote', 'client_env_rules_json')


def _home(monkeypatch):
    from gateway import run
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(CONFIG)
    monkeypatch.setattr(run, '_hermes_home', home)
    return home


def _older_gateway_reference(policy):
    """The reference the first release of frozen config secrets wrote for *policy*: the whole-policy hash
    over the fields it had, no ``bound`` endpoints."""
    from gateway.session_policy_credentials import PREFIX
    source = json.loads(policy.config_secret_ref[len(PREFIX):])
    data = {k: v for k, v in asdict(policy).items()
            if k not in ('credential_ref', 'config_secret_ref', *_LATER_FIELDS)}
    source['policy'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    source.pop('bound', None)
    return PREFIX + json.dumps(source, sort_keys=True)


def _store_reference(store, sid, ref):
    from hermes_state_local import POLICY_PREFIX
    def write(conn):
        saved = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?',
                                        (POLICY_PREFIX + sid,)).fetchone()[0])
        saved['policy']['config_secret_ref'] = ref
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(saved), POLICY_PREFIX + sid))
    store._db._execute_write(write)


def _stored_policy(store, sid):
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    return restore_policy(local_receipt(store._db, sid)['policy'])


async def _older_session(tmp_path, monkeypatch):
    """A session whose receipt an older gateway stamped, adopted by a restarted current owner."""
    home = _home(monkeypatch)
    store, first = await _owner_authority(home, 'first')
    owner = _connect(first)
    sid = (await _call(owner, 'session.create', request_id='s', source='cli', cwd=str(tmp_path)))['session_id']
    await owner.close()
    _store_reference(store, sid, _older_gateway_reference(_stored_policy(store, sid)))
    store.close_all_db_handles()
    store, cold = await _owner_authority(home, 'cold')
    return home, store, cold, sid


@pytest.mark.asyncio
async def test_a_reference_stamped_by_an_older_gateway_still_runs_its_session(tmp_path, monkeypatch):
    home, store, cold, sid = await _older_session(tmp_path, monkeypatch)
    try:
        config = _stored_policy(store, sid).config(cold)
        assert config['model']['api_key'] == 'route-token'
        assert config['mcp_servers']['docs']['env'] == {'DOCS_TOKEN': 'mcp-token'}
        # Its values are still gated by their fingerprints: the route key rotated is refused by name.
        (home / 'config.yaml').write_text(CONFIG.replace('route-token', 'rotated-token'))
        fresh = SimpleNamespace(db=store._db, profile_id=cold.profile_id)
        from hermes_state_runtime import RuntimeStoreError
        with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable') as refused:
            _stored_policy(store, sid).config(fresh)
        assert refused.value.field == 'model.api_key'
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_model_re_stamps_a_reference_an_older_gateway_wrote(tmp_path, monkeypatch):
    from hermes_cli import model_switch
    from gateway.session_policy_credentials import PREFIX, policy_identity
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model='m2', target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    _home_dir, store, cold, sid = await _older_session(tmp_path, monkeypatch)
    viewer = _connect(cold)
    try:
        snap = store._db.get_session(sid)
        reply = await viewer.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': sid, 'request_id': 'switch', 'operation': 'model', 'payload': {'model': 'm2'},
            'expected_revision': snap['runtime_revision'], 'expected_generation': snap['runtime_generation']}})
        assert 'result' in reply, reply
        switched = _stored_policy(store, sid)
        assert switched.model == 'm2'
        source = json.loads(switched.config_secret_ref[len(PREFIX):])
        assert source['policy'] == policy_identity(switched) and 'bound' in source
        # The next turn's read, on an owner with nothing cached, recovers the same secrets.
        restarted = SimpleNamespace(db=store._db, profile_id=cold.profile_id)
        assert switched.config(restarted)['model']['api_key'] == 'route-token'
    finally:
        await viewer.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('edit', ['base_url', 'reasoning', 'mcp_list'])
async def test_a_non_secret_profile_edit_keeps_the_session_and_its_model_switch(tmp_path, monkeypatch, edit):
    """The profile edit the issue names, then a restart, a resume (its settings refresh re-stamps the
    reference) and the session's next turn read and ``/model``."""
    from hermes_cli import model_switch
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model='m2', target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    home = _home(monkeypatch)
    store, first = await _owner_authority(home, 'first')
    owner = _connect(first)
    sid = (await _call(owner, 'session.create', request_id='s', source='cli', cwd=str(tmp_path)))['session_id']
    await owner.close()
    store.close_all_db_handles()
    (home / 'config.yaml').write_text({
        'base_url': CONFIG.replace('127.0.0.1:9', '127.0.0.1:10'),
        'reasoning': CONFIG.replace('reasoning_effort: high', 'reasoning_effort: low'),
        'mcp_list': CONFIG.replace('}}}\n', '}}, extra: {command: extra-mcp}}\n')}[edit])
    store, cold = await _owner_authority(home, 'cold')
    viewer = _connect(cold)
    try:
        resumed = await _call(viewer, 'session.resume', session_id=sid)
        assert _stored_policy(store, sid).config(cold)['model']['api_key'] == 'route-token'
        reply = await viewer.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': sid, 'request_id': 'switch', 'operation': 'model', 'payload': {'model': 'm2'},
            'expected_revision': resumed['revision'], 'expected_generation': resumed['execution_generation']}})
        assert 'result' in reply, reply
        assert _stored_policy(store, sid).config(cold)['model']['api_key'] == 'route-token'
    finally:
        await viewer.close()
        store.close_all_db_handles()


def _bound_policy(tmp_path, monkeypatch):
    from gateway.session_policy import bind_launch_key, build_policy
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    cfg = {'model': {'provider': 'custom', 'api_key': 'route-token', 'base_url': 'http://127.0.0.1:1234/v1'},
           'mcp_servers': {'docs': {'command': 'docs-mcp', 'env': {'DOCS_TOKEN': 'mcp-token'}}}}
    (home / 'config.yaml').write_text(json.dumps(cfg))

    def owner(path=home, profile='owned'):  # a fresh authority per read, like a restarted owner
        return SimpleNamespace(db=SimpleNamespace(db_path=path / 'state.db'), profile_id=profile,
                               instance_id='new', epoch=2)
    private = {}
    policy = build_policy({'cwd': str(home), 'model': 'frozen'}, cfg, private_secrets=private)
    return home, cfg, owner, bind_launch_key(owner(), 'session', policy, None, config_secrets=private)


def test_a_non_secret_policy_change_keeps_the_reference(tmp_path, monkeypatch):
    """Fields no secret depends on (model name, reasoning, toolsets, a policy field a later release adds)
    move without invalidating the reference."""
    _home_dir, _cfg, owner, policy = _bound_policy(tmp_path, monkeypatch)
    config = json.loads(policy.config_json)
    config.setdefault('agent', {})['reasoning_effort'] = 'low'
    config['mcp_servers']['extra'] = {'command': 'extra-mcp'}
    moved = replace(policy, model='other', toolsets=('web',), skills_prompt='later field',
                    config_json=json.dumps(config))
    recovered = moved.config(owner())
    assert recovered['model']['api_key'] == 'route-token'
    assert recovered['mcp_servers']['docs']['env'] == {'DOCS_TOKEN': 'mcp-token'}


@pytest.mark.parametrize('scope', ['home', 'profile'])
def test_another_home_or_profile_is_refused_by_name(tmp_path, monkeypatch, scope):
    from hermes_state_runtime import RuntimeStoreError
    _home_dir, _cfg, owner, policy = _bound_policy(tmp_path, monkeypatch)
    wrong = owner(path=tmp_path) if scope == 'home' else owner(profile='other')
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable') as refused:
        policy.config(wrong)
    assert refused.value.field == scope and refused.value.detail


def test_a_revoked_route_secret_is_still_refused(tmp_path, monkeypatch):
    from hermes_state_runtime import RuntimeStoreError
    home, cfg, owner, policy = _bound_policy(tmp_path, monkeypatch)
    del cfg['model']['api_key']
    (home / 'config.yaml').write_text(json.dumps(cfg))
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable') as refused:
        policy.config(owner())
    assert refused.value.field == 'model.api_key'


@pytest.mark.parametrize('moved', ['model', 'mcp'])
def test_a_secret_paired_with_a_moved_endpoint_is_refused_by_name(tmp_path, monkeypatch, moved):
    """The scoped check still binds each secret to the endpoint it authenticates in the frozen policy: a
    policy rewrite that repoints it without re-binding never sends the key to the new endpoint."""
    from hermes_state_runtime import RuntimeStoreError
    _home_dir, _cfg, owner, policy = _bound_policy(tmp_path, monkeypatch)
    config = json.loads(policy.config_json)
    if moved == 'model':
        config['model']['base_url'] = 'http://attacker.invalid/v1'
        field = 'model.api_key'
    else:
        config['mcp_servers']['docs']['command'] = 'other-mcp'
        field = 'mcp_servers.docs.env'
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable') as refused:
        replace(policy, config_json=json.dumps(config)).config(owner())
    assert refused.value.field == field and 'endpoint' in refused.value.detail


def _bind(tmp_path, monkeypatch, cfg):
    from gateway.session_policy import bind_launch_key, build_policy
    home = tmp_path / 'profile'
    home.mkdir(exist_ok=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text(json.dumps(cfg))

    def owner():
        return SimpleNamespace(db=SimpleNamespace(db_path=home / 'state.db'), profile_id='owned',
                               instance_id='new', epoch=2)
    private = {}
    policy = build_policy({'cwd': str(home), 'model': 'frozen'}, cfg, private_secrets=private)
    return owner, bind_launch_key(owner(), 'session', policy, None, config_secrets=private)


def _repointed(policy, mutate):
    config = json.loads(policy.config_json)
    mutate(config)
    return replace(policy, config_json=json.dumps(config))


# Each secret authenticates the endpoint its route RESOLVES to, which can live in another block: a keyless
# ``providers:`` entry the ``model:`` route names, a fallback's provider entry, an entry's ``api`` URL.
_ROUTED = {
    'entry_api_url': ({'model': {'provider': 'p2', 'default': 'm'},
                       'providers': {'p2': {'api': 'http://good.invalid/v1', 'api_key': 'K-p2'}}},
                      lambda c: c['providers']['p2'].__setitem__('api', 'http://evil.invalid/v1'),
                      'providers.p2.api_key'),
    'model_key_keyless_entry': ({'model': {'provider': 'myprov', 'default': 'm', 'api_key': 'K-model'},
                                 'providers': {'myprov': {'base_url': 'http://good.invalid/v1'}}},
                                lambda c: c['providers']['myprov'].__setitem__('base_url', 'http://evil.invalid/v1'),
                                'model.api_key'),
    'fallback_key_via_entry': ({'model': {'provider': 'custom', 'default': 'm', 'base_url': 'http://good.invalid/v1'},
                                'providers': {'fb': {'base_url': 'http://good-fb.invalid/v1'}},
                                'fallback_providers': [{'provider': 'fb', 'model': 'x', 'api_key': 'K-fb'}]},
                               lambda c: c['providers']['fb'].__setitem__('base_url', 'http://evil.invalid/v1'),
                               'fallback_providers.0.api_key'),
}


@pytest.mark.parametrize('case', sorted(_ROUTED))
def test_a_secret_is_bound_to_the_endpoint_its_route_resolves_to(tmp_path, monkeypatch, case):
    """Independent review P2-2: moving the URL the secret actually authenticates, even when it lives in
    another block or under ``api``, refuses the secret by name instead of pairing it with the new URL."""
    from hermes_state_runtime import RuntimeStoreError
    cfg, mutate, field = _ROUTED[case]
    owner, policy = _bind(tmp_path, monkeypatch, cfg)
    assert policy.config(owner())  # the unmoved policy recovers
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable') as refused:
        _repointed(policy, mutate).config(owner())
    assert refused.value.field == field and 'endpoint' in refused.value.detail


def _legacy(policy):
    """*policy* carrying the reference an older gateway wrote for it."""
    return replace(policy, config_secret_ref=_older_gateway_reference(policy))


def test_a_legacy_reference_keeps_the_endpoint_guard(tmp_path, monkeypatch):
    """Independent review P2-3: a reference without ``bound`` still matches its own policy (the later
    fields at their defaults are not hashed), but a moved endpoint, or any other policy rewrite its whole-
    policy hash covers, is refused instead of degrading to the fingerprints alone."""
    from hermes_state_runtime import RuntimeStoreError
    _home_dir, _cfg, owner, policy = _bound_policy(tmp_path, monkeypatch)
    legacy = _legacy(policy)
    assert legacy.config(owner())['model']['api_key'] == 'route-token'
    for moved in (_repointed(legacy, lambda c: c['model'].__setitem__('base_url', 'http://evil.invalid/v1')),
                  replace(legacy, skills_prompt='set after the stamp')):
        with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable') as refused:
            moved.config(owner())
        assert refused.value.field == 'config_secret_ref' and refused.value.detail


# Independent review round 2: every route shape a consumer reads binds its secret to the route that owns
# it. A dictionary-form fallback is ONE route (its key's owner is the whole dictionary, not the key), and
# an ``auxiliary.<task>`` route or one of its ``fallback_chain`` entries authenticates the endpoint it
# resolves to, a named ``providers:`` entry's URL or, for ``provider: main``, the session's own route.
_A, _B = 'http://good.invalid/v1', 'http://evil.invalid/v1'
_BASE = {'model': {'provider': 'custom', 'default': 'm', 'base_url': _A}}
_SHAPES = {
    'fallback_providers_dict': ({**_BASE, 'fallback_providers': {'provider': 'custom', 'model': 'x',
                                                                 'base_url': _A, 'api_key': 'K-fb'}},
                                lambda c: c['fallback_providers'].__setitem__('base_url', _B),
                                'fallback_providers.api_key'),
    'fallback_providers_dict_via_entry': ({**_BASE, 'providers': {'fb': {'base_url': _A}},
                                           'fallback_providers': {'provider': 'fb', 'model': 'x', 'api_key': 'K-fb'}},
                                          lambda c: c['providers']['fb'].__setitem__('base_url', _B),
                                          'fallback_providers.api_key'),
    'fallback_model_dict_via_entry': ({**_BASE, 'providers': {'fb': {'base_url': _A}},
                                       'fallback_model': {'provider': 'fb', 'model': 'x', 'api_key': 'K-fb'}},
                                      lambda c: c['providers']['fb'].__setitem__('base_url', _B),
                                      'fallback_model.api_key'),
    'aux_task_via_entry': ({**_BASE, 'providers': {'team': {'base_url': _A}},
                            'auxiliary': {'compression': {'provider': 'team', 'model': 'm', 'api_key': 'K-aux'}}},
                           lambda c: c['providers']['team'].__setitem__('base_url', _B),
                           'auxiliary.compression.api_key'),
    'aux_task_own_url': ({**_BASE, 'auxiliary': {'vision': {'provider': 'custom', 'model': 'm',
                                                            'base_url': _A, 'api_key': 'K-aux'}}},
                         lambda c: c['auxiliary']['vision'].__setitem__('base_url', _B),
                         'auxiliary.vision.api_key'),
    'aux_task_on_main': ({**_BASE, 'auxiliary': {'compression': {'provider': 'main', 'api_key': 'K-aux'}}},
                         lambda c: c['model'].__setitem__('base_url', _B),
                         'auxiliary.compression.api_key'),
    'aux_fallback_chain_via_entry': ({**_BASE, 'providers': {'team': {'base_url': _A}},
                                      'auxiliary': {'compression': {'provider': 'custom', 'model': 'm', 'base_url': _A,
                                                                    'fallback_chain': [{'provider': 'team', 'model': 'm',
                                                                                        'api_key': 'K-chain'}]}}},
                                     lambda c: c['providers']['team'].__setitem__('base_url', _B),
                                     'auxiliary.compression.fallback_chain.0.api_key'),
    # alt-glitch critical: after ``/model <alias>`` with a literal key, the key lives at the alias entry and
    # ``model.api_key_from`` points at it; config() projects it into ``model.api_key``, so it authenticates
    # the session's ``model:`` route, and moving that route must refuse it.
    'alias_key_via_api_key_from': ({'model': {**_BASE['model'], 'api_key_from': ['model_aliases', 'alt', 'api_key']},
                                    'model_aliases': {'alt': {'provider': 'custom', 'model': 'x', 'api_key': 'K-alias'}}},
                                   lambda c: c['model'].__setitem__('base_url', _B),
                                   'model_aliases.alt.api_key'),
    'nested_alias_key_via_api_key_from': ({'model': {**_BASE['model'], 'api_key_from': ['model', 'aliases', 'alt', 'api_key'],
                                                     'aliases': {'alt': {'provider': 'custom', 'model': 'x',
                                                                         'api_key': 'K-alias'}}}},
                                          lambda c: c['model'].__setitem__('base_url', _B),
                                          'model.aliases.alt.api_key'),
    'model_alias_own_url': ({'model': {**_BASE['model'], 'aliases': {'alt': {'provider': 'custom', 'model': 'x',
                                                                             'base_url': _A, 'api_key': 'K-alias'}}}},
                            lambda c: c['model']['aliases']['alt'].__setitem__('base_url', _B),
                            'model.aliases.alt.api_key'),
}


@pytest.mark.parametrize('case', sorted(_SHAPES))
def test_every_route_shape_binds_its_secret_to_the_owning_route(tmp_path, monkeypatch, case):
    from hermes_state_runtime import RuntimeStoreError
    cfg, mutate, field = _SHAPES[case]
    owner, policy = _bind(tmp_path, monkeypatch, json.loads(json.dumps(cfg)))
    assert policy.config(owner())  # the unmoved policy recovers
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable') as refused:
        _repointed(policy, mutate).config(owner())
    assert refused.value.field == field and 'endpoint' in refused.value.detail


def test_an_unrelated_route_edit_keeps_an_auxiliary_and_fallback_key(tmp_path, monkeypatch):
    """Binding the owning route is not binding the policy: another task's or entry's move is no change."""
    cfg = {**_BASE, 'providers': {'team': {'base_url': _A}, 'other': {'base_url': _A}},
           'fallback_providers': {'provider': 'team', 'model': 'x', 'api_key': 'K-fb'},
           'auxiliary': {'compression': {'provider': 'team', 'model': 'm', 'api_key': 'K-aux'},
                         'vision': {'provider': 'other', 'model': 'm'}}}
    owner, policy = _bind(tmp_path, monkeypatch, cfg)

    def unrelated(c):
        c['providers']['other']['base_url'] = _B
        c['auxiliary']['compression']['timeout'] = 30
        c['fallback_providers']['model'] = 'y'
    recovered = _repointed(policy, unrelated).config(owner())
    assert recovered['auxiliary']['compression']['api_key'] == 'K-aux'
    assert recovered['fallback_providers']['api_key'] == 'K-fb'


@pytest.mark.parametrize('spelling', ['http://GOOD.invalid/v1', 'http://good.invalid/v1/', 'HTTP://good.invalid:80/v1'])
def test_a_cosmetic_url_respelling_keeps_the_key(tmp_path, monkeypatch, spelling):
    """@unsupportedpastels minor: host case, a trailing slash or the scheme's default port is the same
    endpoint (normalize_route_base_url), so re-spelling it is not a moved endpoint."""
    cfg = {**_BASE, 'model': {**_BASE['model'], 'api_key': 'K-model'},
           'providers': {'team': {'base_url': _A, 'api_key': 'K-team'}},
           'auxiliary': {'compression': {'provider': 'team', 'model': 'm', 'api_key': 'K-aux'}}}
    owner, policy = _bind(tmp_path, monkeypatch, cfg)

    def respell(c):
        c['model']['base_url'] = spelling
        c['providers']['team']['base_url'] = spelling
    recovered = _repointed(policy, respell).config(owner())
    assert recovered['model']['api_key'] == 'K-model'
    assert recovered['providers']['team']['api_key'] == 'K-team'
    assert recovered['auxiliary']['compression']['api_key'] == 'K-aux'


def _loopback():
    """Two real loopback HTTP servers recording the Authorization each request carries."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = {'A': [], 'B': []}
    servers, urls = [], {}
    for name in ('A', 'B'):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self, name=name):
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
                seen[name].append(self.headers.get('Authorization'))
                body = json.dumps({'id': 'x', 'object': 'chat.completion', 'created': 1, 'model': 'm', 'choices': [
                    {'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        urls[name] = f'http://127.0.0.1:{server.server_port}/v1'
    return seen, urls, servers


def _send(client, model):
    try:
        client.with_options(timeout=3, max_retries=0).chat.completions.create(
            model=model, messages=[{'role': 'user', 'content': 'probe'}], max_tokens=1)
    finally:
        client.close()


@pytest.mark.parametrize('route', ['fallback_providers_dict', 'auxiliary_named_entry'])
def test_a_moved_route_never_sends_its_frozen_key_to_the_new_endpoint(tmp_path, monkeypatch, route):
    """The review's wire repro: bind at loopback A, move the frozen route to loopback B without a re-bind,
    and drive the real consumer (the main runtime's fallback walker / the auxiliary client). A's key must
    reach A while the route is unmoved and never reach B."""
    from hermes_state_runtime import RuntimeStoreError
    seen, urls, servers = _loopback()
    try:
        if route == 'fallback_providers_dict':
            cfg = {'model': {'provider': 'anthropic', 'default': 'm'},
                   'fallback_providers': {'provider': 'custom', 'model': 'backup', 'base_url': urls['A'],
                                          'api_key': 'SENTINEL-FB-A'}}
            move = lambda c: c['fallback_providers'].__setitem__('base_url', urls['B'])
        else:
            cfg = {'model': {'provider': 'custom', 'default': 'm', 'base_url': urls['A']},
                   'providers': {'team': {'base_url': urls['A']}},
                   'auxiliary': {'compression': {'provider': 'team', 'model': 'm', 'api_key': 'SENTINEL-AUX-A'}}}
            move = lambda c: c['providers']['team'].__setitem__('base_url', urls['B'])
        for name in ('ANTHROPIC_API_KEY', 'ANTHROPIC_TOKEN', 'CLAUDE_CODE_OAUTH_TOKEN', 'OPENAI_API_KEY'):
            monkeypatch.delenv(name, raising=False)
        owner, policy = _bind(tmp_path, monkeypatch, cfg)

        def drive(frozen):
            authority = owner()
            if route == 'fallback_providers_dict':
                from openai import OpenAI
                from gateway.run_turn_prepare import _resolve_policy_agent_runtime
                from gateway.session_policy import policy_scope
                with policy_scope(frozen, authority=authority):
                    model, runtime = _resolve_policy_agent_runtime(SimpleNamespace(session_authority=authority), frozen)
                    _send(OpenAI(api_key=runtime['api_key'], base_url=runtime['base_url']), model)
            else:
                from agent import auxiliary_client
                from agent.auxiliary_task_config import session_auxiliary_scope
                from hermes_cli import runtime_provider
                config = frozen.config(authority)
                with runtime_provider.frozen_runtime_config(config), session_auxiliary_scope(config['auxiliary']):
                    client, model = auxiliary_client.get_text_auxiliary_client('compression')
                    _send(client, model)

        drive(policy)  # control: the unmoved route sends A's key to A
        assert seen['A'] and seen['A'][-1].endswith('-A')
        with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
            drive(_repointed(policy, move))
        assert seen['B'] == []
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
