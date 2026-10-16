"""An exact retry of an alias ``session.create`` is the same request whatever the owner's ``.env`` now
expands the alias to: retry identity is what the client sent (the alias name and its own fields),
checked the same way by a retry that sees the committed session and by one that overlaps the first
create's staging. A payload the client really changed is still refused."""
import json

import pytest
import pytest_asyncio

CONFIG = (
    'platform_toolsets: {cli: []}\n'
    'model: {provider: custom, default: m, base_url: http://default.invalid/v1}\n'
    'model_aliases:\n'
    '  alias-key: {model: model-alias, provider: custom, base_url: "${P5_ENDPOINT}", api_key: "${P5_ALIAS_KEY}"}\n'
    '  twin-key: {model: model-alias, provider: custom, base_url: "${P5_ENDPOINT}", api_key: "${P5_ALIAS_KEY}"}\n'
    '  same-name: {model: same-name, provider: custom, base_url: "${P5_ENDPOINT}"}\n')


def _env(home, n):
    (home / '.env').write_text(f'P5_ENDPOINT=http://endpoint-{n}.invalid/v1\nP5_ALIAS_KEY=alias-key-{n}\n')


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    from hermes_cli import model_switch
    monkeypatch.setattr(model_switch, 'DIRECT_ALIASES', {})
    for name in ('P5_ENDPOINT', 'P5_ALIAS_KEY'):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(CONFIG)
    _env(home, 1)
    running = [home, *await _start(home, monkeypatch)]
    try:
        yield running
    finally:
        await _stop(running)


async def _start(home, monkeypatch):
    from tests.gateway.test_local_create_env_refs import _owner
    return await _owner(home, monkeypatch)


async def _stop(running):
    await running[2].close()
    running[1].db.close()


def _params(home, **fields):
    return {'request_id': 'retry', 'source': 'cli', 'cwd': str(home), 'toolsets': [], **fields}


def _create(authority, connection, params):
    from gateway.session_local import create_local_session
    return create_local_session(authority, connection.actor, dict(params))


def _stage(authority, connection, params):
    from gateway.session_local import stage_local_session
    return stage_local_session(authority, connection.actor, dict(params))


def _commit(authority, staged):
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_local import commit_local_session
    commit_local_session(authority.db, epoch=authority.epoch, receipt=staged)
    return restore_local_session(authority, staged['session_id'])


def _policy(authority, ref):
    from hermes_state_local import local_receipt
    return local_receipt(authority.db, ref.session_id)['policy']


def _refusal(call, *args):
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError) as refused:
        call(*args)
    return refused.value.reason


@pytest.mark.asyncio
async def test_exact_alias_retry_after_an_owner_key_rotation_returns_the_original(owner):
    """``api_key: "${VAR}"`` rotated in ``.env`` between create and retry: the owner's new expansion is
    not the client's payload, so the retry returns the original session, which keeps its own key."""
    from gateway.session_policy import launch_key, restore_policy
    home, authority, connection = owner
    ref = _create(authority, connection, _params(home, model='alias-key'))
    saved = _policy(authority, ref)
    assert launch_key(authority, restore_policy(saved)) == 'alias-key-1'
    _env(home, 2)
    assert _create(authority, connection, _params(home, model='alias-key')) == ref
    assert _policy(authority, ref) == saved
    assert launch_key(authority, restore_policy(saved)) == 'alias-key-1'


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', [{'api_key': 'alias-key-1'}, {'api_key': 'other'},
                                     {'base_url': 'http://endpoint-1.invalid/v1'}])
async def test_an_alias_retry_that_adds_a_client_route_field_is_refused(owner, changed):
    """The alias's key and endpoint were the owner's; a retry that now supplies its own is a new request,
    even when the value happens to equal what the owner expanded."""
    home, authority, connection = owner
    _create(authority, connection, _params(home, model='alias-key'))
    assert _refusal(_create, authority, connection, _params(home, model='alias-key', **changed)) in {
        'invalid_params', 'admission_conflict'}


@pytest.mark.asyncio
async def test_a_changed_client_credential_is_still_refused(owner):
    home, authority, connection = owner
    params = _params(home, model='plain-model', base_url='http://client.invalid/v1', api_key='client-1')
    ref = _create(authority, connection, params)
    _env(home, 2)
    assert _create(authority, connection, params) == ref
    assert _refusal(_create, authority, connection, dict(params, api_key='client-2')) == 'admission_conflict'


@pytest.mark.asyncio
@pytest.mark.parametrize('alias', ['same-name', 'alias-key'])
@pytest.mark.parametrize('order', ['sequential', 'overlap-first-wins', 'overlap-second-wins'])
async def test_exact_alias_retry_after_an_endpoint_rotation_returns_the_original(owner, alias, order):
    """Same identity check whether the retry sees the committed session (sequential) or both attempts
    staged before either commits; ``same-name`` resolves to the model of its own name."""
    home, authority, connection = owner
    params = _params(home, model=alias)
    if order == 'sequential':
        ref = _create(authority, connection, params)
        saved = _policy(authority, ref)
        _env(home, 2)
        assert _create(authority, connection, params) == ref
    else:
        staged = [_stage(authority, connection, params)]
        _env(home, 2)  # the alias's key rotates too: the retry takes the key its slot holds
        staged.append(_stage(authority, connection, params))
        assert json.loads(staged[0]['policy']['config_json']) != json.loads(staged[1]['policy']['config_json'])
        ordered = staged if order == 'overlap-first-wins' else staged[::-1]
        ref = _commit(authority, ordered[0])
        saved = _policy(authority, ref)
        assert _commit(authority, ordered[1]) == ref
    assert _policy(authority, ref) == saved
    assert restore_policy_url(saved) == f'http://endpoint-{1 if order != "overlap-second-wins" else 2}.invalid/v1'


def restore_policy_url(saved):
    from gateway.session_policy import restore_policy
    return restore_policy(saved).config()['model']['base_url']


@pytest.mark.asyncio
@pytest.mark.parametrize('order', ['sequential', 'overlap'])
@pytest.mark.parametrize('first, second', [('alias-key', 'twin-key'), ('same-name', 'm'),
                                           ('alias-key', 'model-alias'), ('same-name', 'other-model')])
async def test_a_different_alias_or_model_under_one_request_id_is_refused(owner, order, first, second):
    """Two aliases that resolve to the same model and endpoint are still two requests, and so is an
    explicit model id in place of the alias (even the one the alias resolves to)."""
    home, authority, connection = owner
    if order == 'sequential':
        _create(authority, connection, _params(home, model=first))
        assert _refusal(_create, authority, connection, _params(home, model=second)) == 'invalid_params'
    else:
        winner = _stage(authority, connection, _params(home, model=first))
        loser = _stage(authority, connection, _params(home, model=second))
        _commit(authority, winner)
        assert _refusal(_commit, authority, loser) == 'invalid_params'


@pytest.mark.asyncio
@pytest.mark.parametrize('alias, old_shape', [
    ('alias-key', lambda request: request.pop('route_derived', None)),  # stamped model_requested only
    ('same-name', lambda request: [request.pop(key, None) for key in ('model_requested', 'route_derived')]),
])
async def test_a_receipt_committed_before_this_change_still_takes_its_exact_retry(owner, monkeypatch, alias,
                                                                                 old_shape):
    """Receipts written before ``route_derived`` (and, for a same-name alias, before ``model_requested``
    was stamped for it), retried after the owner restarts on this code: the unchanged retry still
    returns the original session."""
    from hermes_state_local import POLICY_PREFIX
    home, authority, connection = owner
    params = _params(home, model=alias)
    ref = _create(authority, connection, params)

    def rewrite(conn):
        key = POLICY_PREFIX + ref.session_id
        receipt = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
        request = json.loads(receipt['policy']['request_json'])
        old_shape(request)
        receipt['policy']['request_json'] = json.dumps(request, sort_keys=True)
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(receipt), key))
    authority.db._execute_write(rewrite)
    await _stop(owner)
    owner[1:] = await _start(home, monkeypatch)
    authority, connection = owner[1:]
    saved = _policy(authority, ref)
    assert _create(authority, connection, params) == ref
    if alias == 'alias-key':  # model_requested already marked the alias's key and endpoint as the owner's
        _env(home, 2)
        assert _create(authority, connection, params) == ref
    assert _policy(authority, ref) == saved


def _attempt(call, *args):
    from hermes_state_runtime import RuntimeStoreError
    try:
        return call(*args)
    except RuntimeStoreError as exc:
        return exc.reason


def _rotate_key(home):
    (home / '.env').write_text('P5_ENDPOINT=http://endpoint-1.invalid/v1\nP5_ALIAS_KEY=alias-key-2\n')


@pytest.mark.asyncio
@pytest.mark.parametrize('order', ['first-wins', 'second-wins'])
async def test_an_overlapping_exact_alias_retry_after_a_key_rotation_reuses_the_held_key(owner, order):
    """Both attempts share one key slot: the retry staged after ``.env`` rotated the alias's key runs with
    the key the first attempt holds, and whichever commits stores that key's fingerprint."""
    from agent.credential_persistence import fingerprint_secret_value
    from gateway.session_policy import launch_key, restore_policy
    home, authority, connection = owner
    params = _params(home, model='alias-key')
    staged = [_stage(authority, connection, params)]
    _rotate_key(home)
    staged.append(_attempt(_stage, authority, connection, params))
    assert isinstance(staged[1], dict), staged[1]
    ordered = staged if order == 'first-wins' else staged[::-1]
    ref = _commit(authority, ordered[0])
    assert _commit(authority, ordered[1]) == ref
    with authority.db._read_ctx() as conn:
        assert conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0] == 1
    policy = restore_policy(_policy(authority, ref))
    assert launch_key(authority, policy) == 'alias-key-1'
    assert policy.credential_ref.partition('#')[2] == fingerprint_secret_value('alias-key-1')


@pytest.mark.asyncio
@pytest.mark.parametrize('first, second', [
    ({'model': 'alias-key'}, {'model': 'alias-key', 'api_key': 'other'}),
    ({'model': 'alias-key'}, {'model': 'twin-key'}),  # another request under the same id
    ({'model': 'plain-model', 'api_key': 'client-1'}, {'model': 'plain-model', 'api_key': 'client-2'}),
])
async def test_an_overlapping_retry_with_another_key_is_still_refused(owner, first, second):
    """Only the same request's owner-expanded key takes the held one: a key the client sends is compared
    exactly, and another request staged under the same id never inherits the slot's key."""
    home, authority, connection = owner
    _stage(authority, connection, _params(home, **first))
    _rotate_key(home)
    assert _attempt(_stage, authority, connection, _params(home, **second)) == 'admission_conflict'


PROMPT = 'HERMES_EPHEMERAL_SYSTEM_PROMPT'  # kept by the rules of a client the gateway did not prove local


@pytest.mark.asyncio
@pytest.mark.parametrize('first, second', [({PROMPT: 'P'}, {PROMPT: 'Q'}), ({PROMPT: 'P'}, None),
                                           (None, {PROMPT: 'P'}), ({PROMPT: 'P'}, {})])
async def test_an_overlapping_alias_retry_with_another_client_env_never_takes_the_held_key(owner, first, second):
    """The held key is for the same create request only, and a client_env the slot holder's rules tell
    apart is another request: refused at staging, so it gets neither the key nor a session, even when
    the holder never commits."""
    home, authority, connection = owner
    with_env = (lambda env: _params(home, model='alias-key', **({} if env is None else {'client_env': env})))
    _stage(authority, connection, with_env(first))
    _rotate_key(home)
    assert _attempt(_stage, authority, connection, with_env(second)) == 'admission_conflict'
    with authority.db._read_ctx() as conn:
        assert conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_an_overlapping_alias_retry_differing_only_in_a_dropped_name_reuses_the_held_key(owner):
    """A name the holder's rules drop (PATH from a client not proven local) does not make another request."""
    from gateway.session_policy import launch_key, restore_policy
    home, authority, connection = owner
    staged = [_stage(authority, connection, _params(home, model='alias-key', client_env={PROMPT: 'P'}))]
    _rotate_key(home)
    staged.append(_stage(authority, connection, _params(home, model='alias-key',
                                                        client_env={PROMPT: 'P', 'PATH': '/client/bin'})))
    ref = _commit(authority, staged[1])
    assert _commit(authority, staged[0]) == ref
    assert launch_key(authority, restore_policy(_policy(authority, ref))) == 'alias-key-1'
