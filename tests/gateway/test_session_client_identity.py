"""S4: a client names itself on ``session.create`` (``client: {name, version?}``).

The label is attribution only: it rides the frozen creation request, is echoed as
``session.info.client`` (absent when none was sent), and no policy decision reads it.
"""
import json
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_client_env import _home, _owner

TERN = {'name': 'hermes-tern', 'version': '0.4.1+g1a2b'}


def _create(owner, actor, home, request_id, source='tui', **extra):
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    from hermes_agent.hermes_state_local import local_receipt
    ref = create_local_session(owner, actor, {'request_id': request_id, 'source': source, 'cwd': str(home),
                                              'toolsets': [], **extra})
    return ref, restore_policy(local_receipt(owner.db, ref.session_id)['policy'])


def _info(owner, ref, policy, monkeypatch):
    from gateway import session_policy
    from gateway.session_local import local_session_info
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)
    return local_session_info(owner, ref)


@pytest.mark.asyncio
async def test_a_named_client_reaches_session_info_and_launch_request(tmp_path, monkeypatch):
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        ref, policy = _create(owner, connection.actor, home, 'named', client=dict(TERN))
        info = _info(owner, ref, policy, monkeypatch)
        assert info['client'] == TERN
        assert info['launch_request']['client'] == TERN
        assert info['source'] == 'tui'
        ref, policy = _create(owner, connection.actor, home, 'bare-name', client={'name': 'x'})
        assert _info(owner, ref, policy, monkeypatch)['client'] == {'name': 'x'}
        from tui_gateway.contracts.registry import CANONICAL_METHODS
        CANONICAL_METHODS['session.create'].params.model_validate(
            {'request_id': 'r', 'source': 'tui', 'client': dict(TERN)})
        assert CANONICAL_METHODS['session.info'].result.model_fields['client']
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('source', ['tui', 'gui', 'cli', 'acp', 'tool'])
async def test_a_create_without_client_keeps_its_request_and_info_shape(tmp_path, monkeypatch, source):
    """Ink, Desktop, the CLI, ACP and tools send no ``client``: no ``client`` key anywhere."""
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        ref, policy = _create(owner, connection.actor, home, f'plain-{source}', source=source)
        assert json.loads(policy.request_json) == {'source': source, 'cwd': str(home), 'toolsets': []}
        info = _info(owner, ref, policy, monkeypatch)
        assert 'client' not in info and 'client' not in info['launch_request']
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [
    'hermes-tern', None, [], {}, {'version': '1'}, {'name': 'x', 'extra': 1}, {'name': ''},
    {'name': 'Hermes'}, {'name': '-x'}, {'name': 'a b'}, {'name': 'x' * 65}, {'name': 7},
    {'name': 'x', 'version': ''}, {'name': 'x', 'version': 'v 1'}, {'name': 'x', 'version': 'v' * 65},
    {'name': 'x', 'version': None}, {'name': 'x', 'version': 1}, {'name': 'x', 'version': 'caf\u00e9'},
])
async def test_a_malformed_client_is_refused_as_invalid_params(tmp_path, monkeypatch, bad):
    from hermes_agent.hermes_state_runtime import RuntimeStoreError
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        with pytest.raises(RuntimeStoreError) as exc:
            _create(owner, connection.actor, home, 'bad', client=bad)
        assert (exc.value.reason, exc.value.field) == ('invalid_params', 'client')
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.parametrize('source', ['cli', 'tui', 'gui', 'acp', 'tool', 'oneshot'])
def test_the_client_label_changes_no_policy_decision(tmp_path, monkeypatch, source):
    from gateway.local_viewer_events import launch_shows_reasoning
    from gateway.session_policy import DAYBREAK_LABELS, build_policy
    from tools.approval_context import _LOCAL_INTERACTIVE_SOURCES
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    config = {'model': {'default': 'm', 'provider': 'custom', 'base_url': 'http://x.invalid/v1'},
              'platform_toolsets': {'cli': ['terminal', 'file'], 'acp': ['file']}}
    params = {'source': source, 'cwd': str(tmp_path)}
    plain, named = build_policy(dict(params), config), build_policy({**params, 'client': dict(TERN)}, config)
    fields = ('source', 'platform', 'cwd', 'model', 'toolsets', 'config_json', 'terminal_json', 'safe_mode',
              'ignore_user_config', 'coding_posture', 'client_env_json', 'edit_approval_policy')
    assert {f: getattr(named, f) for f in fields} == {f: getattr(plain, f) for f in fields}
    assert json.loads(named.request_json) == {**json.loads(plain.request_json), 'client': TERN}
    assert launch_shows_reasoning(named, None) == launch_shows_reasoning(plain, None)
    assert (named.source in DAYBREAK_LABELS) == (plain.source in DAYBREAK_LABELS)
    assert (named.source in _LOCAL_INTERACTIVE_SOURCES) == (plain.source in _LOCAL_INTERACTIVE_SOURCES)


@pytest.mark.asyncio
async def test_an_exact_retry_must_repeat_the_same_client(tmp_path, monkeypatch):
    """The label is part of the exact-retry identity, like every other create field."""
    from gateway.session_local import create_local_session
    from hermes_agent.hermes_state_runtime import RuntimeStoreError
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    params = {'request_id': 'retry', 'source': 'tui', 'cwd': str(home), 'toolsets': [], 'client': dict(TERN)}
    try:
        ref = create_local_session(owner, connection.actor, dict(params))
        assert create_local_session(owner, connection.actor, json.loads(json.dumps(params))) == ref
        for other in ({**params, 'client': {'name': 'other'}}, {k: v for k, v in params.items() if k != 'client'}):
            with pytest.raises(RuntimeStoreError) as exc:
                create_local_session(owner, connection.actor, other)
            # main's refusal for any changed field of a resumed request id
            assert (exc.value.reason, exc.value.field) == ('invalid_params', 'request_id')
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_a_named_session_restores_after_restart(tmp_path, monkeypatch):
    """The label lives only in ``request_json``: the receipt's policy keys are main's, so an older
    owner's ``restore_policy`` (checked against a base tree in the PR) restores it too."""
    from dataclasses import fields
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_policy import LocalSessionPolicy, restore_policy
    from hermes_agent.hermes_state_local import local_receipt
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        ref, policy = _create(owner, connection.actor, home, 'restart', client=dict(TERN))
        stored = local_receipt(owner.db, ref.session_id)['policy']
        assert set(stored) == {f.name for f in fields(LocalSessionPolicy)}
        assert json.loads(stored['request_json'])['client'] == TERN
        owner.sessions.clear()
        assert restore_local_session(owner, ref.session_id) == ref
        assert restore_policy(local_receipt(owner.db, ref.session_id)['policy']) == policy
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_runtime_describe_advertises_the_client_parameter(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_local_source_labels import _authority
    authority, _store = await _authority(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_HOME', authority.profile_id)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        described = (await conn.dispatch({'id': 0, 'method': 'runtime.describe', 'params': {}}))['result']
        assert 'session-client-v1' in described['capabilities']
        assert 'client' in described['session_create']['parameters']
        refused = await conn.dispatch({'id': 1, 'method': 'session.create', 'params': {
            'request_id': 'w', 'source': 'tui', 'client': {'name': 'x', 'extra': 1}}})
        assert refused['error']['message'] == 'invalid_params', refused
    finally:
        await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_a_named_create_and_resume_return_snapshots_that_match_their_contract(tmp_path, monkeypatch):
    """``session.create`` / ``session.resume`` return ``local_session_info`` as the snapshot's ``info``."""
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_local_source_labels import _authority
    from tui_gateway.contracts.registry import CANONICAL_METHODS
    authority, _store = await _authority(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_HOME', authority.profile_id)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        created = await conn.dispatch({'id': 1, 'method': 'session.create', 'params': {
            'request_id': 'named', 'source': 'tui', 'cwd': authority.profile_id, 'client': dict(TERN)}})
        resumed = await conn.dispatch({'id': 2, 'method': 'session.resume',
                                       'params': {'session_id': created['result']['session_id']}})
        for method, reply in (('session.create', created), ('session.resume', resumed)):
            CANONICAL_METHODS[method].result.model_validate(reply['result'])
            assert reply['result']['info']['client'] == TERN
    finally:
        await conn.close()
        authority.db.close()


def test_the_classic_clis_new_does_not_repeat_another_clients_name():
    """``/new`` copies the frozen launch request; the creating client's label is not the terminal's."""
    import asyncio
    from hermes_cli import gateway_chat_commands
    sent = {}

    async def rpc(method, **params):
        if method == 'session.info':
            return {'launch_request': {'source': 'tui', 'toolsets': ['file'], 'client': dict(TERN)}, 'cwd': '/tmp'}
        if method == 'runtime.describe':
            return {'session_create': {'parameters': ['source', 'cwd', 'toolsets', 'client']}}
        sent.update(params)
        return {'stored_session_id': 'new'}

    async def adopt(snapshot):
        return None
    view = SimpleNamespace(client=SimpleNamespace(rpc=rpc), session_id='old', adopt=adopt, launch_api_key=None)
    asyncio.run(gateway_chat_commands._new(view, ''))
    assert {k: v for k, v in sent.items() if k != 'request_id'} == {'source': 'tui', 'toolsets': ['file'], 'cwd': '/tmp'}


@pytest.mark.asyncio
async def test_each_profile_home_sees_only_its_own_sessions_client(tmp_path, monkeypatch):
    """Multiplexed homes: two owners' turns run side by side; each plugin read answers for its own."""
    from agent.system_prompt import _plugin_session_info
    from gateway.session_policy import policy_scope
    from hermes_cli.plugin_session import session_client
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    home_a, home_b = _home(tmp_path / 'a'), _home(tmp_path / 'b')
    owner_a, conn_a = await _owner(home_a, monkeypatch)
    _, named = _create(owner_a, conn_a.actor, home_a, 'a', client=dict(TERN))
    owner_b, conn_b = await _owner(home_b, monkeypatch)
    _, plain = _create(owner_b, conn_b.actor, home_b, 'b')
    seen, barrier = {}, threading.Barrier(2)

    def turn(label, owner, policy):
        with policy_scope(policy, authority=owner):
            barrier.wait()
            info = _plugin_session_info(SimpleNamespace(session_id=label))
            seen[label] = (session_client(), info['client'], info['client_version'])
    try:
        threads = [threading.Thread(target=turn, args=('a', owner_a, named)),
                   threading.Thread(target=turn, args=('b', owner_b, plain))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert seen == {'a': (TERN, 'hermes-tern', '0.4.1+g1a2b'), 'b': (None, '', '')}
        assert session_client() is None  # outside any turn
    finally:
        for conn, owner in ((conn_a, owner_a), (conn_b, owner_b)):
            await conn.close()
            owner.db.close()
