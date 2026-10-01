"""Hosted settings write native stores and preserve auth/profile boundaries."""
import os

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    from hermes_cli import web_server
    # Native gateway loading mirrors YAML policies into process env.
    # Do not inherit those mirrors from a previous test.
    for key in list(os.environ):
        if key.startswith('TELEGRAM_'):
            monkeypatch.delenv(key)
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'codex'))
    return TestClient(web_server.app, headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN})


def test_profile_and_group_settings_reach_native_readers(client):
    from hermes_cli.config import load_config
    from agent.prompt_builder import load_soul_md
    from hermes_cli.web_settings import telegram_values
    response = client.put('/api/settings/profile', json={'instructions': 'Keep replies brief.', 'timezone': 'Europe/Prague'})
    assert response.status_code == 200, response.text
    config = load_config()
    assert load_soul_md() == 'Keep replies brief.'
    response = client.put('/api/settings/groups/-100123', json={'mode': 'all', 'instructions': 'English only', 'topics': [{'id': '5', 'mode': 'silent'}]})
    assert response.status_code == 200, response.text
    tg = load_config()['telegram']
    assert '-100123' in tg['group_allowed_chats']
    assert '-100123' in tg['free_response_chats']
    assert tg['silent_topics'] == ['-100123:5']
    assert telegram_values()['groups'][0]['topics'][0]['mode'] == 'silent'
    assert client.get('/api/settings').json()['profile']['instructions'] == 'Keep replies brief.'
    assert client.put('/api/settings/groups/-100123', json={'remove': True}).status_code == 200
    assert '-100123' not in load_config()['telegram']['group_allowed_chats']


def test_key_failure_preserves_previous_secret_and_success_updates_memory_source(client, monkeypatch):
    import httpx
    from hermes_cli.config import save_env_value
    from deploy.railway.hindsight_settings import current
    save_env_value('OPENROUTER_API_KEY', 'old-key')
    async def reject(self, *args, **kwargs):
        return httpx.Response(401, json={'error': 'bad'})
    monkeypatch.setattr(httpx.AsyncClient, 'request', reject)
    response = client.put('/api/settings/keys/OPENROUTER_API_KEY', json={'value': 'new-key'})
    assert response.status_code == 400
    assert current()['openrouter_api_key'] == 'old-key'
    async def accept(self, *args, **kwargs):
        return httpx.Response(200, json={'data': {}})
    monkeypatch.setattr(httpx.AsyncClient, 'request', accept)
    before = current()['revision']
    assert client.put('/api/settings/keys/OPENROUTER_API_KEY', json={'value': 'new-key'}).status_code == 200
    assert current()['openrouter_api_key'] == 'new-key'
    assert current()['revision'] != before
    assert 'new-key' not in client.get('/api/settings').text


def test_approve_decline_revoke_use_native_pairing(client):
    from gateway.pairing import PairingStore
    store = PairingStore()
    store.generate_code('telegram', '12345', 'Jana')
    request = store.list_pending('telegram')[0]
    response = client.post('/api/settings/people', json={'action': 'decline', 'user_id': '12345', 'request_id': request['request_id']})
    assert response.status_code == 200, response.text
    assert not store.list_pending('telegram')
    assert client.post('/api/settings/people', json={'action': 'add', 'user_id': '12345'}).status_code == 200
    assert PairingStore().is_approved('telegram', '12345')
    assert client.post('/api/settings/people', json={'action': 'remove', 'user_id': '12345'}).status_code == 200
    assert not PairingStore().is_approved('telegram', '12345')


def test_settings_require_token(client):
    client.headers.pop('X-Hermes-Session-Token', None)
    assert client.put('/api/settings/profile', json={'instructions': '', 'timezone': 'UTC'}).status_code in {401, 403}


def test_password_rotation_persists_and_invalidates_sessions(client, monkeypatch):
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password, _settings
    from hermes_cli.dashboard_auth import registry
    provider = BasicAuthProvider(username='admin', password_hash=hash_password('old-password'), secret=b'x' * 32)
    monkeypatch.setitem(registry._providers, 'basic', provider)
    session = provider.complete_password_login(username='admin', password='old-password')
    body = {'username': 'owner', 'current_password': 'wrong', 'new_password': 'new-password-123'}
    assert client.post('/api/settings/admin', json=body).status_code == 403
    body['current_password'] = 'old-password'
    assert client.post('/api/settings/admin', json=body).status_code == 200
    assert provider.verify_session(access_token=session.access_token) is None
    monkeypatch.setenv('HERMES_DASHBOARD_BASIC_AUTH_PASSWORD', 'old-bootstrap-password')
    monkeypatch.setenv('HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS', '3600')
    resolved = _settings()
    assert resolved['ttl_seconds'] == 3600
    restored = BasicAuthProvider(**resolved)
    assert restored.complete_password_login(username='owner', password='new-password-123')


def test_two_profile_memory_settings_and_secrets_do_not_cross(monkeypatch, tmp_path):
    from hermes_cli.web_settings import save_memory, memory_values
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.config import save_env_value
    from deploy.railway.hindsight_settings import current
    from agent.secret_scope import set_multiplex_active
    a, b = tmp_path / 'A', tmp_path / 'B'
    a.mkdir(); b.mkdir()
    set_multiplex_active(True)
    try:
        for home, effort in [(a, 'low'), (b, 'high')]:
            token = set_hermes_home_override(home)
            try:
                save_memory('gpt-5.6-luna', effort, 'medium')
                save_env_value('OPENROUTER_API_KEY', home.name)
            finally:
                reset_hermes_home_override(token)
        for home, effort in [(a, 'low'), (b, 'high'), (a, 'low')]:
            token = set_hermes_home_override(home)
            try:
                assert memory_values()['llm_reasoning_effort'] == effort
                assert current()['openrouter_api_key'] == home.name
            finally:
                reset_hermes_home_override(token)
    finally:
        set_multiplex_active(False)


def test_access_reads_nested_native_settings_and_retains_other_group_policy(client):
    from hermes_cli.config import save_config, load_config
    from hermes_cli.web_settings import telegram_values
    save_config({'platforms': {'telegram': {'extra': {
        'group_allowed_chats': ['-100', '-200'], 'require_mention': False,
        'channel_prompts': {'-200': 'Keep this'}, 'allow_from': ['12345', '67890']}}}})
    access = telegram_values()
    assert {p['user_id'] for p in access['people']} == {'12345', '67890'}
    assert all(g['mode'] == 'all' for g in access['groups'])
    assert client.put('/api/settings/groups/-100', json={'mode': 'mention'}).status_code == 200
    tg = load_config()['telegram']
    assert tg['free_response_chats'] == ['-200']
    assert tg['channel_prompts']['-200'] == 'Keep this'
    assert client.post('/api/settings/people', json={'action': 'remove', 'user_id': '12345'}).json()['restart']
    assert load_config()['platforms']['telegram']['extra']['allow_from'] == ['67890']


def test_memory_catalog_validation_and_status_acknowledgement(client, monkeypatch):
    from hermes_cli import models
    from deploy.railway import codex_inference
    from deploy.railway.hindsight_settings import current, status
    monkeypatch.setattr(models, 'provider_model_ids', lambda *a, **kw: ['gpt-5.6-luna'])
    body = {'model': 'gpt-5.6-luna', 'learning': 'high', 'recall': 'low'}
    assert client.put('/api/settings/memory', json=body).status_code == 200
    assert current()['llm_reasoning_effort'] == 'high'
    assert client.put('/api/settings/memory', json={**body, 'model': 'unknown'}).status_code == 400
    monkeypatch.setenv('HINDSIGHT_INFERENCE_KEY', 'private-test-key')
    service = TestClient(codex_inference.app)
    assert service.get('/v1/settings').status_code == 401
    headers = {'Authorization': 'Bearer private-test-key'}
    response = service.get('/v1/settings', headers=headers)
    assert response.status_code == 200
    revision = response.json()['revision']
    assert service.post('/v1/settings/status', json={'revision': revision, 'state': 'ready'}, headers=headers).status_code == 200
    assert status()['state'] == 'ready'
    assert client.put('/api/settings/memory', json={**body, 'learning': 'low'}).status_code == 200
    assert status()['state'] == 'applying'


def test_invalid_profile_does_not_write(client, monkeypatch):
    from hermes_cli.config import load_config
    before = load_config()
    assert client.put('/api/settings/profile', json={'instructions': '', 'timezone': 'Unknown/Zone'}).status_code == 400
    assert load_config() == before


@pytest.mark.parametrize('key', ['TELEGRAM_ALLOWED_USERS', 'TELEGRAM_GROUP_ALLOWED_USERS'])
@pytest.mark.parametrize('allowlist', ['12345,67890', '["12345", "67890"]', '[12345, 67890]'])
def test_removing_environment_grant_requires_gateway_restart(client, allowlist, key):
    from hermes_cli.config import save_env_value
    from agent.secret_scope import build_profile_secret_scope
    from hermes_constants import get_hermes_home
    save_env_value(key, allowlist)
    response = client.post('/api/settings/people', json={'action': 'remove', 'user_id': '12345'})
    assert response.status_code == 200
    assert response.json()['restart'] is True
    assert build_profile_secret_scope(get_hermes_home())[key] == '67890'


def test_group_edits_use_effective_values_and_preserve_extra_bindings(client, monkeypatch):
    from hermes_cli.config import save_config, load_config
    from hermes_cli.web_settings import telegram_values
    from gateway.config import load_gateway_config, Platform
    monkeypatch.setenv('SETTINGS_TEST_GROUP', '-200')
    save_config({'telegram': {'group_allowed_chats': ['-100', '${SETTINGS_TEST_GROUP}'],
                             'extra': {'group_topics': {'-100': [{'thread_id': 5, 'name': 'Old'}]}}}})
    assert {g['id'] for g in telegram_values()['groups']} == {'-100', '-200'}
    response = client.put('/api/settings/groups/-100', json={'topics': [{'id': '6', 'mode': 'inherit'}]})
    assert response.status_code == 200
    tg = load_gateway_config().platforms[Platform.TELEGRAM].extra
    assert tg['group_topics']['-100'] == [{'thread_id': 5, 'name': 'Old'}]
    assert tg['group_allowed_chats'] == ['-200', '-100']
    assert load_config()['telegram']['extra']['group_topics']['-100'][0]['name'] == 'Old'


def test_json_list_policy_updates_preserve_unrelated_topics(client):
    from hermes_cli.config import save_env_value
    from hermes_cli.web_settings import telegram_values
    from agent.secret_scope import build_profile_secret_scope
    from hermes_constants import get_hermes_home
    save_env_value('TELEGRAM_GROUP_ALLOWED_CHATS', '["-100", "-200"]')
    save_env_value('TELEGRAM_FREE_RESPONSE_TOPICS', '["-100:5", "-200:6"]')
    assert {g['id'] for g in telegram_values()['groups']} == {'-100', '-200'}
    assert client.put('/api/settings/groups/-100', json={'topics': [{'id': '7', 'mode': 'all'}]}).status_code == 200
    assert set(build_profile_secret_scope(get_hermes_home())['TELEGRAM_FREE_RESPONSE_TOPICS'].split(',')) == {'-200:6', '-100:7'}


def test_profile_identity_uses_native_frozen_soul_not_ephemeral_prompt(client, tmp_path):
    from unittest.mock import MagicMock
    from hermes_state import SessionDB
    from hermes_cli.config import load_config, save_config
    from hermes_cli.web_settings import profile_values
    from agent.prompt_builder import load_soul_md
    from agent.conversation_loop import _restore_or_build_system_prompt
    from gateway.run import GatewayRunner
    save_config({'agent': {'system_prompt': 'Existing native overlay'}})
    body = {'instructions': 'Keep the original instructions.', 'timezone': 'UTC'}
    assert client.put('/api/settings/profile', json=body).status_code == 200
    before_config = load_config()
    before = load_soul_md()
    with SessionDB(db_path=tmp_path / 'sessions.db') as db:
        db.create_session('existing', source='telegram')
        db.update_system_prompt('existing', before)
        assert client.put('/api/settings/profile', json={**body, 'instructions': 'New instructions.'}).status_code == 200
        assert profile_values(load_config())['instructions'] == 'New instructions.'
        assert load_config()['agent']['system_prompt'] == 'Existing native overlay'
        assert GatewayRunner._extract_cache_busting_config(load_config()) == GatewayRunner._extract_cache_busting_config(before_config)
        def agent(sid):
            result = MagicMock()
            result.session_id = sid
            result._session_db = db
            result._use_prompt_caching = False
            result.enabled_toolsets = result.disabled_toolsets = None
            result._build_system_prompt.side_effect = lambda _: load_soul_md()
            return result
        existing = agent('existing')
        _restore_or_build_system_prompt(existing, None, [{'role': 'user', 'content': 'Continue'}])
        assert existing._cached_system_prompt == before
        existing._build_system_prompt.assert_not_called()
        db.create_session('new', source='telegram')
        fresh = agent('new')
        _restore_or_build_system_prompt(fresh, None, [])
        assert fresh._cached_system_prompt == 'New instructions.'


def test_topics_are_discovered_from_native_sessions_and_bindings_stay_intact(client, monkeypatch):
    from gateway import channel_directory
    from hermes_cli.config import save_config, load_config
    from hermes_cli.web_settings import telegram_values
    bindings = {-100: [{'thread_id': 7, 'name': 'Planning', 'skill': 'existing'}]}
    save_config({'telegram': {'group_allowed_chats': ['-100'], 'group_topics': bindings}})
    entries = channel_directory._entries_from_origins('telegram', 'test', lambda: iter([
        ({'chat_id': '-100', 'chat_name': 'Team', 'thread_id': '5'}, 'group'),
        ({'chat_id': '-100', 'chat_name': 'Team', 'thread_id': '6', 'chat_topic': 'Support'}, 'group'),
    ]))
    monkeypatch.setattr(channel_directory, 'load_directory', lambda: {'platforms': {'telegram': entries}})
    group = telegram_values()['groups'][0]
    assert group['name'] == 'Team'
    assert {t['id']: t['name'] for t in group['topics']} == {'5': '5', '6': 'Support', '7': 'Planning'}
    assert client.put('/api/settings/groups/-100', json={'topics': [{'id': '5', 'mode': 'silent'}]}).status_code == 200
    assert load_config()['telegram']['group_topics'] == bindings
    assert client.put('/api/settings/groups/-100', json={'remove': True}).status_code == 200
    assert load_config()['telegram']['group_topics'] == bindings


@pytest.mark.parametrize('method,path,body', [
    ('put', 'profile', {'instructions': 'Changed', 'timezone': 'UTC'}),
    ('put', 'memory', {'model': 'gpt-5.6-luna', 'learning': 'low', 'recall': 'medium'}),
    ('put', 'groups/-100', {}),
    ('post', 'people', {'action': 'remove', 'user_id': '123'}),
    ('put', 'keys/OPENROUTER_API_KEY', {'value': 'test-key'}),
])
def test_settings_writes_reject_ambiguous_profile(client, monkeypatch, method, path, body):
    monkeypatch.setattr('agent.secret_scope.is_multiplex_active', lambda: True)
    response = getattr(client, method)('/api/settings/' + path, json=body)
    assert response.status_code == 400
    assert 'explicit profile' in response.json()['detail']


def test_timezone_change_clears_local_cache_and_requests_gateway_restart(client):
    from hermes_time import get_timezone_name, reset_cache
    from hermes_cli.config import save_config
    save_config({'timezone': 'UTC'})
    reset_cache()
    assert get_timezone_name() == 'UTC'
    body = {'instructions': 'Keep replies brief.', 'timezone': 'Europe/Prague'}
    response = client.put('/api/settings/profile', json=body)
    assert response.status_code == 200
    assert response.json()['restart'] is True
    assert get_timezone_name() == 'Europe/Prague'
    assert client.put('/api/settings/profile', json=body).json()['restart'] is False


def test_top_level_mention_policy_survives_editing_another_group(client):
    from hermes_cli.config import save_config
    from hermes_cli.web_settings import telegram_values
    from gateway.config import load_gateway_config, Platform
    save_config({'require_mention': True, 'telegram': {'group_allowed_chats': ['-100', '-200']}}, strip_defaults=False)
    assert all(group['mode'] == 'mention' for group in telegram_values()['groups']), telegram_values()['groups']
    assert client.put('/api/settings/groups/-100', json={'mode': 'all'}).status_code == 200
    tg = load_gateway_config().platforms[Platform.TELEGRAM].extra
    assert tg['require_mention'] is True
    assert tg['free_response_chats'] == ['-100']


@pytest.mark.parametrize('nested', ['platforms', 'gateway'])
def test_clearing_last_nested_silent_topic_reaches_native_gateway(client, nested):
    from hermes_cli.config import save_config
    from gateway.config import load_gateway_config, Platform
    platform = {'telegram': {'extra': {'group_allowed_chats': ['-100'], 'silent_topics': ['-100:5']}}}
    config = {'platforms': platform} if nested == 'platforms' else {'gateway': {'platforms': platform}}
    save_config(config)
    response = client.put('/api/settings/groups/-100', json={'topics': [{'id': '5', 'mode': 'inherit'}]})
    assert response.status_code == 200
    assert load_gateway_config().platforms[Platform.TELEGRAM].extra['silent_topics'] == []


def test_secondary_profile_cannot_edit_unsupervised_memory(client, monkeypatch, tmp_path):
    from deploy.railway.hindsight_settings import status
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    secondary = tmp_path / 'secondary'
    secondary.mkdir()
    monkeypatch.setattr('hermes_cli.web_server_profiles._resolve_profile_dir', lambda _: secondary)
    response = client.put('/api/settings/memory?profile=secondary', json={
        'model': 'gpt-5.6-luna', 'learning': 'high', 'recall': 'medium',
    })
    assert response.status_code == 400
    assert 'owns this server' in response.json()['detail']
    assert not (secondary / 'config.yaml').exists()
    token = set_hermes_home_override(secondary)
    try:
        assert status()['state'] == 'unmanaged'
    finally:
        reset_hermes_home_override(token)


def test_empty_native_topic_and_instruction_sections_are_editable(client):
    from hermes_cli.config import save_config
    from hermes_cli.web_settings import telegram_values
    save_config({'gateway': None, 'telegram': {'group_allowed_chats': ['-100'],
        'group_topics': None, 'channel_prompts': None}})
    group = telegram_values()['groups'][0]
    assert group['topics'] == []
    assert group['instructions'] == ''
    assert client.put('/api/settings/groups/-100', json={'instructions': 'Be brief'}).status_code == 200
    assert telegram_values()['groups'][0]['instructions'] == 'Be brief'
    assert client.post('/api/settings/people', json={'action': 'add', 'user_id': '123'}).status_code == 200


def test_chat_reasoning_reads_effective_override_and_native_save_updates_it(client):
    from hermes_cli.config import load_config, save_config
    from hermes_constants import resolve_reasoning_config
    model = 'gpt-5.6-luna'
    save_config({'model': {'default': model}, 'agent': {'reasoning_effort': 'medium',
        'reasoning_overrides': {'openai/' + model: 'low', 'another-model': 'high'}}})
    assert client.get('/api/settings').json()['chat']['effort'] == 'low'
    response = client.put('/api/config', json={'config': {'agent': {'reasoning_overrides': {model: 'high'}}}})
    assert response.status_code == 200
    assert resolve_reasoning_config(load_config(), model)['effort'] == 'high'
    assert client.get('/api/settings').json()['chat']['effort'] == 'high'
    assert load_config()['agent']['reasoning_overrides']['another-model'] == 'high'


def test_instruction_save_preserves_unset_native_timezone(client):
    from hermes_cli.config import read_raw_config, save_config
    save_config({'agent': {'system_prompt': 'Existing overlay'}})
    profile = client.get('/api/settings').json()['profile']
    assert profile['timezone'] == ''
    response = client.put('/api/settings/profile', json={**profile, 'instructions': 'New instructions'})
    assert response.status_code == 200
    assert response.json()['restart'] is False
    assert 'timezone' not in read_raw_config()


@pytest.mark.parametrize('lock', ['installation', 'secret'])
def test_managed_key_refusal_is_reported_before_probe(client, monkeypatch, lock):
    from unittest.mock import AsyncMock
    import httpx
    from hermes_cli.config import save_env_value, load_env
    save_env_value('OPENROUTER_API_KEY', 'original')
    monkeypatch.setattr('hermes_cli.config.is_managed', lambda: lock == 'installation')
    monkeypatch.setattr('hermes_cli.config.managed_scope.is_env_managed', lambda _: lock == 'secret')
    probe = AsyncMock()
    monkeypatch.setattr(httpx.AsyncClient, 'request', probe)
    response = client.put('/api/settings/keys/OPENROUTER_API_KEY', json={'value': 'replacement'})
    assert response.status_code == 409
    assert 'managed' in response.json()['detail']
    assert load_env()['OPENROUTER_API_KEY'] == 'original'
    probe.assert_not_called()


@pytest.mark.parametrize('lock', ['installation', 'credential'])
def test_managed_password_rotation_preserves_live_credentials_and_sessions(client, monkeypatch, lock):
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password
    from hermes_cli.dashboard_auth import registry
    from hermes_cli.config import read_raw_config
    provider = BasicAuthProvider(username='admin', password_hash=hash_password('original-password'), secret=b'x' * 32)
    monkeypatch.setitem(registry._providers, 'basic', provider)
    session = provider.complete_password_login(username='admin', password='original-password')
    before = read_raw_config()
    monkeypatch.setattr('hermes_cli.config.is_managed', lambda: lock == 'installation')
    monkeypatch.setattr('hermes_cli.managed_scope.managed_config_keys',
                        lambda: {'dashboard.basic_auth.password_hash'} if lock == 'credential' else set())
    response = client.post('/api/settings/admin', json={'username': 'owner',
        'current_password': 'original-password', 'new_password': 'replacement-password'})
    assert response.status_code == 409
    assert 'managed' in response.json()['detail']
    assert read_raw_config() == before
    assert provider.verify_session(access_token=session.access_token) is not None
    assert provider.complete_password_login(username='admin', password='original-password')


@pytest.mark.parametrize('route,body,key', [
    ('profile', {'instructions': 'Changed', 'timezone': 'Europe/Prague'}, 'timezone'),
    ('memory', {'model': 'gpt-5.6-luna', 'learning': 'low', 'recall': 'medium'}, 'hindsight.llm_model'),
    ('groups/-100', {'mode': 'all'}, 'telegram.group_allowed_chats'),
])
@pytest.mark.parametrize('lock', ['installation', 'setting'])
def test_managed_settings_fail_without_writes(client, monkeypatch, route, body, key, lock):
    from hermes_cli.config import read_raw_config
    from hermes_constants import get_hermes_home
    before = read_raw_config()
    soul = get_hermes_home() / 'SOUL.md'
    previous_soul = soul.read_bytes() if soul.exists() else None
    monkeypatch.setattr('hermes_cli.config.is_managed', lambda: lock == 'installation')
    monkeypatch.setattr('hermes_cli.managed_scope.managed_config_keys', lambda: {key} if lock == 'setting' else set())
    response = client.put('/api/settings/' + route, json=body)
    assert response.status_code == 400
    assert 'managed' in response.json()['detail']
    assert read_raw_config() == before
    assert (soul.read_bytes() if soul.exists() else None) == previous_soul


def test_managed_allowlist_rejection_precedes_pairing_grant(client, monkeypatch):
    from gateway.pairing import PairingStore
    monkeypatch.setattr('hermes_cli.managed_scope.managed_config_keys', lambda: {'telegram.allow_from'})
    response = client.post('/api/settings/people', json={'action': 'add', 'user_id': '123'})
    assert response.status_code == 400
    assert not PairingStore().is_approved('telegram', '123')


def test_instruction_only_save_keeps_managed_timezone(client, monkeypatch):
    from hermes_cli.config import read_raw_config
    monkeypatch.setattr('hermes_cli.managed_scope.load_managed_config', lambda: {'timezone': 'Europe/Prague'})
    profile = client.get('/api/settings').json()['profile']
    assert profile['timezone'] == 'Europe/Prague'
    response = client.put('/api/settings/profile', json={**profile, 'instructions': 'New instructions'})
    assert response.status_code == 200
    assert response.json()['restart'] is False
    assert 'timezone' not in read_raw_config()


def test_unconfigured_native_sections_can_be_set_up(client, monkeypatch):
    from hermes_cli.config import save_config
    from hermes_cli.dashboard_auth import registry
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password
    save_config({'model': None, 'hindsight': None, 'telegram': None, 'dashboard': None})
    assert client.get('/api/settings').status_code == 200
    assert client.put('/api/settings/memory', json={'model': 'gpt-5.6-luna', 'learning': 'low', 'recall': 'medium'}).status_code == 200
    assert client.put('/api/settings/groups/-100', json={'mode': 'mention'}).status_code == 200
    provider = BasicAuthProvider(username='admin', password_hash=hash_password('original-password'), secret=b'x' * 32)
    monkeypatch.setitem(registry._providers, 'basic', provider)
    assert client.post('/api/settings/admin', json={'username': 'owner', 'current_password': 'original-password',
        'new_password': 'replacement-password'}).status_code == 200


def test_profile_controls_write_only_the_edited_field(client):
    from hermes_cli.config import save_config, load_config
    from agent.prompt_builder import load_soul_md
    save_config({'timezone': 'America/New_York'})
    response = client.put('/api/settings/profile', json={'instructions': 'Latest instructions'})
    assert response.status_code == 200
    assert response.json()['restart'] is False
    assert load_config()['timezone'] == 'America/New_York'
    response = client.put('/api/settings/profile', json={'timezone': 'Europe/Prague'})
    assert response.status_code == 200
    assert response.json()['restart'] is True
    assert load_soul_md() == 'Latest instructions'


@pytest.mark.parametrize('pending', [False, True])
def test_invalid_config_cannot_grant_access_or_consume_request(client, pending):
    from gateway.pairing import PairingStore
    from hermes_constants import get_hermes_home
    store = PairingStore()
    store.generate_code('telegram', '12345', 'Jana')
    request = store.list_pending('telegram')[0]
    (get_hermes_home() / 'config.yaml').write_text('telegram: [', encoding='utf-8')
    response = client.post('/api/settings/people', json={
        'action': 'add', 'user_id': '12345',
        'request_id': request['request_id'] if pending else '',
    })
    assert response.status_code >= 400
    assert not store.is_approved('telegram', '12345')
    assert store.list_pending('telegram')[0]['request_id'] == request['request_id']
