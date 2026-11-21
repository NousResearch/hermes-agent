"""HTTP delegation uses real discovery and never routes provider effects to the hub."""
import json

import pytest
from starlette.testclient import TestClient

from hermes_constants import get_hermes_home
from tests.hermes_cli.test_skill_catalog_service import fixture


@pytest.fixture
def client(_isolate_hermes_home, monkeypatch):
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_cli.web_routers import skills
    from tools.skills_hub_official import OptionalSkillSource
    from tools import skills_hub_install
    monkeypatch.setattr(OptionalSkillSource, 'list_local', lambda self: [])
    monkeypatch.setattr(skills, '_hub_sources', lambda profile: [])
    monkeypatch.setattr(skills_hub_install, 'check_for_skill_updates', list)
    http = TestClient(app)
    http.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    yield http
    http.close()


@pytest.fixture
def catalog(client, request):
    manager, entry, plan = fixture(get_hermes_home(), isolation=getattr(request, 'param', 'in_process'))
    try:
        yield entry, plan
    finally:
        manager.unload()


def tripwires(monkeypatch):
    from hermes_cli.web_routers import skills
    def forbidden(*args, **kwargs):
        pytest.fail('provider identifier reached the stock bundle/spawn pipeline')
    monkeypatch.setattr(skills, '_resolve_hub_skill', forbidden)
    monkeypatch.setattr(skills, 'spawn_profile_action', forbidden)


@pytest.mark.parametrize('catalog', ['in_process', 'host'], indirect=True)
def test_http_review_confirm_operation_and_spent_replay(client, catalog, monkeypatch):
    entry, plan = catalog
    tripwires(monkeypatch)
    body = dict(identifier=plan['identifier'], action='install')
    reviewed = client.post('/api/skills/hub/review', json=body)
    assert reviewed.status_code == 200, reviewed.text
    assert reviewed.json() == plan
    body.update(review_token=plan['review_token'], acknowledgements=[])
    result = client.post('/api/skills/hub/confirm', json=body)
    assert result.status_code == 200, result.text
    assert result.json()['state'] == 'completed'
    assert (get_hermes_home() / 'skills/new/SKILL.md').is_file()
    replay = client.post('/api/skills/hub/confirm', json=body)
    assert replay.json() == result.json()
    assert entry.provider.effects == 1
    operation = client.get('/api/skills/hub/operation', params=dict(
        identifier=plan['identifier'], operation_id=result.json()['operation_id']))
    assert operation.status_code == 200, operation.text
    assert operation.json() == dict(result.json(), settlement='settled')


@pytest.mark.parametrize('route', ['install', 'update', 'uninstall'])
def test_legacy_effect_with_provider_identifier_requires_review(client, catalog, monkeypatch, route):
    _entry, plan = catalog
    tripwires(monkeypatch)
    response = client.post('/api/skills/hub/' + route, json={'identifier': plan['identifier']})
    assert response.status_code == 409, response.text
    assert response.json()['error']['code'] == 'review_required'


def test_legacy_install_with_token_uses_same_confirmation(client, catalog, monkeypatch):
    entry, plan = catalog
    tripwires(monkeypatch)
    response = client.post('/api/skills/hub/install', json=dict(identifier=plan['identifier'],
        review_token=plan['review_token'], acknowledgements=[]))
    assert response.status_code == 200, response.text
    assert response.json()['outcome'] == 'installed'
    assert entry.provider.effects == 1


@pytest.mark.parametrize('route', ['preview', 'scan'])
def test_provider_read_failure_does_not_fall_back_to_bundle(client, catalog, monkeypatch, route):
    _entry, plan = catalog
    tripwires(monkeypatch)
    response = client.get('/api/skills/hub/' + route, params={'identifier': plan['identifier']})
    assert response.status_code == (404 if route == 'preview' else 422), response.text
    assert response.json()['error']['code'] == ('not_found' if route == 'preview' else 'unsupported')


def test_official_and_sources_add_owner_qualified_source(client, catalog):
    entry, _plan = catalog
    official = client.get('/api/skills/hub/official')
    assert official.status_code == 200
    source, = official.json()['provider_sources']
    assert source['id'] == entry.source_id
    assert source['provider'] == dict(id='team', plugin_id='fixture-catalog')
    sources = client.get('/api/skills/hub/sources').json()
    assert sources['sources'] == [source]
    assert sources['featured'] == []


def test_unregistered_extensions_match_frozen_pure_main_before_validation(client):
    from pathlib import Path
    fixture = Path(__file__).parents[1] / 'fixtures/skill_catalog/stock_extensions.json'
    frozen = json.loads(fixture.read_text())
    for case in frozen['requests']:
        response = client.request(case['method'], case['path'],
            **({'content': 'not json'} if case['method'] == 'POST' else {}))
        assert response.status_code == case['status']
        assert response.content == case['body'].encode()
        assert dict(response.headers) == case['headers']


@pytest.mark.parametrize('payload', [[], {'identifier': 'bad', 'action': 'install'},
    {'identifier': 'bad', 'action': 'install', 'context': {'profile_key': 'attacker'}}])
def test_active_extension_strict_validation(client, catalog, payload):
    response = client.post('/api/skills/hub/review', json=payload)
    assert response.status_code == 400, response.text
    assert response.json()['error']['code'] in ('invalid_request', 'invalid_identifier')


def test_profile_conflict_rejected_before_effect(client, catalog):
    entry, plan = catalog
    response = client.post('/api/skills/hub/review?profile=default', json=dict(
        identifier=plan['identifier'], action='install', profile='another'))
    assert response.status_code == 400, response.text
    assert entry.provider.effects == 0


def test_extension_uses_existing_auth_boundary(client, catalog):
    # Clear the actual session header, independently of its configured spelling.
    from hermes_cli.web_server import _SESSION_HEADER_NAME
    client.headers.pop(_SESSION_HEADER_NAME, None)
    response = client.post('/api/skills/hub/review', json={})
    assert response.status_code in (401, 403)


def test_updates_and_update_all_return_candidates_without_provider_effect(client, catalog, monkeypatch):
    from hermes_cli.web_routers import skills
    entry, plan = catalog
    calls = []
    update = dict(identifier=plan['identifier'], installed_version='1', latest_version='2',
                  state='update_available', installation_revision='old')
    def check(request):
        calls.append(request)
        return dict(ok=True, data=dict(items=[update], next_cursor=None))
    monkeypatch.setattr(entry.provider, 'check', check)
    monkeypatch.setattr(skills, 'spawn_profile_action', lambda *args, **kw: {'ok': True, 'pid': 123, 'name': 'skills-update'})
    result = client.get('/api/skills/hub/updates')
    assert result.json() == {'updates': [update]}
    result = client.post('/api/skills/hub/update', json={})
    assert result.json()['provider_review_candidates'] == [update]
    assert entry.provider.effects == 0
    assert len(calls) == 2


def test_managed_bare_name_uninstall_returns_reference_not_spawn(client, monkeypatch):
    from tests.hermes_cli.test_skill_catalog_installed import _provider, _skill
    _skill(get_hermes_home(), 'nested/folder', 'shown-name')
    manager, record = _provider(get_hermes_home())
    try:
        tripwires(monkeypatch)
        response = client.post('/api/skills/hub/uninstall', json={'name': 'shown-name'})
        assert response.status_code == 409, response.text
        assert response.json()['error']['identifier'] == record['identifier']
        assert (get_hermes_home() / 'skills/nested/folder/SKILL.md').is_file()
    finally:
        manager.unload()


def test_update_check_consumes_every_page(client, catalog, monkeypatch):
    entry, plan = catalog
    seen = []
    def check(request):
        cursor = request['cursor']
        seen.append(cursor)
        row = dict(identifier=plan['identifier'] + ('2' if cursor else ''), installed_version='1',
                   latest_version='2', state='update_available', installation_revision='old')
        return dict(ok=True, data=dict(items=[row], next_cursor=None if cursor else 'c' * 16))
    monkeypatch.setattr(entry.provider, 'check', check)
    response = client.get('/api/skills/hub/updates')
    assert len(response.json()['updates']) == 2
    assert seen == [None, 'c' * 16]


def test_added_fields_have_correct_content_length(client, catalog):
    response = client.get('/api/skills/hub/official')
    assert int(response.headers['content-length']) == len(response.content)


def test_provider_browse_pages_and_scope_bound_cursor(client, catalog, monkeypatch):
    from tests.hermes_cli.test_skill_catalog_validation import entry as item
    entry, _plan = catalog
    calls = []
    def search(request):
        calls.append(request)
        value = item()
        value['id'] = 'page-2' if request['cursor'] else 'page-1'
        value['identifier'] = value['source'] + ':' + value['id']
        return dict(ok=True, data=dict(items=[value], next_cursor=None if request['cursor'] else 'c' * 16))
    monkeypatch.setattr(entry.provider, 'search', search)
    response = client.get('/api/skills/hub/search', params={'source': entry.source_id})
    assert response.status_code == 200, response.text
    assert response.json()['results'][0]['identifier'] == entry.source_id + ':page-1'
    cursor = response.json()['next_cursor']
    assert cursor
    second = client.get('/api/skills/hub/search', params={'source': entry.source_id, 'cursor': cursor})
    assert second.json()['results'][0]['identifier'] == entry.source_id + ':page-2'
    assert second.json()['next_cursor'] is None
    wrong_query = client.get('/api/skills/hub/search', params={'source': entry.source_id, 'cursor': cursor, 'q': 'other'})
    assert wrong_query.status_code == 400
    assert len(calls) == 2


def test_provider_search_failure_is_not_an_empty_library(client, catalog, monkeypatch):
    entry, _plan = catalog
    def offline(request):
        raise RuntimeError('private exception text')
    monkeypatch.setattr(entry.provider, 'search', offline)
    response = client.get('/api/skills/hub/search', params={'source': entry.source_id})
    assert response.status_code == 200
    assert response.json()['provider_errors'][0]['code'] == 'provider_error'
    assert 'private exception text' not in response.text


def test_public_search_fallback_does_not_wait_for_provider(client, catalog, monkeypatch):
    import threading
    from tools import skills_hub_search
    entry, _plan = catalog
    public_finished, release = threading.Event(), threading.Event()
    provider_started = threading.Event()
    def public(*args, **kwargs):
        assert provider_started.wait(2)
        public_finished.set()
        release.set()
        return [], {}, []
    def provider(request):
        provider_started.set()
        assert release.wait(2)
        return dict(ok=True, data=dict(items=[], next_cursor=None))
    monkeypatch.setattr(skills_hub_search, 'parallel_search_sources', public)
    monkeypatch.setattr(entry.provider, 'search', provider)
    try:
        response = client.get('/api/skills/hub/search', params={'q': 'example'})
        assert response.status_code == 200
        assert public_finished.is_set()
        assert not response.json().get('provider_errors')
    finally:
        release.set()


def test_installed_http_error_maps_to_named_safe_503(client, monkeypatch):
    from tests.hermes_cli.test_skill_catalog_installed import _provider, _skill
    from hermes_cli.skill_catalog.registry import snapshot
    _skill(get_hermes_home(), 'nested/folder', 'shown-name')
    manager, _record = _provider(get_hermes_home())
    entry, = snapshot()
    try:
        def offline(request):
            raise RuntimeError('private provider failure')
        monkeypatch.setattr(entry.provider, 'installed', offline)
        response = client.get('/api/skills')
        assert response.status_code == 503, response.text
        assert response.json()['error']['code'] == 'ownership_state_unavailable'
        assert 'private provider failure' not in response.text
    finally:
        manager.unload()


def test_named_profile_managed_bare_name_cannot_fall_through_to_spawn(client, monkeypatch):
    from hermes_cli import profiles
    from tests.hermes_cli.test_skill_catalog_installed import _provider, _skill
    default = get_hermes_home()
    other = default / 'profiles/other'
    _skill(other, 'nested/folder', 'shown-name')
    monkeypatch.setattr(profiles, '_get_profiles_root', lambda: default / 'profiles')
    monkeypatch.setattr(profiles, '_get_default_hermes_home', lambda: default)
    manager, record = _provider(other)
    try:
        tripwires(monkeypatch)
        result = client.post('/api/skills/hub/uninstall?profile=other', json={'name': 'shown-name'})
        assert result.status_code == 409, result.text
        assert result.json()['error']['identifier'] == record['identifier']
    finally:
        manager.unload()


def test_modified_cursor_cannot_change_provider_page(client, catalog, monkeypatch):
    import base64
    entry, _plan = catalog
    calls = []
    def search(request):
        calls.append(request)
        return dict(ok=True, data=dict(items=[], next_cursor='c' * 16))
    monkeypatch.setattr(entry.provider, 'search', search)
    first = client.get('/api/skills/hub/search', params={'source': entry.source_id}).json()
    cursor = first['next_cursor']
    payload = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
    payload['sources'][entry.source_id] = 'attacker_cursor_123'
    changed = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
    response = client.get('/api/skills/hub/search', params={'source': entry.source_id, 'cursor': changed})
    assert response.status_code == 400, response.text
    assert len(calls) == 1


def test_hidden_provider_does_not_leak_search_rows(client, catalog, monkeypatch):
    entry, _plan = catalog
    def hidden(request):
        return dict(ok=True, data=dict(state='hidden', message=''))
    def search(request):
        pytest.fail('hidden source was searched')
    monkeypatch.setattr(entry.provider, 'availability', hidden)
    monkeypatch.setattr(entry.provider, 'search', search)
    result = client.get('/api/skills/hub/search', params={'source': entry.source_id})
    assert result.status_code == 200
    assert result.json()['results'] == []
    assert result.json()['provider_errors'] == []
    assert 'provider_sources' not in client.get('/api/skills/hub/official').json()


@pytest.mark.parametrize('route', ['preview', 'scan'])
def test_provider_read_returns_exact_metadata_without_hub_fetch(client, catalog, monkeypatch, route):
    from tests.hermes_cli.test_skill_catalog_validation import entry as item, scan
    entry, plan = catalog
    calls = []
    value = item()
    value['id'] = plan['identifier'].split(':')[1]
    value['identifier'] = value['source'] + ':' + value['id']
    payload = dict(entry=value, scan=scan()) if route == 'scan' else dict(
        entry=value, skill_md='# approved', files=[], preview_complete=True)
    def read(request):
        calls.append(request)
        return dict(ok=True, data=payload)
    monkeypatch.setattr(entry.provider, 'inspect' if route == 'preview' else 'scan', read)
    tripwires(monkeypatch)
    response = client.get('/api/skills/hub/' + route,
        params={'identifier': plan['identifier'], 'version': '1'})
    assert response.status_code == 200, response.text
    assert response.json()['entry']['identifier'] == plan['identifier']
    assert calls[0]['version'] == '1'
    assert calls[0]['context']['plugin_key'] == 'fixture-catalog'


@pytest.mark.parametrize('body', [dict(review_token='short', acknowledgements=[]),
    dict(review_token='t' * 16, acknowledgements=['same', 'same'])])
def test_malformed_confirmation_cannot_spawn_or_modify_files(client, catalog, monkeypatch, body):
    entry, plan = catalog
    tripwires(monkeypatch)
    result = client.post('/api/skills/hub/confirm', json=dict(identifier=plan['identifier'], action='install', **body))
    assert result.status_code == 400
    assert entry.provider.effects == 0
    assert not (get_hermes_home() / 'skills/new').exists()


def test_no_registration_official_has_no_provider_fields(client):
    response = client.get('/api/skills/hub/official')
    assert response.status_code == 200
    assert json.loads(response.content) == {'skills': []}
