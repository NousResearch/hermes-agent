"""Provider HTTP admission rejects ambiguous selectors before hub or plugin work."""
import json

import pytest

from hermes_constants import get_hermes_home

pytest_plugins = ['tests.hermes_cli.test_skill_catalog_http']


def _no_work(monkeypatch, entry):
    from hermes_cli.web_routers import skills
    calls = []
    original = entry.provider.review

    def review(request):
        calls.append(request)
        return original(request)

    def forbidden(*args, **kwargs):
        pytest.fail('ambiguous catalog request reached the hub')

    monkeypatch.setattr(entry.provider, 'review', review)
    monkeypatch.setattr(skills, 'spawn_profile_action', forbidden)
    monkeypatch.setattr(skills, '_resolve_hub_skill', forbidden)
    return calls


@pytest.mark.parametrize('query', [
    'profile=default&profile=default',
    'profile=current&profile=default',
    'profile=default&profile=missing',
])
def test_duplicate_query_profile_refuses_before_any_effect(client, catalog, monkeypatch, query):
    entry, plan = catalog
    calls = _no_work(monkeypatch, entry)
    result = client.post('/api/skills/hub/review?' + query,
        json=dict(identifier=plan['identifier'], action='install'))
    assert result.status_code == 400, result.text
    assert result.json()['error']['code'] == 'invalid_request'
    assert calls == []
    assert entry.provider.effects == 0
    assert not (get_hermes_home() / 'skills/new').exists()


@pytest.mark.parametrize('profiles', [
    '"profile":"default","profile":"default"',
    '"profile":"current","profile":"default"',
    '"profile":"default","profile":"missing"',
])
def test_duplicate_body_profile_refuses_before_any_effect(client, catalog, monkeypatch, profiles):
    entry, plan = catalog
    calls = _no_work(monkeypatch, entry)
    raw = '{"identifier":' + json.dumps(plan['identifier']) + ',"action":"install",' + profiles + '}'
    result = client.post('/api/skills/hub/review', content=raw)
    assert result.status_code == 400, result.text
    assert result.json()['error']['code'] == 'invalid_request'
    assert calls == []
    assert entry.provider.effects == 0


@pytest.mark.parametrize('profile', [None, 0, False, [], {}])
def test_explicit_nonstring_profile_is_not_an_implicit_current_profile(client, catalog, profile):
    _entry, plan = catalog
    result = client.post('/api/skills/hub/review', json=dict(
        identifier=plan['identifier'], action='install', profile=profile))
    assert result.status_code == 400, result.text
    assert result.json()['error']['code'] == 'invalid_request'


def test_matching_query_and_body_profile_remain_valid(client, catalog):
    _entry, plan = catalog
    result = client.post('/api/skills/hub/review?profile=default', json=dict(
        identifier=plan['identifier'], action='install', profile='default'))
    assert result.status_code == 200, result.text
    assert result.json() == plan


@pytest.mark.parametrize('query', ['profile=default&profile=default', 'profile=current&profile=default'])
def test_providerless_duplicate_profile_preserves_stock_extension(client, query):
    result = client.post('/api/skills/hub/review?' + query, content='not json')
    assert result.status_code == 405
    assert result.content == b'{"detail":"Method Not Allowed"}'
    assert result.headers['allow'] == 'GET'


def test_providerless_duplicate_body_profile_preserves_stock_extension(client):
    result = client.post('/api/skills/hub/review', content='{"profile":"default","profile":"default"}')
    assert result.status_code == 405
    assert result.content == b'{"detail":"Method Not Allowed"}'


def test_authentication_precedes_body_ingestion_and_catalog_lookup(client, catalog, monkeypatch):
    from hermes_cli import web_server
    from hermes_cli.skill_catalog import http
    calls = []

    def lookup(*args, **kwargs):
        calls.append('lookup')
        pytest.fail('unauthenticated request looked up a catalog')

    def body():
        calls.append('body')
        yield b'{"identifier":'

    monkeypatch.setattr(http, '_service', lookup)
    client.headers.pop(web_server._SESSION_HEADER_NAME)
    result = client.post('/api/skills/hub/review', content=body())
    assert result.status_code in (401, 403)
    assert calls == []


def test_ambiguous_destructive_profile_cannot_start_hub_or_review(client, catalog, monkeypatch):
    from agent import secret_scope
    entry, plan = catalog
    calls = _no_work(monkeypatch, entry)
    monkeypatch.setattr(secret_scope, 'is_multiplex_active', lambda: True)
    result = client.post('/api/skills/hub/review', json=dict(identifier=plan['identifier'], action='install'))
    assert result.status_code == 400, result.text
    assert 'explicit profile' in result.json()['detail']
    assert calls == []
    assert entry.provider.effects == 0


@pytest.mark.parametrize('source', ['all', 'provider'])
def test_duplicate_search_profile_cannot_launch_public_or_provider_work(client, catalog, monkeypatch, source):
    from hermes_cli.web_routers import skills
    entry, _plan = catalog
    calls = []
    original = skills._hub_sources

    def public(profile):
        calls.append('public')
        return original(profile)

    def availability(request):
        calls.append('provider')
        return dict(ok=True, data=dict(state='ready', message=''))

    monkeypatch.setattr(skills, '_hub_sources', public)
    monkeypatch.setattr(entry.provider, 'availability', availability)
    selected = entry.source_id if source == 'provider' else 'all'
    result = client.get('/api/skills/hub/search', params=[('q', 'example'),
        ('source', selected), ('profile', 'default'), ('profile', 'default')])
    assert result.status_code == 400, result.text
    assert calls == []


def test_sources_installed_map_preserves_stock_entry_value_shape(client, catalog, monkeypatch):
    entry, plan = catalog
    record = dict(identifier=plan['identifier'], name='shown', target_relative_path='new',
        installed_version='1', latest_version='1', managed_state='installed',
        installation_revision='installed-1', actions=['update', 'uninstall'])
    monkeypatch.setattr(entry.provider, 'installed', lambda request: dict(
        ok=True, data=dict(items=[record], next_cursor=None)))
    result = client.get('/api/skills/hub/sources')
    assert result.status_code == 200, result.text
    assert result.json()['installed'][plan['identifier']] == dict(
        name='shown', trust_level='community', scan_verdict='unknown')


def test_update_all_returns_only_update_available_candidates(client, catalog, monkeypatch):
    from hermes_cli.web_routers import skills
    entry, _plan = catalog
    states = ['current', 'update_available', 'removed', 'pending', 'unavailable', 'blocked']
    rows = [dict(identifier=entry.source_id + ':' + state, installed_version='1', latest_version='2',
        state=state, installation_revision='installed-1') for state in states]
    monkeypatch.setattr(entry.provider, 'check', lambda request: dict(
        ok=True, data=dict(items=rows, next_cursor=None)))
    monkeypatch.setattr(skills, 'spawn_profile_action', lambda *args, **kwargs: dict(ok=True, pid=123))
    result = client.post('/api/skills/hub/update', json={})
    assert result.status_code == 200, result.text
    assert result.json()['provider_review_candidates'] == [rows[1]]
    assert entry.provider.effects == 0
