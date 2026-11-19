"""One caller deadline survives bootstrap, successive reads and local admission."""
import time

import pytest

pytest_plugins = ['tests.hermes_cli.test_skill_catalog_http']


class Clock:
    def __init__(self):
        self.value = time.monotonic()
        self.wall = time.time()
        self.start = self.value

    def monotonic(self):
        return self.value

    def time(self):
        return self.wall + self.value - self.start

    def advance(self, seconds):
        self.value += seconds


def _clock(monkeypatch):
    from hermes_cli.skill_catalog import http, service, worker, installed
    clock = Clock()
    for module in (http, service, worker, installed):
        monkeypatch.setattr(module, 'time', clock)
    return clock


@pytest.mark.parametrize('route,budget', [('preview', 15), ('scan', 30), ('review', 60),
    ('confirm', 60), ('operation', 3), ('install', 60)])
def test_bootstrap_spends_callers_budget_before_provider_dispatch(client, catalog, monkeypatch, route, budget):
    from hermes_cli.skill_catalog import http
    from hermes_cli.skill_catalog.service import Service
    entry, plan = catalog
    clock = _clock(monkeypatch)
    calls = []

    def lookup(*args, **kwargs):
        clock.advance(budget + 1)
        return Service(entry.scope)

    def invoked(request):
        calls.append(request)
        return dict(ok=False, error=dict(code='not_found', message='Absent'))

    monkeypatch.setattr(http, '_service', lookup)
    for method in ('inspect', 'scan', 'review', 'reviewed', 'operation'):
        monkeypatch.setattr(entry.provider, method, invoked)
    body = dict(identifier=plan['identifier'])
    if route in ('review', 'confirm'):
        body['action'] = 'install'
    if route in ('confirm', 'install'):
        body.update(review_token=plan['review_token'], acknowledgements=[])
    if route == 'operation':
        body['operation_id'] = 'a' * 32
    response = (client.get('/api/skills/hub/' + route, params=body) if route in ('preview', 'scan', 'operation')
                else client.post('/api/skills/hub/' + route, json=body))
    assert response.status_code == 504, response.text
    assert response.json()['error']['code'] == 'timeout'
    assert calls == []
    assert entry.provider.effects == 0


def test_confirm_authority_read_and_effect_share_original_deadline(client, catalog, monkeypatch):
    entry, plan = catalog
    clock = _clock(monkeypatch)
    original = entry.mutations.reconcile

    def reconcile(**kwargs):
        result = original(**kwargs)
        clock.advance(61)
        return result

    monkeypatch.setattr(entry.mutations, 'reconcile', reconcile)
    response = client.post('/api/skills/hub/confirm', json=dict(identifier=plan['identifier'],
        action='install', review_token=plan['review_token'], acknowledgements=[]))
    assert response.status_code == 504, response.text
    assert entry.provider.effects == 0
    assert entry.mutations.projection.read()['reservations'] == {}


@pytest.mark.parametrize('method', ['sources', 'installed'])
def test_sequential_provider_reads_do_not_multiply_budget(tmp_path, monkeypatch, method):
    from hermes_cli.skill_catalog import service, installed, registry
    from hermes_cli.skill_catalog.service import Service
    from tests.hermes_cli.test_skill_catalog_workers import make_entry, Catalog
    from hermes_cli.plugins_loader import _plugin_home_scope
    clock = _clock(monkeypatch)
    calls = []

    class Reader(Catalog):
        def availability(self, request):
            calls.append(self)
            clock.advance(2)
            return dict(ok=True, data=dict(state='ready', message=''))

        def installed(self, request):
            calls.append(self)
            clock.advance(2)
            return dict(ok=True, data=dict(items=[], next_cursor=None))

    entries = [make_entry(tmp_path, Reader(), key='catalog-' + str(n)) for n in range(3)]
    monkeypatch.setattr(service, 'snapshot', lambda **kwargs: tuple(entries))
    if method == 'sources':
        rows = Service(tmp_path).sources()
        assert [row['state'] for row in rows] == ['ready', 'unavailable', 'unavailable']
    else:
        from hermes_cli.skill_catalog.projection import Projection
        from hermes_cli.skill_catalog.validation import CatalogError
        monkeypatch.setattr(registry, 'snapshot', lambda **kwargs: tuple(entries))
        monkeypatch.setattr(installed, 'snapshot', lambda **kwargs: tuple(entries))
        handles = [Projection(tmp_path).start(entry) for entry in entries]
        try:
            for entry in entries:
                Projection(tmp_path).publish(entry, [])
            with _plugin_home_scope(tmp_path), pytest.raises(CatalogError):
                installed.extend_installed_rows([])
        finally:
            for handle in handles:
                handle.close()
    assert len(calls) == 2


@pytest.mark.parametrize('route', ['updates', 'update', 'official', 'sources'])
def test_public_work_does_not_grant_provider_a_replacement_budget(client, catalog, monkeypatch, route):
    from hermes_cli.web_routers import skills
    entry, _plan = catalog
    clock = _clock(monkeypatch)
    calls = []

    def public(*args, **kwargs):
        clock.advance(20)
        return dict(ok=True, pid=123)

    def available(request):
        calls.append('availability')
        return dict(ok=True, data=dict(state='ready', message=''))

    def check(request):
        calls.append('check')
        return dict(ok=True, data=dict(items=[], next_cursor=None))

    monkeypatch.setattr(entry.provider, 'availability', available)
    monkeypatch.setattr(entry.provider, 'check', check)
    monkeypatch.setattr(skills, 'spawn_profile_action', public)
    if route == 'updates':
        from tools import skills_hub_install
        monkeypatch.setattr(skills_hub_install, 'check_for_skill_updates', lambda: (public(), [])[1])
    elif route == 'sources':
        monkeypatch.setattr(skills, '_hub_sources', lambda profile: (public(), [])[1])
    elif route == 'official':
        from tools.skills_hub_official import OptionalSkillSource
        monkeypatch.setattr(OptionalSkillSource, 'list_local', lambda self: (public(), [])[1])
    response = (client.post('/api/skills/hub/update', json={}) if route == 'update'
                else client.get('/api/skills/hub/' + route))
    assert response.status_code == 200, response.text
    assert calls == []
    data = response.json()
    if route in ('update', 'updates'):
        assert data['provider_errors'][0]['code'] == 'timeout'
    else:
        assert data.get('provider_sources', data.get('sources'))[0]['state'] == 'unavailable'
