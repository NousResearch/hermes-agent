"""A failed initial installed snapshot must recover on explicit demand, not lose authority."""
import asyncio
import time
from types import SimpleNamespace

import pytest

from hermes_cli.plugins import PluginManager
from hermes_cli.skill_catalog import registry
from hermes_cli.skill_catalog.identity import source_id
from hermes_cli.skill_catalog.projection import Projection
from hermes_cli.skill_catalog.service import Service
from hermes_cli.skill_catalog.validation import CatalogError
from tests.hermes_cli.test_skill_catalog_contract import make_plugin
from tests.hermes_cli.test_skill_catalog_projection import review
from tests.hermes_cli.test_skill_catalog_service import FIXTURE


def _interrupted(home, records=(), *, isolation='in_process'):
    owner = SimpleNamespace(source_id=source_id('fixture-catalog', 'team'))
    plan = review(owner, expires_at_ms=int(time.time() * 1000) + 60000)
    installed = '''    def installed(self, request):
        self.reads = getattr(self, 'reads', 0) + 1
        if self.reads == 1 or getattr(self, 'offline', False):
            return {'ok': False, 'error': {'code': 'provider_unavailable', 'message': 'Offline'}}
        return {'ok': True, 'data': {'items': RECORDS, 'next_cursor': None}}
'''
    original = "    def installed(self, request):\n        return {'ok': True, 'data': {'items': [], 'next_cursor': None}}\n"
    assert original in FIXTURE
    body = 'PLAN = ' + repr(plan) + '\nRECORDS = ' + repr(list(records)) + '\n'
    make_plugin(home, body=body + FIXTURE.replace(original, installed))
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    config = read_user_config_raw(home / 'config.yaml')
    config['plugins']['isolation'] = isolation
    atomic_config_write(home / 'config.yaml', config)
    manager = PluginManager(scope_key=str(home))
    manager.discover_and_load()
    entry, = registry.snapshot(scope=str(home))
    assert Projection(home).read()['registrations'][entry.registration_id]['state'] == 'reconciling'
    return manager, entry, plan


@pytest.mark.parametrize('lane', ['confirm', 'native', 'generic'])
@pytest.mark.parametrize('isolation', ['in_process', 'host'])
def test_explicit_mutation_retries_failed_initial_snapshot(tmp_path, lane, isolation):
    from agent.runtime_scope import bind_profile_scope
    from hermes_cli.skill_catalog.admission import generic
    from hermes_cli.skill_catalog.facade import ContextMutations

    manager, entry, plan = _interrupted(tmp_path, isolation=isolation)
    try:
        with bind_profile_scope(tmp_path):
            if lane == 'confirm':
                result = Service(tmp_path).confirm(plan['identifier'], 'install', plan['review_token'], [])
                assert result['outcome'] == 'installed'
                assert (tmp_path / 'skills/new/SKILL.md').is_file()
            elif lane == 'native':
                api = ContextMutations(SimpleNamespace(plugin_id='fixture-catalog',
                    _manager=SimpleNamespace(scope_key=str(tmp_path))))
                result = api.admit_native(plan, {})
                assert result['admission']['registration_id'] == entry.registration_id
            else:
                target = tmp_path / 'skills/other/SKILL.md'
                with generic(target.parent):
                    target.parent.mkdir(parents=True)
                    target.write_text('local')
                assert target.read_text() == 'local'
        assert Projection(tmp_path).read()['registrations'][entry.registration_id]['state'] == 'ready'
        assert entry.provider.reads == 2
    finally:
        manager.unload()


def test_installed_read_recovers_failed_initial_snapshot_without_losing_provenance(_isolate_hermes_home):
    from hermes_constants import get_hermes_home
    from hermes_cli.web_routers.skills import get_skills
    from tests.hermes_cli.test_skill_catalog_installed import _skill

    home = get_hermes_home()
    _skill(home, 'owned', 'managed')
    record = dict(identifier=source_id('fixture-catalog', 'team') + ':owned', name='managed',
        target_relative_path='owned', installed_version='1', latest_version=None,
        managed_state='installed', installation_revision='installed-1', actions=['uninstall'])
    manager, entry, _ = _interrupted(home, [record])
    try:
        rows = asyncio.run(get_skills())
        row, = rows
        assert row['provenance'] == 'provider'
        assert row['identifier'] == record['identifier']
        assert entry.provider.reads >= 2
    finally:
        manager.unload()


def test_late_initial_snapshot_cannot_erase_a_newer_published_claim(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli.skill_catalog import installed

    manager, entry, _ = _interrupted(tmp_path)
    captured, release = threading.Event(), threading.Event()
    original = installed.installed_records
    first = [True]

    def delayed(owner, **kwargs):
        records = original(owner, **kwargs)
        if first[0]:
            first[0] = False
            captured.set()
            assert release.wait(10)
        return records

    monkeypatch.setattr(installed, 'installed_records', delayed)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(registry.retry_initialization, tmp_path)
            assert captured.wait(10)
            target = tmp_path / 'skills/owned'
            target.mkdir(parents=True)
            record = dict(identifier=entry.source_id + ':owned', name='managed',
                target_relative_path='owned', installed_version='1', latest_version=None,
                managed_state='installed', installation_revision='new-1', actions=['uninstall'])
            entry.provider.installed = lambda request: dict(ok=True, data=dict(items=[record], next_cursor=None))
            registry.retry_initialization(tmp_path)
            release.set()
            future.result(10)
        claim = Projection(tmp_path).read()['targets']['owned']
        assert claim['installation_revision'] == 'new-1'
    finally:
        release.set()
        manager.unload()


def test_failed_retry_stays_fenced_until_a_complete_snapshot_is_available(tmp_path):
    manager, entry, plan = _interrupted(tmp_path)
    try:
        entry.provider.offline = True
        with pytest.raises(CatalogError):
            Service(tmp_path).confirm(plan['identifier'], 'install', plan['review_token'], [])
        assert entry.provider.effects == 0
        assert Projection(tmp_path).read()['registrations'][entry.registration_id]['state'] == 'reconciling'
        entry.provider.offline = False
        assert Service(tmp_path).confirm(plan['identifier'], 'install', plan['review_token'], [])['outcome'] == 'installed'
    finally:
        manager.unload()
