"""Common service routes exact authority before filesystem prerequisites."""
import uuid
import time

import pytest

from hermes_cli.plugins import PluginManager
from hermes_cli.skill_catalog.identity import source_id
from hermes_cli.skill_catalog.validation import CatalogError
from tests.hermes_cli.test_skill_catalog_contract import make_plugin
from tests.hermes_cli.test_skill_catalog_projection import review
from types import SimpleNamespace


FIXTURE = r'''
import uuid
from pathlib import Path
from hermes_constants import get_hermes_home
from hermes_cli.skill_catalog import SkillCatalogProvider
class Catalog(SkillCatalogProvider):
    name = 'team'
    label = 'Team skills'
    implementation_revision = '1'
    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = PLAN
        self.saved = None
        self.effects = 0
    def search(self, request):
        return {'ok': True, 'data': {'items': [], 'next_cursor': None}}
    def installed(self, request):
        return {'ok': True, 'data': {'items': [], 'next_cursor': None}}
    def inspect(self, request):
        return {'ok': False, 'error': {'code': 'not_found', 'message': 'Absent'}}
    def review(self, request):
        return {'ok': True, 'data': self.plan}
    def reviewed(self, request):
        return {'ok': True, 'data': {'kind': 'completed' if self.saved else 'unstarted',
                'review': self.plan, 'operation': self.saved}}
    def operation(self, request):
        return {'ok': True, 'data': self.saved}
    def confirm(self, request):
        if self.saved:
            return {'ok': True, 'data': self.saved}
        admission = request['admission']
        op = uuid.uuid4().hex
        self.ctx.skill_catalog_mutations.bind({k: admission[k] for k in (
            'registration_id', 'review_id', 'reservation_id', 'expected_path_revision',
            'expected_installation_revision')} | {'operation_id': op})
        with self.ctx.skill_catalog_mutations.commit(admission, op) as commit:
            target = get_hermes_home() / 'skills' / commit.target_relative_path
            target.mkdir(parents=True)
            (target / 'SKILL.md').write_text('---\nname: approved\ndescription: approved\n---\nApproved')
            self.effects += 1
            self.saved = {'operation_id': op, 'identifier': self.plan['identifier'],
                'action': 'install', 'state': 'completed', 'outcome': 'installed',
                'managed_state': 'installed', 'message': '', 'installation_revision': 'new-1'}
            commit.settle(self.saved)
        return {'ok': True, 'data': self.saved}
def register(ctx):
    ctx.register_skill_catalog(Catalog(ctx))
'''


def fixture(home, *, isolation='in_process'):
    from hermes_cli.skill_catalog.registry import snapshot
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    owner = SimpleNamespace(source_id=source_id('fixture-catalog', 'team'))
    plan = review(owner, expires_at_ms=int(time.time() * 1000) + 60000)
    # repr keeps nested data literal and the fixture remains a real discovered plugin.
    body = 'PLAN = ' + repr(plan) + '\n' + FIXTURE
    make_plugin(home, body=body)
    config = read_user_config_raw(home / 'config.yaml')
    config['plugins']['isolation'] = isolation
    atomic_config_write(home / 'config.yaml', config)
    manager = PluginManager(scope_key=str(home))
    manager.discover_and_load()
    entry, = snapshot(scope=str(home))
    return manager, entry, plan


@pytest.mark.parametrize('isolation', ['in_process', 'host'])
def test_real_plugin_review_confirm_effect_then_completed_replay(tmp_path, isolation):
    from hermes_cli.skill_catalog.service import Service
    manager, entry, plan = fixture(tmp_path, isolation=isolation)
    try:
        service = Service(tmp_path)
        assert service.review(plan['identifier'], 'install') == plan
        result = service.confirm(plan['identifier'], 'install', plan['review_token'], [])
        assert result['state'] == 'completed'
        assert (tmp_path / 'skills/new/SKILL.md').is_file()
        assert entry.provider.effects == 1
        assert service.confirm(plan['identifier'], 'install', plan['review_token'], []) == result
        assert entry.provider.effects == 1
        assert service.operation(plan['identifier'], result['operation_id'])['settlement'] == 'settled'
        assert entry.mutations.projection.read()['reservations'] == {}
    finally:
        manager.unload()


def test_operation_read_reconciles_terminal_journal_without_second_effect(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog.service import Service
    from hermes_cli.skill_catalog import projection as storage
    manager, entry, plan = fixture(tmp_path)
    original_stage = storage._stage
    original_operation = entry.provider.operation
    try:
        def fail_stage(*args):
            raise OSError('disk failure')
        def offline(request):
            raise RuntimeError('offline')
        monkeypatch.setattr(entry.provider, 'operation', offline)
        # Fail only the terminal projection save, after files and provider journal.
        original_settle = entry.mutations.settle
        def failed_settle(payload):
            with monkeypatch.context() as scoped:
                scoped.setattr(storage, '_stage', fail_stage)
                return original_settle(payload)
        monkeypatch.setattr(entry.mutations, 'settle', failed_settle)
        service = Service(tmp_path)
        with pytest.raises(CatalogError):
            service.confirm(plan['identifier'], 'install', plan['review_token'], [])
        saved = entry.provider.saved
        before = entry.mutations.projection.read()
        assert next(iter(before['reservations'].values()))['state'] == 'unknown'
        # Token replay is strictly read-only, even while projection finalization is pending.
        assert service.confirm(plan['identifier'], 'install', plan['review_token'], []) == saved
        assert entry.mutations.projection.read() == before
        monkeypatch.setattr(entry.provider, 'operation', original_operation)
        result = service.operation(plan['identifier'], saved['operation_id'])
        assert result['settlement'] == 'settled'
        assert not entry.mutations.projection.read()['reservations']
        assert entry.provider.effects == 1
    finally:
        monkeypatch.setattr(storage, '_stage', original_stage)
        manager.unload()


@pytest.mark.parametrize('isolation', ['in_process', 'host'])
def test_one_plugin_multiple_catalogs_keep_permit_routing_unambiguous(tmp_path, isolation):
    from hermes_cli.skill_catalog.service import Service
    from hermes_cli.skill_catalog import registry
    body = FIXTURE + '''
_original_register = register
class OtherCatalog(Catalog):
    name = 'other'
def register(ctx):
    _original_register(ctx)
    ctx.register_skill_catalog(OtherCatalog(ctx))
'''
    owner = SimpleNamespace(source_id=source_id('fixture-catalog', 'team'))
    plan = review(owner, expires_at_ms=int(time.time() * 1000) + 60000)
    make_plugin(tmp_path, body='PLAN = ' + repr(plan) + '\n' + body)
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    config = read_user_config_raw(tmp_path / 'config.yaml')
    config['plugins']['isolation'] = isolation
    atomic_config_write(tmp_path / 'config.yaml', config)
    manager = PluginManager(scope_key=str(tmp_path))
    manager.discover_and_load()
    try:
        assert len(registry.snapshot(scope=str(tmp_path))) == 2
        result = Service(tmp_path).confirm(plan['identifier'], 'install', plan['review_token'], [])
        assert result['outcome'] == 'installed'
        assert (tmp_path / 'skills/new/SKILL.md').is_file()
        from hermes_cli.skill_catalog.projection import Projection
        assert not Projection(tmp_path).read()['reservations']
        if isolation == 'host':
            from hermes_cli.plugins_manifest import manifest_key
            context = manager._plugin_host()._contexts[manifest_key(manager._plugins['fixture-catalog'].manifest)]
            api = context.skill_catalog_mutations
            assert not api._permits
            assert api.release({'permit_id': 'already-released'}) is True
    finally:
        manager.unload()


def test_confirm_wrong_ack_or_identity_has_zero_effect(tmp_path):
    from hermes_cli.skill_catalog.service import Service
    manager, entry, plan = fixture(tmp_path)
    try:
        service = Service(tmp_path)
        for identifier, acknowledgements in ((plan['identifier'], ['unknown']),
                                             (plan['identifier'] + 'other', [])):
            with pytest.raises(CatalogError):
                service.confirm(identifier, 'install', plan['review_token'], acknowledgements)
        assert entry.provider.effects == 0
        assert entry.mutations.projection.read()['reservations'] == {}
    finally:
        manager.unload()
