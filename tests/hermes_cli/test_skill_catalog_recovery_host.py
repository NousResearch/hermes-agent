"""Real discovered providers recover in their own writer process after restart."""
import json
import time

import pytest

from hermes_cli.config import atomic_config_write, read_user_config_raw
from hermes_cli.plugins import PluginManager
from hermes_cli.skill_catalog.identity import source_id
from hermes_cli.skill_catalog.registry import snapshot
from hermes_cli.skill_catalog.service import Service
from tests.hermes_cli.test_skill_catalog_contract import make_plugin
from tests.hermes_cli.test_skill_catalog_projection import review
from types import SimpleNamespace


PROVIDER = r'''
import json
from pathlib import Path
from hermes_constants import get_hermes_home
from hermes_cli.skill_catalog import SkillCatalogProvider
class Catalog(SkillCatalogProvider):
    name = 'team'
    label = 'Team'
    implementation_revision = '1'
    def __init__(self, ctx):
        self.ctx = ctx
        self.home = get_hermes_home()
        self.plan = json.loads((self.home / 'plan.json').read_text())
        self.op = 'original_operation_0001'
    def pending(self):
        return dict(operation_id=self.op, identifier=self.plan['identifier'], action='install',
            state='pending', outcome=None, managed_state='installed', message='', installation_revision='revision-1')
    def installed(self, request):
        target = self.home / 'skills/new'
        rows = [] if not target.exists() else [dict(identifier=self.plan['identifier'], name='New',
            target_relative_path='new', installed_version='1', latest_version='1', managed_state='installed',
            installation_revision='revision-1', actions=['uninstall'])]
        return dict(ok=True, data=dict(items=rows, next_cursor=None))
    def reviewed(self, request):
        pending = (self.home / 'journal.json').exists()
        return dict(ok=True, data=dict(kind='pending' if pending else 'unstarted', review=self.plan,
            operation=self.pending() if pending else None))
    def operation(self, request):
        return dict(ok=True, data=self.pending())
    def confirm(self, request):
        api = self.ctx.skill_catalog_mutations
        admission = request['admission']
        api.bind({key: admission[key] for key in ('registration_id','review_id','reservation_id',
            'expected_path_revision','expected_installation_revision')} | dict(operation_id=self.op))
        journal = self.home / 'journal.json'
        if not journal.exists():
            with api.commit(admission, self.op):
                target = self.home / 'skills/new'
                target.mkdir(parents=True)
                (target / 'SKILL.md').write_bytes(b'approved')
                journal.write_text(json.dumps(dict(digest=EXPECTED_DIGEST)))
                raise RuntimeError('interrupt after rename before local marker')
        with api.recover(self.op) as recovery:
            assert recovery.observed_revision == json.loads(journal.read_text())['digest']
            recovery.decide(dict(kind='reconciled', journal_digest=EXPECTED_DIGEST, result=self.pending()))
        return dict(ok=True, data=self.pending())
def register(ctx):
    ctx.register_skill_catalog(Catalog(ctx))
'''


@pytest.mark.parametrize('isolation', ['host', 'in_process'])
@pytest.mark.parametrize('restart', [False, True])
def test_discovered_provider_recovers_same_operation_after_rename(tmp_path, isolation, restart):
    from tests.hermes_cli.test_skill_catalog_recovery import tree_digest
    plan = review(SimpleNamespace(source_id=source_id('fixture-catalog', 'team')),
                  expires_at_ms=int(time.time() * 1000) + 60000)
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    body = 'EXPECTED_DIGEST = ' + repr(tree_digest({'SKILL.md': b'approved'})) + '\n' + PROVIDER
    make_plugin(tmp_path, body=body)
    cfg = read_user_config_raw(tmp_path / 'config.yaml')
    cfg['plugins']['isolation'] = isolation
    atomic_config_write(tmp_path / 'config.yaml', cfg)
    manager = PluginManager(scope_key=str(tmp_path))
    manager.discover_and_load()
    try:
        service = Service(tmp_path)
        with pytest.raises(Exception):
            service.confirm(plan['identifier'], 'install', plan['review_token'], [])
        target = tmp_path / 'skills/new/SKILL.md'
        before = target.stat()
        if restart:
            old_registration = snapshot(scope=str(tmp_path))[0].registration_id
            manager.unload()
            manager = PluginManager(scope_key=str(tmp_path))
            manager.discover_and_load()
            assert snapshot(scope=str(tmp_path))[0].registration_id != old_registration
        result = service.confirm(plan['identifier'], 'install', plan['review_token'], [])
        assert result['state'] == 'pending'
        assert result['operation_id'] == 'original_operation_0001'
        after = target.stat()
        assert (after.st_ino, after.st_mtime_ns, after.st_ctime_ns, after.st_size) == (
            before.st_ino, before.st_mtime_ns, before.st_ctime_ns, before.st_size)
        row, = snapshot(scope=str(tmp_path))[0].mutations.projection.read()['reservations'].values()
        assert row['local_result'] == result
    finally:
        manager.unload()
