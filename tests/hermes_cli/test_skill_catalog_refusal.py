"""Generic provider policy limits: one target, metadata-only author enrollment."""
import json
import time
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home
from hermes_cli.config import atomic_config_write, read_user_config_raw
from hermes_cli.plugins import PluginManager
from hermes_cli.skill_catalog.identity import source_id
from hermes_cli.skill_catalog.registry import snapshot
from hermes_cli.skill_catalog.service import Service
from tests.hermes_cli.test_skill_catalog_contract import PLUGIN, make_plugin
from tests.hermes_cli.test_skill_catalog_installed import _skill
from tests.hermes_cli.test_skill_catalog_projection import record, review


REFUSING_PROVIDER = '''
import json
import uuid
from hermes_constants import get_hermes_home
class RefusingCatalog(Catalog):
    def __init__(self, ctx):
        self.ctx = ctx
        self.saved = None
    def installed(self, request):
        return dict(ok=True, data=dict(items=[RECORD], next_cursor=None))
    def review(self, request):
        return dict(ok=True, data=PLAN)
    def reviewed(self, request):
        return dict(ok=True, data=dict(kind='completed' if self.saved else 'unstarted',
            review=PLAN, operation=self.saved))
    def operation(self, request):
        return dict(ok=True, data=self.saved)
    def confirm(self, request):
        admission = request['admission']
        operation_id = uuid.uuid4().hex
        self.ctx.skill_catalog_mutations.bind({key: value for key, value in admission.items()
            if key != 'generation'} | dict(operation_id=operation_id))
        with self.ctx.skill_catalog_mutations.commit(admission, operation_id) as hold:
            target = get_hermes_home() / 'skills' / hold.target_relative_path
            if (target / 'SKILL.md').read_text() != 'original installed bytes':
                self.saved = dict(operation_id=operation_id, identifier=PLAN['identifier'],
                    action='update', state='failed', outcome='refused', managed_state='installed',
                    message='Local edits present at managed. Archive or rename it in Skills, then update.',
                    installation_revision=RECORD['installation_revision'])
                hold.settle(self.saved)
                return dict(ok=True, data=self.saved)
            raise AssertionError('fixture must contain local edits')
    def enroll_author(self, relative):
        # Publication metadata is the provider's record, not a filesystem effect
        # on skills and not an installed() claim.
        path = get_hermes_home() / 'author-enrollment.json'
        path.write_text(json.dumps(dict(target=relative, published=True)))
        return True
def register(ctx):
    ctx.register_skill_catalog(RefusingCatalog(ctx))
'''


def _files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('isolation', ['in_process', 'host'])
def test_modified_update_refuses_second_target_without_changing_files_or_claims(
        _isolate_hermes_home, isolation):
    home = get_hermes_home()
    _skill(home, 'managed', 'edited')
    _skill(home, 'authored', 'authored')
    owner = SimpleNamespace(source_id=source_id('fixture-catalog', 'team'))
    installed = record(owner, target='managed')
    plan = review(owner, target='managed', action='update', installation_revision='receipt-1',
                  expires_at_ms=int(time.time() * 1000) + 60000)
    body = 'RECORD = ' + repr(installed) + '\nPLAN = ' + repr(plan) + '\n' + PLUGIN + REFUSING_PROVIDER
    make_plugin(home, body=body)
    config = read_user_config_raw(home / 'config.yaml')
    config['plugins']['isolation'] = isolation
    atomic_config_write(home / 'config.yaml', config)
    manager = PluginManager(scope_key=str(home))
    manager.discover_and_load()
    entry, = snapshot(scope=str(home))
    try:
        service = Service(home)
        before_files = _files(home / 'skills')
        before = entry.mutations.projection.read()
        assert service.review(plan['identifier'], 'update') == plan
        result = service.confirm(plan['identifier'], 'update', plan['review_token'], [])
        assert result['state'] == 'failed'
        assert result['outcome'] == 'refused'
        assert 'Archive or rename' in result['message']
        assert result['installation_revision'] == 'receipt-1'
        after = entry.mutations.projection.read()
        assert after['targets'] == before['targets']
        assert after['path_revisions'] == before['path_revisions']
        assert after['reservations'] == {}
        assert _files(home / 'skills') == before_files
        assert service.confirm(plan['identifier'], 'update', plan['review_token'], []) == result
        assert _files(home / 'skills') == before_files
        assert entry.mutations.projection.read() == after

        assert entry.provider.enroll_author('authored') is True
        assert json.loads((home / 'author-enrollment.json').read_text()) == dict(
            target='authored', published=True)
        assert _files(home / 'skills') == before_files
        assert entry.mutations.projection.read() == after
        from hermes_cli.skill_catalog.worker import call
        assert call(entry, 'installed', dict(cursor=None))['items'] == [installed]
        assert 'authored' not in after['targets']
    finally:
        manager.unload()
