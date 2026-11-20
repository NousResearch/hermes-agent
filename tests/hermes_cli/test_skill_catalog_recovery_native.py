"""Native re-admission must consult a matching pending provider journal after restart."""
import hashlib

import pytest

from hermes_constants import hermes_home_key
from hermes_cli.skill_catalog.validation import CatalogError
from tests.hermes_cli.test_skill_catalog_native import context
from tests.hermes_cli.test_skill_catalog_service import fixture
from tests.hermes_cli.test_skill_catalog_mutations import operation


@pytest.mark.parametrize('effect_present', [True, False])
def test_native_after_reregistration_revalidates_exact_pending_operation(tmp_path, monkeypatch, effect_present):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    manager, entry, plan = fixture(tmp_path)
    api = context(entry).skill_catalog_mutations
    first = api.admit_native(plan, {'acknowledgements': []})
    manager.unload()
    if effect_present:
        (tmp_path / 'skills/new').mkdir(parents=True)
    from hermes_cli.plugins import PluginManager
    from hermes_cli.skill_catalog.registry import snapshot
    manager = PluginManager(scope_key=str(tmp_path))
    manager.discover_and_load()
    entry, = snapshot(scope=str(tmp_path))
    try:
        pending = operation(entry, first['operation_id'], state='pending', outcome=None)
        calls = []
        def journal(request):
            calls.append(request['operation_id'])
            return dict(ok=True, data=pending)
        monkeypatch.setattr(entry.provider, 'operation', journal)
        result = context(entry).skill_catalog_mutations.admit_native(plan, {'acknowledgements': []})
        assert result['operation_id'] == first['operation_id']
        assert result['admission']['registration_id'] != first['admission']['registration_id']
        assert calls == [first['operation_id']]
        row = entry.mutations.projection.read()['reservations'][result['admission']['reservation_id']]
        assert row['recovery'] is True
        with pytest.raises(CatalogError):
            with context(entry).skill_catalog_mutations.commit(result['admission'], result['operation_id']):
                pytest.fail('recovery-only native reservation entered ordinary commit')
        with context(entry).skill_catalog_mutations.recover(result['operation_id']) as recovery:
            if effect_present:
                recovery.decide(dict(kind='reconciled', journal_digest=recovery.observed_revision, result=pending))
            else:
                recovery.decide(dict(kind='retry_effect'))
                (tmp_path / 'skills/new').mkdir(parents=True)
                recovery.settle(pending)
    finally:
        manager.unload()


@pytest.mark.parametrize('change', [dict(state='completed', outcome='installed'),
                                    dict(operation_id='another_operation_0001')])
def test_native_no_binding_never_adopts_wrong_or_terminal_journal(tmp_path, monkeypatch, change):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    manager, entry, plan = fixture(tmp_path)
    try:
        (tmp_path / 'skills/new').mkdir(parents=True)
        op = hashlib.sha256((hermes_home_key(tmp_path) + '\0' + entry.source_id + '\0' + plan['review_id']).encode()).hexdigest()
        pending = operation(entry, op, state='pending', outcome=None, **change) if 'state' not in change else operation(entry, op, **change)
        monkeypatch.setattr(entry.provider, 'operation', lambda request: dict(ok=True, data=pending))
        with pytest.raises(CatalogError) as caught:
            context(entry).skill_catalog_mutations.admit_native(plan, {'acknowledgements': []})
        assert caught.value.code == 'review_changed'
        assert not entry.mutations.projection.read()['reservations']
    finally:
        manager.unload()
