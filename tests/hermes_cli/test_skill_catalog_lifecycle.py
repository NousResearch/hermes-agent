"""Real plugin lifecycle refuses incomplete drain without forgetting ownership."""
import time

import pytest

from hermes_cli.skill_catalog import registry
from hermes_cli.skill_catalog.locks import FileLease
from hermes_cli.skill_catalog.projection import Projection
from hermes_cli.skill_catalog.validation import CatalogError
from tests.hermes_cli.test_skill_catalog_locks import _held_writer
from tests.hermes_cli.test_skill_catalog_service import fixture


@pytest.mark.parametrize('all_plugins', [False, True])
def test_busy_unload_retains_ledger_and_can_retry_after_real_writer_exits(tmp_path, all_plugins):
    manager, entry, _ = fixture(tmp_path)
    catalog_handle, = [r for r in manager._registration_order if r.kind == 'skill_catalog']
    try:
        with _held_writer(tmp_path, ('new',)):
            with pytest.raises(CatalogError) as caught:
                manager.unload(None if all_plugins else 'fixture-catalog')
            assert caught.value.code == 'busy'
            assert catalog_handle.active
            assert catalog_handle in manager._ownership_ledger['fixture-catalog']
            assert 'fixture-catalog' in manager._plugins
            assert entry.stopped
            assert registry.snapshot(scope=str(tmp_path)) == ()
            with pytest.raises(CatalogError) as active:
                with FileLease(entry.projection_handle.projection.directory / 'presence.lock',
                               deadline=time.monotonic()):
                    pytest.fail('live writer lost presence barrier')
            assert active.value.code == 'busy'
        assert manager.unload(None if all_plugins else 'fixture-catalog')
        assert not catalog_handle.active
        assert Projection(tmp_path).read()['registrations'] == {}
        assert not entry.mutations._thread.is_alive()
    finally:
        manager.unload()


def test_direct_handle_disposal_can_retry_after_writer_drain(tmp_path):
    manager, entry, _ = fixture(tmp_path)
    handle, = [r for r in manager._registration_order if r.kind == 'skill_catalog']
    try:
        with _held_writer(tmp_path, ('new',)):
            with pytest.raises(CatalogError) as caught:
                handle.dispose()
            assert caught.value.code == 'busy'
            assert handle.active
        handle.dispose()
        assert not handle.active
        assert not Projection(tmp_path).read()['registrations']
        assert not entry.mutations._thread.is_alive()
    finally:
        manager.unload()


def test_unload_fences_new_commits_but_waits_for_entered_writer_release(tmp_path):
    manager, entry, plan = fixture(tmp_path)
    api = entry.mutations
    admission = api.projection.reserve(entry, plan)
    api.bind({key: admission[key] for key in ('registration_id', 'review_id', 'reservation_id',
        'expected_path_revision', 'expected_installation_revision')} | {'operation_id': 'operation_0123456789'})
    acquired = api.acquire(dict(admission=admission, operation_id='operation_0123456789'))
    try:
        with pytest.raises(CatalogError) as caught:
            manager.unload('fixture-catalog')
        assert caught.value.code == 'busy'
        assert api._thread.is_alive()
        with pytest.raises(CatalogError):
            api.validate(dict(permit_id=acquired['permit_id']))
        assert api.release(dict(permit_id=acquired['permit_id']))
        assert manager.unload('fixture-catalog')
        assert not api._thread.is_alive()
        assert not Projection(tmp_path).read()['registrations']
    finally:
        api.release(dict(permit_id=acquired['permit_id'])) if not api._closed else None
        manager.unload()


def test_other_registration_disposal_keeps_existing_best_effort_semantics():
    from hermes_cli.plugins_ledger import PluginRegistration
    calls = []

    def failed_release():
        calls.append('release')
        raise RuntimeError('fixture')

    handle = PluginRegistration('hook', 'fixture', failed_release)
    with pytest.raises(RuntimeError):
        handle.dispose()
    handle.dispose()
    assert not handle.active
    assert calls == ['release']
