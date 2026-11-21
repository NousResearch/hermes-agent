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


HOST_WRITER = '''
    def hold_commit(self, admission):
        operation_id = 'operation_0123456789'
        self.ctx.skill_catalog_mutations.bind({key: admission[key] for key in (
            'registration_id', 'review_id', 'reservation_id', 'expected_path_revision',
            'expected_installation_revision')} | {'operation_id': operation_id})
        import threading
        self.release_writer = threading.Event()
        ready = threading.Event()
        def write():
            with self.ctx.skill_catalog_mutations.commit(admission, operation_id):
                ready.set()
                self.release_writer.wait(60)
        self.writer = threading.Thread(target=write, daemon=False)
        self.writer.start()
        assert ready.wait(10)
        return True
    def release_commit(self):
        self.release_writer.set()
        self.writer.join(10)
        return True
'''


def hosted_writer(home):
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    from hermes_cli.plugins import PluginManager
    from tests.hermes_cli.test_skill_catalog_contract import make_plugin
    from tests.hermes_cli.test_skill_catalog_projection import review
    from tests.hermes_cli.test_skill_catalog_service import FIXTURE
    from hermes_cli.skill_catalog.identity import source_id
    from types import SimpleNamespace

    plan = review(SimpleNamespace(source_id=source_id('fixture-catalog', 'team')),
                  expires_at_ms=int(time.time() * 1000) + 60000)
    body = 'PLAN = ' + repr(plan) + '\n' + FIXTURE.replace('def register(ctx):', HOST_WRITER + '\ndef register(ctx):')
    make_plugin(home, body=body)
    config = read_user_config_raw(home / 'config.yaml')
    config['plugins']['isolation'] = 'host'
    atomic_config_write(home / 'config.yaml', config)
    manager = PluginManager(scope_key=str(home))
    manager.discover_and_load()
    entry, = registry.snapshot(scope=str(home))
    admission = entry.mutations.projection.reserve(entry, plan)
    assert entry.provider.hold_commit(admission)
    return manager, entry


def test_verified_host_exit_releases_parent_permit_without_claiming_settlement(tmp_path, monkeypatch):
    from tests.waiting import wait_until
    manager, entry = hosted_writer(tmp_path)
    host = manager._plugin_host()
    monkeypatch.setattr(host, '_restart', lambda keys: None)
    proc = host._proc
    try:
        proc.kill()
        proc.wait(timeout=10)
        assert wait_until(lambda: entry.mutations._permit is None, timeout=5)
        doc = Projection(tmp_path).read()
        reservation, = doc['reservations'].values()
        assert reservation['state'] == 'unknown'
        assert not doc['targets']
        assert manager.unload('fixture-catalog')
        assert not entry.mutations._thread.is_alive()
        assert not Projection(tmp_path).read()['registrations']
    finally:
        manager.unload()
        host.shutdown()


def test_broken_channel_never_releases_a_still_live_host_writer(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog.locks import writer_hold
    manager, entry = hosted_writer(tmp_path)
    host = manager._plugin_host()
    monkeypatch.setattr(host, '_restart', lambda keys: None)
    proc = host._proc
    try:
        # Mark the parent's transport unavailable without closing its OS pipes:
        # a real pipe close asks the stock host main() to os._exit immediately.
        host._channel._close('fixture channel failure')
        assert proc.poll() is None
        with pytest.raises(CatalogError) as caught:
            manager.unload('fixture-catalog')
        assert caught.value.code == 'busy'
        assert entry.mutations._permit is not None
        with pytest.raises(CatalogError):
            with writer_hold(tmp_path, ('new',), deadline=time.monotonic()):
                pytest.fail('live host writer exclusion was revoked')
    finally:
        proc.kill()
        proc.wait(timeout=10)
        from tests.waiting import wait_until
        wait_until(lambda: entry.mutations._permit is None, timeout=5)
        manager.unload()
        host.shutdown()


def test_host_restart_rebuilds_catalog_after_dead_writer_permit_cleanup(tmp_path):
    from tests.waiting import wait_until
    manager, entry = hosted_writer(tmp_path)
    host = manager._plugin_host()
    old_process = host._proc
    try:
        old_process.kill()
        old_process.wait(timeout=10)
        def renewed():
            rows = registry.snapshot(scope=str(tmp_path))
            return rows and rows[0].generation != entry.generation
        wait_until(renewed, timeout=10)
        replacement, = registry.snapshot(scope=str(tmp_path))
        assert replacement.source_id == entry.source_id
        assert host._proc is not old_process
        assert not entry.mutations._thread.is_alive()
        assert not replacement.mutations.projection.read()['reservations']
        assert not (tmp_path / 'skills/new').exists()
    finally:
        manager.unload()
        host.shutdown()


@pytest.mark.parametrize('isolation', ['in_process', 'host'])
def test_provider_registration_handle_retains_retry_after_busy_disposal(tmp_path, isolation):
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    from hermes_cli.plugins import PluginManager
    from tests.hermes_cli.test_skill_catalog_contract import make_plugin, PLUGIN
    body = PLUGIN.replace('def register(ctx):', '''
    def dispose_registration(self):
        self.handle.dispose()
        return self.handle.active
def register(ctx):''').replace('    ctx.register_skill_catalog(Catalog())',
        '    provider = Catalog()\n    provider.handle = ctx.register_skill_catalog(provider)')
    make_plugin(tmp_path, body=body)
    config = read_user_config_raw(tmp_path / 'config.yaml')
    config['plugins']['isolation'] = isolation
    atomic_config_write(tmp_path / 'config.yaml', config)
    manager = PluginManager(scope_key=str(tmp_path))
    manager.discover_and_load()
    entry, = registry.snapshot(scope=str(tmp_path))
    try:
        with _held_writer(tmp_path, ('new',)):
            with pytest.raises(Exception):
                entry.provider.dispose_registration()
        assert entry.provider.dispose_registration() is False
        assert not Projection(tmp_path).read()['registrations']
    finally:
        manager.unload()
        if isolation == 'host':
            manager._plugin_host().shutdown()


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
