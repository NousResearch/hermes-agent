"""HTTP cold activation respects ordinary isolation, consent and loader ownership."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_constants import get_hermes_home
from hermes_cli.config import atomic_config_write, read_user_config_raw
from hermes_cli.plugins import PluginManager
from tests.hermes_cli.test_skill_catalog_contract import make_plugin, PLUGIN
pytest_plugins = ['tests.hermes_cli.test_skill_catalog_http']


@pytest.fixture
def cold(monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.plugins_loader import _plugin_home_scope
    home = get_hermes_home()
    managers = []
    def setup(body=PLUGIN, *, isolation='host', granted=True, enabled=True):
        make_plugin(home, body=body, granted=granted)
        config = read_user_config_raw(home / 'config.yaml')
        config['plugins']['isolation'] = isolation
        config['plugins']['load_timeout_seconds'] = 0
        if not enabled:
            config['plugins']['enabled'] = []
        atomic_config_write(home / 'config.yaml', config)
        manager = PluginManager(scope_key=str(home))
        managers.append(manager)
        monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: manager)
        return manager
    yield setup
    for manager in managers:
        with _plugin_home_scope(home):
            manager.unload()
            host = manager._running_plugin_host()
            if host is not None:
                host.shutdown()


def test_cold_host_official_bootstraps_real_normal_manager(client, cold):
    from hermes_cli.plugin_host import is_hosted_object
    from hermes_cli.skill_catalog.registry import snapshot
    manager = cold()
    result = client.get('/api/skills/hub/official')
    assert result.status_code == 200, result.text
    source, = result.json()['provider_sources']
    assert source['provider']['plugin_id'] == 'fixture-catalog'
    entry, = snapshot(scope=manager.scope_key)
    assert is_hosted_object(entry.provider)
    assert manager._discovered


@pytest.mark.parametrize('mode', ['in_process', 'disabled', 'ungranted', 'unrelated'])
def test_ineligible_cold_request_never_imports_plugin(client, cold, mode):
    marker = get_hermes_home() / 'imported'
    body = f'from pathlib import Path\nPath({str(marker)!r}).touch()\n' + PLUGIN
    manager = cold(body, isolation='in_process' if mode == 'in_process' else 'host',
                   granted=mode != 'ungranted', enabled=mode != 'disabled')
    if mode == 'unrelated':
        manifest = get_hermes_home() / 'plugins/fixture-catalog/plugin.yaml'
        manifest.write_text('name: fixture-catalog\nversion: "1.0.0"\n')
    result = client.get('/api/skills/hub/official')
    assert result.status_code == 200
    assert 'provider_sources' not in result.json()
    assert not marker.exists()
    assert not manager._discovered


@pytest.mark.parametrize('ending', ['return', 'raise RuntimeError("private load error")'])
def test_host_no_registration_keeps_stock_and_contains_output(client, cold, capfd, ending):
    import logging
    import sys
    logger = logging.getLogger('hermes_cli.plugins')
    console = logging.StreamHandler(sys.stderr)
    logger.addHandler(console)
    manager = cold('def register(ctx):\n    print("PRIVATE CATALOG OUTPUT", flush=True)\n    ' + ending + '\n')
    try:
        baseline = client.get('/api/skills/hub/official')
        manager._running_plugin_host().shutdown()
    finally:
        logger.removeHandler(console)
    assert baseline.status_code == 200
    assert 'provider_sources' not in baseline.json()
    assert manager._discovered
    assert 'PRIVATE CATALOG OUTPUT' not in baseline.text
    output = capfd.readouterr()
    assert 'PRIVATE CATALOG OUTPUT' not in output.out + output.err
    assert 'private load error' not in output.out + output.err
    diagnostic = get_hermes_home() / 'logs/skill-catalog.log'
    assert 'PRIVATE CATALOG OUTPUT' in diagnostic.read_text()


def test_lock_wait_is_bounded_shared_and_late_stale_registration_refuses(client, cold):
    from hermes_cli.skill_catalog.http import _service
    from hermes_cli.skill_catalog.registry import snapshot
    manager = cold()
    entered, release = threading.Event(), threading.Event()
    def hold():
        with manager._discovery_lock:
            entered.set()
            assert release.wait(10)
    holder = threading.Thread(target=hold)
    holder.start()
    assert entered.wait(5)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            calls = [executor.submit(_service, None, deadline=time.monotonic() + .15) for _ in range(4)]
            assert [call.result(timeout=3) for call in calls] == [None] * 4
        from hermes_cli.skill_catalog.bootstrap import _active
        bootstrap_thread = _active[manager.scope_key].thread
        _service(None, deadline=time.monotonic() + .1)
        assert _active[manager.scope_key].thread is bootstrap_thread
        # Disable while the original bootstrap remains parked behind the lock.
        config = read_user_config_raw(get_hermes_home() / 'config.yaml')
        config['plugins']['enabled'] = []
        atomic_config_write(get_hermes_home() / 'config.yaml', config)
    finally:
        release.set()
        holder.join(5)
    # Join the original bootstrap, not a replacement worker after each timeout.
    assert _service(None, deadline=time.monotonic() + 5) is None
    assert snapshot(scope=manager.scope_key) == ()


@pytest.mark.parametrize('change', ['manager', 'consent', 'disabled'])
def test_late_bootstrap_cannot_publish_stale_authority(client, cold, monkeypatch, change):
    from hermes_cli import plugins
    from hermes_cli.skill_catalog.http import _service
    from hermes_cli.skill_catalog.registry import snapshot
    manager = cold()
    entered, release = threading.Event(), threading.Event()
    original = manager._publish_staged_registrations
    def publish(ctx):
        entered.set()
        assert release.wait(10)
        return original(ctx)
    monkeypatch.setattr(manager, '_publish_staged_registrations', publish)
    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(_service, None, deadline=time.monotonic() + .3)
        assert entered.wait(5)
        if change == 'manager':
            monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: PluginManager(scope_key=manager.scope_key))
        else:
            config = read_user_config_raw(get_hermes_home() / 'config.yaml')
            if change == 'disabled':
                config['plugins']['enabled'] = []
            else:
                config['plugins']['entries']['fixture-catalog']['granted_capabilities'] = []
            atomic_config_write(get_hermes_home() / 'config.yaml', config)
        assert task.result(timeout=3) is None
        release.set()
    from hermes_cli.skill_catalog.bootstrap import _active
    _active[manager.scope_key].thread.join(5)
    assert snapshot(scope=manager.scope_key) == ()


def test_isolation_flip_while_waiting_never_imports_cold_plugin(client, cold, monkeypatch):
    from hermes_cli.skill_catalog.http import _service
    marker = get_hermes_home() / 'imported'
    manager = cold(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n' + PLUGIN)
    entered, release = threading.Event(), threading.Event()
    original = manager.discover_and_load
    def discover():
        entered.set()
        assert release.wait(10)
        return original()
    monkeypatch.setattr(manager, 'discover_and_load', discover)
    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(_service, None, deadline=time.monotonic() + .3)
        assert entered.wait(5)
        config = read_user_config_raw(get_hermes_home() / 'config.yaml')
        config['plugins']['isolation'] = 'in_process'
        atomic_config_write(get_hermes_home() / 'config.yaml', config)
        release.set()
        assert task.result(timeout=5) is None
    assert not marker.exists()


def test_cold_search_public_work_starts_before_bootstrap_join(client, cold, monkeypatch):
    from hermes_cli.skill_catalog import http
    from tools import skills_hub_search
    manager = cold()
    public_finished, boot_started = threading.Event(), threading.Event()
    deadlines = []
    original = http._service
    def bootstrap(profile, **kwargs):
        deadlines.append(kwargs.get('deadline'))
        boot_started.set()
        assert public_finished.wait(3), 'public fanout was held behind cold bootstrap'
        return original(profile, **kwargs)
    def public(*args, **kwargs):
        public_finished.set()
        return [], {}, []
    monkeypatch.setattr(http, '_service', bootstrap)
    monkeypatch.setattr(skills_hub_search, 'parallel_search_sources', public)
    result = client.get('/api/skills/hub/search?q=hello')
    assert result.status_code == 200, result.text
    assert boot_started.is_set()
    assert deadlines[0] is not None
    assert manager._discovered


def test_official_uses_one_absolute_bootstrap_and_availability_deadline(client, cold, monkeypatch):
    from hermes_cli.skill_catalog import http
    manager = cold()
    deadlines = []
    original = http._invoke
    def invoke(service, method, *args, **kwargs):
        if method == 'sources':
            deadlines.append(kwargs.get('deadline'))
        return original(service, method, *args, **kwargs)
    monkeypatch.setattr(http, '_invoke', invoke)
    result = client.get('/api/skills/hub/official')
    assert result.status_code == 200
    assert deadlines and deadlines[0] is not None
    assert manager._discovered


def test_hung_host_keeps_one_bootstrap_until_real_exit(client, cold):
    from hermes_cli.skill_catalog import bootstrap
    from hermes_cli.skill_catalog.http import _service
    from tests.waiting import wait_until
    home = get_hermes_home()
    entered, release = home / 'entered', home / 'release'
    body = ('import threading\nfrom pathlib import Path\n' + PLUGIN +
            f'\n_original = register\ndef register(ctx):\n    Path({str(entered)!r}).touch()\n'
            f'    while not Path({str(release)!r}).exists():\n        threading.Event().wait(.01)\n'
            '    _original(ctx)\n')
    manager = cold(body)
    try:
        assert _service(None, deadline=time.monotonic() + .2) is None
        assert wait_until(entered.exists, timeout=5)
        thread = bootstrap._active[manager.scope_key].thread
        assert thread.is_alive()
        for _ in range(3):
            assert _service(None, deadline=time.monotonic() + .05) is None
            assert bootstrap._active[manager.scope_key].thread is thread
        # No completed response is mutated when this valid late load is published.
    finally:
        release.touch()
        bootstrap._active[manager.scope_key].thread.join(5)
    assert _service(None, deadline=time.monotonic() + 5) is not None


def test_unrelated_printing_plugin_in_eligible_sweep_is_contained(client, cold, capfd):
    import logging
    import sys
    manager = cold()
    path = get_hermes_home() / 'plugins/sidecar'
    path.mkdir()
    (path / 'plugin.yaml').write_text('name: sidecar\nversion: "1.0.0"\n')
    (path / '__init__.py').write_text('def register(ctx):\n    print("SIDECAR PRIVATE", flush=True)\n')
    config = read_user_config_raw(get_hermes_home() / 'config.yaml')
    config['plugins']['enabled'].append('sidecar')
    atomic_config_write(get_hermes_home() / 'config.yaml', config)
    logger = logging.getLogger('hermes_cli.plugins')
    console = logging.StreamHandler(sys.stderr)
    logger.addHandler(console)
    try:
        result = client.get('/api/skills/hub/official')
        assert result.status_code == 200
        manager._running_plugin_host().shutdown()
    finally:
        logger.removeHandler(console)
    assert 'SIDECAR PRIVATE' not in result.text
    output = capfd.readouterr()
    assert 'SIDECAR PRIVATE' not in output.out + output.err
    assert 'SIDECAR PRIVATE' in (get_hermes_home() / 'logs/skill-catalog.log').read_text()

