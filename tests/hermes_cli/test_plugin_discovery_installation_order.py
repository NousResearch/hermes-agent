"""Real registration callbacks must compose with installation/snapshot transactions."""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("entry", ["eager", "direct", "deferred", "memory", "unload"])
def test_registration_and_toolset_toggle_complete(tmp_path, entry):
    # A subprocess bounds a broken implementation without leaking wedged threads into pytest.
    probe = r'''
import os, sys, threading, traceback, types
from contextlib import contextmanager
from pathlib import Path
import yaml
home = Path(sys.argv[1])
os.environ['HERMES_HOME'] = os.environ['HOME'] = str(home)
home.mkdir()
(home / 'config.yaml').write_text('plugins:\n  enabled: [order_probe]\n')
from hermes_cli import plugins, plugins_cmd as cmd, plugin_installation
from hermes_cli.plugins_loader import _plugin_home_scope
plugin = home / 'plugins' / 'order_probe'
plugin.mkdir(parents=True)
(plugin / '__init__.py').write_text('def register(ctx):\n    from order_sync import callback\n    callback(ctx)\n')
sync = types.ModuleType('order_sync')
sync.entered, sync.attempted = threading.Event(), threading.Event()
sync.calls = []
def callback(ctx):
    sync.entered.set()
    assert sync.attempted.wait(10), 'toggle did not reach installation entry'
    ctx.set_config('ready', True)
    if sys.argv[2] != 'unload':
        ctx.register_tool('order_probe_tool', 'order_probe_tools',
                          {'name': 'order_probe_tool', 'description': 'probe', 'parameters': {}},
                          lambda **kwargs: 'ok')
    sync.calls.append(True)
sync.callback = callback
sys.modules['order_sync'] = sync
manager = plugins.get_plugin_manager()
manifest = plugins.PluginManifest(name='order_probe', source='user', path=plugin)
# Observe the competing real transaction BEFORE it acquires any lock. Waiting for
# lookup inside the transaction would require the very inversion we are fixing.
original = plugin_installation.plugin_installation_lock
@contextmanager
def observed(home=None):
    if threading.current_thread().name != 'toggle':
        with original(home):
            yield
        return
    from hermes_cli import plugins_state
    from hermes_constants import get_hermes_home
    path = Path(home or get_hermes_home()) / '.plugin-installation.lock'
    with plugins_state._PLUGIN_STATE_LOCKS_GUARD:
        mutex = plugins_state._PLUGIN_STATE_LOCKS.setdefault(str(path.resolve()), threading.RLock())
    # Force the old inversion when discovery does not own installation yet;
    # otherwise prove the competitor has reached the already-owned mutex.
    acquired = mutex.acquire(blocking=False)
    try:
        if not acquired:
            sync.attempted.set()
        with original(home):
            sync.attempted.set()
            yield
    finally:
        if acquired:
            mutex.release()
plugin_installation.plugin_installation_lock = observed
entry = sys.argv[2]
if entry == 'eager':
    manager._collect_directory_manifests = lambda: [manifest]
    manager._scan_entry_points = lambda: []
    action = manager.discover_and_load
elif entry == 'direct':
    manager._discovered = True
    action = lambda: manager._load_plugin(manifest)
elif entry == 'deferred':
    from gateway.platform_registry import platform_registry
    manager._discovered = True
    manager._register_deferred_platform(manifest)
    action = lambda: platform_registry.get('order_probe')
elif entry == 'memory':
    from plugins.memory import _ProviderCollector
    manager._discovered = True
    module = manager._load_directory_module(manifest)
    collector = _ProviderCollector('order_probe')
    # The collector intentionally only forwards register_*; use its real context
    # from this native registration callback, just as a provider can obtain it.
    action = lambda: collector.collect(lambda ctx: module.register(ctx._plugin_context()))
else:
    manager._discovered = True
    context = plugins.PluginContext(manifest, manager)
    context.on_unload(lambda: callback(context))
    action = lambda: manager.unload('order_probe')
errors = []
def run(fn):
    try:
        with _plugin_home_scope(home):
            fn()
    except BaseException as exc:
        errors.append(repr(exc))
a = threading.Thread(target=run, args=(action,), daemon=True, name='registration')
b = threading.Thread(target=run, args=(lambda: cmd._toggle_plugin_toolset('order_probe', enable=True),), daemon=True, name='toggle')
a.start()
assert sync.entered.wait(10), 'registration callback did not run'
b.start()
a.join(10)
b.join(10)
for t in (a, b):
    if t.is_alive():
        traceback.print_stack(sys._current_frames()[t.ident])
assert not a.is_alive() and not b.is_alive(), 'discovery/installation deadlock'
assert not errors, errors
assert sync.calls, 'registration swallowed a failure'
config = yaml.safe_load((home / 'config.yaml').read_text())
assert config['plugins']['entries']['order_probe']['settings']['ready'] is True
if entry in ('eager', 'direct', 'deferred'):
    assert config['platform_toolsets']['cli'] == ['order_probe_tools']
'''
    result = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path / "home"), entry],
        capture_output=True, text=True, timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_transaction_does_not_join_discovery_waiting_for_its_lock(tmp_path):
    probe = r'''
import os, sys, threading
from pathlib import Path
home = Path(sys.argv[1])
os.environ['HERMES_HOME'] = os.environ['HOME'] = str(home)
from hermes_cli import plugins, plugin_installation
manager = plugins.get_plugin_manager()
scans = []
manager._collect_directory_manifests = lambda: scans.append(True) or []
manager._scan_entry_points = lambda: []
attempted = threading.Event()
from contextlib import contextmanager
original = plugin_installation.plugin_installation_lock
@contextmanager
def observed(home=None):
    if threading.current_thread().name == 'plugin-discovery':
        attempted.set()
    with original(home):
        yield
plugin_installation.plugin_installation_lock = observed
# A join is an invalid wait edge here, even if its timeout eventually breaks it.
class ObservedThread(threading.Thread):
    def join(self, timeout=None):
        if threading.current_thread().name == 'MainThread' and inside[0]:
            raise AssertionError('joining discovery while holding its prerequisite lock')
        return super().join(timeout)
inside = [False]
plugins.threading.Thread = ObservedThread
with plugin_installation.plugin_installation_lock(home):
    inside[0] = True
    plugins.start_background_plugin_discovery()
    assert attempted.wait(10)
    plugins.discover_plugins()
    plugins.unload_plugins()
    inside[0] = False
plugins._background_discovery_thread.join(10)
assert not plugins._background_discovery_thread.is_alive()
assert scans == [True], 'pending background discovery resurrected unloaded plugins'
assert not manager._discovered
'''
    result = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path / "home")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
