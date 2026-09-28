"""Secondary platform names share the owning plugin's lifecycle and scope."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Barrier

import pytest

from gateway.platform_registry import platform_registry
from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_manifest import parse_manifest_file
from hermes_constants import set_hermes_home_override, reset_hermes_home_override


def _manifest(tmp_path, body=''):
    plugin = tmp_path / 'alias-fixture'
    plugin.mkdir()
    (plugin / 'plugin.yaml').write_text('name: alias_fixture-platform\nkind: platform\nplatform_aliases: [alias_secondary, alias_secondary]\n')
    (plugin / '__init__.py').write_text('''
calls = 0
def register(ctx):
    global calls
    calls += 1
    for name in ('alias_fixture', 'alias_secondary'):
        ctx.register_platform(name=name, label=name, adapter_factory=lambda cfg: cfg,
                              check_fn=lambda: True)
''' + body)
    return parse_manifest_file(plugin / 'plugin.yaml', plugin, 'bundled', '')


@pytest.mark.parametrize('first', ['alias_fixture', 'alias_secondary'])
def test_aliases_load_once_and_unload_under_own_home(tmp_path, first):
    manifest = _manifest(tmp_path)
    home_a, home_b = tmp_path / 'a', tmp_path / 'b'
    home_a.mkdir(); home_b.mkdir()
    token = set_hermes_home_override(home_a)
    manager = PluginManager()
    try:
        manager._register_deferred_platform(manifest)
        entry = platform_registry.get(first)
        assert entry is not None
        sibling = platform_registry.get('alias_secondary' if first == 'alias_fixture' else 'alias_fixture')
        assert sibling is not None
        assert manager._plugins[manifest.key].module.calls == 1
        other = set_hermes_home_override(home_b)
        try:
            assert platform_registry.get('alias_secondary') is None
            assert platform_registry.get('alias_fixture') is None
        finally:
            reset_hermes_home_override(other)
        assert platform_registry.get(first) is entry
        manager.unload(manifest)
        assert platform_registry.get('alias_secondary') is None
        assert platform_registry.get('alias_fixture') is None
    finally:
        manager.unload(manifest)
        reset_hermes_home_override(token)


def test_unload_before_lookup_removes_all_alias_leases(tmp_path):
    manifest = _manifest(tmp_path)
    manager = PluginManager()
    manager._register_deferred_platform(manifest)
    manager.unload(manifest)
    assert platform_registry.get('alias_secondary') is None
    assert platform_registry.get('alias_fixture') is None


@pytest.mark.parametrize('metadata', ['', 'platform_aliases: null\n', 'platform_aliases: []\n',
                                      'platform_aliases: alias_secondary\n'])
def test_absent_or_invalid_alias_list_preserves_primary_loading(tmp_path, metadata):
    manifest = _manifest(tmp_path)
    plugin = tmp_path / 'alias-fixture'
    (plugin / 'plugin.yaml').write_text('name: alias_fixture-platform\nkind: platform\n' + metadata)
    manifest = parse_manifest_file(plugin / 'plugin.yaml', plugin, 'bundled', '')
    assert manifest.platform_aliases == []
    manager = PluginManager()
    try:
        manager._register_deferred_platform(manifest)
        assert not platform_registry.is_registered('alias_secondary')
        assert platform_registry.get('alias_fixture') is not None
    finally:
        manager.unload(manifest)


def test_malformed_alias_entries_do_not_discard_valid_sibling(tmp_path):
    _manifest(tmp_path)
    plugin = tmp_path / 'alias-fixture'
    (plugin / 'plugin.yaml').write_text(
        'name: alias_fixture-platform\nkind: platform\n'
        'platform_aliases: [null, 12, false, "", "  ", " alias_secondary "]\n')
    manifest = parse_manifest_file(plugin / 'plugin.yaml', plugin, 'bundled', '')
    assert manifest.platform_aliases == ['alias_secondary']
    manager = PluginManager()
    try:
        manager._register_deferred_platform(manifest)
        assert platform_registry.get('alias_secondary') is not None
        assert manager._plugins[manifest.key].module.calls == 1
    finally:
        manager.unload(manifest)


def test_concurrent_primary_and_alias_materialize_one_plugin(tmp_path):
    manifest = _manifest(tmp_path)
    manager = PluginManager()
    start = Barrier(2)

    def lookup(name):
        start.wait(timeout=10)
        return platform_registry.get(name)

    try:
        manager._register_deferred_platform(manifest)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_context().run, lookup, name)
                       for name in ('alias_fixture', 'alias_secondary')]
            entries = [future.result(timeout=15) for future in futures]
        assert all(entry is not None for entry in entries)
        assert manager._plugins[manifest.key].module.calls == 1
        manager.unload(manifest)
        assert not platform_registry.is_registered('alias_fixture')
        assert not platform_registry.is_registered('alias_secondary')
    finally:
        manager.unload(manifest)


def test_two_active_profile_managers_keep_alias_ownership_separate(tmp_path):
    manifest = _manifest(tmp_path)
    managers = []
    entries = []
    homes = [tmp_path / 'a', tmp_path / 'b']
    for home in homes:
        home.mkdir()
    try:
        for home in homes:
            token = set_hermes_home_override(home)
            try:
                manager = PluginManager()
                managers.append(manager)
                manager._register_deferred_platform(manifest)
                entries.append(platform_registry.get('alias_secondary'))
                assert entries[-1] is not None
                assert manager._plugins[manifest.key].module.calls == 1
            finally:
                reset_hermes_home_override(token)
        assert entries[0] is not entries[1]
        token = set_hermes_home_override(homes[0])
        try:
            assert platform_registry.get('alias_secondary') is entries[0]
            managers[0].unload(manifest)
            assert platform_registry.get('alias_secondary') is None
        finally:
            reset_hermes_home_override(token)
        token = set_hermes_home_override(homes[1])
        try:
            assert platform_registry.get('alias_secondary') is entries[1]
            assert managers[1]._plugins[manifest.key].module.calls == 1
        finally:
            reset_hermes_home_override(token)
    finally:
        for manager in managers:
            manager.unload(manifest)


def test_failed_register_does_not_leave_alias_callable(tmp_path):
    manifest = _manifest(tmp_path, '    raise RuntimeError("fixture failure")\n')
    manager = PluginManager()
    manager._register_deferred_platform(manifest)
    try:
        assert platform_registry.get('alias_secondary') is None
        assert platform_registry.get('alias_fixture') is None
    finally:
        manager.unload(manifest)
