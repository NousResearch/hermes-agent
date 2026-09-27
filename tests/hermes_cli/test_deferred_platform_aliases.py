"""Secondary platform names share the owning plugin's lifecycle and scope."""
from pathlib import Path

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


def test_failed_register_does_not_leave_alias_callable(tmp_path):
    manifest = _manifest(tmp_path, '    raise RuntimeError("fixture failure")\n')
    manager = PluginManager()
    manager._register_deferred_platform(manifest)
    try:
        assert platform_registry.get('alias_secondary') is None
        assert platform_registry.get('alias_fixture') is None
    finally:
        manager.unload(manifest)
