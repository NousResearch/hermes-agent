"""Experimental catalog identity and consent contracts through the real plugin loader."""

import hashlib
import json

import pytest

from hermes_cli.config import atomic_config_write
from hermes_cli.plugins import PluginManager


PLUGIN = '''from hermes_cli.skill_catalog import SkillCatalogProvider
class Catalog(SkillCatalogProvider):
    name = "team"
    label = "Team skills"
    implementation_revision = "1"
    def search(self, request):
        return {"ok": True, "data": {"items": [], "next_cursor": None}}
    def installed(self, request):
        return {"ok": True, "data": {"items": [], "next_cursor": None}}
    def inspect(self, request):
        return {"ok": False, "error": {"code": "not_found", "message": "Absent"}}
    review = inspect
    reviewed = inspect
    confirm = inspect
    operation = inspect

def register(ctx):
    ctx.register_skill_catalog(Catalog())
'''


def make_plugin(home, key="fixture-catalog", body=PLUGIN, *, granted=True):
    path = home / "plugins" / key
    path.mkdir(parents=True, exist_ok=True)
    (path / "plugin.yaml").write_text(
        f'name: {key}\nversion: "1.0.0"\ncapabilities:\n  - skills.catalog\n')
    (path / "__init__.py").write_text(body)
    atomic_config_write(home / "config.yaml", {"plugins": {"enabled": [key], "entries": {
        key: {"enabled": True, "granted_capabilities": ["skills.catalog"] if granted else []}}}})
    return path


def test_catalog_loads_with_owner_qualified_identity_and_unloads(tmp_path):
    from hermes_cli.skill_catalog.registry import snapshot

    make_plugin(tmp_path)
    manager = PluginManager(scope_key=str(tmp_path))
    try:
        manager.discover_and_load()
        assert "fixture-catalog" in manager._plugins, sorted(manager._plugins)
        assert manager._plugins["fixture-catalog"].error is None
        rows = snapshot(scope=manager.scope_key)
        expected = "plugin." + hashlib.sha256(b"fixture-catalog").hexdigest() + ".team"
        assert [row.source_id for row in rows] == [expected]
        assert rows[0].plugin_key == "fixture-catalog"
        assert rows[0].descriptor.label == "Team skills"
        manager.unload("fixture-catalog")
        assert snapshot(scope=manager.scope_key) == ()
    finally:
        manager.unload()


def test_missing_consent_never_registers(tmp_path):
    from hermes_cli.skill_catalog.registry import snapshot

    make_plugin(tmp_path, granted=False)
    manager = PluginManager(scope_key=str(tmp_path))
    try:
        manager.discover_and_load()
        assert snapshot(scope=manager.scope_key) == ()
    finally:
        manager.unload()


def test_failed_load_never_publishes_catalog(tmp_path):
    from hermes_cli.skill_catalog.registry import snapshot

    make_plugin(tmp_path, body=PLUGIN + '\n    raise RuntimeError("failed load")\n')
    manager = PluginManager(scope_key=str(tmp_path))
    try:
        manager.discover_and_load()
        assert snapshot(scope=manager.scope_key) == ()
    finally:
        manager.unload()


@pytest.mark.parametrize("override", [
    '    @property\n    def name(self):\n        raise AssertionError("evaluated identity")\n',
    '    def __getattribute__(self, key):\n        raise AssertionError("evaluated attribute")\n',
    '    def __getattr__(self, key):\n        raise AssertionError("evaluated fallback")\n',
])
def test_dynamic_descriptor_is_rejected_without_execution(override):
    from hermes_cli.skill_catalog import SkillCatalogProvider
    from hermes_cli.skill_catalog.identity import describe

    namespace = {"SkillCatalogProvider": SkillCatalogProvider}
    exec('class Catalog(SkillCatalogProvider):\n' + override, namespace)
    with pytest.raises(ValueError):
        describe(namespace["Catalog"]())


def test_exact_plugin_key_and_same_catalog_names_partition_identity():
    from hermes_cli.skill_catalog.identity import source_id

    assert source_id("Plugin A", "team") != source_id("plugin a", "team")
    assert source_id("A", "team") != source_id("B", "team")
    with pytest.raises(ValueError):
        source_id("A\n", "team")
    with pytest.raises(ValueError):
        source_id("A", "Team")


def test_live_enablement_rechecks_the_loader_denylist(tmp_path):
    from hermes_cli.config import read_user_config_raw
    from hermes_cli.skill_catalog.registry import snapshot

    make_plugin(tmp_path)
    manager = PluginManager(scope_key=str(tmp_path))
    try:
        manager.discover_and_load()
        row, = snapshot(scope=manager.scope_key)
        assert row.authorized()
        config = read_user_config_raw(tmp_path / "config.yaml")
        config["plugins"]["disabled"] = ["fixture-catalog"]
        atomic_config_write(tmp_path / "config.yaml", config)
        assert not row.authorized()
    finally:
        manager.unload()


def test_new_capability_has_explicit_disable_consent():
    from hermes_cli.plugin_capabilities import CAPABILITY_REGISTRY

    copy = CAPABILITY_REGISTRY["skills.catalog"].description
    assert "ordinary local skills" in copy
    assert "confirm" in copy


def test_descriptor_roundtrip_is_plain_data():
    from hermes_cli.skill_catalog.identity import Descriptor

    original = {"schema_version": 1, "name": "team", "label": "Team", "implementation_revision": "1"}
    assert Descriptor.from_wire(json.loads(json.dumps(original))).to_wire() == original
    with pytest.raises(ValueError):
        Descriptor.from_wire(dict(original, schema_version=True))
