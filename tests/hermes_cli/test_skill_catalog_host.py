"""Real host-discovery descriptors never evaluate provider identity in the parent."""

import os

import pytest

from hermes_cli.config import atomic_config_write
from hermes_cli.plugins import PluginManager
from tests.hermes_cli.test_skill_catalog_contract import PLUGIN, make_plugin


@pytest.mark.platforms("posix")
def test_host_catalog_uses_frozen_descriptor_and_dispatches_without_attribute_rpc(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog.registry import snapshot

    make_plugin(tmp_path, body=PLUGIN.replace('"items": []', '"items": [{"pid": __import__("os").getpid()}]', 1))
    from hermes_cli.config import read_user_config_raw
    config = read_user_config_raw(tmp_path / "config.yaml")
    config["plugins"]["isolation"] = "host"
    atomic_config_write(tmp_path / "config.yaml", config)
    manager = PluginManager(scope_key=str(tmp_path))
    try:
        manager.discover_and_load()
        assert manager._plugins["fixture-catalog"].error is None
        rows = snapshot(scope=manager.scope_key)
        assert len(rows) == 1
        host = manager._plugin_host()
        monkeypatch.setattr(host, "obj_getattr", lambda *a, **kw: pytest.fail("identity attribute RPC"))
        assert snapshot(scope=manager.scope_key)[0].descriptor.name == "team"
        result = rows[0].provider.search({"context": {}})
        assert result["data"]["items"][0]["pid"] != os.getpid()
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


def test_all_abandoned_registrars_include_catalog():
    from hermes_cli.plugins import PluginContext, PluginManifest

    ctx = PluginContext.__new__(PluginContext)
    ctx._load_abandoned = True
    ctx.manifest = PluginManifest(name="late")
    for cls in PluginContext.__mro__:
        for name in vars(cls):
            if name.startswith("register_") or name in {"subscribe", "on_unload"}:
                assert getattr(ctx, name)(object()) is None
