"""``ctx.record_data_use``: a plugin tool marks its result so Nous routes it only to qualifying endpoints."""

import json
import os

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager
from tools.connectors.data_use import collect_data_use, parse_data_use_requirements

HEALTH_PLUGIN = '''
import json, os

def register(ctx):
    def report(args):
        return json.dumps({"pid": os.getpid()})

    def health_sync(args, **kw):
        ctx.record_data_use(iter(args["requires"]))
        return report(args)

    async def health_async(args, **kw):
        ctx.record_data_use(args["requires"])
        return report(args)

    def health_bare_string(args, **kw):
        try:
            ctx.record_data_use(args["requires"][0])
        except ValueError as exc:
            return json.dumps({"rejected": type(exc).__name__})
        return json.dumps({"rejected": None})

    for name, handler in (("health_sync", health_sync), ("health_async", health_async),
                          ("health_bare_string", health_bare_string)):
        ctx.register_tool(name=name, toolset="healthprobe", schema={"name": name, "description": "d",
                          "parameters": {"type": "object", "properties": {}}},
                          handler=handler, is_async=name == "health_async")
'''

# What the Portal must receive for each requirement (plugins/model-providers/nous).
NOUS_ROUTE = {"no_training": {"data_collection": "deny"},
              "zero_retention": {"data_collection": "deny", "zdr": True}}


def _home_with_plugin(tmp_path, monkeypatch, isolation):
    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "healthprobe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "healthprobe", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(HEALTH_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": ["healthprobe"], "isolation": isolation}}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)


@pytest.mark.parametrize("requires,parsed", [
    (["no_training"], ("no_training",)),
    (("zero_retention", "no_training", "zero_retention"), ("no_training", "zero_retention")),
    (iter(["zero_retention"]), ("zero_retention",)),
    ([], ()),
])
def test_valid_requirements_parse_sorted_and_deduplicated(requires, parsed):
    assert parse_data_use_requirements(requires) == parsed


@pytest.mark.parametrize("requires", ["no_training", "", b"no_training", ["no_training", "no_storage"], [None], 3])
def test_a_bare_string_an_unknown_name_or_a_non_list_is_refused(requires):
    with pytest.raises(ValueError):
        parse_data_use_requirements(requires)


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_a_plugin_tool_result_routes_nous_to_qualifying_endpoints(tmp_path, monkeypatch, isolation):
    from providers import get_provider_profile
    from tools.registry import registry

    _home_with_plugin(tmp_path, monkeypatch, isolation)
    manager = PluginManager()
    manager.discover_and_load()
    nous = get_provider_profile("nous")
    try:
        assert manager._plugins["healthprobe"].error is None
        for tool in ("health_sync", "health_async"):
            for requirement, route in NOUS_ROUTE.items():
                with collect_data_use() as data_use:
                    result = json.loads(registry.dispatch(tool, {"requires": [requirement]}, scope=manager.scope_key))
                assert (result["pid"] != os.getpid()) == (isolation == "host"), result
                assert data_use == {requirement}, (tool, requirement)
                assert nous.build_extra_body(data_use=frozenset(data_use))["provider"] == route

        with collect_data_use() as data_use:
            refused = json.loads(registry.dispatch("health_bare_string", {"requires": ["no_training"]},
                                                   scope=manager.scope_key))
        assert (refused, data_use) == ({"rejected": "ValueError"}, set())
    finally:
        manager.unload()
        if isolation == "host":
            manager._plugin_host().shutdown()
