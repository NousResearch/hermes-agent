"""A plugin toolset registered with ``default_platforms`` defaults on only there; an explicit choice wins."""

import pytest

import hermes_yaml as yaml

from hermes_cli.plugins import discover_plugins, get_plugin_toolset_default_platforms, get_plugin_toolsets
from hermes_cli.tools_config import _get_platform_tools

_PLUGIN = '''
def _schema(name):
    return {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}

def register(ctx):
    ctx.register_tool(name="scoped_stub_read", toolset="scoped_stub", schema=_schema("scoped_stub_read"),
                      handler=lambda args, **kw: "{}", default_platforms=("slack",))
    ctx.register_tool(name="anywhere_stub_read", toolset="anywhere_stub", schema=_schema("anywhere_stub_read"),
                      handler=lambda args, **kw: "{}")
    # Siblings in one toolset: one scoped, one undeclared (= every platform).
    ctx.register_tool(name="mixed_stub_slack", toolset="mixed_stub", schema=_schema("mixed_stub_slack"),
                      handler=lambda args, **kw: "{}", default_platforms=("slack",))
    ctx.register_tool(name="mixed_stub_any", toolset="mixed_stub", schema=_schema("mixed_stub_any"),
                      handler=lambda args, **kw: "{}")
'''


def test_default_platforms_scope_a_plugin_toolset_until_the_user_chooses(tmp_path, monkeypatch):
    home = tmp_path / "home"
    plugin_dir = home / "plugins" / "scoped_stub"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "scoped_stub", "version": "0.1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["scoped_stub"]}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    discover_plugins()
    assert {"scoped_stub", "anywhere_stub"} <= {key for key, _, _ in get_plugin_toolsets()}

    new_home = {}
    assert "scoped_stub" in _get_platform_tools(new_home, "slack")
    for platform in ("cli", "telegram"):
        enabled = _get_platform_tools(new_home, platform)
        assert "scoped_stub" not in enabled
        assert "anywhere_stub" in enabled  # an undeclared toolset stays on everywhere
        assert "mixed_stub" in enabled  # one scoped tool does not hide its undeclared sibling
    assert "mixed_stub" not in get_plugin_toolset_default_platforms()  # the one default cache + gateway carry

    chosen = {"platform_toolsets": {"cli": ["hermes-cli", "scoped_stub"]}}
    assert "scoped_stub" in _get_platform_tools(chosen, "cli")


_FIRST = '''
def register(ctx):
    ctx.register_tool(name="dup_stub_read", toolset="first_stub", handler=lambda args, **kw: "{}",
                      schema={"name": "dup_stub_read", "description": "d", "parameters": {"type": "object"}},
                      default_platforms=("slack",))
'''
_SECOND_FAILS = '''
def register(ctx):
    ctx.register_tool(name="dup_stub_read", toolset="second_stub", handler=lambda args, **kw: "{}",
                      schema={"name": "dup_stub_read", "description": "d", "parameters": {"type": "object"}},
                      override=True)
    raise RuntimeError("second_stub fails after replacing the tool")
'''


def test_a_restored_tool_gets_its_own_default_platforms_back(tmp_path, monkeypatch):
    """A plugin replaces a same-named tool without ``default_platforms`` and then fails to load: the
    registry restores the first plugin's tool, and its toolset must be scoped to its platforms again."""
    home = tmp_path / "home"
    for name, body in (("aa_first_stub", _FIRST), ("zz_second_stub", _SECOND_FAILS)):
        plugin_dir = home / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "0.1.0"}), encoding="utf-8")
        (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["aa_first_stub", "zz_second_stub"],
        "entries": {"zz_second_stub": {"allow_tool_override": True}},
    }}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    discover_plugins()
    from tools.registry import registry
    assert registry.get_entry("dup_stub_read").toolset == "first_stub"  # the failed load was rolled back
    assert get_plugin_toolset_default_platforms().get("first_stub") == frozenset({"slack"})
    assert "first_stub" not in _get_platform_tools({}, "cli")


class _Discovering:
    """Background discovery still importing for the first ``alive`` probes, finished after."""

    def __init__(self, alive: int):
        self.alive = alive

    def is_alive(self):
        self.alive -= 1
        return self.alive >= 0


def _cache(home, blob):
    import json
    (home / "cache").mkdir(parents=True, exist_ok=True)
    (home / "cache" / "plugin_toolset_keys.json").write_text(json.dumps(blob), encoding="utf-8")


def test_names_and_default_platforms_come_from_one_read(tmp_path, monkeypatch):
    """Greptile #159651: discovery finishing between the names read (last launch's cache) and the rules read
    (live registry, which renamed the toolset) left the cached name with no rule: on for every platform."""
    from hermes_cli import plugins
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _cache(home, {"toolset_keys": ["renamed_stub"], "portable_mcp": [],
                  "toolset_default_platforms": {"renamed_stub": ["slack"]}})
    manager = plugins.get_plugin_manager()
    monkeypatch.setattr(manager, "_discovered", True)
    monkeypatch.setattr(plugins, "_background_discovery_thread", _Discovering(alive=1))
    monkeypatch.setattr(plugins, "get_plugin_toolsets", list)
    monkeypatch.setattr(plugins, "get_plugin_toolset_default_platforms", dict)
    assert "renamed_stub" not in _get_platform_tools({}, "telegram")


def test_a_cache_from_before_default_platforms_does_not_wait_for_discovery(tmp_path, monkeypatch):
    """Greptile #159651: the first start after the upgrade reads a cache with names but no rules field; that
    cache's names had no rule, so it is read as is instead of joining the discovery thread."""
    from hermes_cli import plugins
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _cache(home, {"toolset_keys": ["old_stub"], "portable_mcp": []})
    monkeypatch.setattr(plugins.get_plugin_manager(), "_discovered", False)
    monkeypatch.setattr(plugins, "_background_discovery_thread", _Discovering(alive=99))
    monkeypatch.setattr(plugins, "discover_plugins", lambda *a, **k: pytest.fail("joined discovery"))
    assert "old_stub" in _get_platform_tools({}, "telegram")
