"""A plugin toolset registered with ``default_platforms`` defaults on only there; an explicit choice wins."""

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
