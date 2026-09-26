"""``PluginContext.register_toolset`` / ``add_to_toolset``: a plugin can define a composite toolset and
join a tool to a built-in bundle whose tool list is static; both resolve through ``toolsets`` like
built-in membership, stay in the registering profile, and leave on unload."""

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from toolsets import TOOLSETS, resolve_toolset, validate_toolset


def _ctx():
    return PluginContext(PluginManifest(name="bundle-probe", source="user"), PluginManager())


def test_plugin_toolset_and_bundle_membership_resolve_and_unload():
    ctx = _ctx()
    assert "probe_tool" not in resolve_toolset("hermes-cli")

    defined = ctx.register_toolset("probe_core", "Probe bundle", tools=["probe_tool"], includes=["todo"])
    joined = ctx.add_to_toolset("hermes-cli", "probe_tool")

    assert validate_toolset("probe_core")
    assert set(resolve_toolset("probe_core")) == {"probe_tool", *resolve_toolset("todo")}
    assert "probe_tool" in resolve_toolset("hermes-cli")
    assert "probe_tool" not in TOOLSETS["hermes-cli"]["tools"]  # the static bundle is not mutated

    defined.dispose()
    joined.dispose()
    assert not validate_toolset("probe_core")
    assert "probe_tool" not in resolve_toolset("hermes-cli")


def test_plugin_toolset_is_scoped_to_its_profile(tmp_path, monkeypatch):
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    handle = _ctx().register_toolset("probe_scoped", "A-only", tools=["probe_tool"])
    try:
        assert validate_toolset("probe_scoped")
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        assert not validate_toolset("probe_scoped")
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        assert resolve_toolset("probe_scoped") == ["probe_tool"]
    finally:
        handle.dispose()
