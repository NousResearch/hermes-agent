"""Real plugin toggles preserve inherited tools, saved selections and credential defaults."""

import pytest

from hermes_cli import config as config_module
from hermes_cli import plugins_cmd
from hermes_cli.plugins import get_plugin_manager
from hermes_cli.tools_config import _get_platform_tools
from tests.hermes_cli.plugin_worker_support import (
    isolated_python as isolated_python,
    plugin_world as plugin_world,
)

PLUGIN = "toggle-fixture"
TOOLSET = "toggle_fixture"


def _install_fixture(home):
    plugin = home / "plugins" / PLUGIN
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        f"name: {PLUGIN}\nversion: 1.0.0\nprovides_tools: [toggle_fixture_tool]\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_tool(name='toggle_fixture_tool', toolset='toggle_fixture',\n"
        "        schema={'name': 'toggle_fixture_tool', 'description': 'Offline fixture',\n"
        "                'parameters': {'type': 'object', 'properties': {}}},\n"
        "        handler=lambda args, **kwargs: '{\"ok\": true}')\n",
        encoding="utf-8",
    )


def _selected():
    # Disable is intentionally config-only for open chats. Rediscovery exercises the next-session path.
    get_plugin_manager().discover_and_load(force=True)
    return _get_platform_tools(config_module.load_config(), "cli")


@pytest.mark.parametrize("selection", [
    None, {}, {"cli": []}, {"cli": ["file"]}, {"cli": ["hermes-cli"]},
    {"cli": ["file", "no_mcp"]}, {"telegram": ["file"]},
    {"cli": "['file', 'no_mcp']"}, {"cli": [TOOLSET]},
])
@pytest.mark.parametrize("credentials", [False, True])
def test_dashboard_round_trip_preserves_selection_and_credentials(plugin_world, monkeypatch, selection, credentials):
    _install_fixture(plugin_world.home)
    if credentials:
        monkeypatch.setenv("XAI_API_KEY", "nonfunctional-test-fixture")
    config_module.save_config({
        "platform_toolsets": selection,
        "plugins": {"enabled": [], "disabled": [PLUGIN]},
        "agent": {"disabled_toolsets": ["memory"]},
    })
    before = config_module.read_raw_config()
    selected_before = _selected() - {TOOLSET}
    for enabled in (True, False, True, False):
        result = plugins_cmd.dashboard_set_agent_plugin_enabled(PLUGIN, enabled=enabled)
        assert result["ok"] and not result["unchanged"]
        assert result["name"] == PLUGIN
        selected = _selected()
        assert selected - {TOOLSET} == selected_before
        assert (TOOLSET in selected) is enabled
        saved = config_module.read_raw_config()
        assert saved["agent"] == before["agent"]
        assert (PLUGIN in saved["plugins"]["enabled"]) is enabled
        assert (PLUGIN in saved["plugins"]["disabled"]) is not enabled
        if not selection:
            # Upstream fixed the original bug by leaving inherited selections implicit.
            assert saved.get("platform_toolsets") == before.get("platform_toolsets")
        else:
            from hermes_cli.toolset_validation import parse_platform_toolsets_value
            for platform, original in selection.items():
                original = parse_platform_toolsets_value(original)
                expected = [item for item in original if item != TOOLSET]
                assert saved["platform_toolsets"][platform] == (expected + [TOOLSET] if enabled else expected)
            if "cli" not in selection:
                assert "cli" not in saved["platform_toolsets"]
        unchanged = plugins_cmd.dashboard_set_agent_plugin_enabled(PLUGIN, enabled=enabled)
        assert unchanged["ok"] and unchanged["unchanged"]
        assert config_module.read_raw_config() == saved


@pytest.mark.parametrize("opt_out", [False, True])
def test_mixed_composite_credentials_are_profile_scoped(plugin_world, monkeypatch, opt_out):
    homes = [plugin_world.home, plugin_world.root / "other-home"]
    for home in homes:
        _install_fixture(home)
    states = []
    for home in (homes[0], homes[1], homes[0]):
        monkeypatch.setenv("HERMES_HOME", str(home))
        has_credentials = home == homes[0]
        if has_credentials:
            monkeypatch.setenv("XAI_API_KEY", "nonfunctional-test-fixture")
        else:
            monkeypatch.delenv("XAI_API_KEY", raising=False)
        config_module.save_config({
            "platform_toolsets": {"cli": ["hermes-cli"]},
            "plugins": {"enabled": [], "disabled": [PLUGIN]},
            "agent": {"disabled_toolsets": ["x_search"] if opt_out else []},
        })
        before = _selected()
        assert {"terminal", "file", "web"} <= before
        assert ("x_search" in before) is (has_credentials and not opt_out)
        for enabled in (True, False):
            assert plugins_cmd.dashboard_set_agent_plugin_enabled(PLUGIN, enabled=enabled)["ok"]
            selected = _selected()
            assert selected - {TOOLSET} == before
            assert (TOOLSET in selected) is enabled
        states.append(before)
    assert states[0] == states[2]
    assert states[0] - {"x_search"} == states[1]
