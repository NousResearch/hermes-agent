"""Plugin toggles must not replace inherited core tools with a plugin-only list."""

import pytest

from hermes_cli import config as config_module
from hermes_cli import plugins_cmd
from hermes_cli.tools_config import _get_platform_tools


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("initial", [{}, {"platform_toolsets": {}}, {"platform_toolsets": None}])
def test_enable_preserves_inherited_core_tools(plugin_home, initial):
    config_module.save_config(initial)
    before = _get_platform_tools(config_module.load_config(), "cli")
    assert {"terminal", "file", "web"} <= before
    assert "spotify" not in before

    plugins_cmd._toggle_plugin_toolset("spotify", enable=True)

    saved = config_module.read_raw_config()
    after = _get_platform_tools(config_module.load_config(), "cli")
    assert before <= after
    assert "spotify" in after
    assert "spotify" in saved["platform_toolsets"]["cli"]

    plugins_cmd._toggle_plugin_toolset("spotify", enable=False)
    assert _get_platform_tools(config_module.load_config(), "cli") == before
    assert "spotify" not in config_module.read_raw_config()["platform_toolsets"]["cli"]


@pytest.mark.parametrize("selection", [[], ["file"], ["hermes-cli"], ["file", "no_mcp"]])
def test_dashboard_round_trip_preserves_explicit_selections(plugin_home, selection):
    config_module.save_config({
        "platform_toolsets": {"cli": selection, "telegram": ["web"], "discord": []},
        "agent": {"disabled_toolsets": ["memory"]},
    })
    before = config_module.read_raw_config()
    tools_before = _get_platform_tools(config_module.load_config(), "cli")

    result = plugins_cmd.dashboard_set_agent_plugin_enabled("spotify", enabled=True)
    assert result == {"ok": True, "name": "spotify", "unchanged": False}
    enabled = config_module.read_raw_config()
    for platform, original in before["platform_toolsets"].items():
        assert enabled["platform_toolsets"][platform] == [*original, "spotify"]
    assert "spotify" in enabled["plugins"]["enabled"]
    assert "spotify" not in enabled["plugins"].get("disabled", [])
    assert "spotify" in _get_platform_tools(config_module.load_config(), "cli")
    assert enabled["agent"] == before["agent"]

    assert plugins_cmd.dashboard_set_agent_plugin_enabled("spotify", enabled=True)["unchanged"]
    assert config_module.read_raw_config() == enabled
    result = plugins_cmd.dashboard_set_agent_plugin_enabled("spotify", enabled=False)
    assert result == {"ok": True, "name": "spotify", "unchanged": False}
    disabled = config_module.read_raw_config()
    assert disabled["platform_toolsets"] == before["platform_toolsets"]
    assert "spotify" in disabled["plugins"]["disabled"]
    assert "spotify" not in disabled["plugins"]["enabled"]
    assert _get_platform_tools(config_module.load_config(), "cli") == tools_before
    assert disabled["agent"] == before["agent"]
    assert plugins_cmd.dashboard_set_agent_plugin_enabled("spotify", enabled=False)["unchanged"]
    assert config_module.read_raw_config() == disabled


@pytest.mark.parametrize("initial", [{}, {"platform_toolsets": {}}, {"platform_toolsets": {"cli": []}}])
def test_disable_does_not_invent_a_selection(plugin_home, initial):
    config_module.save_config(initial)
    before = config_module.read_raw_config()
    tools_before = _get_platform_tools(config_module.load_config(), "cli")
    plugins_cmd._toggle_plugin_toolset("spotify", enable=False)
    assert config_module.read_raw_config() == before
    assert _get_platform_tools(config_module.load_config(), "cli") == tools_before


def test_dashboard_inherited_round_trip_preserves_core_defaults(plugin_home):
    config_module.save_config({"agent": {"disabled_toolsets": ["memory"]}})
    before = _get_platform_tools(config_module.load_config(), "cli")
    for enabled in (True, False, True, False):
        assert plugins_cmd.dashboard_set_agent_plugin_enabled("spotify", enabled=enabled)["ok"]
        selected = _get_platform_tools(config_module.load_config(), "cli")
        assert before <= selected
        assert ("spotify" in selected) is enabled
        assert "memory" not in selected
        saved = config_module.read_raw_config()
        assert ("spotify" in saved["plugins"]["enabled"]) is enabled
        assert ("spotify" in saved["plugins"]["disabled"]) is not enabled


def test_disabling_explicit_plugin_only_selection_keeps_empty_list(plugin_home):
    config_module.save_config({"platform_toolsets": {"cli": ["spotify"]}})
    plugins_cmd._toggle_plugin_toolset("spotify", enable=False)
    # There is no provenance proving this was the historical broken writer.
    # Do not silently turn an explicit plugin-only choice into core defaults.
    assert config_module.read_raw_config()["platform_toolsets"]["cli"] == []
    assert "terminal" not in _get_platform_tools(config_module.load_config(), "cli")


def test_other_platform_selection_does_not_materialize_inherited_cli(plugin_home):
    config_module.save_config({"platform_toolsets": {"telegram": ["file"]}})
    before = _get_platform_tools(config_module.load_config(), "cli")
    for enabled in (True, False):
        plugins_cmd._toggle_plugin_toolset("spotify", enable=enabled)
        saved = config_module.read_raw_config()["platform_toolsets"]
        assert "cli" not in saved
        assert saved["telegram"] == (["file", "spotify"] if enabled else ["file"])
        assert before <= _get_platform_tools(config_module.load_config(), "cli")


def test_plugin_without_tools_leaves_toolset_config_untouched(plugin_home):
    config_module.save_config({"platform_toolsets": {"cli": []}})
    before = config_module.read_raw_config()
    for enabled in (True, False):
        plugins_cmd._toggle_plugin_toolset("nonexistent-no-tools-plugin", enable=enabled)
        assert config_module.read_raw_config() == before
