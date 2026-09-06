"""Plugin changes preserve raw profile settings outside the selected leaf."""

import copy

import pytest
import yaml

from hermes_cli import config, plugins_cmd


@pytest.mark.parametrize("operation", ["enable", "disable", "flag"])
def test_plugin_mutation_preserves_raw_profile_settings(tmp_path, monkeypatch, operation):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(config, "is_managed", lambda: False)
    raw = {
        "providers": {"custom": {"models": None}},
        "auxiliary": {"compression": {"extra_body": {}}},
        "api_base": "https://synthetic.invalid/v1",
        "custom_setting": {"empty": {}, "template": "${SYNTHETIC_UNSET}"},
        "plugins": {"enabled": [], "disabled": ["example"], "entries": {"other": {"settings": {}}}},
    }
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    expected = copy.deepcopy(raw)
    monkeypatch.setattr(plugins_cmd, "_resolve_plugin_key_and_source", lambda _: ("example", "user"))
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: [("example", "1", "", "user", home, "example")])
    monkeypatch.setattr(plugins_cmd, "_declared_capabilities_for_key", lambda _: [])
    if operation == "enable":
        plugins_cmd.cmd_enable("example", allow_tool_override=False)
        expected["plugins"]["enabled"] = ["example"]
        expected["plugins"]["disabled"] = []
        expected["plugins"]["entries"]["example"] = {"allow_tool_override": False}
    elif operation == "disable":
        plugins_cmd._save_disabled_set({"example", "another"})
        expected["plugins"]["disabled"] = ["another", "example"]
    else:
        plugins_cmd._set_plugin_entry_flag("example", "allow_tool_override", False)
        expected["plugins"]["entries"]["example"] = {"allow_tool_override": False}
    assert yaml.safe_load(path.read_text()) == expected


def test_capability_consent_preserves_raw_profile_settings(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(config, "is_managed", lambda: False)
    raw = {
        "providers": {"custom": {"models": None}},
        "auxiliary": {"compression": {"extra_body": {}}},
        "api_base": "https://synthetic.invalid/v1",
        "custom_setting": {"empty": {}, "template": "${SYNTHETIC_UNSET}"},
        "plugins": {"entries": {"other": {"settings": {}}}},
    }
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(raw))

    from hermes_cli.plugin_capabilities import record_consent
    record_consent("example", ["tools.override"], ["tools.override"])

    actual = yaml.safe_load(path.read_text())
    assert actual["providers"] == raw["providers"]
    assert actual["auxiliary"] == raw["auxiliary"]
    assert actual["api_base"] == raw["api_base"]
    assert actual["custom_setting"] == raw["custom_setting"]
    assert actual["plugins"]["entries"]["other"] == raw["plugins"]["entries"]["other"]
    assert actual["plugins"]["entries"]["example"]["granted_capabilities"] == ["tools.override"]
    assert actual["plugins"]["entries"]["example"]["allow_tool_override"] is True


def test_plugin_toolset_toggle_preserves_raw_profile_settings(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(config, "is_managed", lambda: False)
    raw = {
        "providers": {"custom": {"models": None}},
        "auxiliary": {"compression": {"extra_body": {}}},
        "api_base": "https://synthetic.invalid/v1",
        "custom_setting": {"empty": {}, "template": "${SYNTHETIC_UNSET}"},
        "platform_toolsets": {"cli": ["file"], "web": None},
        "legacy_root": {},
    }
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(plugins_cmd, "_get_plugin_toolset_key", lambda _: "example_tools")

    plugins_cmd._toggle_plugin_toolset("example", enable=True)

    actual = yaml.safe_load(path.read_text())
    assert actual["providers"] == raw["providers"]
    assert actual["auxiliary"] == raw["auxiliary"]
    assert actual["api_base"] == raw["api_base"]
    assert actual["custom_setting"] == raw["custom_setting"]
    assert actual["legacy_root"] == raw["legacy_root"]
    assert actual["platform_toolsets"] == {"cli": ["file", "example_tools"], "web": None}


@pytest.mark.parametrize("invalid", ["[unterminated", "- not-a-mapping", "managed", "managed-key"])
def test_plugin_mutation_refuses_unreadable_document(tmp_path, monkeypatch, invalid):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(config, "is_managed", lambda: False)
    path = tmp_path / "config.yaml"
    original = "plugins: {enabled: []}\n" if invalid.startswith("managed") else invalid
    path.write_text(original)
    if invalid == "managed":
        monkeypatch.setattr(config, "is_managed", lambda: True)
        plugins_cmd._save_enabled_set({"example"})
    elif invalid == "managed-key":
        monkeypatch.setattr(config.managed_scope, "is_key_managed", lambda key: key == "plugins.enabled")
        with pytest.raises(SystemExit):
            plugins_cmd._save_enabled_set({"example"})
    else:
        with pytest.raises(RuntimeError):
            plugins_cmd._save_enabled_set({"example"})
    assert path.read_text() == original
