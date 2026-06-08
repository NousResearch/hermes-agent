import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import plugins_cmd


def _args(**kwargs):
    defaults = {
        "enabled": False,
        "user": False,
        "no_bundled": False,
        "plain": False,
        "json": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_filter_plugin_entries_enabled_only():
    entries = [
        ("disk-cleanup", "disk-cleanup", "2.0.0", "Bundled", "bundled", None),
        ("web-search-plus", "web-search-plus", "2.2.0", "Search", "git", None),
        ("old-plugin", "old-plugin", "1.0.0", "Old", "user", None),
    ]

    filtered = plugins_cmd._filter_plugin_entries(
        entries,
        _args(enabled=True),
        enabled={"disk-cleanup", "web-search-plus"},
        disabled={"old-plugin"},
    )

    assert [entry[0] for entry in filtered] == ["disk-cleanup", "web-search-plus"]


def test_filter_plugin_entries_no_bundled():
    entries = [
        ("disk-cleanup", "disk-cleanup", "2.0.0", "Bundled", "bundled", None),
        ("drawthings-grpc", "drawthings-grpc", "0.3.0", "Draw Things", "user", None),
        ("web-search-plus", "web-search-plus", "2.2.0", "Search", "git", None),
    ]

    filtered = plugins_cmd._filter_plugin_entries(
        entries,
        _args(no_bundled=True),
        enabled=set(),
        disabled=set(),
    )

    assert [entry[0] for entry in filtered] == ["drawthings-grpc", "web-search-plus"]


def test_cmd_list_plain_compact_output(monkeypatch, capsys):
    entries = [
        ("disk-cleanup", "disk-cleanup", "2.0.0", "Bundled", "bundled", None),
        ("web-search-plus", "web-search-plus", "2.2.0", "Search", "git", None),
    ]
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {"web-search-plus"})
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())

    plugins_cmd.cmd_list(_args(plain=True, no_bundled=True))

    out = capsys.readouterr().out
    assert "web-search-plus" in out
    assert "enabled" in out
    assert "disk-cleanup" not in out
    assert "Search" not in out  # plain mode stays compact, no descriptions


def test_cmd_list_json_output(monkeypatch, capsys):
    entries = [("web-search-plus", "web-search-plus", "2.2.0", "Search", "git", None)]
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {"web-search-plus"})
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())

    plugins_cmd.cmd_list(_args(json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "name": "web-search-plus",
            "status": "enabled",
            "version": "2.2.0",
            "description": "Search",
            "source": "git",
        }
    ]


def test_cmd_list_json_output_marks_nested_plugin_enabled_via_legacy_name(monkeypatch, capsys):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {"nemo_relay"})
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())

    plugins_cmd.cmd_list(_args(json=True, enabled=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "name": "observability/nemo_relay",
            "status": "enabled",
            "version": "0.1.0",
            "description": "Relay observability",
            "source": "bundled",
        }
    ]


def test_plugin_exists_accepts_legacy_nested_name(monkeypatch):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)

    assert plugins_cmd._plugin_exists("nemo_relay") is True
    assert plugins_cmd._plugin_exists("observability/nemo_relay") is True


def test_cmd_enable_accepts_legacy_nested_name(monkeypatch, capsys):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    enabled = set()
    disabled = set()

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled))
    monkeypatch.setattr(plugins_cmd, "_save_enabled_set", lambda value: enabled.clear() or enabled.update(value))
    monkeypatch.setattr(plugins_cmd, "_save_disabled_set", lambda value: disabled.clear() or disabled.update(value))

    plugins_cmd.cmd_enable("nemo_relay")

    assert enabled == {"observability/nemo_relay"}
    assert disabled == set()
    assert "enabled" in capsys.readouterr().out


def test_cmd_disable_accepts_legacy_nested_name(monkeypatch, capsys):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    enabled = {"nemo_relay"}
    disabled = set()

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled))
    monkeypatch.setattr(plugins_cmd, "_save_enabled_set", lambda value: enabled.clear() or enabled.update(value))
    monkeypatch.setattr(plugins_cmd, "_save_disabled_set", lambda value: disabled.clear() or disabled.update(value))

    plugins_cmd.cmd_disable("nemo_relay")

    assert enabled == set()
    assert disabled == {"observability/nemo_relay"}
    assert "disabled" in capsys.readouterr().out


def test_cmd_enable_clears_mixed_nested_alias_disable_state(monkeypatch, capsys):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    enabled = set()
    disabled = {"observability/nemo_relay"}

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled))
    monkeypatch.setattr(plugins_cmd, "_save_enabled_set", lambda value: enabled.clear() or enabled.update(value))
    monkeypatch.setattr(plugins_cmd, "_save_disabled_set", lambda value: disabled.clear() or disabled.update(value))

    plugins_cmd.cmd_enable("nemo_relay")

    assert enabled == {"observability/nemo_relay"}
    assert disabled == set()
    assert "enabled" in capsys.readouterr().out


def test_cmd_enable_resolves_split_nested_alias_state(monkeypatch, capsys):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    enabled = {"observability/nemo_relay"}
    disabled = {"nemo_relay"}

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled))
    monkeypatch.setattr(plugins_cmd, "_save_enabled_set", lambda value: enabled.clear() or enabled.update(value))
    monkeypatch.setattr(plugins_cmd, "_save_disabled_set", lambda value: disabled.clear() or disabled.update(value))

    plugins_cmd.cmd_enable("observability/nemo_relay")

    assert enabled == {"observability/nemo_relay"}
    assert disabled == set()
    assert "enabled" in capsys.readouterr().out


def test_cmd_disable_resolves_split_nested_alias_state(monkeypatch, capsys):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    enabled = {"nemo_relay"}
    disabled = {"observability/nemo_relay"}

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled))
    monkeypatch.setattr(plugins_cmd, "_save_enabled_set", lambda value: enabled.clear() or enabled.update(value))
    monkeypatch.setattr(plugins_cmd, "_save_disabled_set", lambda value: disabled.clear() or disabled.update(value))

    plugins_cmd.cmd_disable("observability/nemo_relay")

    assert enabled == set()
    assert disabled == {"observability/nemo_relay"}
    assert "disabled" in capsys.readouterr().out


def test_dashboard_enable_resolves_split_nested_alias_state(monkeypatch):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    enabled = {"observability/nemo_relay"}
    disabled = {"nemo_relay"}
    toggled: list[tuple[str, bool]] = []

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled))
    monkeypatch.setattr(plugins_cmd, "_save_enabled_set", lambda value: enabled.clear() or enabled.update(value))
    monkeypatch.setattr(plugins_cmd, "_save_disabled_set", lambda value: disabled.clear() or disabled.update(value))
    monkeypatch.setattr(plugins_cmd, "_toggle_plugin_toolset", lambda name, enable: toggled.append((name, enable)))

    result = plugins_cmd.dashboard_set_agent_plugin_enabled("observability/nemo_relay", enabled=True)

    assert result == {"ok": True, "name": "observability/nemo_relay", "unchanged": False}
    assert enabled == {"observability/nemo_relay"}
    assert disabled == set()
    assert toggled == [("observability/nemo_relay", True)]


def test_dashboard_disable_resolves_split_nested_alias_state(monkeypatch):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    enabled = {"nemo_relay"}
    disabled = {"observability/nemo_relay"}
    toggled: list[tuple[str, bool]] = []

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled))
    monkeypatch.setattr(plugins_cmd, "_save_enabled_set", lambda value: enabled.clear() or enabled.update(value))
    monkeypatch.setattr(plugins_cmd, "_save_disabled_set", lambda value: disabled.clear() or disabled.update(value))
    monkeypatch.setattr(plugins_cmd, "_toggle_plugin_toolset", lambda name, enable: toggled.append((name, enable)))

    result = plugins_cmd.dashboard_set_agent_plugin_enabled("nemo_relay", enabled=False)

    assert result == {"ok": True, "name": "observability/nemo_relay", "unchanged": False}
    assert enabled == set()
    assert disabled == {"observability/nemo_relay"}
    assert toggled == [("observability/nemo_relay", False)]


def test_cmd_toggle_does_not_crash_on_nested_plugin_entry_shape(monkeypatch, capsys):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd.sys, "stdin", SimpleNamespace(isatty=lambda: False))

    plugins_cmd.cmd_toggle()

    assert "Interactive mode requires a terminal." in capsys.readouterr().out


def test_cmd_toggle_marks_nested_plugin_selected_via_legacy_alias(monkeypatch):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {"nemo_relay"})
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd.sys, "stdin", SimpleNamespace(isatty=lambda: True))

    def _fake_ui(_curses, plugin_names, _plugin_labels, plugin_selected, plugin_refs, disabled, _categories, _console):
        captured["plugin_names"] = plugin_names
        captured["plugin_selected"] = plugin_selected
        captured["plugin_refs"] = plugin_refs
        captured["disabled"] = disabled

    monkeypatch.setattr(plugins_cmd, "_run_composite_ui", _fake_ui)

    plugins_cmd.cmd_toggle()

    assert captured["plugin_names"] == ["observability/nemo_relay"]
    assert captured["plugin_selected"] == {0}
    assert captured["plugin_refs"] == [("observability/nemo_relay", {"observability/nemo_relay", "nemo_relay"})]
    assert captured["disabled"] == set()


def test_normalize_plugin_selection_cleans_mixed_nested_alias_state():
    plugin_refs = [("observability/nemo_relay", {"observability/nemo_relay", "nemo_relay"})]

    new_enabled, new_disabled = plugins_cmd._normalize_plugin_selection(
        plugin_refs,
        {0},
        {"nemo_relay"},
    )

    assert new_enabled == {"observability/nemo_relay"}
    assert new_disabled == set()


def test_dashboard_remove_user_plugin_rejects_bundled_nested_alias_without_crash(monkeypatch, tmp_path: Path):
    entries = [
        (
            "observability/nemo_relay",
            "nemo_relay",
            "0.1.0",
            "Relay observability",
            "bundled",
            None,
        )
    ]
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: tmp_path)

    result = plugins_cmd.dashboard_remove_user_plugin("nemo_relay")

    assert result == {"ok": False, "error": "Bundled plugins cannot be removed from the dashboard."}


def test_discover_all_plugins_includes_nested_bundled_keys(monkeypatch, tmp_path: Path):
    bundled_dir = tmp_path / "bundled"
    nested_plugin_dir = bundled_dir / "observability" / "nemo_relay"
    nested_plugin_dir.mkdir(parents=True)
    (nested_plugin_dir / "plugin.yaml").write_text(
        "\n".join(
            [
                "name: nemo_relay",
                "version: '0.1.0'",
                "description: nested bundled plugin",
            ]
        ),
        encoding="utf-8",
    )

    user_dir = tmp_path / "user"
    user_dir.mkdir()

    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_bundled_plugins_dir",
        lambda: bundled_dir,
    )

    entries = plugins_cmd._discover_all_plugins()

    assert (
        "observability/nemo_relay",
        "nemo_relay",
        "0.1.0",
        "nested bundled plugin",
        "bundled",
        nested_plugin_dir,
    ) in entries
