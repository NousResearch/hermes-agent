import argparse
import json
from pathlib import Path

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
