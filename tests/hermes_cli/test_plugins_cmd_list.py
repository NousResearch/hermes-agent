import argparse
import json

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
        ("disk-cleanup", "2.0.0", "Bundled", "bundled", None),
        ("web-search-plus", "2.2.0", "Search", "git", None),
        ("old-plugin", "1.0.0", "Old", "user", None),
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
        ("disk-cleanup", "2.0.0", "Bundled", "bundled", None),
        ("drawthings-grpc", "0.3.0", "Draw Things", "user", None),
        ("web-search-plus", "2.2.0", "Search", "git", None),
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
        ("disk-cleanup", "2.0.0", "Bundled", "bundled", None),
        ("web-search-plus", "2.2.0", "Search", "git", None),
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
    entries = [("web-search-plus", "2.2.0", "Search", "git", None)]
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
