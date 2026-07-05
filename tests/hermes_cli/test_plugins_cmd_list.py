import argparse
import json
from unittest.mock import patch, MagicMock

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
        ("disk-cleanup", "2.0.0", "Bundled", "bundled", None, "disk-cleanup"),
        ("web-search-plus", "2.2.0", "Search", "git", None, "web-search-plus"),
        ("old-plugin", "1.0.0", "Old", "user", None, "old-plugin"),
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
        ("disk-cleanup", "2.0.0", "Bundled", "bundled", None, "disk-cleanup"),
        ("drawthings-grpc", "0.3.0", "Draw Things", "user", None, "drawthings-grpc"),
        ("web-search-plus", "2.2.0", "Search", "git", None, "web-search-plus"),
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
        ("disk-cleanup", "2.0.0", "Bundled", "bundled", None, "disk-cleanup"),
        ("web-search-plus", "2.2.0", "Search", "git", None, "web-search-plus"),
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
    entries = [("web-search-plus", "2.2.0", "Search", "git", None, "web-search-plus")]
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


def test_discover_entry_point_plugins_returns_tuples():
    """Entry-point plugins should produce 6-tuples like directory plugins."""
    fake_ep = MagicMock()
    fake_ep.name = "rtk-rewrite"
    fake_ep.value = "rtk_hermes"
    fake_ep.group = "hermes_agent.plugins"
    fake_ep.dist.name = "rtk-hermes"

    fake_dist_meta = MagicMock()
    fake_dist_meta.get = lambda key, default="": {
        "Version": "1.2.3",
        "Summary": "RTK rewrite plugin",
    }.get(key, default)

    with patch("importlib.metadata.entry_points", return_value=MagicMock(
            select=MagicMock(return_value=[fake_ep]))), \
         patch("importlib.metadata.metadata", return_value=fake_dist_meta):
        entries = plugins_cmd._discover_entry_point_plugins()

    assert len(entries) == 1
    name, version, description, source, path, key = entries[0]
    assert name == "rtk-rewrite"
    assert version == "1.2.3"
    assert description == "RTK rewrite plugin"
    assert source == "entrypoint"
    assert key == "rtk-rewrite"


def test_discover_all_plugins_includes_entry_points():
    """_discover_all_plugins should include entry-point plugins."""
    ep_entry = ("rtk-rewrite", "1.2.3", "RTK rewrite", "entrypoint", None, "rtk-rewrite")

    with patch("hermes_cli.plugins_cmd._discover_entry_point_plugins",
               return_value=[ep_entry]), \
         patch("hermes_cli.plugins.get_bundled_plugins_dir",
               return_value=MagicMock(is_dir=MagicMock(return_value=False))), \
         patch("hermes_cli.plugins_cmd._plugins_dir",
               return_value=MagicMock(is_dir=MagicMock(return_value=False))):
        entries = plugins_cmd._discover_all_plugins()

    keys = [e[5] for e in entries]
    assert "rtk-rewrite" in keys
