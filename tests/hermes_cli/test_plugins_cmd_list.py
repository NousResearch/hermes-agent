import importlib.metadata
import argparse
import json
import logging
import os
from types import SimpleNamespace

import pytest

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


def test_discover_all_plugins_includes_entrypoint_plugins(monkeypatch, tmp_path):
    bundled_dir = tmp_path / "bundled"
    user_dir = tmp_path / "user"
    bundled_dir.mkdir()
    user_dir.mkdir()

    dist = SimpleNamespace(
        version="0.1.0",
        metadata={"Summary": "Karpathy-style LLM Wikis for Hermes"},
    )
    entry_point = SimpleNamespace(
        name="wiki",
        value="adapters.hermes.cli_plugin",
        group="hermes_agent.plugins",
        dist=dist,
    )

    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_bundled_plugins_dir",
        lambda: bundled_dir,
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: [entry_point],
    )

    entries = plugins_cmd._discover_all_plugins()

    assert entries == [
        (
            "wiki",
            "0.1.0",
            "Karpathy-style LLM Wikis for Hermes",
            "entrypoint",
            "adapters.hermes.cli_plugin",
            "wiki",
        )
    ]


def test_declared_capabilities_for_entrypoint_uses_distribution_metadata(
    monkeypatch, tmp_path
):
    bundled_dir = tmp_path / "bundled"
    user_dir = tmp_path / "user"
    bundled_dir.mkdir()
    user_dir.mkdir()
    plugin_ep = SimpleNamespace(
        name="thread-namer",
        value="thread_namer.plugin:register",
        group="hermes_agent.plugins",
        dist=SimpleNamespace(version="1.0", metadata={"Summary": ""}),
    )
    capability_ep = SimpleNamespace(
        name="thread-namer.gateway.platform_actions",
        value="thread_namer.plugin:register",
        group="hermes_agent.plugin_capabilities",
    )
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled_dir
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: [plugin_ep, capability_ep],
    )

    assert plugins_cmd._declared_capabilities_for_key("thread-namer") == [
        "gateway.platform_actions"
    ]


@pytest.mark.skipif(os.name == "nt", reason="chmod is a no-op on Windows")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores file permissions")
def test_unreadable_plugin_dir_is_skipped_by_every_manifest_scan(monkeypatch, tmp_path, caplog):
    """One plugin directory the process cannot stat() into (Windows WinError 5, POSIX mode 000)
    must be warned about and skipped — not abort discovery for every other plugin (#111804).
    Covers the loader scan (``scan_directory``) and the list/hub scan (``_scan_level``)."""
    from hermes_cli.plugins_discovery import scan_directory

    user_dir = tmp_path / "plugins"
    bundled_dir = tmp_path / "bundled"
    bundled_dir.mkdir()
    for name in ("denied", "good"):
        (user_dir / name).mkdir(parents=True)
        (user_dir / name / "plugin.yaml").write_text(f"name: {name}\nversion: 1.0.0\n", encoding="utf-8")
    (user_dir / "denied").chmod(0)
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled_dir)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: [])

    try:
        with caplog.at_level(logging.WARNING):
            loader_names = [m.name for m in scan_directory(user_dir, "user")]
            listed_names = [entry[0] for entry in plugins_cmd._discover_all_plugins()]
    finally:
        (user_dir / "denied").chmod(0o700)

    assert loader_names == ["good"]
    assert listed_names == ["good"]
    assert caplog.text.count("Skipping unreadable plugin directory") == 2
