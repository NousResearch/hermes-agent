import argparse
import json
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
        plugins_cmd.importlib.metadata,
        "entry_points",
        lambda: [entry_point],
    )
    monkeypatch.setattr("hermes_cli.plugins._get_extra_plugin_paths", lambda: [])

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


def test_discover_all_plugins_includes_external_collection_and_checkout(
    monkeypatch, tmp_path
):
    bundled_dir = tmp_path / "bundled"
    user_dir = tmp_path / "user"
    collection = tmp_path / "external-collection"
    direct = tmp_path / "external-direct"
    bundled_dir.mkdir()
    user_dir.mkdir()
    plugin_dir = collection / "analytics"
    plugin_dir.mkdir(parents=True)
    direct.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        "name: analytics\nversion: 1.2.3\ndescription: External analytics\n"
    )
    (direct / "plugin.yaml").write_text(
        "name: direct-plugin\nversion: 2.0.0\ndescription: Direct checkout\n"
    )

    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled_dir)
    monkeypatch.setattr(
        "hermes_cli.plugins._get_extra_plugin_paths",
        lambda: [collection, direct],
    )
    monkeypatch.setattr(plugins_cmd.importlib.metadata, "entry_points", lambda: [])

    entries = plugins_cmd._discover_all_plugins()

    assert entries == [
        (
            "analytics",
            "1.2.3",
            "External analytics",
            "external",
            plugin_dir,
            "analytics",
        ),
        (
            "direct-plugin",
            "2.0.0",
            "Direct checkout",
            "external",
            direct,
            "direct-plugin",
        ),
    ]


def test_discover_all_plugins_unenabled_external_does_not_shadow_bundled(
    monkeypatch, tmp_path
):
    bundled_dir = tmp_path / "bundled"
    user_dir = tmp_path / "user"
    external_dir = tmp_path / "external"
    bundled_plugin = bundled_dir / "shared-plugin"
    external_plugin = external_dir / "shared-plugin"
    user_dir.mkdir()
    bundled_plugin.mkdir(parents=True)
    external_plugin.mkdir(parents=True)
    (bundled_plugin / "plugin.yaml").write_text(
        "name: shared-plugin\nversion: 1.0.0\ndescription: Bundled\n"
    )
    (external_plugin / "plugin.yaml").write_text(
        "name: shared-plugin\nversion: 2.0.0\ndescription: External\n"
    )

    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled_dir)
    monkeypatch.setattr("hermes_cli.plugins._get_extra_plugin_paths", lambda: [external_dir])
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd.importlib.metadata, "entry_points", lambda: [])

    entries = plugins_cmd._discover_all_plugins()

    assert entries == [
        (
            "shared-plugin",
            "1.0.0",
            "Bundled",
            "bundled",
            bundled_plugin,
            "shared-plugin",
        )
    ]


def test_discover_all_plugins_includes_enabled_project_plugins(monkeypatch, tmp_path):
    bundled_dir = tmp_path / "bundled"
    user_dir = tmp_path / "user"
    project_plugin = tmp_path / ".hermes" / "plugins" / "project-tool"
    bundled_dir.mkdir()
    user_dir.mkdir()
    project_plugin.mkdir(parents=True)
    (project_plugin / "plugin.yaml").write_text(
        "name: project-tool\nversion: 1.0.0\ndescription: Project tool\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "true")
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled_dir)
    monkeypatch.setattr("hermes_cli.plugins._get_extra_plugin_paths", lambda: [])
    monkeypatch.setattr(plugins_cmd.importlib.metadata, "entry_points", lambda: [])

    entries = plugins_cmd._discover_all_plugins()

    assert entries == [
        (
            "project-tool",
            "1.0.0",
            "Project tool",
            "project",
            project_plugin,
            "project-tool",
        )
    ]


def test_discover_all_plugins_warns_for_missing_external_root(
    monkeypatch, tmp_path, caplog
):
    bundled_dir = tmp_path / "bundled"
    user_dir = tmp_path / "user"
    missing = tmp_path / "missing-external"
    bundled_dir.mkdir()
    user_dir.mkdir()
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled_dir)
    monkeypatch.setattr("hermes_cli.plugins._get_extra_plugin_paths", lambda: [missing])
    monkeypatch.setattr(plugins_cmd.importlib.metadata, "entry_points", lambda: [])

    with caplog.at_level("WARNING", logger="hermes_cli.plugins"):
        entries = plugins_cmd._discover_all_plugins()

    assert entries == []
    assert "does not exist or is not a directory" in caplog.text


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
        plugins_cmd.importlib.metadata,
        "entry_points",
        lambda: [plugin_ep, capability_ep],
    )
    monkeypatch.setattr("hermes_cli.plugins._get_extra_plugin_paths", lambda: [])

    assert plugins_cmd._declared_capabilities_for_key("thread-namer") == [
        "gateway.platform_actions"
    ]
