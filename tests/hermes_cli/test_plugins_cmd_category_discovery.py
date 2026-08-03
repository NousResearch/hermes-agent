"""Tests for the nested category plugin discovery fix (issue #41066).

Verifies that _discover_all_plugins() recurses into category directories
(up to 2 levels deep) and that _plugin_status() checks both manifest name
and path-derived key against the enabled/disabled sets.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plugin_dir(parent: Path, name: str, manifest: dict) -> Path:
    """Create a minimal plugin directory with a plugin.yaml."""
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    import yaml
    (d / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
    (d / "__init__.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    return d


def _make_category_plugin(
    parent: Path, category: str, name: str, manifest: dict
) -> Path:
    """Create a category-namespaced plugin: <parent>/<category>/<name>/plugin.yaml."""
    return _make_plugin_dir(parent / category, name, manifest)


# ---------------------------------------------------------------------------
# _read_manifest_info
# ---------------------------------------------------------------------------


class TestReadManifestInfo:
    def test_flat_plugin(self, tmp_path):
        from hermes_cli.plugins_cmd import _read_manifest_info

        d = _make_plugin_dir(tmp_path, "my-plugin", {
            "name": "my-plugin", "version": "1.0.0", "description": "test"
        })
        result = _read_manifest_info(d, "")
        assert result is not None
        name, version, description, key, kind = result
        assert name == "my-plugin"
        assert version == "1.0.0"
        assert description == "test"
        assert key == "my-plugin"  # flat: key == name
        assert kind == "standalone"


    def test_no_manifest(self, tmp_path):
        from hermes_cli.plugins_cmd import _read_manifest_info

        d = tmp_path / "empty-dir"
        d.mkdir()
        assert _read_manifest_info(d, "") is None

    def test_yml_extension(self, tmp_path):
        from hermes_cli.plugins_cmd import _read_manifest_info

        d = tmp_path / "my-plugin"
        d.mkdir()
        import yaml
        (d / "plugin.yml").write_text(yaml.dump({"name": "my-plugin"}), encoding="utf-8")
        result = _read_manifest_info(d, "")
        assert result is not None
        assert result[0] == "my-plugin"


# ---------------------------------------------------------------------------
# _discover_all_plugins — recursive discovery
# ---------------------------------------------------------------------------


class TestDiscoverAllPlugins:
    @pytest.mark.parametrize("external_source", ["user", "project"])
    def test_inactive_external_collision_reports_active_bundled_winner(
        self,
        tmp_path,
        monkeypatch,
        capsys,
        external_source,
    ):
        from hermes_cli import plugins_cmd

        bundled = tmp_path / "bundled"
        hermes_home = tmp_path / "home"
        user_plugins = hermes_home / "plugins"
        user_plugins.mkdir(parents=True)
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  enabled: []\n  disabled: []\n",
            encoding="utf-8",
        )
        _make_plugin_dir(
            bundled,
            "shared",
            {
                "name": "shared",
                "version": "1.0.0",
                "kind": "backend",
            },
        )

        if external_source == "user":
            external_root = user_plugins
            monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
        else:
            project = tmp_path / "project"
            project.mkdir()
            monkeypatch.chdir(project)
            monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
            external_root = project / ".hermes" / "plugins"
        _make_plugin_dir(
            external_root,
            "shared",
            {
                "name": "shared",
                "version": "9.0.0",
                "kind": "backend",
            },
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
        monkeypatch.setattr(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            lambda: bundled,
        )
        monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_plugins)
        monkeypatch.setattr(plugins_cmd, "_discover_entrypoint_plugins", lambda: [])

        entries = plugins_cmd._discover_all_plugins()
        shared = [entry for entry in entries if entry[5] == "shared"]

        assert len(shared) == 1
        assert shared[0][3] == "bundled"
        assert shared[0][1] == "1.0.0"

        args = MagicMock(
            json=True,
            plain=False,
            no_bundled=False,
            user=False,
            enabled=False,
        )
        plugins_cmd.cmd_list(args)
        payload = json.loads(capsys.readouterr().out)
        listed = next(plugin for plugin in payload if plugin["key"] == "shared")
        assert listed["source"] == "bundled"
        assert listed["status"] == "enabled"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_flat_plugins_still_discovered(self, mock_user_dir, mock_bundled_dir, tmp_path):
        from hermes_cli.plugins_cmd import _discover_all_plugins

        _make_plugin_dir(tmp_path, "disk-cleanup", {
            "name": "disk-cleanup", "version": "1.0.0"
        })
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]
        assert "disk-cleanup" in keys


    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_mixed_flat_and_category(self, mock_user_dir, mock_bundled_dir, tmp_path):
        from hermes_cli.plugins_cmd import _discover_all_plugins

        _make_plugin_dir(tmp_path, "disk-cleanup", {
            "name": "disk-cleanup", "version": "1.0.0"
        })
        _make_category_plugin(tmp_path, "web", "tavily", {
            "name": "web-tavily", "version": "1.0.0"
        })
        _make_category_plugin(tmp_path, "web", "exa", {
            "name": "web-exa", "version": "1.0.0"
        })
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]
        assert "disk-cleanup" in keys
        assert "web/tavily" in keys
        assert "web/exa" in keys
        assert len(entries) == 3

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_depth_cap_at_two(self, mock_user_dir, mock_bundled_dir, tmp_path):
        """Plugins nested 3 levels deep should NOT be discovered."""
        from hermes_cli.plugins_cmd import _discover_all_plugins

        # 2 levels: should be found
        _make_category_plugin(tmp_path, "web", "tavily", {
            "name": "web-tavily", "version": "1.0.0"
        })
        # 3 levels: should NOT be found
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        import yaml
        (deep / "plugin.yaml").write_text(
            yaml.dump({"name": "too-deep"}), encoding="utf-8"
        )
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]
        assert "web/tavily" in keys
        assert "a/b/c" not in keys

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_bundled_platform_key_matches_runtime_loader(
        self,
        mock_user_dir,
        mock_bundled_dir,
        tmp_path,
    ):
        from hermes_cli.plugins_cmd import _discover_all_plugins

        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        user.mkdir()
        _make_category_plugin(
            bundled,
            "platforms",
            "buzz",
            {
                "name": "buzz-platform",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        mock_bundled_dir.return_value = bundled
        mock_user_dir.return_value = user

        entries = _discover_all_plugins()
        buzz = next(entry for entry in entries if entry[0] == "buzz-platform")

        assert buzz[5] == "buzz-platform"
        assert buzz[6] == "platform"


# ---------------------------------------------------------------------------
# _plugin_status — key-aware status
# ---------------------------------------------------------------------------


class TestPluginStatus:


    def test_key_in_disabled(self):
        from hermes_cli.plugins_cmd import _plugin_status
        assert _plugin_status("web-tavily", set(), {"web/tavily"}, key="web/tavily") == "disabled"

    def test_neither_name_nor_key(self):
        from hermes_cli.plugins_cmd import _plugin_status
        assert _plugin_status("unknown", {"other"}, set(), key="cat/unknown") == "not enabled"


# ---------------------------------------------------------------------------
# Integration: _filter_plugin_entries with category plugins
# ---------------------------------------------------------------------------


class TestFilterPluginEntries:
    def test_enabled_filter_uses_key(self):
        from hermes_cli.plugins_cmd import _filter_plugin_entries

        entries = [
            ("web-tavily", "1.0.0", "search", "user", Path("/tmp"), "web/tavily", "standalone"),
            (
                "disk-cleanup",
                "1.0.0",
                "cleanup",
                "bundled",
                Path("/tmp"),
                "disk-cleanup",
                "standalone",
            ),
        ]
        args = MagicMock()
        args.no_bundled = False
        args.user = False
        args.enabled = True

        result = _filter_plugin_entries(entries, args, {"web/tavily"}, set())
        assert len(result) == 1
        assert result[0][5] == "web/tavily"


# ---------------------------------------------------------------------------
# Integration: cmd_list JSON output includes category plugins
# ---------------------------------------------------------------------------


class TestCmdListJson:
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_json_output_includes_category_plugins(self, mock_user_dir, mock_bundled_dir, tmp_path, capsys):
        from hermes_cli.plugins_cmd import cmd_list

        _make_category_plugin(tmp_path, "web", "tavily", {
            "name": "web-tavily", "version": "1.0.0", "description": "search"
        })
        _make_plugin_dir(tmp_path, "disk-cleanup", {
            "name": "disk-cleanup", "version": "2.0.0", "description": "cleanup"
        })
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        args = MagicMock()
        args.json = True
        args.plain = False
        args.no_bundled = False
        args.user = False
        args.enabled = False

        cmd_list(args)
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        names = [p["name"] for p in payload]
        assert "web-tavily" in names
        assert "disk-cleanup" in names
        by_name = {p["name"]: p for p in payload}
        assert by_name["web-tavily"]["key"] == "web/tavily"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_json_status_uses_key(self, mock_user_dir, mock_bundled_dir, tmp_path, capsys):
        from hermes_cli.plugins_cmd import cmd_list

        _make_category_plugin(tmp_path, "web", "tavily", {
            "name": "web-tavily", "version": "1.0.0"
        })
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        # Patch config to return web/tavily as enabled
        with patch("hermes_cli.plugins_cmd._get_enabled_set", return_value={"web/tavily"}):
            args = MagicMock()
            args.json = True
            args.plain = False
            args.no_bundled = False
            args.user = False
            args.enabled = False

            cmd_list(args)
            captured = capsys.readouterr()
            payload = json.loads(captured.out)
            assert len(payload) == 1
            assert payload[0]["status"] == "enabled"
