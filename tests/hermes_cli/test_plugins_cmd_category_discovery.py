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
        name, version, description, key = result
        assert name == "my-plugin"
        assert version == "1.0.0"
        assert description == "test"
        assert key == "my-plugin"  # flat: key == name


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
    def test_bundled_model_providers_skipped(self, mock_user_dir, mock_bundled_dir, tmp_path):
        """``plugins/model-providers/`` has its own provider registry loader.

        Bundled providers should not appear in ``hermes plugins list`` as
        general opt-in plugins, or users get a misleading enable/disable
        surface for providers selected via ``model.provider`` / ``--provider``.
        Same rationale as the existing bundled memory/context_engine skip.
        """
        from hermes_cli.plugins_cmd import _discover_all_plugins

        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        user.mkdir()
        _make_category_plugin(bundled, "model-providers", "openrouter", {
            "name": "openrouter-provider", "version": "1.0.0",
            "kind": "model-provider",
        })
        _make_category_plugin(bundled, "memory", "letta", {
            "name": "letta-memory", "version": "1.0.0"
        })
        _make_category_plugin(bundled, "context_engine", "compressor", {
            "name": "compressor", "version": "1.0.0"
        })
        _make_category_plugin(bundled, "observability", "langfuse", {
            "name": "langfuse", "version": "1.0.0"
        })
        mock_user_dir.return_value = user
        mock_bundled_dir.return_value = bundled

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]
        assert "model-providers/openrouter" not in keys
        assert "memory/letta" not in keys
        assert "context_engine/compressor" not in keys
        assert "observability/langfuse" in keys

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_user_model_providers_subdir_is_still_scanned(
        self, mock_user_dir, mock_bundled_dir, tmp_path
    ):
        """The model-providers skip only applies to *bundled* — a user plugin
        at ``~/.hermes/plugins/model-providers/<x>/`` is still discovered so
        ``hermes plugins list`` shows what the user installed."""
        from hermes_cli.plugins_cmd import _discover_all_plugins

        bundled = tmp_path / "bundled"
        bundled.mkdir()
        user = tmp_path / "user"
        _make_category_plugin(user, "model-providers", "acme", {
            "name": "acme-provider", "version": "0.1.0",
            "kind": "model-provider",
        })
        mock_user_dir.return_value = user
        mock_bundled_dir.return_value = bundled

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]
        assert "model-providers/acme" in keys


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
            ("web-tavily", "1.0.0", "search", "user", Path("/tmp"), "web/tavily"),
            ("disk-cleanup", "1.0.0", "cleanup", "bundled", Path("/tmp"), "disk-cleanup"),
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
