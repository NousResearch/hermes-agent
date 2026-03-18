"""Tests for hermes_cli.plugins_cmd — the ``hermes plugins`` CLI subcommand."""

from __future__ import annotations

import logging
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_cli.plugins_cmd import (
    _read_manifest,
    _repo_name_from_url,
    _resolve_git_url,
    _sanitize_plugin_name,
    plugins_command,
)


# ── _sanitize_plugin_name ─────────────────────────────────────────────────


class TestSanitizePluginName:
    """Reject path-traversal attempts while accepting valid names."""

    def test_valid_simple_name(self, tmp_path):
        target = _sanitize_plugin_name("my-plugin", tmp_path)
        assert target == (tmp_path / "my-plugin").resolve()

    def test_valid_name_with_hyphen_and_digits(self, tmp_path):
        target = _sanitize_plugin_name("plugin-v2", tmp_path)
        assert target.name == "plugin-v2"

    def test_rejects_dot_dot(self, tmp_path):
        with pytest.raises(ValueError, match="must not contain"):
            _sanitize_plugin_name("../../etc/passwd", tmp_path)

    def test_rejects_single_dot_dot(self, tmp_path):
        with pytest.raises(ValueError, match="must not contain"):
            _sanitize_plugin_name("..", tmp_path)

    def test_rejects_forward_slash(self, tmp_path):
        with pytest.raises(ValueError, match="must not contain"):
            _sanitize_plugin_name("foo/bar", tmp_path)

    def test_rejects_backslash(self, tmp_path):
        with pytest.raises(ValueError, match="must not contain"):
            _sanitize_plugin_name("foo\\bar", tmp_path)

    def test_rejects_absolute_path(self, tmp_path):
        with pytest.raises(ValueError, match="must not contain"):
            _sanitize_plugin_name("/etc/passwd", tmp_path)

    def test_rejects_empty_name(self, tmp_path):
        with pytest.raises(ValueError, match="must not be empty"):
            _sanitize_plugin_name("", tmp_path)


# ── _resolve_git_url ──────────────────────────────────────────────────────


class TestResolveGitUrl:
    """Shorthand and full-URL resolution."""

    def test_owner_repo_shorthand(self):
        url = _resolve_git_url("owner/repo")
        assert url == "https://github.com/owner/repo.git"

    def test_https_url_passthrough(self):
        url = _resolve_git_url("https://github.com/x/y.git")
        assert url == "https://github.com/x/y.git"

    def test_ssh_url_passthrough(self):
        url = _resolve_git_url("git@github.com:x/y.git")
        assert url == "git@github.com:x/y.git"

    def test_http_url_passthrough(self):
        url = _resolve_git_url("http://example.com/repo.git")
        assert url == "http://example.com/repo.git"

    def test_file_url_passthrough(self):
        url = _resolve_git_url("file:///tmp/repo")
        assert url == "file:///tmp/repo"

    def test_invalid_single_word_raises(self):
        with pytest.raises(ValueError, match="Invalid plugin identifier"):
            _resolve_git_url("justoneword")

    def test_invalid_three_parts_raises(self):
        with pytest.raises(ValueError, match="Invalid plugin identifier"):
            _resolve_git_url("a/b/c")


# ── _repo_name_from_url ──────────────────────────────────────────────────


class TestRepoNameFromUrl:
    """Extract plugin directory name from Git URLs."""

    def test_https_with_dot_git(self):
        assert _repo_name_from_url("https://github.com/owner/my-plugin.git") == "my-plugin"

    def test_https_without_dot_git(self):
        assert _repo_name_from_url("https://github.com/owner/my-plugin") == "my-plugin"

    def test_trailing_slash(self):
        assert _repo_name_from_url("https://github.com/owner/repo/") == "repo"

    def test_ssh_style(self):
        assert _repo_name_from_url("git@github.com:owner/repo.git") == "repo"

    def test_ssh_protocol(self):
        assert _repo_name_from_url("ssh://git@github.com/owner/repo.git") == "repo"


# ── plugins_command dispatch ──────────────────────────────────────────────


class TestPluginsCommandDispatch:
    """Verify alias routing in plugins_command()."""

    def _make_args(self, action, **extras):
        args = MagicMock()
        args.plugins_action = action
        for k, v in extras.items():
            setattr(args, k, v)
        return args

    @patch("hermes_cli.plugins_cmd.cmd_remove")
    def test_rm_alias(self, mock_remove):
        args = self._make_args("rm", name="some-plugin")
        plugins_command(args)
        mock_remove.assert_called_once_with("some-plugin")

    @patch("hermes_cli.plugins_cmd.cmd_remove")
    def test_uninstall_alias(self, mock_remove):
        args = self._make_args("uninstall", name="some-plugin")
        plugins_command(args)
        mock_remove.assert_called_once_with("some-plugin")

    @patch("hermes_cli.plugins_cmd.cmd_list")
    def test_ls_alias(self, mock_list):
        args = self._make_args("ls")
        plugins_command(args)
        mock_list.assert_called_once()

    @patch("hermes_cli.plugins_cmd.cmd_list")
    def test_none_falls_through_to_list(self, mock_list):
        args = self._make_args(None)
        plugins_command(args)
        mock_list.assert_called_once()

    @patch("hermes_cli.plugins_cmd.cmd_install")
    def test_install_dispatches(self, mock_install):
        args = self._make_args("install", identifier="owner/repo", force=False)
        plugins_command(args)
        mock_install.assert_called_once_with("owner/repo", force=False)

    @patch("hermes_cli.plugins_cmd.cmd_update")
    def test_update_dispatches(self, mock_update):
        args = self._make_args("update", name="foo")
        plugins_command(args)
        mock_update.assert_called_once_with("foo")

    @patch("hermes_cli.plugins_cmd.cmd_remove")
    def test_remove_dispatches(self, mock_remove):
        args = self._make_args("remove", name="bar")
        plugins_command(args)
        mock_remove.assert_called_once_with("bar")


# ── _read_manifest ────────────────────────────────────────────────────────


class TestReadManifest:
    """Manifest reading edge cases."""

    def test_valid_yaml(self, tmp_path):
        manifest = {"name": "cool-plugin", "version": "1.0.0"}
        (tmp_path / "plugin.yaml").write_text(yaml.dump(manifest))
        result = _read_manifest(tmp_path)
        assert result["name"] == "cool-plugin"
        assert result["version"] == "1.0.0"

    def test_missing_file_returns_empty(self, tmp_path):
        result = _read_manifest(tmp_path)
        assert result == {}

    def test_invalid_yaml_returns_empty_and_logs(self, tmp_path, caplog):
        (tmp_path / "plugin.yaml").write_text(": : : bad yaml [[[")
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins_cmd"):
            result = _read_manifest(tmp_path)
        assert result == {}
        assert any("Failed to read plugin.yaml" in r.message for r in caplog.records)

    def test_empty_file_returns_empty(self, tmp_path):
        (tmp_path / "plugin.yaml").write_text("")
        result = _read_manifest(tmp_path)
        assert result == {}
