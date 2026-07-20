"""Tests for hermes_cli.plugins_cmd — the ``hermes plugins`` CLI subcommand."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_cli.plugins_cmd import (
    PluginOperationError,
    _copy_example_files,
    _find_sole_plugin_subdir,
    _has_plugin_marker,
    _read_manifest,
    _repo_name_from_url,
    _resolve_git_executable,
    _resolve_git_url,
    _resolve_subdir_within,
    _sanitize_plugin_name,
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
        with pytest.raises(ValueError, match="must not reference the plugins directory itself"):
            _sanitize_plugin_name("..", tmp_path)

    def test_rejects_single_dot(self, tmp_path):
        with pytest.raises(ValueError, match="must not reference the plugins directory itself"):
            _sanitize_plugin_name(".", tmp_path)

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

    # ── allow_subdir=True ──

    def test_allow_subdir_accepts_single_slash(self, tmp_path):
        target = _sanitize_plugin_name(
            "observability/langfuse", tmp_path, allow_subdir=True
        )
        assert target == (tmp_path / "observability" / "langfuse").resolve()

    def test_allow_subdir_strips_leading_trailing_slash(self, tmp_path):
        target = _sanitize_plugin_name(
            "/image_gen/openai/", tmp_path, allow_subdir=True
        )
        assert target == (tmp_path / "image_gen" / "openai").resolve()

    def test_allow_subdir_still_rejects_dot_dot(self, tmp_path):
        with pytest.raises(ValueError, match="must not contain"):
            _sanitize_plugin_name("foo/../bar", tmp_path, allow_subdir=True)

    def test_allow_subdir_still_rejects_backslash(self, tmp_path):
        with pytest.raises(ValueError, match="must not contain"):
            _sanitize_plugin_name("foo\\bar", tmp_path, allow_subdir=True)

    def test_allow_subdir_rejects_empty_after_strip(self, tmp_path):
        with pytest.raises(ValueError, match="must not be empty"):
            _sanitize_plugin_name("///", tmp_path, allow_subdir=True)

    def test_allow_subdir_resolves_inside_plugins_dir(self, tmp_path):
        target = _sanitize_plugin_name("a/b/c", tmp_path, allow_subdir=True)
        assert target.is_relative_to(tmp_path.resolve())


# ── _resolve_git_url ──────────────────────────────────────────────────────


class TestResolveGitUrl:
    """Shorthand and full-URL resolution, with optional subdirectory."""

    def test_owner_repo_shorthand(self):
        url, subdir = _resolve_git_url("owner/repo")
        assert url == "https://github.com/owner/repo.git"
        assert subdir is None

    def test_https_url_passthrough(self):
        url, subdir = _resolve_git_url("https://github.com/x/y.git")
        assert url == "https://github.com/x/y.git"
        assert subdir is None

    def test_ssh_url_passthrough(self):
        url, subdir = _resolve_git_url("git@github.com:x/y.git")
        assert url == "git@github.com:x/y.git"
        assert subdir is None

    def test_http_url_passthrough(self):
        url, subdir = _resolve_git_url("http://example.com/repo.git")
        assert url == "http://example.com/repo.git"
        assert subdir is None

    def test_file_url_passthrough(self):
        url, subdir = _resolve_git_url("file:///tmp/repo")
        assert url == "file:///tmp/repo"
        assert subdir is None

    def test_invalid_single_word_raises(self):
        with pytest.raises(ValueError, match="Invalid plugin identifier"):
            _resolve_git_url("justoneword")

    def test_shorthand_with_subdir(self):
        url, subdir = _resolve_git_url("owner/repo/my-plugin")
        assert url == "https://github.com/owner/repo.git"
        assert subdir == "my-plugin"

    def test_shorthand_with_nested_subdir(self):
        url, subdir = _resolve_git_url("owner/repo/path/to/plugin")
        assert url == "https://github.com/owner/repo.git"
        assert subdir == "path/to/plugin"

    def test_shorthand_with_subdir_trailing_slash(self):
        url, subdir = _resolve_git_url("owner/repo/my-plugin/")
        assert url == "https://github.com/owner/repo.git"
        assert subdir == "my-plugin"

    def test_https_url_with_subdir(self):
        url, subdir = _resolve_git_url("https://github.com/owner/repo.git/my-plugin")
        assert url == "https://github.com/owner/repo.git"
        assert subdir == "my-plugin"

    def test_https_url_with_nested_subdir(self):
        url, subdir = _resolve_git_url(
            "https://github.com/owner/repo.git/path/to/plugin"
        )
        assert url == "https://github.com/owner/repo.git"
        assert subdir == "path/to/plugin"

    def test_url_with_fragment_subdir(self):
        url, subdir = _resolve_git_url("https://github.com/owner/repo.git#my-plugin")
        assert url == "https://github.com/owner/repo.git"
        assert subdir == "my-plugin"

    def test_file_url_with_fragment_subdir(self):
        url, subdir = _resolve_git_url("file:///tmp/repo#path/to/plugin")
        assert url == "file:///tmp/repo"
        assert subdir == "path/to/plugin"

    def test_ssh_url_with_fragment_subdir(self):
        url, subdir = _resolve_git_url("git@github.com:owner/repo.git#sub")
        assert url == "git@github.com:owner/repo.git"
        assert subdir == "sub"

    @pytest.mark.parametrize(
        "identifier",
        [
            "https://github.com/owner/repo/tree/main",
            "https://github.com/owner/repo/blob/main/README.md",
            "https://github.com/owner/repo/pull/123",
            "https://github.com/owner/repo/commit/abc123def",
            "https://github.com/owner/repo/releases/tag/v1.0",
            "https://github.com/owner/repo/issues/42",
        ],
    )
    def test_github_browser_url_normalized_to_repo(self, identifier):
        url, subdir = _resolve_git_url(identifier)
        assert url == "https://github.com/owner/repo.git"
        assert subdir is None

    @pytest.mark.parametrize(
        ("identifier", "expected_subdir"),
        [
            ("https://github.com/owner/repo/tree/main/plugins/foo", "plugins/foo"),
            ("https://github.com/owner/repo/tree/feature-branch/plugin", "plugin"),
            ("https://github.com/owner/repo/tree/main/plugins/foo?plain=1", "plugins/foo"),
            ("https://github.com/owner/repo.git/tree/main/plugins/foo", "plugins/foo"),
        ],
    )
    def test_github_tree_browser_url_preserves_subdir(self, identifier, expected_subdir):
        url, subdir = _resolve_git_url(identifier)
        assert url == "https://github.com/owner/repo.git"
        assert subdir == expected_subdir

    @pytest.mark.parametrize(
        "identifier",
        [
            "https://github.com/owner/repo",
            "https://github.com/owner/repo.git",
            "https://github.com/owner",
            "https://github.com/owner/repo/branches",
            "https://github.com/owner//tree/main",
            "https://gitlab.com/owner/repo/tree/main",
            "git@github.com:owner/repo.git",
            "file:///tmp/repo/tree/main",
        ],
    )
    def test_non_browser_urls_passthrough(self, identifier):
        url, subdir = _resolve_git_url(identifier)
        assert url == identifier
        assert subdir is None


# ── _resolve_subdir_within ──────────────────────────────────────────────────


class TestResolveSubdirWithin:
    """Subdirectory resolution stays within the clone and rejects traversal."""

    def test_valid_subdir(self, tmp_path):
        (tmp_path / "my-plugin").mkdir()
        result = _resolve_subdir_within(tmp_path, "my-plugin")
        assert result == (tmp_path / "my-plugin").resolve()

    def test_valid_nested_subdir(self, tmp_path):
        (tmp_path / "a" / "b" / "c").mkdir(parents=True)
        result = _resolve_subdir_within(tmp_path, "a/b/c")
        assert result == (tmp_path / "a" / "b" / "c").resolve()

    def test_rejects_dot_dot_escape(self, tmp_path):
        clone = tmp_path / "clone"
        clone.mkdir()
        (tmp_path / "secret").mkdir()
        with pytest.raises(PluginOperationError, match="escapes the repository"):
            _resolve_subdir_within(clone, "../secret")

    def test_rejects_absolute_path_escape(self, tmp_path):
        clone = tmp_path / "clone"
        clone.mkdir()
        # An absolute path resolves outside the clone root.
        with pytest.raises(PluginOperationError, match="escapes the repository"):
            _resolve_subdir_within(clone, "/etc")

    def test_rejects_symlink_escape(self, tmp_path):
        clone = tmp_path / "clone"
        clone.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (clone / "link").symlink_to(outside)
        with pytest.raises(PluginOperationError, match="escapes the repository"):
            _resolve_subdir_within(clone, "link")

    def test_rejects_missing_subdir(self, tmp_path):
        with pytest.raises(PluginOperationError, match="does not exist"):
            _resolve_subdir_within(tmp_path, "nope")

    def test_rejects_file_not_dir(self, tmp_path):
        (tmp_path / "afile").write_text("x")
        with pytest.raises(PluginOperationError, match="not a directory"):
            _resolve_subdir_within(tmp_path, "afile")


class TestPluginSubdirDiscovery:
    def _make_plugin(self, path: Path, *, manifest: bool = True, init: bool = True):
        path.mkdir()
        if manifest:
            (path / "plugin.yaml").write_text("name: candidate\n")
        if init:
            (path / "__init__.py").write_text("")

    def test_requires_manifest_and_init(self, tmp_path):
        complete = tmp_path / "complete"
        manifest_only = tmp_path / "manifest-only"
        init_only = tmp_path / "init-only"
        self._make_plugin(complete)
        self._make_plugin(manifest_only, init=False)
        self._make_plugin(init_only, manifest=False)

        assert _has_plugin_marker(complete) is True
        assert _has_plugin_marker(manifest_only) is False
        assert _has_plugin_marker(init_only) is False

    def test_finds_exactly_one_loadable_child(self, tmp_path):
        plugin = tmp_path / "plugin"
        self._make_plugin(plugin)
        self._make_plugin(tmp_path / "ordinary-package", manifest=False)

        assert _find_sole_plugin_subdir(tmp_path) == plugin

    def test_ambiguous_children_are_not_selected(self, tmp_path):
        self._make_plugin(tmp_path / "plugin-a")
        self._make_plugin(tmp_path / "plugin-b")

        assert _find_sole_plugin_subdir(tmp_path) is None

    def test_hidden_child_is_not_selected(self, tmp_path):
        self._make_plugin(tmp_path / ".plugin")

        assert _find_sole_plugin_subdir(tmp_path) is None


# ── _resolve_git_executable ─────────────────────────────────────────────────


class TestResolveGitExecutable:
    """Fallback resolution when bare ``git`` is not discoverable via ``PATH``."""

    def teardown_method(self):
        _resolve_git_executable.cache_clear()

    def test_prefers_shutil_which(self):
        import hermes_cli.plugins_cmd as pc

        _resolve_git_executable.cache_clear()
        with patch.object(pc.shutil, "which", return_value="/usr/local/bin/git"):
            assert pc._resolve_git_executable() == "/usr/local/bin/git"

    def test_fallback_posix_first_matching_path(self):
        import hermes_cli.plugins_cmd as pc

        _resolve_git_executable.cache_clear()

        def _isfile(p: str) -> bool:
            return p == "/usr/local/bin/git"

        with patch.object(pc.shutil, "which", return_value=None):
            with patch.object(pc.os, "name", "posix"):
                with patch.object(pc.os.path, "isfile", side_effect=_isfile):
                    assert pc._resolve_git_executable() == "/usr/local/bin/git"

    def test_returns_none_when_unavailable(self):
        import hermes_cli.plugins_cmd as pc

        _resolve_git_executable.cache_clear()
        with patch.object(pc.shutil, "which", return_value=None):
            with patch.object(pc.os, "name", "posix"):
                with patch.object(pc.os.path, "isfile", return_value=False):
                    assert pc._resolve_git_executable() is None

    def test_git_pull_uses_resolved_executable(self, tmp_path):
        import hermes_cli.plugins_cmd as pc

        _resolve_git_executable.cache_clear()
        with patch.object(
            pc,
            "_resolve_git_executable",
            return_value="/resolved/git",
        ):
            with patch.object(pc.subprocess, "run") as run:
                run.return_value = MagicMock(returncode=0, stdout="Already up to date\n", stderr="")
                ok, msg = pc._git_pull_plugin_dir(tmp_path)
        assert ok is True
        run.assert_called_once()
        assert run.call_args[0][0][0] == "/resolved/git"

    def test_install_core_raises_when_git_unresolved(self):
        import hermes_cli.plugins_cmd as pc

        _resolve_git_executable.cache_clear()
        with patch.object(pc, "_resolve_git_executable", return_value=None):
            with pytest.raises(PluginOperationError, match="git is not installed"):
                pc._install_plugin_core("owner/repo", force=True)


# ── _repo_name_from_url ──────────────────────────────────────────────────


class TestRepoNameFromUrl:
    """Extract plugin directory name from Git URLs."""

    def test_https_with_dot_git(self):
        assert (
            _repo_name_from_url("https://github.com/owner/my-plugin.git") == "my-plugin"
        )

    def test_https_without_dot_git(self):
        assert _repo_name_from_url("https://github.com/owner/my-plugin") == "my-plugin"

    def test_trailing_slash(self):
        assert _repo_name_from_url("https://github.com/owner/repo/") == "repo"

    def test_ssh_style(self):
        assert _repo_name_from_url("git@github.com:owner/repo.git") == "repo"

    def test_ssh_protocol(self):
        assert _repo_name_from_url("ssh://git@github.com/owner/repo.git") == "repo"


# ── plugins_command dispatch ──────────────────────────────────────────────


# ── _read_manifest ────────────────────────────────────────────────────────


class TestReadManifest:
    """Manifest reading edge cases."""

    def test_valid_yaml(self, tmp_path):
        manifest = {"name": "cool-plugin", "version": "1.0.0"}
        (tmp_path / "plugin.yaml").write_text(yaml.dump(manifest))
        result = _read_manifest(tmp_path)
        assert result["name"] == "cool-plugin"
        assert result["version"] == "1.0.0"

    def test_valid_yml_fallback(self, tmp_path):
        (tmp_path / "plugin.yml").write_text("name: yml-plugin\n")

        assert _read_manifest(tmp_path)["name"] == "yml-plugin"

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


# ── cmd_install tests ─────────────────────────────────────────────────────────


class TestCmdInstall:
    """Test the install command."""

    def test_install_requires_identifier(self):
        from hermes_cli.plugins_cmd import cmd_install

        with pytest.raises(SystemExit):
            cmd_install("")

    @patch("hermes_cli.plugins_cmd._resolve_git_url")
    def test_install_validates_identifier(self, mock_resolve):
        from hermes_cli.plugins_cmd import cmd_install

        mock_resolve.side_effect = ValueError("Invalid identifier")

        with pytest.raises(SystemExit) as exc_info:
            cmd_install("invalid")
        assert exc_info.value.code == 1

    @patch("hermes_cli.plugins_cmd._display_after_install")
    @patch("hermes_cli.plugins_cmd.shutil.move")
    @patch("hermes_cli.plugins_cmd.shutil.rmtree")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._clone_plugin_source")
    def test_install_rejects_manifest_name_pointing_at_plugins_root(
        self,
        mock_clone_source,
        mock_plugins_dir,
        mock_rmtree,
        mock_move,
        mock_display_after_install,
        tmp_path,
    ):
        from hermes_cli.plugins_cmd import cmd_install

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        mock_plugins_dir.return_value = plugins_dir
        mock_clone_source.return_value = (
            tmp_path / "source",
            {"name": "."},
            ".",
            None,
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_install("owner/repo", force=True)

        assert exc_info.value.code == 1
        assert plugins_dir not in [call.args[0] for call in mock_rmtree.call_args_list]
        mock_move.assert_not_called()
        mock_display_after_install.assert_not_called()


# ── cmd_update tests ─────────────────────────────────────────────────────────


class TestCmdUpdate:
    """Test the update command."""

    @patch("hermes_cli.plugins_cmd._git_pull_plugin_dir")
    def test_root_checkout_ignores_source_metadata_name_collision(
        self, mock_git_pull, tmp_path
    ):
        from hermes_cli import plugins_cmd as pc

        target = tmp_path / "plugin"
        target.mkdir()
        (target / ".git").mkdir()
        (target / pc._SOURCE_METADATA_FILENAME).write_text("not installer metadata")
        mock_git_pull.return_value = (True, "Already up to date.")

        assert pc._update_plugin_dir(target) == (
            True,
            "Already up to date.",
            True,
        )

    @patch("hermes_cli.plugins_cmd._update_plugin_dir")
    @patch("hermes_cli.plugins_cmd._sanitize_plugin_name")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_update_git_pull_success(
        self,
        mock_plugins_dir,
        mock_sanitize,
        mock_update,
    ):
        from hermes_cli.plugins_cmd import cmd_update

        mock_plugins_dir_val = MagicMock()
        mock_plugins_dir.return_value = mock_plugins_dir_val
        mock_target = MagicMock()
        mock_target.exists.return_value = True
        mock_sanitize.return_value = mock_target
        mock_update.return_value = (True, "Updated", False)

        cmd_update("test-plugin")

        mock_update.assert_called_once_with(mock_target)

    @patch("hermes_cli.plugins_cmd._sanitize_plugin_name")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_update_plugin_not_found(self, mock_plugins_dir, mock_sanitize):
        from hermes_cli.plugins_cmd import cmd_update

        mock_plugins_dir_val = MagicMock()
        mock_plugins_dir_val.iterdir.return_value = []
        mock_plugins_dir.return_value = mock_plugins_dir_val
        mock_target = MagicMock()
        mock_target.exists.return_value = False
        mock_sanitize.return_value = mock_target

        with pytest.raises(SystemExit) as exc_info:
            cmd_update("nonexistent-plugin")

        assert exc_info.value.code == 1


# ── cmd_remove tests ─────────────────────────────────────────────────────────


class TestCmdRemove:
    """Test the remove command."""

    @patch("hermes_cli.plugins_cmd._sanitize_plugin_name")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd.shutil.rmtree")
    def test_remove_deletes_plugin(self, mock_rmtree, mock_plugins_dir, mock_sanitize):
        from hermes_cli.plugins_cmd import cmd_remove

        mock_plugins_dir.return_value = MagicMock()
        mock_target = MagicMock()
        mock_target.exists.return_value = True
        mock_sanitize.return_value = mock_target

        cmd_remove("test-plugin")

        mock_rmtree.assert_called_once_with(mock_target)

    @patch("hermes_cli.plugins_cmd._sanitize_plugin_name")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_remove_plugin_not_found(self, mock_plugins_dir, mock_sanitize):
        from hermes_cli.plugins_cmd import cmd_remove

        mock_plugins_dir_val = MagicMock()
        mock_plugins_dir_val.iterdir.return_value = []
        mock_plugins_dir.return_value = mock_plugins_dir_val
        mock_target = MagicMock()
        mock_target.exists.return_value = False
        mock_sanitize.return_value = mock_target

        with pytest.raises(SystemExit) as exc_info:
            cmd_remove("nonexistent-plugin")

        assert exc_info.value.code == 1


# ── cmd_list tests ─────────────────────────────────────────────────────────


class TestCmdList:
    """Test the list command."""

    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_list_empty_plugins_dir(self, mock_plugins_dir):
        from hermes_cli.plugins_cmd import cmd_list

        mock_plugins_dir_val = MagicMock()
        mock_plugins_dir_val.iterdir.return_value = []
        mock_plugins_dir.return_value = mock_plugins_dir_val

        cmd_list()

    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._read_manifest")
    def test_list_with_plugins(self, mock_read_manifest, mock_plugins_dir):
        from hermes_cli.plugins_cmd import cmd_list

        mock_plugins_dir_val = MagicMock()
        mock_plugin_dir = MagicMock()
        mock_plugin_dir.name = "test-plugin"
        mock_plugin_dir.is_dir.return_value = True
        mock_plugin_dir.__truediv__ = lambda self, x: MagicMock(
            exists=MagicMock(return_value=False)
        )
        mock_plugins_dir_val.iterdir.return_value = [mock_plugin_dir]
        mock_plugins_dir.return_value = mock_plugins_dir_val
        mock_read_manifest.return_value = {"name": "test-plugin", "version": "1.0.0"}

        cmd_list()


# ── _copy_example_files tests ─────────────────────────────────────────────────


class TestCopyExampleFiles:
    """Test example file copying."""

    def test_copies_example_files(self, tmp_path):
        from unittest.mock import MagicMock

        console = MagicMock()

        # Create example file
        example_file = tmp_path / "config.yaml.example"
        example_file.write_text("key: value")

        _copy_example_files(tmp_path, console)

        # Should have created the file
        assert (tmp_path / "config.yaml").exists()
        console.print.assert_called()

    def test_skips_existing_files(self, tmp_path):
        from unittest.mock import MagicMock

        console = MagicMock()

        # Create both example and real file
        example_file = tmp_path / "config.yaml.example"
        example_file.write_text("key: value")
        real_file = tmp_path / "config.yaml"
        real_file.write_text("existing: true")

        _copy_example_files(tmp_path, console)

        # Should NOT have overwritten
        assert real_file.read_text() == "existing: true"

    def test_handles_copy_error_gracefully(self, tmp_path):
        from unittest.mock import MagicMock, patch

        console = MagicMock()

        # Create example file
        example_file = tmp_path / "config.yaml.example"
        example_file.write_text("key: value")

        # Mock shutil.copy2 to raise an error
        with patch(
            "hermes_cli.plugins_cmd.shutil.copy2",
            side_effect=OSError("Permission denied"),
        ):
            # Should not raise, just warn
            _copy_example_files(tmp_path, console)

        # Should have printed a warning
        assert any("Warning" in str(c) for c in console.print.call_args_list)


class TestPromptPluginEnvVars:
    """Tests for _prompt_plugin_env_vars."""

    def test_skips_when_no_requires_env(self):
        from hermes_cli.plugins_cmd import _prompt_plugin_env_vars
        from unittest.mock import MagicMock

        console = MagicMock()
        _prompt_plugin_env_vars({}, console)
        console.print.assert_not_called()

    def test_skips_already_set_vars(self, monkeypatch):
        from hermes_cli.plugins_cmd import _prompt_plugin_env_vars
        from unittest.mock import MagicMock, patch

        console = MagicMock()
        with patch("hermes_cli.config.get_env_value", return_value="already-set"):
            _prompt_plugin_env_vars({"requires_env": ["MY_KEY"]}, console)
        # No prompt should appear — all vars are set
        console.print.assert_not_called()

    def test_prompts_for_missing_var_simple_format(self):
        from hermes_cli.plugins_cmd import _prompt_plugin_env_vars
        from unittest.mock import MagicMock, patch

        console = MagicMock()
        manifest = {
            "name": "test_plugin",
            "requires_env": ["MY_API_KEY"],
        }

        with patch("hermes_cli.config.get_env_value", return_value=None), \
             patch("builtins.input", return_value="sk-test-123"), \
             patch("hermes_cli.config.save_env_value") as mock_save:
            _prompt_plugin_env_vars(manifest, console)

        mock_save.assert_called_once_with("MY_API_KEY", "sk-test-123")

    def test_prompts_for_missing_var_rich_format(self):
        from hermes_cli.plugins_cmd import _prompt_plugin_env_vars
        from unittest.mock import MagicMock, patch

        console = MagicMock()
        manifest = {
            "name": "langfuse_tracing",
            "requires_env": [
                {
                    "name": "LANGFUSE_PUBLIC_KEY",
                    "description": "Public key",
                    "url": "https://langfuse.com",
                    "secret": False,
                },
            ],
        }

        with patch("hermes_cli.config.get_env_value", return_value=None), \
             patch("builtins.input", return_value="pk-lf-123"), \
             patch("hermes_cli.config.save_env_value") as mock_save:
            _prompt_plugin_env_vars(manifest, console)

        mock_save.assert_called_once_with("LANGFUSE_PUBLIC_KEY", "pk-lf-123")
        # Should show url hint
        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "langfuse.com" in printed

    def test_secret_uses_masked_prompt(self):
        from hermes_cli.plugins_cmd import _prompt_plugin_env_vars
        from unittest.mock import MagicMock, patch

        console = MagicMock()
        manifest = {
            "name": "test",
            "requires_env": [{"name": "SECRET_KEY", "secret": True}],
        }

        with patch("hermes_cli.config.get_env_value", return_value=None), \
             patch("hermes_cli.plugins_cmd.masked_secret_prompt", return_value="s3cret") as mock_prompt, \
             patch("hermes_cli.config.save_env_value"):
            _prompt_plugin_env_vars(manifest, console)

        mock_prompt.assert_called_once()

    def test_empty_input_skips(self):
        from hermes_cli.plugins_cmd import _prompt_plugin_env_vars
        from unittest.mock import MagicMock, patch

        console = MagicMock()
        manifest = {"name": "test", "requires_env": ["OPTIONAL_VAR"]}

        with patch("hermes_cli.config.get_env_value", return_value=None), \
             patch("builtins.input", return_value=""), \
             patch("hermes_cli.config.save_env_value") as mock_save:
            _prompt_plugin_env_vars(manifest, console)

        mock_save.assert_not_called()

    def test_keyboard_interrupt_skips_gracefully(self):
        from hermes_cli.plugins_cmd import _prompt_plugin_env_vars
        from unittest.mock import MagicMock, patch

        console = MagicMock()
        manifest = {"name": "test", "requires_env": ["KEY1", "KEY2"]}

        with patch("hermes_cli.config.get_env_value", return_value=None), \
             patch("builtins.input", side_effect=KeyboardInterrupt), \
             patch("hermes_cli.config.save_env_value") as mock_save:
            _prompt_plugin_env_vars(manifest, console)

        # Should not crash, and not save anything
        mock_save.assert_not_called()


# ── curses_radiolist ─────────────────────────────────────────────────────


class TestCursesRadiolist:
    """Test the curses_radiolist function."""

    def test_non_tty_returns_default(self):
        from hermes_cli.curses_ui import curses_radiolist
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = curses_radiolist("Pick one", ["a", "b", "c"], selected=1)
            assert result == 1

    def test_non_tty_returns_cancel_value(self):
        from hermes_cli.curses_ui import curses_radiolist
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = curses_radiolist("Pick", ["x", "y"], selected=0, cancel_returns=1)
            assert result == 1

    def test_keyboard_interrupt_returns_cancel_value(self):
        from hermes_cli.curses_ui import curses_radiolist

        with patch("sys.stdin") as mock_stdin, patch("curses.wrapper", side_effect=KeyboardInterrupt):
            mock_stdin.isatty.return_value = True
            result = curses_radiolist("Pick", ["x", "y"], selected=0, cancel_returns=-1)
            assert result == -1


# ── Provider discovery helpers ───────────────────────────────────────────


class TestProviderDiscovery:
    """Test provider plugin discovery and config helpers."""

    def test_get_current_memory_provider_default(self, tmp_path, monkeypatch):
        """Empty config returns empty string."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_file = tmp_path / "config.yaml"
        config_file.write_text("memory:\n  provider: ''\n")
        from hermes_cli.plugins_cmd import _get_current_memory_provider
        result = _get_current_memory_provider()
        assert result == ""

    def test_get_current_context_engine_default(self, tmp_path, monkeypatch):
        """Default config returns 'compressor'."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_file = tmp_path / "config.yaml"
        config_file.write_text("context:\n  engine: compressor\n")
        from hermes_cli.plugins_cmd import _get_current_context_engine
        result = _get_current_context_engine()
        assert result == "compressor"

    def test_save_memory_provider(self, tmp_path, monkeypatch):
        """Saving a memory provider persists to config.yaml."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_file = tmp_path / "config.yaml"
        config_file.write_text("memory:\n  provider: ''\n")
        from hermes_cli.plugins_cmd import _save_memory_provider
        _save_memory_provider("honcho")
        content = yaml.safe_load(config_file.read_text())
        assert content["memory"]["provider"] == "honcho"

    def test_save_context_engine(self, tmp_path, monkeypatch):
        """Saving a context engine persists to config.yaml."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_file = tmp_path / "config.yaml"
        config_file.write_text("context:\n  engine: compressor\n")
        from hermes_cli.plugins_cmd import _save_context_engine
        _save_context_engine("lcm")
        content = yaml.safe_load(config_file.read_text())
        assert content["context"]["engine"] == "lcm"

    def test_discover_memory_providers_empty(self):
        """Discovery returns empty list when import fails."""
        with patch("plugins.memory.discover_memory_providers",
                    side_effect=ImportError("no module")):
            from hermes_cli.plugins_cmd import _discover_memory_providers
            result = _discover_memory_providers()
            assert result == []

    def test_discover_context_engines_empty(self):
        """Discovery returns empty list when import fails."""
        with patch("plugins.context_engine.discover_context_engines",
                    side_effect=ImportError("no module")):
            from hermes_cli.plugins_cmd import _discover_context_engines
            result = _discover_context_engines()
            assert result == []


# ── Auto-activation fix ──────────────────────────────────────────────────


class TestNoAutoActivation:
    """Verify that plugin engines don't auto-activate when config says 'compressor'."""

    def test_compressor_default_ignores_plugin(self):
        """When context.engine is 'compressor', a plugin-registered engine should NOT
        be used — only explicit config triggers plugin engines."""
        # This tests the run_agent.py logic indirectly by checking that the
        # code path for default config doesn't call get_plugin_context_engine.
        import run_agent as ra_module
        source = open(ra_module.__file__).read()
        # The old code had: "Even with default config, check if a plugin registered one"
        # The fix removes this. Verify it's gone.
        assert "Even with default config, check if a plugin registered one" not in source


# ── End-to-end subdirectory install ──────────────────────────────────────────


class TestSubdirInstallE2E:
    """Install a plugin that lives in a subdirectory of a real local git repo."""

    @staticmethod
    def _git_env() -> dict[str, str]:
        return {
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        }

    @classmethod
    def _commit(cls, repo_root: Path, message: str) -> None:
        import subprocess as sp

        env = cls._git_env()
        sp.run(["git", "add", "-A"], cwd=repo_root, check=True, env=env)
        sp.run(
            ["git", "commit", "-q", "-m", message],
            cwd=repo_root,
            check=True,
            env=env,
        )

    @staticmethod
    def _make_repo_with_subdir_plugin(repo_root: Path) -> None:
        """Create a git repo where the plugin lives in ``./my-plugin/`` and the
        repo root holds unrelated docs/tests."""
        import subprocess as sp

        repo_root.mkdir(parents=True, exist_ok=True)
        # Root-level noise: docs + tests that should NOT be installed.
        (repo_root / "README.md").write_text("# Monorepo docs\n")
        (repo_root / "tests").mkdir()
        (repo_root / "tests" / "test_x.py").write_text("def test_x():\n    pass\n")
        # The actual plugin in a subdirectory.
        plugin_dir = repo_root / "my-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: my-plugin\nmanifest_version: 1\ndescription: A subdir plugin\n"
        )
        (plugin_dir / "__init__.py").write_text("# plugin entry\n")
        (plugin_dir / "obsolete.py").write_text("OLD = True\n")
        (plugin_dir / "config.yaml.example").write_text("user-owned: false\n")

        env = TestSubdirInstallE2E._git_env()
        sp.run(["git", "init", "-q"], cwd=repo_root, check=True, env=env)
        TestSubdirInstallE2E._commit(repo_root, "init")

    def test_installs_only_the_subdir_plugin(self, tmp_path, monkeypatch):
        if shutil.which("git") is None:
            pytest.skip("git not available")

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "monorepo"
        self._make_repo_with_subdir_plugin(repo_root)

        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        identifier = f"file://{repo_root}#my-plugin"
        target, manifest, name = pc._install_plugin_core(identifier, force=False)

        # Installed under the plugin's own name, not the repo name.
        assert name == "my-plugin"
        assert manifest.get("name") == "my-plugin"
        assert target == (plugins_dir / "my-plugin").resolve()

        # The plugin's files are present...
        assert (target / "plugin.yaml").exists()
        assert (target / "__init__.py").exists()
        # ...and the repo-root noise is NOT.
        assert not (target / "README.md").exists()
        assert not (target / "tests").exists()

    def test_missing_subdir_raises(self, tmp_path, monkeypatch):
        if shutil.which("git") is None:
            pytest.skip("git not available")

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "monorepo"
        self._make_repo_with_subdir_plugin(repo_root)

        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        identifier = f"file://{repo_root}#does-not-exist"
        with pytest.raises(PluginOperationError, match="does not exist"):
            pc._install_plugin_core(identifier, force=False)

    def test_root_install_rejects_ambiguous_plugins(self, tmp_path, monkeypatch):
        if shutil.which("git") is None:
            pytest.skip("git not available")

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "monorepo"
        self._make_repo_with_subdir_plugin(repo_root)
        second = repo_root / "second-plugin"
        second.mkdir()
        (second / "plugin.yaml").write_text("name: second-plugin\n")
        (second / "__init__.py").write_text("")
        self._commit(repo_root, "add second plugin")
        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        with pytest.raises(PluginOperationError, match="multiple loadable plugins"):
            pc._install_plugin_core(f"file://{repo_root}", force=False)

        assert list(plugins_dir.iterdir()) == []

    def test_root_install_rejects_repo_without_loadable_plugin(
        self, tmp_path, monkeypatch
    ):
        if shutil.which("git") is None:
            pytest.skip("git not available")

        import subprocess as sp

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "not-a-plugin"
        repo_root.mkdir()
        (repo_root / "README.md").write_text("not a plugin\n")
        sp.run(["git", "init", "-q"], cwd=repo_root, check=True)
        self._commit(repo_root, "init")
        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        with pytest.raises(PluginOperationError, match="repository root must contain"):
            pc._install_plugin_core(f"file://{repo_root}", force=False)

        assert list(plugins_dir.iterdir()) == []

    def test_root_install_autodetects_and_updates_sole_plugin(
        self, tmp_path, monkeypatch
    ):
        if shutil.which("git") is None:
            pytest.skip("git not available")

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "monorepo"
        self._make_repo_with_subdir_plugin(repo_root)
        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        target, manifest, name = pc._install_plugin_core(
            f"file://{repo_root}", force=False
        )
        metadata = json.loads(
            (target / pc._SOURCE_METADATA_FILENAME).read_text(encoding="utf-8")
        )

        assert name == "my-plugin"
        assert manifest["name"] == "my-plugin"
        assert metadata["subdir"] == "my-plugin"
        assert not (target / "README.md").exists()

        (repo_root / "my-plugin" / "__init__.py").write_text("VERSION = 2\n")
        self._commit(repo_root, "plugin v2")

        result = pc.dashboard_update_user_plugin("my-plugin")

        assert result["ok"] is True
        assert result["unchanged"] is False
        assert (target / "__init__.py").read_text() == "VERSION = 2\n"

    def test_updates_subdir_plugin_without_losing_user_files(
        self, tmp_path, monkeypatch
    ):
        if shutil.which("git") is None:
            pytest.skip("git not available")

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "monorepo"
        self._make_repo_with_subdir_plugin(repo_root)

        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        identifier = f"file://{repo_root}#my-plugin"
        target, _manifest, _name = pc._install_plugin_core(identifier, force=False)
        (target / "config.yaml").write_text("user-owned: true\n")

        (repo_root / "my-plugin" / "__init__.py").write_text("VERSION = 2\n")
        (repo_root / "my-plugin" / "obsolete.py").unlink()
        self._commit(repo_root, "plugin v2")

        result = pc.dashboard_update_user_plugin("my-plugin")

        assert result["ok"] is True
        assert result["unchanged"] is False
        assert (target / "__init__.py").read_text() == "VERSION = 2\n"
        assert not (target / "obsolete.py").exists()
        assert (target / "config.yaml").read_text() == "user-owned: true\n"

    def test_refuses_to_overwrite_local_managed_changes(self, tmp_path, monkeypatch):
        if shutil.which("git") is None:
            pytest.skip("git not available")

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "monorepo"
        self._make_repo_with_subdir_plugin(repo_root)
        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        identifier = f"file://{repo_root}#my-plugin"
        target, _manifest, _name = pc._install_plugin_core(identifier, force=False)
        (target / "__init__.py").write_text("LOCAL = True\n")
        (repo_root / "my-plugin" / "plugin.yaml").write_text(
            "name: my-plugin\nmanifest_version: 1\ndescription: Version two\n"
        )
        self._commit(repo_root, "plugin v2")

        result = pc.dashboard_update_user_plugin("my-plugin")

        assert result["ok"] is False
        assert "local changes in '__init__.py'" in result["error"]
        assert (target / "__init__.py").read_text() == "LOCAL = True\n"

    def test_failed_reclone_leaves_installed_plugin_untouched(
        self, tmp_path, monkeypatch
    ):
        if shutil.which("git") is None:
            pytest.skip("git not available")

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "monorepo"
        self._make_repo_with_subdir_plugin(repo_root)
        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        identifier = f"file://{repo_root}#my-plugin"
        target, _manifest, _name = pc._install_plugin_core(identifier, force=False)
        original_entrypoint = (target / "__init__.py").read_text()
        shutil.rmtree(repo_root)

        result = pc.dashboard_update_user_plugin("my-plugin")

        assert result["ok"] is False
        assert "Git clone failed" in result["error"]
        assert target.is_dir()
        assert (target / "__init__.py").read_text() == original_entrypoint

    def test_activation_failure_restores_installed_plugin(
        self, tmp_path, monkeypatch
    ):
        if shutil.which("git") is None:
            pytest.skip("git not available")

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "monorepo"
        self._make_repo_with_subdir_plugin(repo_root)
        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        identifier = f"file://{repo_root}#my-plugin"
        target, _manifest, _name = pc._install_plugin_core(identifier, force=False)
        original_entrypoint = (target / "__init__.py").read_text()
        (repo_root / "my-plugin" / "__init__.py").write_text("VERSION = 2\n")
        self._commit(repo_root, "plugin v2")

        original_rename = Path.rename

        def fail_replacement_rename(path: Path, destination: Path) -> Path:
            if path.name == "replacement":
                raise OSError("simulated activation failure")
            return original_rename(path, destination)

        monkeypatch.setattr(Path, "rename", fail_replacement_rename)
        result = pc.dashboard_update_user_plugin("my-plugin")

        assert result["ok"] is False
        assert "Could not activate plugin update" in result["error"]
        assert target.is_dir()
        assert (target / "__init__.py").read_text() == original_entrypoint

    def test_rejects_tampered_metadata_path(self, tmp_path, monkeypatch):
        if shutil.which("git") is None:
            pytest.skip("git not available")

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "monorepo"
        self._make_repo_with_subdir_plugin(repo_root)
        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        identifier = f"file://{repo_root}#my-plugin"
        target, _manifest, _name = pc._install_plugin_core(identifier, force=False)
        metadata_path = target / pc._SOURCE_METADATA_FILENAME
        metadata = json.loads(metadata_path.read_text())
        metadata["files"]["../../outside"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))

        result = pc.dashboard_update_user_plugin("my-plugin")

        assert result["ok"] is False
        assert "Invalid managed plugin path '../../outside'" in result["error"]
        assert not (tmp_path / "outside").exists()

    def test_rejects_symlinked_source_metadata(self, tmp_path, monkeypatch):
        if shutil.which("git") is None or os.name == "nt":
            pytest.skip("git and symlink support required")

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "monorepo"
        self._make_repo_with_subdir_plugin(repo_root)
        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        identifier = f"file://{repo_root}#my-plugin"
        target, _manifest, _name = pc._install_plugin_core(identifier, force=False)
        metadata_path = target / pc._SOURCE_METADATA_FILENAME
        external_metadata = tmp_path / "external-metadata.json"
        metadata_path.replace(external_metadata)
        metadata_path.symlink_to(external_metadata)

        result = pc.dashboard_update_user_plugin("my-plugin")

        assert result["ok"] is False
        assert "metadata must not be a symlink" in result["error"]
        assert external_metadata.is_file()
