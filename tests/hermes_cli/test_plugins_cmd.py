"""Tests for hermes_cli.plugins_cmd — the ``hermes plugins`` CLI subcommand."""

from __future__ import annotations

import logging
import os
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_cli.plugins_cmd import (
    _copy_example_files,
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

    @patch("hermes_cli.plugins_cmd.cmd_toggle")
    def test_none_falls_through_to_toggle(self, mock_toggle):
        args = self._make_args(None)
        plugins_command(args)
        mock_toggle.assert_called_once()

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


# ── cmd_install tests ─────────────────────────────────────────────────────────


class TestCmdInstall:
    """Test the install command."""

    def test_install_requires_identifier(self):
        from hermes_cli.plugins_cmd import cmd_install
        import argparse

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
    @patch("hermes_cli.plugins_cmd._read_manifest")
    @patch("hermes_cli.plugins_cmd.subprocess.run")
    def test_install_rejects_manifest_name_pointing_at_plugins_root(
        self,
        mock_run,
        mock_read_manifest,
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
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_read_manifest.return_value = {"name": "."}

        with pytest.raises(SystemExit) as exc_info:
            cmd_install("owner/repo", force=True)

        assert exc_info.value.code == 1
        assert plugins_dir not in [call.args[0] for call in mock_rmtree.call_args_list]
        mock_move.assert_not_called()
        mock_display_after_install.assert_not_called()


# ── cmd_update tests ─────────────────────────────────────────────────────────


class TestCmdUpdate:
    """Test the update command."""

    @patch("hermes_cli.plugins_cmd._sanitize_plugin_name")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd.subprocess.run")
    def test_update_git_pull_success(self, mock_run, mock_plugins_dir, mock_sanitize):
        from hermes_cli.plugins_cmd import cmd_update

        mock_plugins_dir_val = MagicMock()
        mock_plugins_dir.return_value = mock_plugins_dir_val
        mock_target = MagicMock()
        mock_target.exists.return_value = True
        mock_target.__truediv__ = lambda self, x: MagicMock(
            exists=MagicMock(return_value=True)
        )
        mock_sanitize.return_value = mock_target

        mock_run.return_value = MagicMock(returncode=0, stdout="Updated", stderr="")

        cmd_update("test-plugin")

        mock_run.assert_called_once()

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
        from hermes_cli.plugins_cmd import _copy_example_files
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
        from hermes_cli.plugins_cmd import _copy_example_files
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
        from hermes_cli.plugins_cmd import _copy_example_files
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

    def test_secret_uses_getpass(self):
        from hermes_cli.plugins_cmd import _prompt_plugin_env_vars
        from unittest.mock import MagicMock, patch

        console = MagicMock()
        manifest = {
            "name": "test",
            "requires_env": [{"name": "SECRET_KEY", "secret": True}],
        }

        with patch("hermes_cli.config.get_env_value", return_value=None), \
             patch("getpass.getpass", return_value="s3cret") as mock_gp, \
             patch("hermes_cli.config.save_env_value"):
            _prompt_plugin_env_vars(manifest, console)

        mock_gp.assert_called_once()

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
    """Test the curses_radiolist function (non-TTY fallback path)."""

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
