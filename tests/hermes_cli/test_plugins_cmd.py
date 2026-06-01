"""Tests for hermes_cli.plugins_cmd — the ``hermes plugins`` CLI subcommand."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_cli.plugins_cmd import (
    PluginOperationError,
    _SOURCE_SIDECAR_NAME,
    _copy_example_files,
    _discover_manifests_in_clone,
    _filter_manifests_by_names,
    _install_subplugin_dir,
    _prompt_subset_selection,
    _read_manifest,
    _read_source_sidecar,
    _repo_name_from_url,
    _resolve_git_executable,
    _resolve_git_url,
    _sanitize_plugin_name,
    _write_source_sidecar,
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


# ── multi-plugin repo helpers ───────────────────────────────────────────────


def _make_clone(tmp_path: Path, layout: dict[str, dict]) -> Path:
    """Materialize a fake cloned repo on disk.

    *layout* maps "relative path of plugin dir" → manifest dict. A special
    key "<root-extra>" maps to a dict of additional plain files to drop into
    the clone root (e.g. ``{"after-install.md": "hello"}``).
    """
    clone = tmp_path / "clone"
    clone.mkdir()
    extras = layout.pop("<root-extra>", {})
    for rel, manifest in layout.items():
        plugin_dir = clone if rel == "." else clone / rel
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
        (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
    for name, content in extras.items():
        (clone / name).write_text(content, encoding="utf-8")
    return clone


class TestDiscoverManifestsInClone:
    """`_discover_manifests_in_clone` resolves the four layout cases the installer must handle."""

    def test_root_manifest_only_returns_single(self, tmp_path):
        clone = _make_clone(tmp_path, {".": {"name": "solo", "version": "1.0"}})
        is_root, manifests = _discover_manifests_in_clone(clone)
        assert is_root is True
        assert len(manifests) == 1
        assert manifests[0][0] == clone
        assert manifests[0][1]["name"] == "solo"

    def test_nested_manifests_under_plugins_dir(self, tmp_path):
        clone = _make_clone(
            tmp_path,
            {
                "plugins/alpha": {"name": "alpha"},
                "plugins/beta": {"name": "beta"},
            },
        )
        is_root, manifests = _discover_manifests_in_clone(clone)
        assert is_root is False
        assert [m["name"] for _, m in manifests] == ["alpha", "beta"]
        assert {p.name for p, _ in manifests} == {"alpha", "beta"}

    def test_direct_child_manifests_also_discovered(self, tmp_path):
        clone = _make_clone(
            tmp_path,
            {
                "alpha": {"name": "alpha"},
                "beta": {"name": "beta"},
            },
        )
        is_root, manifests = _discover_manifests_in_clone(clone)
        assert is_root is False
        assert [m["name"] for _, m in manifests] == ["alpha", "beta"]

    def test_root_manifest_wins_over_nested(self, tmp_path):
        clone = _make_clone(
            tmp_path,
            {
                ".": {"name": "the-root-one"},
                "plugins/ignored": {"name": "ignored"},
            },
        )
        is_root, manifests = _discover_manifests_in_clone(clone)
        assert is_root is True
        assert len(manifests) == 1
        assert manifests[0][1]["name"] == "the-root-one"

    def test_empty_repo_returns_no_manifests(self, tmp_path):
        clone = tmp_path / "clone"
        clone.mkdir()
        is_root, manifests = _discover_manifests_in_clone(clone)
        assert is_root is False
        assert manifests == []

    def test_dedup_when_same_dir_appears_via_both_scan_roots(self, tmp_path):
        # A plugin only in plugins/<name>/ should not also appear when scanning root.
        clone = _make_clone(tmp_path, {"plugins/only-here": {"name": "only-here"}})
        is_root, manifests = _discover_manifests_in_clone(clone)
        assert is_root is False
        assert len(manifests) == 1
        assert manifests[0][1]["name"] == "only-here"


class TestPromptSubsetSelection:
    """`_prompt_subset_selection` covers the three interactive paths + the TTY-less default."""

    def _mk_manifests(self, n=3):
        return [(Path(f"/tmp/p{i}"), {"name": f"plug{i}"}) for i in range(n)]

    def test_non_interactive_selects_all(self):
        manifests = self._mk_manifests()
        console = MagicMock()
        with patch("hermes_cli.plugins_cmd.sys.stdin") as stdin, \
             patch("hermes_cli.plugins_cmd.sys.stdout") as stdout:
            stdin.isatty.return_value = False
            stdout.isatty.return_value = True
            assert _prompt_subset_selection(manifests, console) == [0, 1, 2]

    def test_install_all_on_yes(self):
        manifests = self._mk_manifests()
        console = MagicMock()
        with patch("hermes_cli.plugins_cmd.sys.stdin") as stdin, \
             patch("hermes_cli.plugins_cmd.sys.stdout") as stdout, \
             patch("builtins.input", return_value="y"):
            stdin.isatty.return_value = True
            stdout.isatty.return_value = True
            assert _prompt_subset_selection(manifests, console) == [0, 1, 2]

    def test_install_all_on_empty_input(self):
        manifests = self._mk_manifests()
        console = MagicMock()
        with patch("hermes_cli.plugins_cmd.sys.stdin") as stdin, \
             patch("hermes_cli.plugins_cmd.sys.stdout") as stdout, \
             patch("builtins.input", return_value=""):
            stdin.isatty.return_value = True
            stdout.isatty.return_value = True
            assert _prompt_subset_selection(manifests, console) == [0, 1, 2]

    def test_select_subset_on_no(self):
        manifests = self._mk_manifests()
        console = MagicMock()
        inputs = iter(["n", "1, 3"])
        with patch("hermes_cli.plugins_cmd.sys.stdin") as stdin, \
             patch("hermes_cli.plugins_cmd.sys.stdout") as stdout, \
             patch("builtins.input", side_effect=lambda *_: next(inputs)):
            stdin.isatty.return_value = True
            stdout.isatty.return_value = True
            assert _prompt_subset_selection(manifests, console) == [0, 2]

    def test_out_of_range_re_prompts(self):
        manifests = self._mk_manifests(2)
        console = MagicMock()
        inputs = iter(["n", "5", "1"])
        with patch("hermes_cli.plugins_cmd.sys.stdin") as stdin, \
             patch("hermes_cli.plugins_cmd.sys.stdout") as stdout, \
             patch("builtins.input", side_effect=lambda *_: next(inputs)):
            stdin.isatty.return_value = True
            stdout.isatty.return_value = True
            assert _prompt_subset_selection(manifests, console) == [0]

    def test_ctrl_c_returns_empty(self):
        manifests = self._mk_manifests(2)
        console = MagicMock()
        with patch("hermes_cli.plugins_cmd.sys.stdin") as stdin, \
             patch("hermes_cli.plugins_cmd.sys.stdout") as stdout, \
             patch("builtins.input", side_effect=KeyboardInterrupt):
            stdin.isatty.return_value = True
            stdout.isatty.return_value = True
            assert _prompt_subset_selection(manifests, console) == []


class TestFilterManifestsByNames:
    """`_filter_manifests_by_names` filters and rejects unknown names."""

    def test_filters_to_subset(self):
        manifests = [
            (Path("/a"), {"name": "a"}),
            (Path("/b"), {"name": "b"}),
            (Path("/c"), {"name": "c"}),
        ]
        result = _filter_manifests_by_names(manifests, ["a", "c"])
        assert [m["name"] for _, m in result] == ["a", "c"]

    def test_unknown_name_raises(self):
        manifests = [(Path("/a"), {"name": "a"})]
        with pytest.raises(PluginOperationError, match="not found in repo"):
            _filter_manifests_by_names(manifests, ["nope"])


class TestSourceSidecar:
    """`_write_source_sidecar` and `_read_source_sidecar` round-trip provenance."""

    def test_roundtrip(self, tmp_path):
        _write_source_sidecar(tmp_path, "https://example.com/r.git", "plugins/foo")
        data = _read_source_sidecar(tmp_path)
        assert data["source_repo"] == "https://example.com/r.git"
        assert data["sub_path"] == "plugins/foo"
        assert "installed_at" in data

    def test_missing_sidecar_returns_none(self, tmp_path):
        assert _read_source_sidecar(tmp_path) is None

    def test_malformed_sidecar_returns_none(self, tmp_path, caplog):
        (tmp_path / _SOURCE_SIDECAR_NAME).write_text("{ not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins_cmd"):
            assert _read_source_sidecar(tmp_path) is None


class TestInstallSubpluginDir:
    """`_install_subplugin_dir` moves a sub-plugin out + writes the sidecar."""

    def test_moves_subdir_and_writes_sidecar(self, tmp_path):
        clone = _make_clone(
            tmp_path,
            {"plugins/foo": {"name": "foo", "version": "1.0"}},
        )
        plugins_dir = tmp_path / "user-plugins"
        plugins_dir.mkdir()

        src_subdir = clone / "plugins" / "foo"
        target, manifest, name = _install_subplugin_dir(
            src_subdir,
            {"name": "foo", "version": "1.0"},
            plugins_dir,
            force=False,
            source_repo="https://example.com/r.git",
            sub_path="plugins/foo",
        )

        assert target == plugins_dir / "foo"
        assert target.is_dir()
        assert (target / "plugin.yaml").exists()
        assert (target / _SOURCE_SIDECAR_NAME).exists()
        sidecar = _read_source_sidecar(target)
        assert sidecar["source_repo"] == "https://example.com/r.git"
        assert sidecar["sub_path"] == "plugins/foo"
        assert manifest["name"] == "foo"
        assert name == "foo"
        # Source subdir was moved out of the clone.
        assert not src_subdir.exists()

    def test_exists_without_force_raises(self, tmp_path):
        clone = _make_clone(tmp_path, {"plugins/foo": {"name": "foo"}})
        plugins_dir = tmp_path / "user-plugins"
        (plugins_dir / "foo").mkdir(parents=True)

        with pytest.raises(PluginOperationError, match="already exists"):
            _install_subplugin_dir(
                clone / "plugins" / "foo",
                {"name": "foo"},
                plugins_dir,
                force=False,
                source_repo="x",
                sub_path="plugins/foo",
            )

    def test_exists_with_force_replaces(self, tmp_path):
        clone = _make_clone(tmp_path, {"plugins/foo": {"name": "foo", "version": "2.0"}})
        plugins_dir = tmp_path / "user-plugins"
        (plugins_dir / "foo").mkdir(parents=True)
        (plugins_dir / "foo" / "stale.txt").write_text("old", encoding="utf-8")

        target, _, _ = _install_subplugin_dir(
            clone / "plugins" / "foo",
            {"name": "foo", "version": "2.0"},
            plugins_dir,
            force=True,
            source_repo="x",
            sub_path="plugins/foo",
        )
        assert not (target / "stale.txt").exists()
        assert (target / "plugin.yaml").exists()

    def test_unsupported_manifest_version_raises(self, tmp_path):
        clone = _make_clone(
            tmp_path,
            {"plugins/foo": {"name": "foo", "manifest_version": 999}},
        )
        plugins_dir = tmp_path / "user-plugins"
        plugins_dir.mkdir()

        with pytest.raises(PluginOperationError, match="manifest_version 999"):
            _install_subplugin_dir(
                clone / "plugins" / "foo",
                {"name": "foo", "manifest_version": 999},
                plugins_dir,
                force=False,
                source_repo="x",
                sub_path="plugins/foo",
            )


class TestCmdInstallMultiPlugin:
    """End-to-end install of a multi-plugin repo using a `file://` clone source."""

    def _setup_source_repo(self, tmp_path: Path, layout: dict[str, dict]) -> str:
        """Create a real local git repo with the given layout and return file:// URL."""
        import subprocess as sp
        src = tmp_path / "src"
        src.mkdir()
        # Build the layout in src/
        for rel, manifest in layout.items():
            plugin_dir = src if rel == "." else src / rel
            plugin_dir.mkdir(parents=True, exist_ok=True)
            (plugin_dir / "plugin.yaml").write_text(
                yaml.dump(manifest), encoding="utf-8"
            )
            (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
        # git init / commit — required so `git clone --depth 1 file://...` works.
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        sp.run(["git", "init", "-q", "-b", "main"], cwd=src, check=True, env=env)
        sp.run(["git", "add", "-A"], cwd=src, check=True, env=env)
        sp.run(["git", "commit", "-q", "-m", "init"], cwd=src, check=True, env=env)
        return "file:///" + str(src).replace("\\", "/").lstrip("/")

    def _patch_plugins_dir(self, plugins_dir: Path):
        return patch("hermes_cli.plugins_cmd._plugins_dir", return_value=plugins_dir)

    def _patch_enable_state(self):
        # Don't actually touch ~/.hermes/config.yaml during tests.
        return (
            patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()),
            patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()),
            patch("hermes_cli.plugins_cmd._save_enabled_set"),
            patch("hermes_cli.plugins_cmd._save_disabled_set"),
        )

    def test_installs_all_sub_plugins_non_interactive(self, tmp_path):
        if not _resolve_git_executable():
            pytest.skip("git not available")
        url = self._setup_source_repo(
            tmp_path,
            {
                "plugins/alpha": {"name": "alpha", "version": "1.0"},
                "plugins/beta": {"name": "beta", "version": "1.0"},
                "<root-extra-after-install>": {},  # placeholder, won't be touched
            } if False else {
                "plugins/alpha": {"name": "alpha", "version": "1.0"},
                "plugins/beta": {"name": "beta", "version": "1.0"},
            },
        )
        plugins_dir = tmp_path / "user-plugins"
        plugins_dir.mkdir()

        from hermes_cli.plugins_cmd import cmd_install

        enabled_set: set = set()
        disabled_set: set = set()

        with self._patch_plugins_dir(plugins_dir), \
             patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=enabled_set), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=disabled_set), \
             patch("hermes_cli.plugins_cmd._save_enabled_set") as save_en, \
             patch("hermes_cli.plugins_cmd._save_disabled_set"), \
             patch("hermes_cli.plugins_cmd.sys.stdin") as stdin, \
             patch("hermes_cli.plugins_cmd.sys.stdout") as stdout, \
             patch("hermes_cli.plugins_cmd._prompt_plugin_env_vars"):
            stdin.isatty.return_value = False
            stdout.isatty.return_value = False
            cmd_install(url, force=False, enable=True)

        assert (plugins_dir / "alpha" / "plugin.yaml").exists()
        assert (plugins_dir / "beta" / "plugin.yaml").exists()
        assert (plugins_dir / "alpha" / _SOURCE_SIDECAR_NAME).exists()
        assert (plugins_dir / "beta" / _SOURCE_SIDECAR_NAME).exists()
        sa = _read_source_sidecar(plugins_dir / "alpha")
        assert sa["sub_path"] == "plugins/alpha"
        save_en.assert_called_once()
        saved = save_en.call_args[0][0]
        assert {"alpha", "beta"}.issubset(saved)

    def test_select_filters_sub_plugins(self, tmp_path):
        if not _resolve_git_executable():
            pytest.skip("git not available")
        url = self._setup_source_repo(
            tmp_path,
            {
                "plugins/alpha": {"name": "alpha"},
                "plugins/beta": {"name": "beta"},
            },
        )
        plugins_dir = tmp_path / "user-plugins"
        plugins_dir.mkdir()
        from hermes_cli.plugins_cmd import cmd_install

        with self._patch_plugins_dir(plugins_dir), \
             patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._save_enabled_set"), \
             patch("hermes_cli.plugins_cmd._save_disabled_set"), \
             patch("hermes_cli.plugins_cmd.sys.stdin") as stdin, \
             patch("hermes_cli.plugins_cmd.sys.stdout") as stdout, \
             patch("hermes_cli.plugins_cmd._prompt_plugin_env_vars"):
            stdin.isatty.return_value = False
            stdout.isatty.return_value = False
            cmd_install(url, enable=False, select=["alpha"])

        assert (plugins_dir / "alpha" / "plugin.yaml").exists()
        assert not (plugins_dir / "beta").exists()

    def test_select_unknown_plugin_errors(self, tmp_path):
        if not _resolve_git_executable():
            pytest.skip("git not available")
        url = self._setup_source_repo(
            tmp_path, {"plugins/alpha": {"name": "alpha"}}
        )
        plugins_dir = tmp_path / "user-plugins"
        plugins_dir.mkdir()
        from hermes_cli.plugins_cmd import cmd_install

        with self._patch_plugins_dir(plugins_dir), \
             patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._save_enabled_set"), \
             patch("hermes_cli.plugins_cmd._save_disabled_set"), \
             patch("hermes_cli.plugins_cmd.sys.stdin") as stdin, \
             patch("hermes_cli.plugins_cmd.sys.stdout") as stdout, \
             patch("hermes_cli.plugins_cmd._prompt_plugin_env_vars"):
            stdin.isatty.return_value = False
            stdout.isatty.return_value = False
            with pytest.raises(SystemExit) as exc:
                cmd_install(url, enable=False, select=["does-not-exist"])
        assert exc.value.code == 1

    def test_no_manifests_errors(self, tmp_path):
        if not _resolve_git_executable():
            pytest.skip("git not available")
        # Empty repo (no plugin.yaml anywhere)
        import subprocess as sp
        src = tmp_path / "src"
        src.mkdir()
        (src / "README.md").write_text("hi", encoding="utf-8")
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        sp.run(["git", "init", "-q", "-b", "main"], cwd=src, check=True, env=env)
        sp.run(["git", "add", "-A"], cwd=src, check=True, env=env)
        sp.run(["git", "commit", "-q", "-m", "init"], cwd=src, check=True, env=env)
        url = "file:///" + str(src).replace("\\", "/").lstrip("/")
        plugins_dir = tmp_path / "user-plugins"
        plugins_dir.mkdir()
        from hermes_cli.plugins_cmd import cmd_install

        with self._patch_plugins_dir(plugins_dir), \
             patch("hermes_cli.plugins_cmd.sys.stdin") as stdin, \
             patch("hermes_cli.plugins_cmd.sys.stdout") as stdout:
            stdin.isatty.return_value = False
            stdout.isatty.return_value = False
            with pytest.raises(SystemExit) as exc:
                cmd_install(url, enable=False)
        assert exc.value.code == 1


class TestCmdInstallSingleRegression:
    """Single-plugin (root manifest) repos behave exactly as before — no sidecar, single after-install."""

    def _setup_source_repo(self, tmp_path: Path, manifest: dict, extras: dict | None = None) -> str:
        import subprocess as sp
        src = tmp_path / "src"
        src.mkdir()
        (src / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
        (src / "__init__.py").write_text("", encoding="utf-8")
        for n, c in (extras or {}).items():
            (src / n).write_text(c, encoding="utf-8")
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        sp.run(["git", "init", "-q", "-b", "main"], cwd=src, check=True, env=env)
        sp.run(["git", "add", "-A"], cwd=src, check=True, env=env)
        sp.run(["git", "commit", "-q", "-m", "init"], cwd=src, check=True, env=env)
        return "file:///" + str(src).replace("\\", "/").lstrip("/")

    def test_single_plugin_unchanged_layout(self, tmp_path):
        if not _resolve_git_executable():
            pytest.skip("git not available")
        url = self._setup_source_repo(tmp_path, {"name": "solo", "version": "1.0"})
        plugins_dir = tmp_path / "user-plugins"
        plugins_dir.mkdir()
        from hermes_cli.plugins_cmd import cmd_install

        with patch("hermes_cli.plugins_cmd._plugins_dir", return_value=plugins_dir), \
             patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._save_enabled_set"), \
             patch("hermes_cli.plugins_cmd._save_disabled_set"), \
             patch("hermes_cli.plugins_cmd.sys.stdin") as stdin, \
             patch("hermes_cli.plugins_cmd.sys.stdout") as stdout, \
             patch("hermes_cli.plugins_cmd._prompt_plugin_env_vars"):
            stdin.isatty.return_value = False
            stdout.isatty.return_value = False
            cmd_install(url, enable=False)

        target = plugins_dir / "solo"
        assert (target / "plugin.yaml").exists()
        # Single-plugin repos must NOT get a sidecar — back-compat.
        assert not (target / _SOURCE_SIDECAR_NAME).exists()
        # The whole clone (including .git) was moved in — cmd_update can still git pull.
        assert (target / ".git").is_dir()


class TestDashboardInstallMultiPlugin:
    """`dashboard_install_plugin` returns the new multi-plugin shape; single shape is unchanged."""

    def _setup_repo(self, tmp_path: Path, layout: dict[str, dict]) -> str:
        import subprocess as sp
        src = tmp_path / "src"
        src.mkdir()
        for rel, manifest in layout.items():
            plugin_dir = src if rel == "." else src / rel
            plugin_dir.mkdir(parents=True, exist_ok=True)
            (plugin_dir / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
            (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        sp.run(["git", "init", "-q", "-b", "main"], cwd=src, check=True, env=env)
        sp.run(["git", "add", "-A"], cwd=src, check=True, env=env)
        sp.run(["git", "commit", "-q", "-m", "init"], cwd=src, check=True, env=env)
        return "file:///" + str(src).replace("\\", "/").lstrip("/")

    def test_multi_plugin_response_shape(self, tmp_path):
        if not _resolve_git_executable():
            pytest.skip("git not available")
        url = self._setup_repo(
            tmp_path,
            {
                "plugins/alpha": {
                    "name": "alpha",
                    "requires_env": ["ALPHA_TOKEN"],
                },
                "plugins/beta": {"name": "beta"},
            },
        )
        plugins_dir = tmp_path / "user-plugins"
        plugins_dir.mkdir()
        from hermes_cli.plugins_cmd import dashboard_install_plugin

        with patch("hermes_cli.plugins_cmd._plugins_dir", return_value=plugins_dir), \
             patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._save_enabled_set"), \
             patch("hermes_cli.plugins_cmd._save_disabled_set"), \
             patch("hermes_cli.config.get_env_value", return_value=None):
            result = dashboard_install_plugin(url, force=False, enable=False)

        assert result["ok"] is True
        assert result["multi"] is True
        names = sorted(p["name"] for p in result["plugins"])
        assert names == ["alpha", "beta"]
        alpha = next(p for p in result["plugins"] if p["name"] == "alpha")
        assert alpha["missing_env"] == ["ALPHA_TOKEN"]
        beta = next(p for p in result["plugins"] if p["name"] == "beta")
        assert beta["missing_env"] == []

    def test_single_plugin_response_shape_unchanged(self, tmp_path):
        if not _resolve_git_executable():
            pytest.skip("git not available")
        url = self._setup_repo(
            tmp_path, {".": {"name": "solo", "requires_env": ["SOLO_KEY"]}},
        )
        plugins_dir = tmp_path / "user-plugins"
        plugins_dir.mkdir()
        from hermes_cli.plugins_cmd import dashboard_install_plugin

        with patch("hermes_cli.plugins_cmd._plugins_dir", return_value=plugins_dir), \
             patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._save_enabled_set"), \
             patch("hermes_cli.plugins_cmd._save_disabled_set"), \
             patch("hermes_cli.config.get_env_value", return_value=None):
            result = dashboard_install_plugin(url, force=False, enable=True)

        assert result["ok"] is True
        assert "multi" not in result
        assert result["plugin_name"] == "solo"
        assert result["missing_env"] == ["SOLO_KEY"]
        assert result["enabled"] is True


# ── cmd_update tests ─────────────────────────────────────────────────────────


class TestCmdUpdate:
    """Test the update command."""

    @patch("hermes_cli.plugins_cmd._read_source_sidecar", return_value=None)
    @patch("hermes_cli.plugins_cmd._sanitize_plugin_name")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd.subprocess.run")
    def test_update_git_pull_success(
        self, mock_run, mock_plugins_dir, mock_sanitize, _mock_sidecar
    ):
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

    def test_update_sub_plugin_re_clones_source(self, tmp_path):
        """When .hermes_source.json is present, cmd_update re-clones the source repo
        and refreshes the sub-plugin subtree (instead of git-pull)."""
        if not _resolve_git_executable():
            pytest.skip("git not available")
        import subprocess as sp
        from hermes_cli.plugins_cmd import cmd_update

        # ── Set up an upstream multi-plugin repo with one sub-plugin ────
        src = tmp_path / "src"
        (src / "plugins" / "foo").mkdir(parents=True)
        (src / "plugins" / "foo" / "plugin.yaml").write_text(
            yaml.dump({"name": "foo", "version": "2.0"}), encoding="utf-8"
        )
        (src / "plugins" / "foo" / "marker.txt").write_text("v2", encoding="utf-8")
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        sp.run(["git", "init", "-q", "-b", "main"], cwd=src, check=True, env=env)
        sp.run(["git", "add", "-A"], cwd=src, check=True, env=env)
        sp.run(["git", "commit", "-q", "-m", "v2"], cwd=src, check=True, env=env)
        url = "file:///" + str(src).replace("\\", "/").lstrip("/")

        # ── Set up a pretend-already-installed sub-plugin with a stale marker ──
        plugins_dir = tmp_path / "user-plugins"
        target = plugins_dir / "foo"
        target.mkdir(parents=True)
        (target / "plugin.yaml").write_text(
            yaml.dump({"name": "foo", "version": "1.0"}), encoding="utf-8"
        )
        (target / "marker.txt").write_text("v1-stale", encoding="utf-8")
        _write_source_sidecar(target, url, "plugins/foo")

        with patch("hermes_cli.plugins_cmd._plugins_dir", return_value=plugins_dir):
            cmd_update("foo")

        assert (target / "marker.txt").read_text(encoding="utf-8") == "v2"
        # Sidecar preserved across refresh.
        sidecar = _read_source_sidecar(target)
        assert sidecar is not None
        assert sidecar["source_repo"] == url
        assert sidecar["sub_path"] == "plugins/foo"


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
