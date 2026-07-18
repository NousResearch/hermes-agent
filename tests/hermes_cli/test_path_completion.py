"""Tests for file path autocomplete in the CLI completer."""

import os
from unittest.mock import MagicMock

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_plain_text

from hermes_cli.commands import SlashCommandCompleter, _file_size_label


def _display_names(completions):
    """Extract plain-text display names from a list of Completion objects."""
    return [to_plain_text(c.display) for c in completions]


def _display_metas(completions):
    """Extract plain-text display_meta from a list of Completion objects."""
    return [to_plain_text(c.display_meta) if c.display_meta else "" for c in completions]


@pytest.fixture
def completer():
    return SlashCommandCompleter()


class TestExtractPathWord:
    def test_relative_path(self):
        assert SlashCommandCompleter._extract_path_word("look at ./src/main.py") == "./src/main.py"

    def test_home_path(self):
        assert SlashCommandCompleter._extract_path_word("edit ~/docs/") == "~/docs/"

    def test_absolute_path(self):
        assert SlashCommandCompleter._extract_path_word("read /etc/hosts") == "/etc/hosts"

    def test_parent_path(self):
        assert SlashCommandCompleter._extract_path_word("check ../config.yaml") == "../config.yaml"

    def test_path_with_slash_in_middle(self):
        assert SlashCommandCompleter._extract_path_word("open src/utils/helpers.py") == "src/utils/helpers.py"

    def test_plain_word_not_path(self):
        assert SlashCommandCompleter._extract_path_word("hello world") is None

    def test_empty_string(self):
        assert SlashCommandCompleter._extract_path_word("") is None

    def test_single_word_no_slash(self):
        assert SlashCommandCompleter._extract_path_word("README.md") is None

    def test_word_after_space(self):
        assert SlashCommandCompleter._extract_path_word("fix the bug in ./tools/") == "./tools/"

    def test_just_dot_slash(self):
        assert SlashCommandCompleter._extract_path_word("./") == "./"

    def test_just_tilde_slash(self):
        assert SlashCommandCompleter._extract_path_word("~/") == "~/"

    def test_url_is_not_treated_as_path(self):
        # A URL contains "/" so the bare slash heuristic would otherwise return
        # it as a path word, firing os.listdir("https:") on every keystroke.
        assert SlashCommandCompleter._extract_path_word("see https://paste.rs/abc") is None

    def test_http_url_is_not_treated_as_path(self):
        assert SlashCommandCompleter._extract_path_word("ref http://example.com/x") is None

    def test_scheme_alone_is_enough_to_reject(self):
        # The "://" scheme separator is the signal, even before any path part
        # has been typed.
        assert SlashCommandCompleter._extract_path_word("ssh://host") is None

    def test_path_word_with_colon_but_no_scheme_still_resolves(self):
        # Only the "://" scheme separator should reject; a bare colon inside a
        # real path token must not regress path detection.
        assert (
            SlashCommandCompleter._extract_path_word("open ./a:b/c.py") == "./a:b/c.py"
        )

    def test_ordinary_path_unaffected_by_url_guard(self):
        assert (
            SlashCommandCompleter._extract_path_word("edit src/pkg/mod.py")
            == "src/pkg/mod.py"
        )


class TestPathCompletions:
    def test_lists_current_directory(self, tmp_path):
        (tmp_path / "file_a.py").touch()
        (tmp_path / "file_b.txt").touch()
        (tmp_path / "subdir").mkdir()

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            completions = list(SlashCommandCompleter._path_completions("./"))
            names = _display_names(completions)
            assert "file_a.py" in names
            assert "file_b.txt" in names
            assert "subdir/" in names
        finally:
            os.chdir(old_cwd)

    def test_filters_by_prefix(self, tmp_path):
        (tmp_path / "alpha.py").touch()
        (tmp_path / "beta.py").touch()
        (tmp_path / "alpha_test.py").touch()

        completions = list(SlashCommandCompleter._path_completions(f"{tmp_path}/alpha"))
        names = _display_names(completions)
        assert "alpha.py" in names
        assert "alpha_test.py" in names
        assert "beta.py" not in names

    def test_directories_have_trailing_slash(self, tmp_path):
        (tmp_path / "mydir").mkdir()
        (tmp_path / "myfile.txt").touch()

        completions = list(SlashCommandCompleter._path_completions(f"{tmp_path}/"))
        names = _display_names(completions)
        metas = _display_metas(completions)
        assert "mydir/" in names
        idx = names.index("mydir/")
        assert metas[idx] == "dir"

    def test_home_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "testfile.md").touch()

        completions = list(SlashCommandCompleter._path_completions("~/test"))
        names = _display_names(completions)
        assert "testfile.md" in names

    def test_nonexistent_dir_returns_empty(self):
        completions = list(SlashCommandCompleter._path_completions("/nonexistent_dir_xyz/"))
        assert completions == []

    def test_respects_limit(self, tmp_path):
        for i in range(50):
            (tmp_path / f"file_{i:03d}.txt").touch()

        completions = list(SlashCommandCompleter._path_completions(f"{tmp_path}/", limit=10))
        assert len(completions) == 10

    def test_case_insensitive_prefix(self, tmp_path):
        (tmp_path / "README.md").touch()

        completions = list(SlashCommandCompleter._path_completions(f"{tmp_path}/read"))
        names = _display_names(completions)
        assert "README.md" in names


class TestIntegration:
    """Test the completer produces path completions via the prompt_toolkit API."""

    def test_slash_commands_still_work(self, completer):
        doc = Document("/hel", cursor_position=4)
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        names = _display_names(completions)
        assert "/help" in names

    def test_path_completion_triggers_on_dot_slash(self, completer, tmp_path):
        (tmp_path / "test.py").touch()
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            doc = Document("edit ./te", cursor_position=9)
            event = MagicMock()
            completions = list(completer.get_completions(doc, event))
            names = _display_names(completions)
            assert "test.py" in names
        finally:
            os.chdir(old_cwd)

    def test_no_completion_for_plain_words(self, completer):
        doc = Document("hello world", cursor_position=11)
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        assert completions == []

    def test_url_does_not_touch_filesystem(self, completer, monkeypatch):
        # Regression for laggy typing: a URL token contains "/", so before the
        # scheme guard it reached _path_completions and called os.listdir on
        # every keystroke. Assert no completions AND that the filesystem is
        # never touched while a URL is under the cursor.
        import hermes_cli.commands as commands_mod

        def _fail(*_args, **_kwargs):
            raise AssertionError("os.listdir must not run for a URL token")

        monkeypatch.setattr(commands_mod.os, "listdir", _fail)

        text = "open https://paste.rs/abc"
        doc = Document(text, cursor_position=len(text))
        event = MagicMock()
        assert list(completer.get_completions(doc, event)) == []

    def test_absolute_path_triggers_completion(self, completer):
        doc = Document("check /etc/hos", cursor_position=14)
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        names = _display_names(completions)
        # /etc/hosts should exist on Linux
        assert any("host" in n.lower() for n in names)


class TestFileSizeLabel:
    def test_bytes(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("hi")
        assert _file_size_label(str(f)) == "2B"

    def test_kilobytes(self, tmp_path):
        f = tmp_path / "medium.txt"
        f.write_bytes(b"x" * 2048)
        assert _file_size_label(str(f)) == "2K"

    def test_megabytes(self, tmp_path):
        f = tmp_path / "large.bin"
        f.write_bytes(b"x" * (2 * 1024 * 1024))
        assert _file_size_label(str(f)) == "2.0M"

    def test_nonexistent(self):
        assert _file_size_label("/nonexistent_xyz") == ""


class TestGetProjectFilesWindowsCrossMount:
    """_get_project_files() should skip paths on a different Windows mount point.

    Regression coverage for #31915 — typing ``@`` to invoke file autocomplete
    on Windows used to crash the prompt_toolkit event loop when ``rg``/``fd``
    returned a path on a different drive or a UNC/device path (e.g.
    ``\\\\.\\nul``), because ``os.path.relpath`` raises ``ValueError`` in
    that case.  These tests simulate the failure on any platform so the
    regression is caught in CI without needing a Windows runner.
    """

    @staticmethod
    def _install_cross_mount_fakes(monkeypatch, bad_path):
        """Patch subprocess/os helpers so ``bad_path`` triggers the cross-mount branch.

        Returns nothing — patches are scoped to the test via ``monkeypatch``.
        """
        import subprocess as _subprocess

        class FakeProc:
            returncode = 0
            stdout = bad_path + "\n"
            stderr = ""

        monkeypatch.setattr(_subprocess, "run", lambda *a, **kw: FakeProc())

        original_relpath = os.path.relpath
        original_isabs = os.path.isabs

        def patched_relpath(path, start=None):
            if path == bad_path:
                raise ValueError("path is on mount 'X:', start on mount 'Y:'")
            return original_relpath(path, start) if start is not None else original_relpath(path)

        monkeypatch.setattr(os.path, "relpath", patched_relpath)
        monkeypatch.setattr(
            os.path, "isabs", lambda p: p == bad_path or original_isabs(p)
        )

    def test_cross_drive_path_is_skipped(self, monkeypatch, tmp_path):
        """Cross-drive path (e.g. ``D:\\file.txt`` when cwd is on ``C:``) is
        skipped without raising, protecting the prompt_toolkit event loop."""
        cwd = str(tmp_path)
        cross_drive_path = "D:\\other\\file.txt" if os.sep != "/" else "/mnt/other/file.txt"

        self._install_cross_mount_fakes(monkeypatch, cross_drive_path)

        completer = SlashCommandCompleter()
        completer._file_cache_cwd = cwd

        files = completer._get_project_files()

        assert cross_drive_path not in files

    def test_unc_device_path_is_skipped(self, monkeypatch, tmp_path):
        """UNC/device paths (``\\\\.\\nul``, ``\\\\?\\C:\\...``) also trigger
        the ``ValueError`` branch and must be skipped without crashing."""
        cwd = str(tmp_path)
        unc_device_path = "\\\\.\\nul"

        self._install_cross_mount_fakes(monkeypatch, unc_device_path)

        completer = SlashCommandCompleter()
        completer._file_cache_cwd = cwd

        files = completer._get_project_files()

        assert unc_device_path not in files

    def test_skip_emits_debug_log(self, monkeypatch, tmp_path, caplog):
        """When a cross-mount path is skipped, a debug log line should be
        emitted so the suppression is observable in future Windows triage."""
        import logging as _logging

        cwd = str(tmp_path)
        cross_drive_path = "D:\\other\\file.txt"

        self._install_cross_mount_fakes(monkeypatch, cross_drive_path)

        completer = SlashCommandCompleter()
        completer._file_cache_cwd = cwd

        with caplog.at_level(_logging.DEBUG, logger="hermes_cli.commands"):
            completer._get_project_files()

        skipped = [
            r for r in caplog.records
            if r.name == "hermes_cli.commands"
            and "Skipping cross-mount path" in r.getMessage()
        ]
        assert skipped, "expected a debug log entry when a cross-mount path is skipped"
        assert cross_drive_path in skipped[0].getMessage()


class TestExplicitAtPathCrossMount:
    """Regression tests for cross-mount handling in the explicit
    ``@file:``/``@folder:`` completion branch (commands.py ~L1623).

    A user browsing an absolute path on another Windows drive must not crash
    the prompt_toolkit event loop when ``os.path.relpath`` raises ``ValueError``.
    """

    def test_cross_drive_explicit_file_path_is_skipped(self, monkeypatch, tmp_path):
        # Create a real entry so listdir/isdir behave, but force relpath to
        # raise for it, emulating a cross-drive absolute path on Windows.
        entry_name = "cross.txt"
        (tmp_path / entry_name).write_text("x")
        search_dir = str(tmp_path)

        original_relpath = os.path.relpath

        def patched_relpath(path, start=None):
            if os.path.basename(str(path)) == entry_name:
                raise ValueError("path is on mount 'X:', start on mount 'Y:'")
            return original_relpath(path, start) if start is not None else original_relpath(path)

        monkeypatch.setattr(os.path, "relpath", patched_relpath)

        completer = SlashCommandCompleter()
        word = f"@file:{search_dir}/"

        # Must not raise, and the un-relative-able entry must be omitted.
        completions = list(completer._context_completions(word))
        names = _display_names(completions)
        assert entry_name not in names
