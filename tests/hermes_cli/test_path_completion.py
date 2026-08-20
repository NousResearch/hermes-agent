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


    def test_directories_have_trailing_slash(self, tmp_path):
        (tmp_path / "mydir").mkdir()
        (tmp_path / "myfile.txt").touch()

        completions = list(SlashCommandCompleter._path_completions(f"{tmp_path}/"))
        names = _display_names(completions)
        metas = _display_metas(completions)
        assert "mydir/" in names
        idx = names.index("mydir/")
        assert metas[idx] == "dir"






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


class TestFileSizeLabel:
    def test_bytes(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("hi")
        assert _file_size_label(str(f)) == "2B"


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
        """Patch subprocess/os helpers so ``bad_path`` triggers the cross-mount branch."""
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
        cwd = str(tmp_path)
        cross_drive_path = "D:\\other\\file.txt" if os.sep != "/" else "/mnt/other/file.txt"

        self._install_cross_mount_fakes(monkeypatch, cross_drive_path)

        completer = SlashCommandCompleter()
        completer._file_cache_cwd = cwd

        files = completer._get_project_files()

        assert cross_drive_path not in files

    def test_unc_device_path_is_skipped(self, monkeypatch, tmp_path):
        cwd = str(tmp_path)
        unc_device_path = "\\\\.\\nul"

        self._install_cross_mount_fakes(monkeypatch, unc_device_path)

        completer = SlashCommandCompleter()
        completer._file_cache_cwd = cwd

        files = completer._get_project_files()

        assert unc_device_path not in files


class TestExplicitAtPathCrossMount:
    """Regression tests for cross-mount handling in the explicit
    ``@file:``/``@folder:`` completion branch.

    A user browsing an absolute path on another Windows drive must not crash
    the prompt_toolkit event loop when ``os.path.relpath`` raises ``ValueError``.
    """

    def test_cross_drive_explicit_file_path_is_skipped(self, monkeypatch, tmp_path):
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

        completions = list(completer._context_completions(word))
        names = _display_names(completions)
        assert entry_name not in names
