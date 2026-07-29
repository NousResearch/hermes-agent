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


    def test_directories_have_trailing_slash(self, tmp_path):
        (tmp_path / "mydir").mkdir()
        (tmp_path / "myfile.txt").touch()

        completions = list(SlashCommandCompleter._path_completions(f"{tmp_path}/"))
        names = _display_names(completions)
        metas = _display_metas(completions)
        assert "mydir/" in names
        idx = names.index("mydir/")
        assert metas[idx] == "dir"


    def test_nonexistent_dir_returns_empty(self):
        completions = list(SlashCommandCompleter._path_completions("/nonexistent_dir_xyz/"))
        assert completions == []


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
