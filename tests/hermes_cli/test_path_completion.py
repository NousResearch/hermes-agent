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




    def test_unanchored_trailing_word_returns_none(self):
        assert SlashCommandCompleter._extract_path_word("write hello world") is None


class TestExtractPathWordWithSpaces:
    """A space-crossing span is ambiguous — `./My Documents/` is one path, but
    `compare ./old dir/ with new/` is an anchor plus an independent later token.
    Only the filesystem separates them, so these use real directories."""

    @pytest.fixture
    def in_tmp_cwd(self, tmp_path):
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            yield tmp_path
        finally:
            os.chdir(old_cwd)

    def test_relative_path_with_space(self, in_tmp_cwd):
        (in_tmp_cwd / "My Documents").mkdir()
        assert (
            SlashCommandCompleter._extract_path_word("./My Documents/")
            == "./My Documents/"
        )

    def test_relative_path_with_space_after_prefix(self, in_tmp_cwd):
        (in_tmp_cwd / "My Documents").mkdir()
        assert (
            SlashCommandCompleter._extract_path_word("look at ./My Documents/")
            == "./My Documents/"
        )

    def test_absolute_path_with_space(self, in_tmp_cwd):
        target = in_tmp_cwd / "My Files"
        target.mkdir()
        assert (
            SlashCommandCompleter._extract_path_word(f"cd {target}/") == f"{target}/"
        )

    def test_home_path_with_space(self, in_tmp_cwd, monkeypatch):
        monkeypatch.setenv("HOME", str(in_tmp_cwd))
        (in_tmp_cwd / "My Notes").mkdir()
        assert (
            SlashCommandCompleter._extract_path_word("edit ~/My Notes/todo.md")
            == "~/My Notes/todo.md"
        )

    def test_multiple_spaces_in_anchored_path(self, in_tmp_cwd):
        (in_tmp_cwd / "My Documents" / "Sub Folder").mkdir(parents=True)
        assert (
            SlashCommandCompleter._extract_path_word(
                "open ./My Documents/Sub Folder/file.py"
            )
            == "./My Documents/Sub Folder/file.py"
        )

    def test_partial_leaf_inside_spaced_dir(self, in_tmp_cwd):
        # Mid-typing: the leaf doesn't exist yet, but its parent does.
        (in_tmp_cwd / "My Documents").mkdir()
        assert (
            SlashCommandCompleter._extract_path_word("open ./My Documents/rep")
            == "./My Documents/rep"
        )

    def test_independent_later_token_wins_over_earlier_anchor(self, in_tmp_cwd):
        # Regression for the sweeper's case: an earlier anchor must not swallow
        # a later independent completion token when the span isn't a real path.
        (in_tmp_cwd / "old dir").mkdir()
        assert (
            SlashCommandCompleter._extract_path_word("compare ./old dir/ with new/")
            == "new/"
        )

    def test_span_wins_when_the_spaced_directory_really_exists(self, in_tmp_cwd):
        # ...but if a directory genuinely IS named "old dir/ with new", the
        # full span is the right answer — hence the on-disk probe.
        (in_tmp_cwd / "old dir" / " with new").mkdir(parents=True)
        assert (
            SlashCommandCompleter._extract_path_word("compare ./old dir/ with new/")
            == "./old dir/ with new/"
        )

    def test_spaced_span_to_nonexistent_dir_does_not_swallow_the_tail(self, in_tmp_cwd):
        # "./src/foo bar.py" — no "./src/foo bar" dir, so the span is rejected
        # and the tail ("bar.py") isn't path-like on its own: no completion.
        assert SlashCommandCompleter._extract_path_word("./src/foo bar.py") is None

    def test_second_completion_inside_spaced_dir_returns_nested_entry(self, in_tmp_cwd):
        # Integration: extraction + completion together resolve a nested entry.
        docs = in_tmp_cwd / "My Documents"
        docs.mkdir()
        (docs / "report.md").touch()

        word = SlashCommandCompleter._extract_path_word("open ./My Documents/")
        assert word == "./My Documents/"
        names = _display_names(list(SlashCommandCompleter._path_completions(word)))
        assert "report.md" in names

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
