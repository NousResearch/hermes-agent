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

    def test_http_url_is_not_a_path(self):
        # URLs contain "/" but must not be treated as local paths — otherwise
        # the completer runs os.listdir on every keystroke while typing a link.
        assert SlashCommandCompleter._extract_path_word("see http://example.com/a") is None

    def test_https_url_is_not_a_path(self):
        assert (
            SlashCommandCompleter._extract_path_word("log https://paste.rs/B3Xws") is None
        )

    def test_scheme_url_without_slash_yet_is_not_a_path(self):
        # As soon as "://" is typed the token is a URL, even before the path part.
        assert SlashCommandCompleter._extract_path_word("ftp://host") is None

    def test_real_path_still_works_after_url_guard(self):
        # The URL guard must not regress ordinary path detection.
        assert (
            SlashCommandCompleter._extract_path_word("open src/utils/helpers.py")
            == "src/utils/helpers.py"
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

    def test_absolute_path_triggers_completion(self, completer):
        doc = Document("check /etc/hos", cursor_position=14)
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        names = _display_names(completions)
        # /etc/hosts should exist on Linux
        assert any("host" in n.lower() for n in names)

    def test_url_yields_no_completions_and_no_fs_access(self, completer, monkeypatch):
        """Typing a URL must not run filesystem completion.

        Regression for laggy typing: a URL token contains "/", so it used to
        reach `_path_completions` and call `os.listdir` on every keystroke.
        Assert both that no completions are produced and that the filesystem is
        never touched for a URL.
        """
        import hermes_cli.commands as commands_mod

        def _boom(*_args, **_kwargs):
            raise AssertionError("os.listdir must not be called for a URL token")

        monkeypatch.setattr(commands_mod.os, "listdir", _boom)

        text = "check https://paste.rs/B3Xws"
        doc = Document(text, cursor_position=len(text))
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        assert completions == []


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
