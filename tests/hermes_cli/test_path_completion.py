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
