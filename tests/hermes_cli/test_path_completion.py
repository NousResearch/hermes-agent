"""Tests for file path autocomplete in the CLI completer."""

import os
from unittest.mock import MagicMock

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_plain_text

from hermes_cli.commands import (
    SlashCommandCompleter,
    _file_size_label,
    _hermesignore_match,
    _token_estimate_label,
)


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

    def test_quoted_path_with_spaces(self):
        assert (
            SlashCommandCompleter._extract_path_word('see "/tmp/Screen Shot 2026')
            == '"/tmp/Screen Shot 2026'
        )

    def test_escaped_spaces_path(self):
        assert (
            SlashCommandCompleter._extract_path_word(
                r"see /tmp/Screen\ Shot\ 2026"
            )
            == r"/tmp/Screen\ Shot\ 2026"
        )

    def test_context_file_reference_path(self):
        assert (
            SlashCommandCompleter._extract_path_word("review @file:src/ma")
            == "@file:src/ma"
        )

    def test_context_folder_reference_path(self):
        assert (
            SlashCommandCompleter._extract_path_word("review @folder:src/")
            == "@folder:src/"
        )


class TestExtractContextWord:
    def test_bare_context_trigger(self):
        assert SlashCommandCompleter._extract_context_word("review @") == "@"

    def test_bare_context_path_fragment(self):
        assert SlashCommandCompleter._extract_context_word("review @src") == "@src"

    def test_explicit_context_file_reference_is_not_bare_trigger(self):
        assert SlashCommandCompleter._extract_context_word("review @file:src/main.py") is None


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

    def test_fuzzy_match_non_contiguous_filename(self, tmp_path):
        (tmp_path / "Screenshot 2026-03-22 at 4.27.40 PM.png").touch()
        (tmp_path / "notes.txt").touch()

        completions = list(SlashCommandCompleter._path_completions(f"{tmp_path}/scsh"))
        names = _display_names(completions)

        assert "Screenshot 2026-03-22 at 4.27.40 PM.png" in names
        assert "notes.txt" not in names

    def test_fuzzy_match_orders_better_match_first(self, tmp_path):
        (tmp_path / "screen-capture.png").touch()
        (tmp_path / "super-complex-screenshot.png").touch()

        completions = list(SlashCommandCompleter._path_completions(f"{tmp_path}/scrcap"))
        names = _display_names(completions)

        assert names[0] == "screen-capture.png"

    def test_preserves_escaped_spaces_in_completion_text(self, tmp_path):
        target = tmp_path / "Screenshot 2026-03-22 at 4.27.40 PM.png"
        target.touch()

        completions = list(
            SlashCommandCompleter._path_completions(
                str(target).replace(" ", r"\ ").rsplit(".", 1)[0]
            )
        )

        assert any(c.text == str(target).replace(" ", r"\ ") for c in completions)

    def test_preserves_opening_quote_in_completion_text(self, tmp_path):
        target = tmp_path / "Screenshot 2026-03-22 at 4.27.40 PM.png"
        target.touch()

        completions = list(
            SlashCommandCompleter._path_completions(f'"{tmp_path}/Screen')
        )

        assert any(c.text == f'"{target}' for c in completions)

    def test_completes_context_file_references(self, tmp_path):
        target = tmp_path / "main.py"
        target.touch()

        completions = list(
            SlashCommandCompleter._path_completions(f"@file:{tmp_path}/ma")
        )

        assert any(c.text == f"@file:{target}" for c in completions)

    def test_completes_context_folder_references(self, tmp_path):
        target = tmp_path / "src"
        target.mkdir()

        completions = list(
            SlashCommandCompleter._path_completions(f"@folder:{tmp_path}/s")
        )

        assert any(c.text == f"@folder:{target}/" for c in completions)

    def test_context_file_completions_exclude_directories(self, tmp_path):
        (tmp_path / "main.py").touch()
        (tmp_path / "main_dir").mkdir()

        completions = list(
            SlashCommandCompleter._path_completions(f"@file:{tmp_path}/ma")
        )
        texts = [c.text for c in completions]

        assert f"@file:{tmp_path}/main.py" in texts
        assert f"@file:{tmp_path}/main_dir/" not in texts

    def test_context_folder_completions_exclude_files(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src.txt").touch()

        completions = list(
            SlashCommandCompleter._path_completions(f"@folder:{tmp_path}/sr")
        )
        texts = [c.text for c in completions]

        assert f"@folder:{tmp_path}/src/" in texts
        assert f"@folder:{tmp_path}/src.txt" not in texts


class TestContextCompletions:
    def test_lists_static_context_references(self):
        completions = list(SlashCommandCompleter._context_completions("@"))
        texts = [c.text for c in completions]
        assert "@diff" in texts
        assert "@staged" in texts
        assert "@file:" in texts
        assert "@folder:" in texts

    def test_bare_context_path_completes_to_file_reference(self, tmp_path):
        target = tmp_path / "main.py"
        target.touch()

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            completions = list(SlashCommandCompleter._context_completions("@ma"))
        finally:
            os.chdir(old_cwd)

        assert any(c.text == "@file:main.py" for c in completions)

    def test_bare_context_path_completes_to_folder_reference(self, tmp_path):
        target = tmp_path / "src"
        target.mkdir()

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            completions = list(SlashCommandCompleter._context_completions("@sr"))
        finally:
            os.chdir(old_cwd)

        assert any(c.text == "@folder:src/" for c in completions)

    def test_bare_context_path_with_spaces_uses_quoted_reference(self, tmp_path):
        target = tmp_path / "Screen Shot.png"
        target.touch()

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            completions = list(SlashCommandCompleter._context_completions("@scrsh"))
        finally:
            os.chdir(old_cwd)

        assert any(c.text == '@file:"Screen Shot.png"' for c in completions)


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

    def test_context_file_reference_triggers_completion(self, completer, tmp_path):
        target = tmp_path / "main.py"
        target.touch()

        doc = Document(f"check @file:{tmp_path}/ma", cursor_position=len(f"check @file:{tmp_path}/ma"))
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))

        assert any(c.text == f"@file:{target}" for c in completions)

    def test_bare_context_trigger_shows_context_completion(self, completer, tmp_path):
        target = tmp_path / "main.py"
        target.touch()

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            doc = Document("check @ma", cursor_position=len("check @ma"))
            event = MagicMock()
            completions = list(completer.get_completions(doc, event))
        finally:
            os.chdir(old_cwd)

        assert any(c.text == "@file:main.py" for c in completions)


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


class TestTokenEstimateLabel:
    def test_small_file(self, tmp_path):
        f = tmp_path / "tiny.txt"
        f.write_text("hello world")
        label = _token_estimate_label(str(f))
        assert label.startswith("~")
        assert "tok" in label

    def test_large_file(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * 40_000)
        label = _token_estimate_label(str(f))
        assert "K tok" in label

    def test_nonexistent(self):
        assert _token_estimate_label("/nonexistent_xyz") == ""


class TestHermesignoreMatch:
    def test_exact_name(self):
        assert _hermesignore_match("node_modules", "node_modules")
        assert not _hermesignore_match("node_modules", "node_mod")

    def test_directory_pattern(self):
        assert _hermesignore_match("build/", "build")
        assert not _hermesignore_match("build/", "builder")

    def test_leading_wildcard(self):
        assert _hermesignore_match("*.pyc", "test.pyc")
        assert not _hermesignore_match("*.pyc", "test.py")

    def test_trailing_wildcard(self):
        assert _hermesignore_match("test_*", "test_foo")
        assert not _hermesignore_match("test_*", "my_test")

    def test_surrounding_wildcards(self):
        assert _hermesignore_match("*cache*", "my_cache_dir")
        assert not _hermesignore_match("*cache*", "my_dir")


class TestIgnoredEntries:
    def test_gitignored_files_excluded_from_completions(self, tmp_path):
        """Files in .gitignore should not appear in completions."""
        import subprocess as sp

        sp.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / ".gitignore").write_text("ignored.txt\n")
        (tmp_path / "ignored.txt").touch()
        (tmp_path / "visible.txt").touch()

        completions = list(
            SlashCommandCompleter._path_completions(f"{tmp_path}/")
        )
        names = _display_names(completions)
        assert "visible.txt" in names
        assert "ignored.txt" not in names

    def test_hermesignore_excludes_files(self, tmp_path):
        """Files matching .hermesignore patterns should not appear."""
        import subprocess as sp

        sp.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / ".hermesignore").write_text("*.log\nsecrets/\n")
        (tmp_path / "app.log").touch()
        (tmp_path / "app.py").touch()
        (tmp_path / "secrets").mkdir()

        completions = list(
            SlashCommandCompleter._path_completions(f"{tmp_path}/")
        )
        names = _display_names(completions)
        assert "app.py" in names
        assert "app.log" not in names
        assert "secrets/" not in names

    def test_nested_hermesignore_excludes_files(self, tmp_path):
        import subprocess as sp

        sp.run(["git", "init"], cwd=tmp_path, capture_output=True)
        nested = tmp_path / "workspace" / "src"
        nested.mkdir(parents=True)
        (tmp_path / "workspace" / ".hermesignore").write_text("*.tmp\ncache/\n")
        (nested / "good.py").touch()
        (nested / "bad.tmp").touch()
        (nested / "cache").mkdir()

        completions = list(
            SlashCommandCompleter._path_completions(f"{nested}/")
        )
        names = _display_names(completions)
        assert "good.py" in names
        assert "bad.tmp" not in names
        assert "cache/" not in names

    def test_context_completions_also_respect_gitignore(self, tmp_path):
        """Bare @ context completions should also filter gitignored entries."""
        import subprocess as sp

        sp.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / ".gitignore").write_text("node_modules\n")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "src").mkdir()

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            completions = list(SlashCommandCompleter._context_completions("@"))
            names = _display_names(completions)
        finally:
            os.chdir(old_cwd)

        assert "src/" in names
        assert "node_modules/" not in names


class TestStaticContextCompletions:
    def test_includes_git_reference(self):
        completions = list(SlashCommandCompleter._context_completions("@gi"))
        texts = [c.text for c in completions]
        assert "@git:" in texts

    def test_includes_url_reference(self):
        completions = list(SlashCommandCompleter._context_completions("@ur"))
        texts = [c.text for c in completions]
        assert "@url:" in texts

    def test_all_static_refs_present(self):
        completions = list(SlashCommandCompleter._context_completions("@"))
        texts = [c.text for c in completions]
        for expected in ("@diff", "@staged", "@file:", "@folder:", "@git:", "@url:"):
            assert expected in texts

    def test_context_completions_show_token_estimate(self, tmp_path):
        target = tmp_path / "main.py"
        target.write_text("print('hello')\n" * 100)

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            completions = list(SlashCommandCompleter._context_completions("@ma"))
            metas = _display_metas(completions)
        finally:
            os.chdir(old_cwd)

        assert any("tok" in m for m in metas)
