"""Unit tests for resolve_rename_path (hermes_cli/web_git.py).

Pure, I/O-free parser that turns git's rename notation into the NEW path:
- no " => "            -> path returned unchanged (stripped)
- "old => new"         -> "new"
- "dir/{old => new}/f" -> "dir/new/f" (brace form), collapsing "//"
"""

import pytest

from hermes_cli.web_git import resolve_rename_path


class TestResolveRenamePath:
    @pytest.mark.parametrize("raw,expected", [
        ("src/foo.py", "src/foo.py"),   # no rename token
        ("", ""),
        ("  a/b.py  ", "a/b.py"),        # stripped
    ])
    def test_non_rename_paths_pass_through(self, raw, expected):
        assert resolve_rename_path(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("old.py => new.py", "new.py"),
        ("a/old.py => a/new.py", "a/new.py"),
        ("  x => y  ", "y"),
    ])
    def test_simple_rename_returns_new_path(self, raw, expected):
        assert resolve_rename_path(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("src/{old => new}/f.py", "src/new/f.py"),
        ("{a => b}/f", "b/f"),
        ("dir/{old => }/f", "dir/f"),   # empty new collapses "//"
    ])
    def test_brace_rename_reconstructs_new_path(self, raw, expected):
        assert resolve_rename_path(raw) == expected
