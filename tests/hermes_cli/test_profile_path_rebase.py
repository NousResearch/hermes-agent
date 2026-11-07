"""Unit tests for the pure path-rebase helper behind profile rename (#136430)."""

import ntpath

import pytest

from hermes_cli.profile_path_rebase import rebase_path

POSIX = [("/h/profiles/old", "/h/profiles/new")]
WINDOWS = [(r"C:\Users\me\.hermes\profiles\old", r"C:\Users\me\.hermes\profiles\new")]


@pytest.mark.parametrize(("path", "expected"), [
    ("/h/profiles/old", "/h/profiles/new"),
    ("/h/profiles/old/projects/x", "/h/profiles/new/projects/x"),
    ("/h/profiles/old2/projects/x", None),  # sibling sharing the prefix
    ("/h/profiles/older", None),
    ("/h/profiles", None),
    ("/elsewhere/profiles/old/x", None),
    ("", None),
    (None, None),
    (42, None),
])
def test_rebase_path_matches_whole_components_only(path, expected):
    assert rebase_path(path, POSIX) == expected


def test_rebase_path_tolerates_a_trailing_separator_on_the_old_prefix():
    assert rebase_path("/h/profiles/old/x", [("/h/profiles/old/", "/h/profiles/new")]) == "/h/profiles/new/x"


def test_rebase_path_accepts_backslash_separators():
    assert rebase_path("/h/profiles/old\\x", POSIX) == "/h/profiles/new\\x"


def test_rebase_path_is_case_insensitive_where_the_platform_is():
    """Windows: ``c:\\users`` and ``C:\\Users`` name the same directory; the stored suffix keeps its spelling."""
    stored = r"c:\users\me\.hermes\profiles\OLD\Projects\X"
    assert rebase_path(stored, WINDOWS, normcase=ntpath.normcase) == r"C:\Users\me\.hermes\profiles\new\Projects\X"
    assert rebase_path(r"c:\users\me\.hermes\profiles\old2\x", WINDOWS, normcase=ntpath.normcase) is None


def test_rebase_path_is_case_sensitive_by_default_on_posix():
    assert rebase_path("/h/profiles/OLD/x", POSIX, normcase=lambda p: p) is None


def test_rebase_path_uses_the_first_matching_pair():
    pairs = [("/link/profiles/old", "/link/profiles/new"), ("/real/profiles/old", "/real/profiles/new")]
    assert rebase_path("/real/profiles/old/x", pairs) == "/real/profiles/new/x"
    assert rebase_path("/link/profiles/old/x", pairs) == "/link/profiles/new/x"
