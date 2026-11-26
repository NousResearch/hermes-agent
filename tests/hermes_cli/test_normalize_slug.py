"""Unit tests for normalize_slug (hermes_cli/projects_db.py).

Pure, I/O-free validator/normalizer for project slugs. Contract
(from _SLUG_RE = ^[a-z0-9][a-z0-9\\-_]{0,63}$):
- None / empty / whitespace-only -> None
- otherwise lowercased + stripped, then validated
- invalid slugs raise ValueError
"""

import pytest

from hermes_cli.projects_db import normalize_slug


class TestNormalizeSlug:
    def test_none_and_empty_return_none(self):
        assert normalize_slug(None) is None
        assert normalize_slug("") is None
        assert normalize_slug("   ") is None

    @pytest.mark.parametrize("raw,expected", [
        ("my-project", "my-project"),
        ("MyProject", "myproject"),       # lowercased
        ("  Foo_Bar-1  ", "foo_bar-1"),   # stripped + lowercased
        ("a", "a"),
        ("abc123", "abc123"),
        ("a" * 64, "a" * 64),             # max length (64)
    ])
    def test_valid_slugs_are_normalized(self, raw, expected):
        assert normalize_slug(raw) == expected

    @pytest.mark.parametrize("raw", [
        "-foo",     # cannot start with hyphen
        "_foo",     # cannot start with underscore
        "foo bar",  # space not allowed
        "foo.bar",  # dot not allowed
        "foo/bar",  # slash not allowed
        "a" * 65,   # exceeds 64 chars
    ])
    def test_invalid_slugs_raise(self, raw):
        with pytest.raises(ValueError):
            normalize_slug(raw)
