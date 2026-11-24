"""Bracket-class syntax stdlib rejects is rejected by ``compile()`` too, never run with another meaning."""
import re as std

import pytest

import hermes_re as hre

from ._support import outcome

_REJECTED = [r"[\d-a]", r"[\w-\s]", r"[a\d-z]", r"[\d--]", r"[^\S-\0]", r"[\D-^]", r"[\s-\d]x"]
_ACCEPTED = [r"[\d-]", r"[^\w-]", r"[-\d]", r"[a-\x7a\d-]", r"[\d\-a]", r"[\d-]]"]


@pytest.mark.parametrize("pattern", _REJECTED)
@pytest.mark.parametrize("kind", [str, bytes])
def test_shorthand_class_as_range_start_is_an_error(pattern, kind):
    source = pattern if kind is str else pattern.encode()
    for flags in (0, hre.I, hre.A) if kind is str else (0, hre.I):
        assert outcome(lambda: std.compile(source, flags)) == "raises error"
        assert outcome(lambda: hre.compile(source, flags)) == "raises error"


@pytest.mark.parametrize("pattern", _ACCEPTED)
def test_trailing_or_escaped_dash_after_a_shorthand_stays_a_member(pattern):
    subject = "-a5z]_ \u0663"
    assert hre.findall(pattern, subject) == std.findall(pattern, subject)


@pytest.mark.parametrize("escape", [r"\u0041", r"\U00000041", r"\N{LATIN SMALL LETTER A}"])
def test_bytes_class_has_no_unicode_escapes(escape):
    for body in ("[{}]", "[a{}]", "[^{}z]", "[{}-z]"):
        source = body.format(escape).encode()
        assert outcome(lambda: std.compile(source)) == "raises error"
        assert outcome(lambda: hre.compile(source)) == "raises error"
