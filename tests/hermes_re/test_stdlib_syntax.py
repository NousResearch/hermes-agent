"""Syntax stdlib refuses is refused at `compile()`, even where RE2 could run it."""
import itertools
import re as std
import sys

import pytest

import hermes_re as hre
from tests.hermes_re._support import outcome

_REFUSED_BY_STDLIB = [
    r"(?<n>a)", r"(?<n>a)$", r"(?U)a+", r"(?U)a+$", r"(?U:a)", r"\pL", r"\p{Greek}", r"\P{L}x", r"a(?i)b",
    r"(?i)a(?m)b", r"(?i:a)(?m)b", r"a|(?i)b", r"(?:(?i)a)", r"(?<!x)(?i)a", r"(?-i)a", r"(?i-i:a)", r"(?au:a)",
    r"(?L)a", r"(?-a:a)", r"(?^)a", r"(?|a)", r"(?R)", r"(?&n)", r"(?P>n)", r"(?ia-:a)", r"(?i",
]


@pytest.mark.parametrize("source", _REFUSED_BY_STDLIB)
def test_syntax_stdlib_refuses_is_refused(source):
    assert outcome(lambda: std.compile(source)) == "raises error"
    with pytest.raises(hre.error) as raised:
        hre.compile(source)
    assert raised.value.pattern == source


def test_z_follows_the_running_stdlib():
    refused = outcome(lambda: hre.compile(r"a\z")) == "raises error"
    assert refused == (outcome(lambda: std.compile(r"a\z")) == "raises error") == (sys.version_info < (3, 14))
    if not refused:
        assert [m.span() for m in hre.finditer(r"a\z", "aba")] == [(2, 3)]


def test_flag_groups_never_accept_what_stdlib_refuses():
    pieces = ["(?i)", "(?m)", "(?s)", "(?U)", "(?a)", "(?u)", "(?L)", "(?-i:", "(?i-m:", "(?i-i:", "(?<n>", "(?P<m>",
              "(?:", ")", "a", "|", "^", r"\pL", "(?#c)"]
    for size in (1, 2, 3):
        for parts in itertools.product(pieces, repeat=size):
            for source in ("".join(parts), "".join(parts).encode()):
                if outcome(lambda: std.compile(source)) == "raises error":
                    assert outcome(lambda: hre.compile(source)) == "raises error", source


def test_what_stdlib_accepts_still_compiles_and_matches():
    for source, subject in ((r"(?i)(?m)^A", "x\na"), (r"(?#c)(?i)a", "A"), (r"(?i:a)(?-i:b)", "Ab"),
                            (r"(?s-i:.)a", "\na"), (r"(?P<n>a)(?=b)", "ab"), (r"(?<=x)a", "xa")):
        ref = std.search(source, subject)
        assert ref is not None and [m.span() for m in hre.finditer(source, subject)][:1] == [ref.span()], source


# Edge-guard shapes around malformed atoms: the guard rewrite turns a guard's body into a class or
# a neighbour test, which must not make syntax stdlib refuses runnable (`(?<!*)a` is not `[^*]a`).
_ATOMS = ["", "*", "+", "?", "*?", "{2}", "{", "{,}", "{1,2}", "{2,1}", "a", "a**", "a*+", "[", "]", "[a", "[]", "[^]",
          "\\", "\\q", "\\x", "\\x4", "\\u12", "\\N", "\\c", "()", "(", ")", "(?", "\\1", "|", "a|", "^*", "$?",
          "\\b*", "\\A+", "[a-", "[z-a]", "[a-\\d]", "(?P<1>a)", "(?P=n)", "a{,}", "(?x)", "\\pL", "]*"]
_GUARDS = ["(?<={})", "(?<!{})", "(?={})", "(?!{})"]
_SHAPES = ["{g}a", "a{g}", "b|{g}a", "a{g}|b", "({g}a)", "{g}a|b{g}", "{g}*a", "x{g}y"]


@pytest.mark.parametrize("guard", _GUARDS)
def test_a_guard_stdlib_refuses_is_refused(guard):
    accepted = []
    for atom, shape in itertools.product(_ATOMS, _SHAPES):
        source = shape.format(g=guard.format(atom))
        for flags in (0, std.IGNORECASE):
            refused = outcome(lambda: std.compile(source, flags)) == "raises error"
            if refused and outcome(lambda: hre.compile(source, flags)) != "raises error":
                accepted.append((source, int(flags)))
    assert accepted == []
