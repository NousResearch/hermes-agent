"""`(?#...)` comment groups are read exactly as stdlib reads them.

A comment ends at the first `)`, may hold `[` or `(`, cannot be nested, must be terminated, and is
ordinary text inside a bracket class. The differential test puts a comment at EVERY offset of a
set of patterns and requires stdlib's verdict and stdlib's results.
"""
import re as std
import warnings

import hermes_re as hre
from tests.hermes_re._support import outcome, same_everywhere

SUBJECTS = ["", "a", "ab", "aab", "abab", "a{2}", "a1", "x-1", "a\n", "(a)", "\x00" + "1", "A", "é a", "aa bb", "k=v;", "a|b"]
COMMENTS = ["(?#c)", "(?# [ note with ( and [ )", "(?#)", "(?#a)(?#b)"]
BASES = [r"a", r"ab+", r"a*b", r"a{2}", r"a{2,3}b", r"a{,2}", r"(a)(b)?", r"(?:a|b)+", r"(?P<n>a)b", r"[ab]c", r"[^a]+", r"\w+\d",
         r"a\.b", r"\x41", r"\0" + "1", r"\101", r"a|b", r"^a$", r"(?i)ab", r"(?i:a)B", r"a??b", r"a+?", r"\bab\b", r"(?<!x)a",
         r"a(?=b)", r"a(?!b)", r"k=(\w);", r"[(?#x)]", r"\(a\)", r"a{2", r"x-\d", r"(?m)^a", r"(?s:.)b"]


def _verdict(module, source):
    try:
        module.compile(source)
    except module.error:
        return "error"
    return "ok"


def test_a_comment_at_every_offset_gives_stdlib_verdict_and_results():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # stdlib warns about some of the generated sources
        _comment_at_every_offset()


def _comment_at_every_offset():
    checks = accepted = refused = 0
    for base in BASES:
        for comment in COMMENTS:
            for offset in range(len(base) + 1):
                source = base[:offset] + comment + base[offset:]
                expected = _verdict(std, source)
                assert _verdict(hre, source) == expected, f"{source!r}: stdlib {expected}"
                if expected == "ok":
                    accepted += 1
                    checks += same_everywhere(source, 0, SUBJECTS)
                else:
                    refused += 1
    assert accepted > 400 and refused > 80 and checks > 50000, (accepted, refused, checks)


def test_comment_rules():
    assert hre.findall(r"a(?#x)*", "aaa b") == std.findall(r"a(?#x)*", "aaa b")  # the quantifier applies to `a`
    assert hre.fullmatch(r"a(?#x){2}", "aa") is not None
    assert hre.fullmatch(r"a{2(?#c)}", "a{2}") is not None and hre.fullmatch(r"a{2(?#c)}", "aa") is None  # no repeat
    assert hre.fullmatch(r"\0(?#c)1", "\x001") is not None                         # not the octal escape \01
    assert hre.fullmatch(r"a(?#[)b", "ab") is not None                             # `[` in a comment is no class
    assert hre.fullmatch(r"[(?#x)]+", "(?#x)") is not None                         # in a class it is ordinary text
    assert hre.fullmatch(r"a(?#b(?#c)d\)", "ad)") is not None                      # ends at the FIRST `)`: no nesting
    assert hre.fullmatch(r"(?#only a comment)", "") is not None
    for bad in (r"a(?#", r"a(?#b", r"((?#c)?:a)", r"a*(?#c)?", r"a{2}(?#c)?", r"(?#c)*a", r"\x4(?#c)1", r"(?i(?#c)m)a",
                r"(?P<na(?#c)me>x)", r"a(?#b(?#c)d)"):
        assert outcome(lambda: std.compile(bad)) == "raises error", bad
        assert outcome(lambda: hre.compile(bad)) == "raises error", bad


def test_the_pattern_registered_by_the_redaction_tests_compiles_and_matches():
    source = r"zzkey-[A-Za-z0-9]{10,}(?# [)"  # tests/agent/test_redaction_registry.py
    compiled = hre.compile(source)
    assert compiled.pattern == source and compiled.groups == 0
    text = "x zzkey-Ab3dE5gH7jK9mN1pQ3s y"
    assert compiled.sub("***", text) == std.sub(source, "***", text) == "x *** y"


def test_comments_keep_groups_spans_and_introspection_right():
    source = r"(?#lead)(a)(?# (not a group )(?P<n>b)?(?#tail)\b"
    mine, ref = hre.compile(source), std.compile(source)
    assert (mine.groups, mine.groupindex) == (ref.groups, dict(ref.groupindex)) == (2, {"n": 2})
    for text in ("ab", "a", "xab c"):
        m, r = mine.search(text), ref.search(text)
        assert (m.span(), m.groups(), m.lastindex, m.lastgroup) == (r.span(), r.groups(), r.lastindex, r.lastgroup)
    assert same_everywhere(rb"a(?#bytes)+b", 0, [b"", b"aab", b"b", b"a(?#bytes)b"]) > 0
    # offsets are reported in the caller's text, comments included
    source = r"x(?#one)\b.(?#two)\By"
    assert hre.inexact_boundaries(source) == [source.index(r"\b"), source.index(r"\B")]
