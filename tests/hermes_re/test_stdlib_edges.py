"""API and syntax edges where the facade must give stdlib's answer, or refuse at `compile()`.

Negative ``count`` / ``maxsplit``, a ``repl`` callback returning ``None``, an escaped `)` inside a
`(?#...)` comment, out-of-range escapes in classes and templates, ``lastindex`` with a group inside
a trailing lookahead; group nesting as deep as stdlib's parser takes, and the refusal past it; and
the registry behind ``build_all`` / the fork clean-up, which must hold every live Pattern even when two are equal.
"""
import contextlib
import gc
import re as std
import sys
import time

import pytest

import hermes_re as hre
from hermes_re._validate import MAX_NESTING
from tests.hermes_re._support import fresh_interpreter, outcome, view


def test_a_negative_count_or_maxsplit_changes_nothing_as_in_stdlib():
    for subject in ("xaxax", "x\u00e9ax\u00e9"):  # ASCII and non-ASCII subjects take different paths
        for module in (std, hre):
            assert module.sub("a", "b", subject, count=-1) == subject
            assert module.subn("a", "b", subject, count=-1) == (subject, 0)
            assert module.split("a", subject, maxsplit=-1) == [subject]
    assert hre.subn(b"a", b"b", b"aa", count=-2) == (b"aa", 0)
    assert hre.split(b"a", b"xax", maxsplit=-5) == [b"xax"]


def test_a_callback_returning_none_removes_the_match_as_in_stdlib():
    for subject in ("xax", "\u00e9a\u00e9", b"xax"):
        pattern = b"a" if isinstance(subject, bytes) else "a"
        assert hre.subn(pattern, lambda m: None, subject) == std.subn(pattern, lambda m: None, subject)


def test_an_escaped_parenthesis_does_not_end_a_comment():
    for source, subject in ((r"a(?#note\))b", "ab"), (r"(?#x\))(a)", "a"), (r"(?#x\\)(a)", "a"),
                            (r"a(?#x\)\))(b)", "ab")):
        assert view(hre.match(source, subject)) == view(std.match(source, subject))
    assert outcome(lambda: hre.compile(r"(a)(?#\)")) == outcome(lambda: std.compile(r"(a)(?#\)")) == "raises error"


def test_an_out_of_range_class_escape_is_refused_at_compile():
    for source in (r"[\U00110000]", r"[a-\U00110000]", r"[\400]", r"[\777]"):
        assert outcome(lambda: std.compile(source)) == "raises error"
        assert outcome(lambda: hre.compile(source)) == "raises error", source
    assert hre.findall(r"[\377\U0010FFFF]", "\xff\U0010ffff") == ["\xff", "\U0010ffff"]


def test_three_octal_digits_in_a_template_are_an_escape_never_a_group_reference():
    many = "(a)" * 41  # group 40 exists, so `\400` must not read as group 40 followed by "0"
    for template in (r"\400", r"\401x", r"\777"):
        assert outcome(lambda: std.sub(many, template, "a" * 41)) == "raises error"
        assert outcome(lambda: hre.sub(many, template, "a" * 41)) == "raises error", template
    for template in (r"\377", r"\40", r"\408", r"\0"):
        assert hre.sub(many, template, "a" * 41) == std.sub(many, template, "a" * 41)


def test_lastindex_counts_a_lookahead_group_as_closing_where_the_lookahead_runs():
    for source in (r"(?P<outer>a(?=(?P<inner>b)))", r"((a)(?=(b)))", r"(a)(?=(b))", r"(a)(?=(b)(c))",
                   r"(?:(a)|(b))+(?=(c))", r"(a)(?=b|(c))"):
        for subject in ("ab", "abc", "bac", "aac", "ac"):
            assert view(hre.search(source, subject)) == view(std.search(source, subject)), (source, subject)


_SUBJECTS = ("ba", "aba", "bba", "cba", "aaba", "ab", "abab", "bab", "a", "b", "")


def test_lastindex_is_an_empty_group_closed_in_a_later_iteration_than_a_group_with_text():
    # `b` ends at 1 in one iteration, `empty` closes at 1 in the next: the empty group closed last
    for source in (r"(?:(?P<empty>)a|(?P<b>b))+", r"(?:(?:()a)|(b))+", r"((?:()a|(b))+)", r"(?:()a|(b)c?)+",
                   r"(?:()a|(b)){2}", r"(x)?(?:()a|(b))*?a"):
        for subject in _SUBJECTS:
            for find in (std.search, std.fullmatch):
                mine = getattr(hre, find.__name__)
                assert view(mine(source, subject)) == view(find(source, subject)), (source, subject, find.__name__)


def test_a_repeat_where_two_empty_groups_can_close_in_either_order_is_refused_at_compile():
    # `x` closes empty at the END of one iteration, `y` empty at the START of the next: both end at
    # one offset and nothing in the spans says which closed last
    for source in (r"(?:(?P<y>)b|a(?P<x>))+", r"(?:()b|a())*", r"(?:(?:()|())a|b())+", r"(?:()b|(c)?())+?",
                   rb"(?:()b|a()){1,3}"):
        assert outcome(lambda: hre.compile(source)) == "raises error", source
    # the later group is certain to follow in the same iteration, or one of them must take text
    for source in (r"(?:()a())*", r"(?:()(?:b|c)())+", r"(?:(a)|()b)*", r"(?:()a|(b))+", r"(?:()(b)?a)+",
                   r"(?:a()|()b)?"):
        for subject in _SUBJECTS:
            assert view(hre.search(source, subject)) == view(std.search(source, subject)), (source, subject)


def test_a_deeply_nested_repeat_compiles_where_stdlib_compiles():
    source = "(" * 400 + "(a)" + ")" * 400 + "*b"
    assert hre.compile(source).groups == std.compile(source).groups == 401
    assert view(hre.search(source, "aab")) == view(std.search(source, "aab"))


# Nesting shapes: each takes the TOTAL group depth `d` (lookarounds count) and gives (pattern, subjects).
_NESTED = {
    "plain": lambda d: ("(" * d + "a" + ")" * d, ["a", "ba"]),
    "dollar": lambda d: ("(" * d + "a" + ")" * d + "$", ["a", "a\n", "ab"]),
    "caret": lambda d: ("^" + "(" * d + "a" + ")" * d, ["a", "ba"]),
    "lookahead_end": lambda d: ("(" * d + "a" + ")" * d + "(?=b)", ["ab", "a"]),
    "negative_lookahead_end": lambda d: ("(" * d + "a" + ")" * d + "(?!b)", ["ab", "a"]),
    "lookbehind_start": lambda d: ("(?<=b)" + "(" * d + "a" + ")" * d, ["ba", "a"]),
    "negative_lookbehind_start": lambda d: ("(?<!b)" + "(" * d + "a" + ")" * d, ["ba", "ca"]),
    "both_guards": lambda d: ("(?<!b)" + "(" * d + "a" + ")" * d + "(?!b)", ["a", "ab", "ca"]),
    "alternation": lambda d: ("(?:x|" * d + "a" + ")" * d + "$", ["a", "x", "b"]),
    "quantified": lambda d: ("(" * d + "a" + ")+" * d + "$", ["aa", "a\n"]),
    "noncapturing": lambda d: ("(?:" * d + "a" + ")" * d + "$", ["a"]),
    "named": lambda d: ("".join(f"(?P<g{k}>" for k in range(d)) + "a" + ")" * d + "$", ["a"]),
    "scoped_flags": lambda d: ("(?<=x)" + "(?i:" * d + "a" + ")" * d, ["xA", "A"]),
    "word_boundaries": lambda d: ("(" * d + "\\ba\\b" + ")" * d, [" a ", "ba"]),
    "guard_inside": lambda d: ("(" * (d - 1) + "a(?=b)" + ")" * (d - 1), ["ab", "a"]),
    "deep_lookahead_body": lambda d: ("a(?=" + "(" * (d - 1) + "b" + ")" * (d - 1) + ")", ["ab"]),
}


@contextlib.contextmanager
def _recursion_limit(limit):
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


def _frames():
    frame, count = sys._getframe(), 0
    while frame is not None:
        frame, count = frame.f_back, count + 1
    return count


def _results(module, pattern, subjects):
    compiled = module.compile(pattern)
    return [(view(compiled.search(s)), view(compiled.match(s)), view(compiled.fullmatch(s)),
             [view(m) for m in compiled.finditer(s)], compiled.sub("<\\g<0>>", s)) for s in subjects]


@pytest.mark.parametrize("shape", sorted(_NESTED))
def test_nesting_as_deep_as_stdlib_compiles_gives_stdlibs_answers_on_a_shallow_stack(shape):
    pattern, subjects = _NESTED[shape](MAX_NESTING)
    with _recursion_limit(10_000):  # stdlib's parser recurses per group; the reference needs the room
        expected = _results(std, pattern, subjects)
    hre.purge()
    with _recursion_limit(_frames() + 60):  # the facade must not: a caller deep in its own stack still compiles
        got = _results(hre, pattern, subjects)
    assert got == expected


def test_nesting_beyond_stdlibs_limit_is_refused_at_compile_with_error():
    patterns = {shape: build(MAX_NESTING + 1)[0] for shape, build in _NESTED.items()}
    for shape, pattern in patterns.items():
        assert outcome(lambda: hre.compile(pattern)) == "raises error", shape
        assert outcome(lambda: hre.compile(pattern.encode())) == "raises error", shape
    # stdlib, from a fresh interpreter at the default recursion limit: depth MAX_NESTING compiles,
    # one more does not (it raises RecursionError, which the facade answers with ``error``)
    code = ("import re, sys\n"
            f"for d in ({MAX_NESTING}, {MAX_NESTING + 1}):\n"
            "    try:\n"
            "        re.compile('(' * d + 'a' + ')' * d + '$')\n"
            "        print(d, 'ok')\n"
            "    except RecursionError:\n"
            "        print(d, 'RecursionError')\n"
            "import hermes_re\n"
            f"print(hermes_re.compile('(' * {MAX_NESTING} + 'a' + ')' * {MAX_NESTING} + '$').search('a\\n').span())\n")
    assert fresh_interpreter(code) == [str(MAX_NESTING), "ok", str(MAX_NESTING + 1), "RecursionError", "(0,", "1)"]


def test_build_all_reaches_every_live_pattern_even_equal_ones():
    hre.purge()
    gc.collect()
    first = hre.compile(r"[\w.-]{1000}" * 4)
    hre.purge()  # the next compile makes a second, equal Pattern
    second = hre.compile(r"[\w.-]{1000}" * 4)
    assert first == second and first is not second
    failures = hre.build_all()
    assert sorted(id(p) for p, _ in failures) == sorted((id(first), id(second)))
    hre.purge()
    first = second = failures = None
    gc.collect()


def test_a_failed_wide_check_leaves_later_matches_as_they_would_be_without_it():
    from hermes_re import _engine
    source, real = r"\babc[a-z]*?\b", _engine.MAX_MEM
    subject = "\u2014abc\u00e9 abcx\u00e9d abcd\u2014"  # needs the exact twin, which does not fit
    hre.purge()
    _engine.MAX_MEM = 20_000  # room for the cheap programs, not for the exact one with the Unicode `\w` class
    try:
        unchecked = [m.span() for m in hre.compile(source).finditer(subject)]  # falls back to the ASCII `\b`
        hre.purge()
        checked = hre.compile(source)
        assert [p for p, _ in hre.build_all(wide=True) if p is checked] == [checked]
        assert outcome(lambda: [m.span() for m in checked.finditer(subject)]) == unchecked
        first = checked.search(subject)
        assert first is not None and first.span() == unchecked[0]
    finally:
        _engine.MAX_MEM = real
        hre.purge()


def test_build_all_wide_checks_a_wide_form_that_matching_already_built():
    from hermes_re import _engine
    source, real = r"\babc[a-z]*?\b", _engine.MAX_MEM
    hre.purge()
    _engine.MAX_MEM = 20_000  # room for the cheap programs, not for the exact one with the Unicode `\w` class
    try:
        compiled = hre.compile(source)
        assert compiled.search("\u2014abc\u2014").span() == (1, 4)  # builds the wide form; its cheap runner suffices
        failures = [("too large" in e.msg) for p, e in hre.build_all(wide=True) if p is compiled]
        assert failures == [True]  # as when nothing had matched yet
    finally:
        _engine.MAX_MEM = real
        hre.purge()


def test_a_non_ascii_digit_after_a_backslash_is_never_a_group_or_octal_digit():
    arabic_one = "\u0661"  # str.isdigit() is true for it; stdlib's parser reads only ASCII 0-9
    for pattern, template, subject in (
        (r"(a)", "\\1" + arabic_one, "a"),  # group 1, then the literal digit
        (r"(a)" * 12, "\\1" + arabic_one, "a" * 12),  # not group 11
        (r"(a)", "\\" + arabic_one, "a"),  # no group at all: the escape stays literal
    ):
        assert hre.sub(pattern, template, subject) == std.sub(pattern, template, subject)
    for pattern, subject in (
        ("x\\b[\\" + arabic_one + "]", "x" + arabic_one),  # an escaped Arabic digit in a class is a word character
        ("(?<=\\" + arabic_one + ")b", arabic_one + "b"),  # a one-character lookbehind guard, not a backreference
    ):
        assert [m.span() for m in hre.finditer(pattern, subject)] == [m.span() for m in std.finditer(pattern, subject)]


def test_a_never_matching_edge_alternative_compiles_in_bytes_patterns():
    # the facade's own never-matching filler must be valid in a bytes pattern, which has no \U escape
    for pattern, subject in ((rb"(?<=a)b|c", b"abcxc"), ((r"(?<=a)b|c", "abcxc"))):
        assert hre.compile(pattern).findall(subject) == std.compile(pattern).findall(subject)


def test_lastindex_reads_many_tied_groups_in_one_pass():
    # 12,000 empty groups all end at 0: comparing each with every other took minutes; one pass takes milliseconds.
    pattern = hre.compile("()" * 12_000)
    found = pattern.match("x")
    started = time.monotonic()
    assert found.lastindex == std.compile("()" * 12_000).match("x").lastindex == 12_000
    assert found.lastgroup is None
    assert time.monotonic() - started < 2
