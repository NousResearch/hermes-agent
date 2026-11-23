"""API and syntax edges where the facade must give stdlib's answer, or refuse at `compile()`.

Negative ``count`` / ``maxsplit``, a ``repl`` callback returning ``None``, an escaped `)` inside a
`(?#...)` comment, out-of-range escapes in classes and templates, ``lastindex`` with a group inside
a trailing lookahead; and the registry behind ``build_all`` / the fork clean-up, which must hold
every live Pattern even when two are equal.
"""
import gc
import re as std

import hermes_re as hre
from tests.hermes_re._support import outcome, view


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
