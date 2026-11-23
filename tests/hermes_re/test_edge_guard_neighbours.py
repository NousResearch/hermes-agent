r"""Edge guards and edge `\b` next to non-ASCII neighbours: stdlib's results at the ASCII path's cost.

A guard on a Unicode class (``(?<!\w)``, ``(?!\w)``, the guards an edge `\b` becomes) is given to
RE2 as a small class that accepts every non-ASCII neighbour; the facade then tests that neighbour
itself. These tests compare every operation with stdlib around the characters where the two
readings differ (letters and digits of other scripts, combining marks, underscore, the three
IGNORECASE specials U+017F, U+0130, U+212A, an emoji, an astral letter, a lone surrogate), and
pin the cost: no program holding the Unicode `\w` class is built for a run-time `\bNAME\b`.
"""
import itertools
import re as std
import time

import hermes_re as hre
from tests.hermes_re._support import same_everywhere, view

# word characters / not word characters for stdlib's Unicode `\w`
WORD = ["a", "Z", "5", "_", "\u00e9", "\u65e5", "\u0663", "\u00b2", "\u212a", "\u017f", "\u0130", "\u0131",
        "\U00010400"]
OTHER = ["", " ", "-", "\n", "\u0301", "\u2014", "\u200d", "\u3000", "\U0001f600", "\ud800"]
NEIGHBOURS = WORD + OTHER
NAMES = ["abc", "k", "s", "i", "K9", "x_y", "a-b", "\u00e9t\u00e9", "\u0663", "\u65e5\u672c", "_"]


def _subjects(cores):
    """Each core between every pair of neighbours, alone and twice in a row sharing a neighbour."""
    out = []
    for core in cores:
        for left, right in itertools.product(NEIGHBOURS, repeat=2):
            out.append(left + core + right)
            out.append(left + core + right + core + left)
    return out


def _cores(name):
    return sorted({name, name.upper(), name.lower(), name.replace("k", "\u212a").replace("s", "\u017f")})


def test_a_runtime_word_bounded_name_matches_like_stdlib_next_to_any_neighbour():
    checks = 0
    for name, flags in itertools.product(NAMES, (0, hre.IGNORECASE)):
        checks += same_everywhere(rf"\b{hre.escape(name)}\b", flags, _subjects(_cores(name)))
    assert checks > 100_000


def test_one_character_guards_on_unicode_classes_match_like_stdlib():
    patterns = [r"(?<!\w)abc(?!\w)", r"(?<=\w)-(?=\w)", r"(?<![\w-])k(?![\w.])", r"(?<!\d)5(?!\d)", r"(?<=\d)s", r"i(?=\d)",
                r"(?<!\W)k(?!\W)", r"(?<!\D)s", r"(?<![^\W\d])abc", r"(?<![\Wk])s(?![\Ws])", r"(?<=[\w\u0301])i(?![\d_])",
                r"(?<!\w)(abc|k)(?!\w)", r"(?<!\w)@?abc\w{0,3}(?!\w)", r"(?<!\w)k+(?!\w)", r"(?<!\w)(?!\w)", r"(?<=\w)(?=\w)"]
    cores = ["abc", "ABC", "k", "K", "\u212a", "s", "\u017f", "i", "\u0130", "5", "-", "", "@abcde"]
    subjects = _subjects(cores)
    checks = 0
    for pattern, flags in itertools.product(patterns, (0, hre.IGNORECASE)):
        checks += same_everywhere(pattern, flags, subjects)
    assert checks > 300_000


def test_patterns_that_need_the_exact_program_after_a_wrong_cheap_report_still_match_like_stdlib():
    # More than one alternative, or a body of variable width: when the cheap guard accepts a neighbour
    # its test refuses, only the exact program can say what matches instead.
    patterns = [r"\babc\w*?\b", r"\babc\b|k\b|(?<!\w)s", r"(?:\babc|.bc)\b", r"\babc\b-?", r"\b(?:abc|k)+\b",
                r"\b\w+?\b", r"\bk{1,3}\b", r"(?<!\w)abc|(?<=\d)abc", r"(?i)\bk\b", r"(?i:\bk\b)", r"(?m)^\babc\b$"]
    cores = ["abc", "abcabc", "k", "kk", "\u212a", "s", "xbc", "abc-"]
    subjects = _subjects(cores)
    checks = 0
    for pattern, flags in itertools.product(patterns, (0, hre.IGNORECASE)):
        checks += same_everywhere(pattern, flags, subjects)
    assert checks > 150_000


def test_guards_are_decided_at_every_start_and_end_position():
    patterns = [r"\babc\b", r"(?<!\w)abc(?!\w)", r"(?<=\w)abc(?=\w)", r"\babc\w*?\b", r"\bk\b|\babc\b"]
    subjects = ["\u00e9abc abc\u00e9 abc \u65e5abc\u65e5abc\u2014abc\u0301 kabc_ abc", "abc\u00e9abc\u0301abc k \u212a"]
    for pattern, flags in itertools.product(patterns, (0, hre.IGNORECASE)):
        mine, ref = hre.compile(pattern, flags), std.compile(pattern, flags)
        for subject in subjects:
            for pos, endpos in itertools.combinations(range(len(subject) + 1), 2):
                for name in ("search", "match", "fullmatch"):
                    expected = view(getattr(ref, name)(subject, pos, endpos))
                    assert view(getattr(mine, name)(subject, pos, endpos)) == expected, (pattern, subject, pos, endpos)
            assert mine.findall(subject, 3) == ref.findall(subject, 3)


def test_a_runtime_word_bounded_name_costs_no_unicode_class_program():
    subject = "u\u017cyj skill-name tutaj, prosz\u0119 \u2014 \u65e5\u672c\u8a9eskill-name\u65e5 skill-name"
    plain, word = hre.compile(hre.escape("skill-name")), hre.compile(r"\w")
    assert plain.search(subject) is not None and word.search(subject) is not None
    for flags in (0, hre.IGNORECASE):
        hre.purge()
        compiled = hre.compile(rf"\b{hre.escape('skill-name')}\b", flags)
        assert [m.span() for m in compiled.finditer(subject)] == [(5, 15), (47, 57)]  # the middle one touches letters
        # Programs built: the pattern as written and the wide form's two. None may hold the Unicode
        # `\w` class, which alone is a program many times the size of all three.
        assert plain._cost < compiled._cost < 4 * plain._cost and compiled._cost < word._cost
        assert compiled._wide[0].exact is None  # the cheap runner settled every neighbour itself
    hre.purge()


def test_the_exact_program_is_built_only_when_a_cheap_report_is_wrong():
    hre.purge()
    source = r"\babc[a-z]*?\b"
    compiled, ref = hre.compile(source), std.compile(source)
    assert compiled.search("\u2014abc\u2014").span() == (1, 4)  # non-ASCII neighbours that are not word characters
    assert compiled.search("\u2014abcd\u0301").span() == (1, 5)
    runner = compiled._wide[0]
    assert runner.exact is None
    cheap = compiled._cost
    subject = "\u2014abc\u00e9 abcx\u00e9d abcd\u2014"  # the cheap class takes the letter after "abc" for a boundary
    spans = [m.span() for m in compiled.finditer(subject)]
    assert spans == [m.span() for m in ref.finditer(subject)] and len(spans) == 1
    assert runner.exact is not None and compiled._cost > 4 * cheap
    assert hre.build(source) is compiled  # build() also proves that the exact program fits the budget
    hre.purge()


def test_build_reports_an_exact_program_over_the_budget_that_matching_would_meet_later():
    from hermes_re import _engine
    source, real = r"(?<!\w)abc[a-z]*?(?!\w)", _engine.MAX_MEM
    hre.purge()
    _engine.MAX_MEM = 20_000  # room for the cheap programs, not for the ones holding the Unicode `\w` class
    try:
        assert hre.compile(source).search(" abc ").span() == (1, 4)  # the cheap programs suffice here
        for call in (lambda: hre.build(source), lambda: hre.search(source, "\u00e9abc ")):
            try:
                call()
            except hre.error as exc:
                assert "too large" in exc.msg and exc.pattern == source
            else:
                raise AssertionError("the exact program fitted")
        assert [(p.pattern, "too large" in e.msg) for p, e in hre.build_all()] == [(source, True)]
    finally:
        _engine.MAX_MEM = real
        hre.purge()


def test_one_search_stays_linear_when_the_cheap_guard_is_refused_before_a_long_lookahead():
    # Each candidate's neighbour passes the cheap class and fails the exact test, and the trailing
    # lookahead reads to the end of the text: retrying one character further after each refusal
    # made ONE search quadratic (about 15 s here at this size). Stdlib takes milliseconds.
    n = 30_000
    cases = [
        (r"(?<=\w)a(?=.*z)", "\u2014a" * n + "z"),
        (r"(?<!\w)ab(?=[^z]*z)", "\u00e9ab" * n + "z"),
        (r"(?<=\d)a(?=(?:..)+$)", "\u2014a" * n),
    ]
    for source, subject in cases:
        compiled, ref = hre.compile(source), std.compile(source)
        began = time.monotonic()
        found = compiled.search(subject)
        spans = [m.span() for m in compiled.finditer(subject)]
        replaced = compiled.sub("", subject)
        elapsed = time.monotonic() - began
        assert found is None and spans == [] and replaced == subject == ref.sub("", subject), source
        assert ref.search(subject) is None
        assert elapsed < 2.0, (source, elapsed)
