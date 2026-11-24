"""Word boundaries: which are exact, and `inexact_boundaries()` that names the ones that are not.

The facade matches a `\\b` exactly as stdlib does when it can decide it from its neighbours or
respell it as an edge guard on `\\w`. Every other `\\b`, and every `\\B`, is left to RE2's ASCII
rule: equal to stdlib on ASCII text, possibly different next to a non-ASCII letter or digit.
`hermes_re.inexact_boundaries(pattern, flags)` lists those by offset. The contract tested here:
replacing exactly the listed boundaries by their ASCII form in the STDLIB reference makes stdlib
agree with the facade on every subject; an empty list therefore means "equal to stdlib".
"""
import itertools
import re as std
import sys

import hermes_re as hre
from tests.hermes_re._support import view

ALPHABET = ["a", "\u00e9", "-", " ", "\u0663", "."]  # incl. a non-ASCII letter and a non-ASCII digit
SUBJECTS = ["".join(parts) for size in range(4) for parts in itertools.product(ALPHABET, repeat=size)]
OLD_NOT_BOUNDARY = sys.version_info < (3, 14)
SINGLE = (lambda p, s: view(p.search(s)), lambda p, s: view(p.match(s)), lambda p, s: view(p.fullmatch(s)))
ITERATING = (lambda p, s: p.findall(s), lambda p, s: p.sub("<>", s), lambda p, s: p.split(s))


def _ascii_at(source, offsets):
    """``source`` with the boundaries at ``offsets`` spelled as ASCII-only boundaries (stdlib syntax)."""
    out, last = [], 0
    for offset in offsets:
        assert source[offset:offset + 2] in (r"\b", r"\B"), (source, offset)
        out.append(source[last:offset] + "(?a:" + source[offset:offset + 2] + ")")
        last = offset + 2
    return "".join(out) + source[last:]


def _same(mine, ref, text):
    """Every single-match operation; the iterating ones only where stdlib finds no EMPTY match
    (after an empty match the two engines continue differently: a known difference)."""
    if any(call(mine, text) != call(ref, text) for call in SINGLE):
        return False
    spans = [m.span() for m in ref.finditer(text)]
    if any(start == end for start, end in spans):
        return True
    return [m.span() for m in mine.finditer(text)] == spans and all(call(mine, text) == call(ref, text) for call in ITERATING)


def _agrees(source, flags, reference, subjects):
    mine, ref = hre.compile(source, flags), std.compile(reference, flags)
    if OLD_NOT_BOUNDARY and r"\B" in source:  # before Python 3.14 stdlib's `\B` never matched in an empty subject
        subjects = [text for text in subjects if text]
    return all(_same(mine, ref, text) for text in subjects)


def test_the_known_gap_is_real_and_is_reported():
    # An edge `\b` next to a class that mixes word and non-word characters cannot be respelled as
    # one guard (it needs a lookbehind AND a lookahead at the same edge): it stays ASCII.
    lead, trail = r"\b[\w./-]+", r"@[A-Za-z0-9-]{3,30}\b"
    assert std.findall(lead, "\u00e9./a") == ["\u00e9./a"] and hre.findall(lead, "\u00e9./a") == ["a"]
    assert std.findall(lead, "-\u00e9 x") == ["\u00e9", "x"] and hre.findall(lead, "-\u00e9 x") == ["x"]
    # at the end the facade matches LESS than stdlib: `-` then a non-ASCII letter is a boundary only for stdlib
    assert std.findall(trail, "@ab-\u00e9") == ["@ab-"] and hre.findall(trail, "@ab-\u00e9") == []
    assert hre.inexact_boundaries(lead) == [0] and hre.inexact_boundaries(trail) == [len(trail) - 2]
    assert hre.findall(lead, "x ./a b") == std.findall(lead, "x ./a b")  # ASCII text: equal


def test_which_boundaries_are_listed():
    listed = {
        r"plain": [], r"\bfoo\b": [], r"\b\w+\b": [], r"(?<![\w.])x\b": [], r"\bfoo\b|\bbar\b": [], r"foo\b\s*=": [],
        r"x\b.": [1], r"\Bfoo": [0], r"<link\b[^>]*>": [5], r"\b[\w./-]+": [0], r"[A-Za-z0-9-]{3,30}\b": [18],
        r"(?:\bab)+": [3], r"a(?=\s+import\b)": [13], r"\b\w+\b(?!-)": [5], r"x\b.\By\b": [1, 4],
    }
    for source, offsets in listed.items():
        assert hre.inexact_boundaries(source) == offsets, source
        assert hre.inexact_boundaries(hre.compile(source)) == offsets, source
    assert hre.inexact_boundaries(r"x\b.", hre.ASCII) == []      # stdlib's rule is ASCII too
    assert hre.inexact_boundaries(rb"x\b.") == []                # bytes patterns as well
    assert hre.inexact_boundaries(r"[\b]x\\b") == []             # a backspace in a class, an escaped backslash
    assert hre.inexact_boundaries(std.compile(r"x\b.")) == [1]   # a compiled stdlib pattern is accepted
    try:
        hre.inexact_boundaries(r"a(?=b)c\b")
    except hre.error:
        pass
    else:
        raise AssertionError("an invalid pattern must raise")


def test_listed_boundaries_are_exactly_what_separates_the_facade_from_stdlib():
    heads = ["", r"\b", r"\B", r"(?<!-)", r"(?:^|-)"]
    bodies = [r"a", r"\w+", r"[\w.-]+", r"[a.]", r"a\b.", r"\w\b-?", r"(?:a|-)", r"a\Ba", r"(?:\b\w)+", r"\w*", r"[^-]", r"1\b_?"]
    tails = ["", r"\b", r"\B", r"\b-", r"(?!\w)", r"$", r"\b(?=\s|$)", r"\b.*"]
    exact = inexact = differing = 0
    for head, body, tail in itertools.product(heads, bodies, tails):
        source = head + body + tail
        for flags in (0, hre.I):
            offsets = hre.inexact_boundaries(source, flags)
            assert _agrees(source, flags, _ascii_at(source, offsets), SUBJECTS), (source, int(flags), offsets)
            if offsets:
                inexact += 1
                differing += not _agrees(source, flags, source, SUBJECTS)
            else:
                exact += 1
    assert exact > 150 and inexact > 300, (exact, inexact)
    assert differing > 200, differing  # the list is not padding: most listed patterns really differ from stdlib


def test_ascii_subjects_never_differ_whatever_is_listed():
    ascii_subjects = [text for text in SUBJECTS if text.isascii()]
    for source in (r"x\b.", r"\Ba", r"\b[\w./-]+", r"[a1-]{1,3}\b", r"a\b.\Ba", r"(?:\ba)+"):
        assert hre.inexact_boundaries(source)
        assert _agrees(source, 0, source, ascii_subjects), source


def test_a_not_boundary_never_matches_inside_a_character():
    # RE2's `\\B` holds between the two bytes of one UTF-8 character; such an empty match is skipped.
    texts = ("a\u00e9a", "\u00e9", "\u00e9\u00e9-", "\u65e5\u672c a", "\U0001f600x\u00e9", "a\u00e9\n")
    for source, flags in ((r"\w*\B", 0), (r"\B", 0), (r"(?:\B)|x", 0), (r"\B(?:-)?", 0), (r"\B", hre.A), (r"\B$", 0)):
        mine = hre.compile(source, flags)
        ref = std.compile(_ascii_at(source, hre.inexact_boundaries(source, flags)), flags)
        for text in texts:
            assert all(call(mine, text) == call(ref, text) for call in SINGLE), (source, text)
            found = [m.span() for m in mine.finditer(text)]
            assert all(0 <= a <= e <= len(text) for a, e in found), (source, text, found)
            assert set(found) <= {m.span() for m in ref.finditer(text)}, (source, text, found)
            assert isinstance(mine.sub("<>", text), str) and isinstance(mine.split(text), list)  # no torn character
            assert "".join(mine.split(text)) + "".join(mine.findall(text)) != "\ufffd"
