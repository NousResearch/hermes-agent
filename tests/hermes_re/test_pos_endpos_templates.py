"""`pos` / `endpos`, offsets on non-ASCII text, and replacement templates."""
import re as std

import hermes_re as hre
from tests.hermes_re._support import outcome, view

TEXTS = ["a1 b22 c333", "caf\u00e9 \u65e5\u672c 42 x", "\U0001f600b\U0001f600 7", "x\ny\n", "a\ud800b 1", "\u00e9" * 1500 + " 12 ab"]
PATTERNS = [r"\d+", r"\w+", r"^\w", r"\w$", r"\b\w", r"(\w)(\d)?", r"\s+", r"(?<!\w)\w+", r"\w+(?!\w)"]


def test_search_match_fullmatch_at_positions():
    for source in PATTERNS:
        mine, ref = hre.compile(source), std.compile(source)
        for text in TEXTS:
            size = len(text)
            spots = sorted({0, 1, 2, 3, size // 2, size - 3, size - 1, size, size + 5, -2} & set(range(-2, size + 6)))
            for pos in spots:
                for end in spots:
                    if pos > end:
                        continue  # stdlib is inconsistent there; see test_known_differences
                    for name in ("search", "match", "fullmatch"):
                        a, b = getattr(ref, name)(text, pos, end), getattr(mine, name)(text, pos, end)
                        assert view(a) == view(b), (source, text[:20], pos, end, name)
                        if a is not None:
                            assert (a.pos, a.endpos, a.regs) == (b.pos, b.endpos, b.regs)


def test_iteration_with_pos_and_endpos():
    for source in PATTERNS:
        mine, ref = hre.compile(source), std.compile(source)
        for text in TEXTS:
            for pos, end in ((0, len(text)), (2, len(text) - 1), (3, 9), (5, 5)):
                assert mine.findall(text, pos, end) == ref.findall(text, pos, end), (source, text[:20], pos, end)
                assert [m.span() for m in mine.finditer(text, pos, end)] == [m.span() for m in ref.finditer(text, pos, end)]


def test_endpos_ends_the_subject_for_anchors_and_boundaries():
    assert hre.compile(r"\w+$").search("ab cd", 0, 2).span() == (0, 2)
    assert hre.compile(r"a\b").search("ab", 0, 1).span() == (0, 1)
    assert hre.compile(r"a(?!b)").search("ab", 0, 1).span() == (0, 1)
    assert hre.compile(r"^b", hre.M).search("ab", 1) is None  # `^` still sees the text before pos


def test_spans_are_character_offsets_on_non_ascii_text():
    text = "\u00e9\u65e5\U0001f600 abc \u00e9"
    assert hre.search("abc", text).span() == (4, 7)
    assert [m.span() for m in hre.finditer(r"\S+", text)] == [m.span() for m in std.finditer(r"\S+", text)]
    spans = [m for m in hre.finditer(r"\w", "\u00e9" * 3000)]
    assert [m.start() for m in reversed(spans)] == list(range(2999, -1, -1))  # any order, any distance


def test_templates_agree_with_stdlib():
    cases = [(r"(a)(b)?", r"<\1|\2>", "ab a"), (r"(?P<x>a)", r"[\g<x>]", "a"), (r"a", "\u00e9\u2192\u65e5\U0001f600", "a"),
             (r"a", r"x\ny\tz\\", "a"), (r"(a)", r"\g<1>0", "a"), (r"a", r"\0", "a"), (r"(a)", "\\1\u2022 ", "a"),
             (r"a", r"<\g<0>>", "a"), (r"(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)", r"\10-\1", "abcdefghij"), (r"a", r"\101", "a"),
             (r"a", r"\g<zz>", "a"), (r"a", r"\d", "a"), (r"(a)", r"\2", "a"), (r"a", "x\\", "a"), (r"a", r"\x41", "a"),
             (rb"(a)", rb"<\1>", b"a"), (r"(\w)$", r"[\1]", "ab\n"), (r"(?<!\w)(\w)", r"\1\1", "ab cd"), (r"x", "", "axb")]
    for pattern, template, text in cases:
        assert outcome(lambda: hre.sub(pattern, template, text)) == outcome(lambda: std.sub(pattern, template, text)), (pattern, template)
        assert outcome(lambda: hre.subn(pattern, template, text)) == outcome(lambda: std.subn(pattern, template, text))


def test_bad_group_names_in_a_template_raise_like_stdlib():
    # A malformed name is `error`, a well-formed name no group has is IndexError, a number too large is `error`.
    for template in (r"\g<a-b>", r"\g<1a>", r"\g<>", r"\g<a b>", r"\g<zz>", r"\g<9>"):
        assert outcome(lambda: hre.sub("(a)", template, "a")) == outcome(lambda: std.sub("(a)", template, "a")), template


def test_callable_replacement_count_and_maxsplit():
    assert hre.sub(r"\d", lambda m: str(int(m.group()) + 1), "a1b2") == "a2b3"
    assert hre.sub(r"(\w)(?!\w)", lambda m: m.group(1).upper(), "ab cd\u00e9") == "aB cd\u00c9"
    assert hre.sub("a", "b", "aaa", count=2) == "bba" and hre.subn("a", "b", "aaa", 2) == ("bba", 2)
    assert hre.split(",", "a,b,c", maxsplit=1) == ["a", "b,c"]
    assert hre.split(r"(,)|(;)", "a,b;c") == std.split(r"(,)|(;)", "a,b;c")
    assert hre.search(r"(?P<n>\w)(\d)?", "a").expand(r"\g<n>-\2-") == "a--"
