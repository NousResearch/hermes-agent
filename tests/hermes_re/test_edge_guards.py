"""Edge guards: a one-character lookbehind at the start, a lookahead at the end, and `\\b` at an edge."""
import re as std

import hermes_re as hre
from tests.hermes_re._support import outcome, same_everywhere, view

SUBJECTS = ["", "*a* *b*", "*a**b*", "x*a*", "*a*x", "--tok --tok- a--tok", "\u00e9--tok --tok\u00e9", "foo foo,foo\nfoo",
            "a. x!  x", ".x", "x", "ab", "abab", "a\nb", "@hermes, Hermes\u00e9 @hermes: x@hermes", "k=1;k=2", "\u65e5a\u65e5",
            "aXb aYb", "foo\n", "a b\n", "1.) 2. 3)\n"]
GUARDED = [
    (r"(?<!\w)\*(\w+)\*(?!\w)", 0), (r"(?<![\w-])--tok\b", 0), (r"(?<=[.!])\s+x", 0), (r"(?<!a)b", 0), (r"a(?!b)", 0),
    (r"a(?=b)", 0), (r"foo(?=,|$)", 0), (r"foo(?=\s|$)", 8), (r"(?:^|(?<=[.!]))\s*x", 0), (r"(?:(?<=^)|(?<=[.!]))x", 0),
    (r"(?P<flag>(?<![\w-])--?t\w+)", 0), (r"(\w+)(?=\s*=)", 0), (r"(?<![\w@])@?hermes\b[,:\-]?", 2), (r"x(?!$)", 0),
    (r"x(?!\Z)", 0), (r"(?<!^)x", 0), (r"(?<!^)x", 8), (r"(?<=\d)[.)](?=\s)", 0), (r"a(?=(b)|c)", 0), (r"(?i)(?<!A)x(?!B)", 0),
    (r"(?<![^a])b(?![^a])", 0), (r"(?<!\w|-)tok", 0), (r"a(?=b\n?$)", 0), (r"(a(?!b))?", 0), (r"(?:a|b(?=a))", 0),
    (r"\w(?=\w$)", 0),
]
BOUNDARIES = [(r"\bfoo\b", 0), (r"\b\w+\b", 0), (r"\w+\b", 0), (r"\b\w+", 0), (r"\bfoo\b", 2), (r"hermes\b[,:\-]?", 2),
              (r"\b(?:tok|foo)\b", 0), (r"\b\w+$", 0), (r"^\w+\b", 8), (r"\b[^\W\d]\w*", 0), (r"\b\.\w+", 0), (r"\w+\.\b", 0)]
BOUNDARY_SUBJECTS = SUBJECTS + ["caf\u00e9!", "\u00e9foo foo\u00e9 foo", "na\u00efve caf\u00e9", "\u65e5foo\u65e5 foo", "foo\u0663 \u0663foo",
                                "a.b .c d.", "herme\u017f hermes\u0130", "foo\u0301 bar"]


def test_guarded_patterns_agree_with_stdlib():
    for pattern, flags in GUARDED:
        same_everywhere(pattern, flags, SUBJECTS)


def test_adjacent_matches_share_the_guard_character():
    # The character consumed as one match's trailing guard is the next match's leading guard.
    assert hre.findall(r"(?<!\w)\*(\w+)\*(?!\w)", "*a* *b*") == ["a", "b"]
    assert hre.findall(r"(?<=,)\w+(?=,)", ",a,b,c,") == ["a", "b", "c"]
    assert hre.sub(r"(?<![\w-])-(?![\w-])", "+", "- - -") == "+ + +"
    assert [m.span() for m in hre.finditer(r"(?<=a)b(?=a)", "ababa")] == [(1, 2), (3, 4)]
    assert hre.split(r"(?<=\d)(?=\D)", "1a2b") == std.split(r"(?<=\d)(?=\D)", "1a2b")


def test_guards_at_every_pos_and_endpos():
    for pattern, flags in GUARDED[:14]:
        mine, ref = hre.compile(pattern, flags), std.compile(pattern, flags)
        for text in ("*a* *b*", "x--tok\u00e9 --tok", "a. x!x", "foo,foo\n"):
            size = len(text)
            for i in range(size + 1):
                assert view(mine.search(text, i)) == view(ref.search(text, i)), (pattern, text, i)
                assert view(mine.match(text, i)) == view(ref.match(text, i)), (pattern, text, i)
                assert view(mine.search(text, 0, i)) == view(ref.search(text, 0, i)), (pattern, text, i)
                for j in range(i, size + 1):
                    assert view(mine.fullmatch(text, i, j)) == view(ref.fullmatch(text, i, j)), (pattern, text, i, j)


def test_guard_classes_use_the_exact_unicode_classes_and_folding():
    # U+1C89 (letter since Unicode 16) is `\w` for Python and not for RE2's own tables; U+212A folds to k.
    same_everywhere(r"(?<!\w)x(?!\w)", 0, ["\u1c89x", "x\u1c89", " x ", "\u0663x", "x\u0663"])
    same_everywhere(r"(?<![a-z])x(?![a-z])", 2, ["\u212ax", "x\u017f", "Ax", "xZ", " x "])
    same_everywhere(r"(?<!\s)x(?!\s)", 0, ["\x1cx", "x\x1f", "\u00a0x", "x\u3000", "ax"])
    same_everywhere(r"(?<!\d)x(?!\d)", 0, ["\u0663x", "x\uff11", "1x", "x "])


def test_word_boundary_at_a_pattern_edge_is_unicode_exact():
    for pattern, flags in BOUNDARIES:
        same_everywhere(pattern, flags, BOUNDARY_SUBJECTS)
    assert hre.findall(r"\b\w+\b", "na\u00efve caf\u00e9") == ["na\u00efve", "caf\u00e9"]
    assert hre.search(r"\bfoo", "\u00e9foo") is None


def test_word_boundary_in_the_interior_stays_ascii():
    # Documented limit: a `\b` that is not at a pattern edge keeps RE2's ASCII meaning.
    assert std.search(r"<link\b[^>]*>", "<link\u00df>") is None
    assert hre.search(r"<link\b[^>]*>", "<link\u00df>") is not None
    same_everywhere(r"<link\b[^>]*>", 0, ["<link rel=x>", "<linkx>", "<link>"])  # exact on ASCII text


def test_guards_on_bytes_patterns():
    same_everywhere(rb"(?<![a-z])x(?![a-z])", 0, [b"x", b"ax", b"xa", b" x ", b"\xe9x"])
    same_everywhere(rb"\w+(?=,)", 0, [b"a,b,c", b"abc"])



def test_negated_guard_class_keeps_its_literal_caret():
    # `[^^a]` excludes a literal `^` and `a`: respelled as the class of the characters the guard
    # lets through, the `^` must stay a member instead of negating that class.
    subjects = ["", "x", "^x", "ax", "bx", "x^", "xa", "xb", "a^x^a", "]x-", "\\x\\"]
    for pattern in (r"(?<![^^a])x", r"x(?![^^a])", r"(?<=[^^a])x", r"x(?=[^^a])", r"(?<![^^])x", r"x(?![^^\]-])",
                    r"a|(?<![^^a])x", r"x(?![^^a])|y", r"(?<![\^a])x", r"(?<!\^|a)x", r"x(?!\^|[\]-])"):
        for flags in (0, std.IGNORECASE, std.ASCII):
            same_everywhere(pattern, flags, subjects)
            same_everywhere(pattern.encode(), flags, [s.encode() for s in subjects])
    assert hre.findall(r"(?<![^^a])x", "ax bx ^x") == ["x", "x"]

def test_unsupported_lookarounds_raise_error_at_compile():
    for pattern in (r"a(?=b)c", r"a(?<!b)c", r"(?<=ab)c", r"(?<!a)(?<!b)x", r"x(?!ab)", r"(?=a)b", r"(?:(?<!x)a)+",
                    r"a(?!.)", r"(?<!.)x", r"((?<!a)x)?y", r"x(?!y)*", r"^(?<!a)x", r"(?<=\b)x", r"(a)\1", r"a++", r"(?>a)",
                    r"(?P<n>a)(?P=n)", r"(a)?(?(1)b|c)"):
        assert outcome(lambda: hre.compile(pattern)) == "raises error", pattern
