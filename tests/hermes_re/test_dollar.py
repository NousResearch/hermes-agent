"""Non-multiline `$`: zero-width, at the end or before one final newline (as stdlib)."""
import re as std

import hermes_re as hre
from tests.hermes_re._support import outcome, same_everywhere

SUBJECTS = ["", "x", "x\n", "x\n\n", "ax\nb", "k=v\n", "a\nx", "xx x\n", "\n", "y\n", "x\r\n", "x \n", "\u00e9x\n", "x\u2028"]
PATTERNS = [r"x$", r"$", r"^$", r"\s*$", r"\w+$", r"^(\w+)=(.*)$", r"(x$)", r"(?P<e>$)", r"(?:x|$)", r"(?:(x)|$)",
            r"(?:x$|(y)$|z)", r"(?:(x)$)?", r"(a$)?", r"\n?$", r"(\n)$", r"x\b$", r"(?i)X$", r"(?:foo|x$)", r"[^\n]*$",
            r"(?-m:x$)", r"x{1,2}$", r"(?:x$){1}", r"(?:x$){0,1}"]
MIXED = [(r"(?m:^a$)|b$", 0), (r"x$", 8), (r"(?m)x$", 0), (r"^x$|y$", 8), (r"(?s).$", 0), (r"x$", 16)]


def test_tail_dollar_agrees_with_stdlib():
    for pattern in PATTERNS:
        same_everywhere(pattern, 0, SUBJECTS)


def test_dollar_with_multiline_and_scoped_flags():
    for pattern, flags in MIXED:
        same_everywhere(pattern, flags, SUBJECTS + ["a\nb\n", "b", "a\n"])


def test_dollar_is_zero_width_in_sub_split_and_spans():
    assert hre.sub(r"x$", r"<\g<0>>", "x\n") == "<x>\n"
    assert hre.search(r"x$", "x\n").span() == (0, 1)
    assert hre.split(r"x$", "ax\n") == ["a", "\n"]
    assert hre.findall(r"$", "a\n") == std.findall(r"$", "a\n") == ["", ""]
    assert hre.search(r"(\w+)$", "ab\n").span(1) == (0, 2)


def test_fullmatch_treats_dollar_as_the_very_end():
    for pattern, text in ((r"x$", "x\n"), (r"x$", "x"), (r"x\n?$", "x\n"), (r"(x$)?", "x\n"), (r"x$|x\n", "x\n")):
        assert outcome(lambda: bool(hre.fullmatch(pattern, text))) == outcome(lambda: bool(std.fullmatch(pattern, text))), (pattern, text)


def test_dollar_with_endpos_sees_endpos_as_the_end():
    for pattern in (r"x$", r"\w+$"):
        mine, ref = hre.compile(pattern), std.compile(pattern)
        for text in ("ax\nb", "xx\n\nx", "\u00e9x\nx"):
            for end in range(len(text) + 1):
                a, b = ref.search(text, 0, end), mine.search(text, 0, end)
                assert (a and a.span()) == (b and b.span()), (pattern, text, end)


def test_dollar_that_is_not_at_the_tail_is_rejected_at_compile():
    for pattern in (r"x$y", r"(x$)+", r"(?:a$)b", r"$a", r"(?:$)+", r"(?:x$){2}"):
        assert outcome(lambda: hre.compile(pattern)) == "raises error", pattern
    try:
        hre.compile(r"x$y")
    except hre.error as exc:
        assert "tail position" in exc.msg and exc.pattern == r"x$y"
