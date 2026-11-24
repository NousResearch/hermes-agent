"""Fixed differential table: every operation of the facade against stdlib `re`."""
import hermes_re as hre
from tests.hermes_re._support import same_everywhere

SUBJECTS = ["", "foo", "foo\n", "foo\n\n", "Foo BAR baz", "a1 b22 c333", "caf\u00e9 \u0663\u0664 \u65e5\u672c\u8a9e x_y",
            "a\u00a0b\u3000c\x0bd\x85e", "line1\nline2\r\nline3", "key=value; other = 'q v'", "\U0001f600b\U0001f600",
            "tab\there", "\u00c0\u00c9\u00ce \u00e0\u00e9\u00ee \u01c5", "https://x.io/a?b=c#d", "  padded  ",
            "\u0131 \u0130 K \u212a \u017f s", "x" * 1100 + "\u00e9 end\n"]
PATTERNS = [
    (r"\d+", 0), (r"\w+", 0), (r"\s+", 0), (r"\S+", 0), (r"\W+", 0), (r"\D+", 0), (r"[\w.-]+", 0), (r"[^\s=]+", 0),
    (r"[^\w\s]", 0), (r"\w+", 256), (r"\s+", 256), (r"\d+", 256), (r"^\w+", 8), (r"\w+$", 8), (r"^.*$", 8),
    (r".+", 16), (r".+", 0), (r"foo", 2), (r"[a-z]+", 2), (r"(?i)bar", 0), (r"(?P<k>\w+)\s*=\s*(?P<v>\S+)", 0),
    (r"(a)|(b)", 0), (r"(\w)(\d)?", 0), (r"x*", 0), (r"k", 2), (r"s", 2), (r"i", 2), ("\u00e9", 2),
    (r"[\d\s]+", 0), (r"[\]\w]+", 0), (r"[]\w]+", 0), (r"[^]\d]+", 0), (r"\\d", 0), (r"\[\w\]", 0),
    (r"(?s:.)+", 0), (r"(?m:^l\w+)", 0), (r"\Aa", 0), (r"[\s\S]+", 0), (r"[\w\W]", 0), (r"\.", 0),
    (r"\w+$", 0), (r"^(\w+)=(.*)$", 0), (r"\s*$", 0), (r"(a)(b?)", 0), (r"(?:x|(y))\Z", 0), (r"\bfoo\b", 0),
    (r"\b\w+\b", 0), (r"(?<![\w-])\w+", 0), (r"\w+(?![\w=])", 0), (r"\x41|\u00e9|\101", 2), (r"[[:alpha:]]", 0),
    # nested groups: `lastindex` is the group whose `)` closes last, not the highest number that took part
    (r"((\w)(\d)?)", 0), (r"(?P<outer>(?P<inner>\w)\s?)+", 0), (r"((a)|(b))(c)?", 0), (r"(\w(\w(\w)))$", 0),
]
BYTES_SUBJECTS = [b"", b"foo", b"foo\n", b"a1 b22\tc333", b"caf\xc3\xa9 x_y", b"\xff\x00\x0b", b"K k S s"]
BYTES_PATTERNS = [(rb"\d+", 0), (rb"\w+", 0), (rb"\s+", 0), (rb"[^\s]+$", 0), (rb"k", 2), (rb"(?P<a>\w)(\d)?", 0),
                  (rb"^\w+", 8), (rb".", 16), (rb"\xff|\x00", 0), (rb"(?<!\w)\w", 0)]


def test_str_patterns_agree_with_stdlib_on_every_operation():
    checks = sum(same_everywhere(pattern, flags, SUBJECTS) for pattern, flags in PATTERNS)
    assert checks == len(PATTERNS) * len(SUBJECTS) * 8


def test_bytes_patterns_agree_with_stdlib_on_every_operation():
    checks = sum(same_everywhere(pattern, flags, BYTES_SUBJECTS) for pattern, flags in BYTES_PATTERNS)
    assert checks == len(BYTES_PATTERNS) * len(BYTES_SUBJECTS) * 8


def test_lone_surrogate_in_the_subject_keeps_offsets_and_text():
    # The subject is scanned through a stand-in; spans and group text must still be the caller's.
    same_everywhere(r"b", 0, ["a\ud800b", "\ud800", "x\udfffbb\n"])
    same_everywhere(r"a(.)b", 16, ["a\ud800b"])
    same_everywhere(r"\w+", 0, ["ab\ud800cd \u00e9"])


def test_open_lower_bound_repeat_means_zero_to_n_as_in_stdlib():
    # RE2 reads `{,n}` as literal text; the translator writes it as `{0,n}`.
    subjects = ["", "a", "aa", "aaaa", "a{,3}", "ab", "b", "aab,"]
    for pattern in (r"a{,3}", r"a{,3}b", r"(?:ab){,2}", r"[ab]{,2}$", r"a{,}b", r"a{,3}?b", r"\w{,2},"):
        same_everywhere(pattern, 0, subjects)
    same_everywhere(rb"a{,3}b", 0, [b"", b"aab", b"a{,3}b"])
    same_everywhere(r"a{,x}", 0, subjects + ["a{,x}"])  # not a repeat in either engine


def test_a_plain_class_with_a_lone_surrogate_compiles_like_the_escaped_spelling():
    # A surrogate named in a pattern never matches (accepted difference, see test_known_differences); the
    # literal spelling must behave like the escaped one whatever the flags are.
    for pattern in ("[\ud800-\udfff]", "[a\udc00]", r"[\ud800-\udfff]"):
        assert hre.build(pattern).search("\ud83d") is None
    assert hre.findall("[a\udc00]", "xa\udc00") == ["a"]
