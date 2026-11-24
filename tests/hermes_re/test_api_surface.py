"""The module surface: what `import hermes_re as re` must provide."""
import copy
import pickle
import re as std

import hermes_re as hre


def test_all_lists_the_stdlib_names_the_code_base_uses():
    expected = {"compile", "search", "match", "fullmatch", "split", "findall", "finditer", "sub", "subn", "escape",
                "purge", "error", "PatternError", "Pattern", "Match", "RegexFlag", "NOFLAG",
                "A", "ASCII", "I", "IGNORECASE", "M", "MULTILINE", "S", "DOTALL", "U", "UNICODE",
                "X", "VERBOSE", "L", "LOCALE", "DEBUG"}
    extra = {"build", "build_all", "inexact_boundaries"}  # the facade's own: forced build, `\\b` introspection
    assert set(hre.__all__) == expected | extra
    assert all(hasattr(hre, name) for name in hre.__all__)


def test_flag_constants_have_stdlib_integer_values():
    for name in ("I", "IGNORECASE", "M", "MULTILINE", "S", "DOTALL", "U", "UNICODE", "A", "ASCII",
                 "X", "VERBOSE", "L", "LOCALE", "DEBUG", "NOFLAG"):
        assert int(getattr(hre, name)) == int(getattr(std, name)), name
    assert hre.I | hre.M == 10
    assert hre.compile("a", 2).flags & hre.IGNORECASE


def test_pattern_attributes_match_stdlib():
    source = r"(?P<key>\w+)=(?P<value>\d+)?(;)?"
    mine, ref = hre.compile(source, hre.I | hre.M), std.compile(source, std.IGNORECASE | std.MULTILINE)
    assert mine.pattern == ref.pattern
    assert int(mine.flags) == int(ref.flags)  # includes the UNICODE bit of a str pattern
    assert mine.groups == ref.groups == 3
    assert dict(mine.groupindex) == dict(ref.groupindex)
    assert list(mine.groupindex) == list(ref.groupindex)  # group-number order
    assert int(hre.compile(b"a").flags) == int(std.compile(b"a").flags) == 0
    assert int(hre.compile("a", hre.A).flags) == int(std.compile("a", std.ASCII).flags)


def test_match_attributes_match_stdlib():
    source, text = r"(?P<key>\w+)=(?P<value>\d+)?(;)?", "x k=12 y"
    mine, ref = hre.compile(source).search(text, 1, 7), std.compile(source).search(text, 1, 7)
    for read in (lambda m: m.group(), lambda m: m.group(1, "value"), lambda m: m["key"], lambda m: m[2],
                 lambda m: m.groups(), lambda m: m.groups("-"), lambda m: m.groupdict(), lambda m: m.groupdict("-"),
                 lambda m: m.span(), lambda m: m.span("value"), lambda m: m.start(3), lambda m: m.end(3),
                 lambda m: m.regs, lambda m: m.lastindex, lambda m: m.lastgroup, lambda m: m.pos, lambda m: m.endpos,
                 lambda m: m.string, lambda m: m.expand(r"\g<key>:\2")):
        assert read(mine) == read(ref)
    assert mine.re is hre.compile(source)


def test_unknown_group_raises_index_error():
    m = hre.search("(a)", "a")
    for bad in (2, -1, "nope"):
        try:
            m.group(bad)
        except IndexError:
            continue
        raise AssertionError(f"group({bad!r}) did not raise IndexError")


def test_annotations_can_subscript_pattern_and_match():
    assert hre.Pattern[str] is not None and hre.Match[bytes] is not None
    assert isinstance(hre.compile("a"), hre.Pattern)
    assert isinstance(hre.search("a", "a"), hre.Match)


def test_repr_looks_like_stdlib():
    assert repr(hre.compile("a+", hre.I | hre.S)) == repr(std.compile("a+", std.IGNORECASE | std.DOTALL))
    assert repr(hre.search("b+", "abb")) == repr(std.search("b+", "abb"))


def test_pickle_and_copy_return_the_cached_pattern():
    compiled = hre.compile(r"(\w+)$", hre.M)
    assert pickle.loads(pickle.dumps(compiled)) is compiled
    assert copy.copy(compiled) is compiled and copy.deepcopy(compiled) is compiled


def test_patterns_compare_by_pattern_and_flags():
    first = hre.compile("ab")
    hre.purge()
    second = hre.compile("ab")
    assert first is not second and first == second and hash(first) == hash(second)
    assert first != hre.compile("ab", hre.I) and first != hre.compile(b"ab")


def test_type_errors_match_stdlib():
    for call in (lambda r: r.compile("a").search(b"a"), lambda r: r.compile(b"a").search("a"),
                 lambda r: r.compile("a").search(None), lambda r: r.compile("a").finditer(b"a"),
                 lambda r: r.compile(123), lambda r: r.compile(None)):
        for module in (std, hre):
            try:
                call(module)
            except TypeError:
                continue
            raise AssertionError(f"{module.__name__} did not raise TypeError")


def test_escape_round_trips_through_the_facade():
    for text in ("a.b*c", "[x]{2}", "é/#~ &", "tab\there", "line\nbreak", "nul\x00", "-^$|()?+\\"):
        assert hre.fullmatch(hre.escape(text), text) is not None, text
        assert std.fullmatch(hre.escape(text), text) is not None, text
    assert hre.escape(b"a.b") == std.escape(b"a.b")


def test_bytes_patterns_match_byte_wise():
    assert hre.search(rb"a.b", b"a\xffb").span() == (0, 3)
    assert hre.findall(rb"\w+", b"caf\xc3\xa9 x") == std.findall(rb"\w+", b"caf\xc3\xa9 x")
    assert hre.sub(rb"(\d)", rb"<\1>", b"a1") == b"a<1>"
