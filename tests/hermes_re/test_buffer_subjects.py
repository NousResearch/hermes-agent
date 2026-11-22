"""A bytearray / memoryview / mmap subject gives the results stdlib gives: bytes, never the buffer type."""
import array
import mmap
import re as std

import pytest

import hermes_re as hre

_SUBJECT = b"xa yb a"
_CALLS = {
    "sub": lambda r, s: r.sub(rb"(a)", rb"[\1]", s),
    "sub_no_match": lambda r, s: r.sub(rb"q", b"", s),
    "sub_callback": lambda r, s: r.sub(rb"a", lambda m: bytearray(m.group() * 2), s),
    "subn": lambda r, s: r.subn(rb"a", b"b", s),
    "split": lambda r, s: r.split(rb"(a)|(q)", s),
    "findall": lambda r, s: r.findall(rb"a", s),
    "findall_groups": lambda r, s: r.findall(rb"(a)|(y)", s),
    "findall_endpos": lambda r, s: r.compile(rb"(a)").findall(s, 0, 3),
    "finditer_pos": lambda r, s: [m.group() for m in r.compile(rb".").finditer(s, 1, 4)],
    "group": lambda r, s: r.search(rb"(a)", s).group(0, 1),
    "getitem": lambda r, s: r.search(rb"(a)", s)[1],
    "groups": lambda r, s: r.search(rb"(a)(z)?", s).groups(),
    "groupdict": lambda r, s: r.search(rb"(?P<q>a)", s).groupdict(),
    "expand": lambda r, s: r.search(rb"(a)", s).expand(rb"<\1>"),
}


@pytest.fixture
def subjects(tmp_path):
    path = tmp_path / "subject"
    path.write_bytes(_SUBJECT)
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        yield {"bytearray": bytearray(_SUBJECT), "memoryview": memoryview(_SUBJECT), "mmap": mapped}


@pytest.mark.parametrize("call", sorted(_CALLS))
def test_a_bytes_like_subject_gives_stdlibs_bytes_results(call, subjects):
    for kind, subject in subjects.items():
        expected = _CALLS[call](std, subject)
        got = _CALLS[call](hre, subject)
        assert got == expected and repr(got) == repr(expected), (kind, got, expected)


_TYPED_SUBJECT = b"xa yb a\xffbaab"  # 12 bytes: whole items for "H" and "I"
_TYPED = {
    "cast_I": lambda: memoryview(_TYPED_SUBJECT).cast("I"),
    "cast_H": lambda: memoryview(_TYPED_SUBJECT).cast("H"),
    "cast_I_slice": lambda: memoryview(_TYPED_SUBJECT).cast("I")[1:],
    "two_dimensional": lambda: memoryview(_TYPED_SUBJECT).cast("B", (3, 4)),
    "array_I": lambda: array.array("I", _TYPED_SUBJECT),
}


def _bounds(m, subject):
    return m.span(1), m.pos, m.endpos, m.string is subject


_TYPED_CALLS = {
    **_CALLS,
    "span_and_bounds": lambda r, s: _bounds(r.search(rb"b(a)", s), s),
    "search_endpos": lambda r, s: r.compile(rb"(b)").search(s, 1, 6).span(),
    "search_last_byte": lambda r, s: r.search(rb"b$", s).span(),
    "fullmatch": lambda r, s: r.fullmatch(rb"(?s).*", s).group(),
    "finditer_spans": lambda r, s: [m.span() for m in r.finditer(rb"a", s)],
    "split_bytes": lambda r, s: r.split(rb"\xff", s),
}


@pytest.mark.parametrize("kind", sorted(_TYPED))
@pytest.mark.parametrize("call", sorted(_TYPED_CALLS))
def test_a_typed_view_is_scanned_by_byte_as_stdlib_does(kind, call):
    # stdlib reads any bytes-like subject as its bytes: offsets, limits and results count bytes,
    # not items, while Match.string stays the caller's object
    subject = _TYPED[kind]()
    expected = _TYPED_CALLS[call](std, subject)
    got = _TYPED_CALLS[call](hre, subject)
    assert got == expected and repr(got) == repr(expected), (kind, got, expected)


def test_a_non_contiguous_view_is_refused_as_stdlib_refuses_it():
    subject = memoryview(_TYPED_SUBJECT)[::2]
    with pytest.raises(TypeError):
        std.search(rb"a", subject)
    with pytest.raises(TypeError):
        hre.search(rb"a", subject)


def test_a_match_does_not_pin_the_subject_buffer():
    # stdlib's Match holds the object, not a buffer export: the bytearray stays resizable
    subject = bytearray(b"xa yb a")
    found = hre.compile(rb"(a)").search(subject, 0, 4)
    subject.extend(b"zz")
    assert found.group(1) == b"a"
