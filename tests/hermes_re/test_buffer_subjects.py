"""A bytearray / memoryview / mmap subject gives the results stdlib gives: bytes, never the buffer type."""
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
