"""Every public call takes the argument types stdlib takes, with stdlib's result or exception type."""
import array
import copy
import pickle
import re as std
import sys

import pytest

import hermes_re as hre

from ._support import outcome


def both(call):
    return outcome(lambda: call(std)), outcome(lambda: call(hre))


def _mv_sub(r):
    return r.sub(rb"(a)", memoryview(rb"<\1>"), b"xa")


@pytest.mark.parametrize("call", [
    lambda r: r.sub(rb"(a)", bytearray(b"x"), b"ya"),
    lambda r: r.sub(rb"(a)", bytearray(rb"<\1>"), bytearray(b"ya")),
    _mv_sub,
    lambda r: r.subn(rb"(a)", memoryview(b"z<\\1>")[1:], b"aa"),
    lambda r: r.sub(rb"(a)", array.array("B", rb"<\1>"), b"a"),
    lambda r: r.sub(rb"(a)", memoryview(b"<\\1>\\1\\1").cast("I"), b"a"),  # ValueError: unhashable view
    lambda r: r.sub(rb"(a)", memoryview(bytearray(rb"\1")), b"a"),  # ValueError: writable view
    lambda r: r.sub(rb"(a)", memoryview(b"xxxx").cast("I"), b"a"),  # no backslash: inserted as it is
    lambda r: r.sub("(a)", bytearray(rb"\q"), "a"),  # template error before the type error
    lambda r: r.sub("(a)", memoryview(rb"\1"), "a"),
    lambda r: r.sub(rb"(a)", memoryview(rb"\9"), b"a"),
    lambda r: r.search(rb"(a)", b"a").expand(memoryview(rb"[\1]")),
    lambda r: r.search("(a)", "a").expand(memoryview(b"x")),  # stdlib returns bytes here
    lambda r: r.search("(a)", "a").expand(1),
    lambda r: r.sub("a", 1, "a"),
], ids=lambda c: "")
def test_buffer_replacements(call):
    ref, mine = both(call)
    assert mine == ref


@pytest.mark.parametrize("value", [memoryview(b"a.b"), memoryview(b"xa.b")[1:], bytearray(b"a.b"),
                                   array.array("B", b"a.b*"), memoryview(b"a.b*").cast("I"),
                                   memoryview(b"a.b.")[::2], 1, None, "a.b"])
def test_escape_accepts_what_stdlib_accepts(value):
    ref, mine = both(lambda r: r.escape(value))
    assert mine == ref


@pytest.mark.parametrize("value", [True, 2.0, "1", None, -10**30, 10**30, sys.maxsize + 1, -sys.maxsize - 1])
def test_pos_endpos_count_maxsplit_are_c_ssize_t(value):
    for call in (lambda r: r.compile("a").search("xa", value), lambda r: r.compile("a").findall("xa", 0, value),
                 lambda r: r.compile(b"a").match(b"xa", endpos=value), lambda r: list(r.compile("a").finditer("aa", value)),
                 lambda r: r.split("a", "xaxa", value), lambda r: r.sub("a", "", "aaa", count=value),
                 lambda r: r.compile("a").search("xa", value).pos if value is True else None):
        ref, mine = both(call)
        assert repr(mine) == repr(ref)


@pytest.mark.parametrize("flags", [1, 512, True, 2.0, None, "i", 1 << 40, -1, sys.maxsize])
def test_flag_values(flags):
    for call in (lambda r: r.compile("a", flags).flags, lambda r: r.search("a", "a", flags) is not None):
        ref, mine = both(call)
        assert mine == ref


def test_pattern_flags_report_leading_inline_flags_as_an_int():
    for source in ("(?i)a", "(?i)(?m)a", b"(?s)a", "(?#c)(?i)a", "(?s:a)", "a"):
        assert hre.compile(source).flags == std.compile(source).flags
        assert type(hre.compile(source).flags) is int
    assert repr(hre.compile("(?i)a")) == repr(std.compile("(?i)a"))


def _assign(groupindex):
    groupindex["x"] = 1


def test_groupindex_is_read_only():
    assert outcome(lambda: _assign(hre.compile("(?P<n>a)").groupindex)) == "raises TypeError"
    assert hre.compile("(?P<n>a)").groupindex == {"n": 1} and hre.compile("a").groupindex == {}


@pytest.mark.parametrize("group", [True, False, 1, -1, 10**30, None, 1.0, b"n", "n", "zz"])
def test_match_group_arguments(group):
    for name in ("group", "__getitem__", "start", "end", "span"):
        ref, mine = both(lambda r: getattr(r.search("(a)(?P<n>b)", "xab"), name)(group))
        assert mine == ref, name


def test_match_copy_and_pickle_as_stdlib():
    m = hre.search("(a)", "a")
    assert copy.copy(m) is m and copy.deepcopy(m) is m
    assert both(lambda r: pickle.dumps(r.search("(a)", "a"))) == ("raises TypeError", "raises TypeError")
