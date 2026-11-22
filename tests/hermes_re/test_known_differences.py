"""Accepted differences from stdlib `re`, pinned so that a change in either direction is noticed.

Each test states stdlib's result and the facade's. If the facade is improved, update the test.
"""
import re as std

import hermes_re as hre
from tests.hermes_re._support import outcome


def test_empty_match_iteration_moves_on_one_character():
    # stdlib retries a non-empty match where an empty one was just found; the facade does not.
    assert std.findall(r"|a", "a") == ["", "a", ""] and hre.findall(r"|a", "a") == ["", ""]
    assert std.sub(r"a*?", "-", "aa") == "-----" and hre.sub(r"a*?", "-", "aa") == "-a-a-"
    assert hre.findall(r"a*", "baac") == std.findall(r"a*", "baac")  # greedy nullable patterns agree


def test_interior_word_boundary_is_ascii():
    assert std.search(r"x\b.", "x\u00e9") is None and hre.search(r"x\b.", "x\u00e9") is not None
    assert std.search(r"\Bfoo", "\u00e9foo") is not None and hre.search(r"\Bfoo", "\u00e9foo") is None


def test_lone_surrogates_named_in_a_pattern_never_match():
    assert std.findall(r"[\ud800-\udfff]", "a\ud83db") == ["\ud83d"] and hre.findall(r"[\ud800-\udfff]", "a\ud83db") == []
    assert outcome(lambda: hre.compile("\ud800")) == "raises error"
    assert std.search("\ufffd", "\ud800") is None and hre.search("\ufffd", "\ud800") is not None


def test_repeated_empty_matchable_group_reports_other_group_text():
    ref, mine = std.fullmatch(r"((b??)+)?", "bbb"), hre.fullmatch(r"((b??)+)?", "bbb")
    assert ref.span() == mine.span() and ref.groups() == ("bbb", "") and mine.groups() == ("bbb", "b")


def test_pos_greater_than_endpos_never_matches():
    assert std.compile("").match("abc", 3, 0) is not None and hre.compile("").match("abc", 3, 0) is None


def test_repeat_counts_over_1000_are_refused_and_huge_programs_fail_at_first_use():
    assert std.compile(r"a{1001}") is not None and outcome(lambda: hre.compile(r"a{1001}")) == "raises error"
    assert std.compile(r"(?:a{40}){30}") is not None and outcome(lambda: hre.compile(r"(?:a{40}){30}")) == "raises error"
    assert hre.build(r"\w{0,512}").search("x" * 9) is not None  # fits the 64 MiB budget (8 MiB: too large)
    huge = hre.compile(r"\w{1000}" * 4)  # compile() cannot know the program size ...
    assert outcome(lambda: huge.search("x")) == "raises error"  # ... RE2 refuses it when it is built
    assert outcome(lambda: hre.build(huge)) == "raises error"
    assert std.compile(r"\w{1000}" * 4) is not None


def test_escape_spells_newline_and_nul_differently():
    assert std.escape("a\n\x00") == "a\\\n\x00" and hre.escape("a\n\x00") == "a\\n\\x00"
    assert hre.fullmatch(hre.escape("a\n\x00"), "a\n\x00") and std.fullmatch(hre.escape("a\n\x00"), "a\n\x00")


def test_missing_api():
    assert hasattr(std.compile("a"), "scanner") and not hasattr(hre.compile("a"), "scanner")
    assert hasattr(std, "Scanner") and not hasattr(hre, "Scanner")
    assert type(hre.compile("(?P<a>x)").groupindex) is dict  # stdlib: a read-only mappingproxy
    assert int(std.compile("(?i)a").flags) == 34 and int(hre.compile("(?i)a").flags) == 32  # inline flags are not reported
