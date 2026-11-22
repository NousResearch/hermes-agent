"""API and syntax edges where the facade must give stdlib's answer, or refuse at `compile()`.

Negative ``count`` / ``maxsplit``, a ``repl`` callback returning ``None``, an escaped `)` inside a
`(?#...)` comment, out-of-range escapes in classes and templates, ``lastindex`` with a group inside
a trailing lookahead; and the registry behind ``build_all`` / the fork clean-up, which must hold
every live Pattern even when two are equal.
"""
import gc
import re as std

import hermes_re as hre
from tests.hermes_re._support import outcome, view


def test_a_negative_count_or_maxsplit_changes_nothing_as_in_stdlib():
    for subject in ("xaxax", "x\u00e9ax\u00e9"):  # ASCII and non-ASCII subjects take different paths
        for module in (std, hre):
            assert module.sub("a", "b", subject, count=-1) == subject
            assert module.subn("a", "b", subject, count=-1) == (subject, 0)
            assert module.split("a", subject, maxsplit=-1) == [subject]
    assert hre.subn(b"a", b"b", b"aa", count=-2) == (b"aa", 0)
    assert hre.split(b"a", b"xax", maxsplit=-5) == [b"xax"]


def test_a_callback_returning_none_removes_the_match_as_in_stdlib():
    for subject in ("xax", "\u00e9a\u00e9", b"xax"):
        pattern = b"a" if isinstance(subject, bytes) else "a"
        assert hre.subn(pattern, lambda m: None, subject) == std.subn(pattern, lambda m: None, subject)


def test_lastindex_counts_a_lookahead_group_as_closing_where_the_lookahead_runs():
    for source in (r"(?P<outer>a(?=(?P<inner>b)))", r"((a)(?=(b)))", r"(a)(?=(b))", r"(a)(?=(b)(c))",
                   r"(?:(a)|(b))+(?=(c))", r"(a)(?=b|(c))"):
        for subject in ("ab", "abc", "bac", "aac", "ac"):
            assert view(hre.search(source, subject)) == view(std.search(source, subject)), (source, subject)
