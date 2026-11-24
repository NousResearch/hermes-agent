"""The compile cache is an LRU bounded by entry count and by the size of the programs it holds."""
import gc
import weakref

import hermes_re as hre
from hermes_re import _budget, _engine


def _with_limits(entries, size, body):
    real = _budget.MAX_ENTRIES, _budget.MAX_BYTES
    hre.purge()
    _budget.MAX_ENTRIES, _budget.MAX_BYTES = entries, size
    try:
        body()
    finally:
        _budget.MAX_ENTRIES, _budget.MAX_BYTES = real
        hre.purge()


def test_the_least_recently_used_pattern_leaves_when_the_cache_is_full():
    def body():
        first = hre.compile("count-0")
        for number in range(1, 8):
            hre.compile(f"count-{number}")
        assert _budget.held()[0] == 8
        assert hre.compile("count-0") is first  # a hit makes it the most recently used
        hre.compile("count-8")                  # one too many: "count-1" is now the oldest
        assert _budget.held()[0] == 8
        assert hre.compile("count-0") is first
        assert [hre.search(f"count-{number}", f"count-{number}").group() for number in range(9)] == [
            f"count-{number}" for number in range(9)]  # an evicted pattern is simply compiled again
    _with_limits(8, _budget.MAX_BYTES, body)


def test_unbuilt_patterns_cost_nothing_and_built_ones_their_programs():
    def body():
        for number in range(50):
            hre.compile(rf"lazy-{number}-\w+")
        assert _budget.held() == (50, 0)
        hre.search(r"lazy-0-\w+", "lazy-0-x")
        entries, size = _budget.held()
        assert entries == 50 and size == hre.compile(r"lazy-0-\w+")._cost > 1000 * _engine.INSTRUCTION_BYTES
    _with_limits(_budget.MAX_ENTRIES, _budget.MAX_BYTES, body)


def test_the_cache_drops_old_patterns_when_their_programs_outgrow_the_memory_budget():
    probe = hre.compile(r"probe-\w+")
    probe.search("probe-x")
    one = probe._cost  # what one such pattern costs once built (it holds the Unicode `\w` class)

    def body():
        sources = [rf"heavy-{number}-\w+" for number in range(40)]
        first = hre.compile(sources[0])
        gone = weakref.ref(hre.compile(sources[1]))
        for source in sources:
            assert hre.search(source, source[:-3] + "x") is not None
        entries, size = _budget.held()
        assert size <= _budget.MAX_BYTES and 5 <= entries <= 10  # far fewer than the 40 the count limit allows
        assert hre.compile(sources[-1])._built          # the newest are still there, built
        gc.collect()
        assert gone() is None                           # an evicted pattern nobody holds is freed
        # A pattern some module still holds stays built and usable; it is no longer the cache's.
        assert first._built and first.search("heavy-0-y").group() == "heavy-0-y"
        assert hre.compile(sources[0]) is not first
    _with_limits(_budget.MAX_ENTRIES, 10 * one + one // 2, body)


def test_the_pattern_in_use_is_never_the_one_dropped():
    def body():
        big = hre.compile(r"alone-\w{0,40}")
        assert big.search("alone-abc").group() == "alone-abc"   # far over the budget on its own
        assert hre.compile(r"alone-\w{0,40}") is big and _budget.held()[0] == 1
        hre.search(r"next-\w+", "next-x")                       # the next build pushes it out
        assert hre.compile(r"alone-\w{0,40}") is not big
    _with_limits(_budget.MAX_ENTRIES, 50_000, body)


def test_wide_forms_and_exact_twins_are_charged_when_they_are_built():
    def body():
        compiled = hre.compile(r"\bcharged[a-z]*?\b")
        compiled.search("charged ascii")
        narrow = compiled._cost
        compiled.search("\u2014charged\u2014")          # first non-ASCII subject: the wide form
        wide = compiled._cost
        compiled.search("\u2014charged\u00e9 x")        # a wrong cheap report: the exact twin
        assert 0 < narrow < wide < compiled._cost == _budget.held()[1]
    _with_limits(_budget.MAX_ENTRIES, _budget.MAX_BYTES, body)
