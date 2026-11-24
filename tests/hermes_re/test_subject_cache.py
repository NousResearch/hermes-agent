"""The subject cache: two slots, and the rule that keeps a huge text from staying pinned.

Slot 1 (1,024 to 4,194,304 characters) is covered in test_caches_errors_engine.py. Here: slot 2,
for longer subjects, which makes a `search(text, pos)` loop over a huge text O(1) per call, and
its retention rule: other subjects pay it off by their length.
"""
import gc
import re as std
import weakref

import hermes_re as hre
from hermes_re import _subject

HUGE = _subject.MAX_CACHED + 1000


class Text(str):
    """A str that can be weakly referenced, to observe when the cache lets go of it."""


def test_a_subject_above_the_slot_1_limit_is_encoded_twice_for_a_whole_pos_loop():
    hre.purge()
    texts = ["ab " * (HUGE // 3 + 1), "a\u00e9 " * (HUGE // 3 + 1), "a\ud800 " * (HUGE // 3 + 1),
             "ab " * 1_700_000, "\u65e5\u672c " * 1_700_000, "ab " * 6_700_000, "a\u00e9 " * 6_700_000]  # 5 M and 20 M characters
    for text in texts:
        word, ref = hre.compile(r"[^ ]+"), std.compile(r"[^ ]+")
        buffers, pos = [], 0
        for _ in range(1000):
            mine, expected = word.search(text, pos), ref.search(text, pos)
            assert mine.span() == expected.span() and mine.group() == expected.group()
            if not buffers or buffers[-1] is not mine._data:
                buffers.append(mine._data)
            pos = mine.end()
        assert len(buffers) == 2, "the text was encoded again inside the loop"  # first sighting + the kept one
        assert _subject._large[0] is text and _subject._last_subject[0] is not text
        far = len(text) - 7
        assert word.search(text, far).span() == ref.search(text, far).span()
        assert word.match(text, far - 2, far + 1).span() == ref.match(text, far - 2, far + 1).span()
    hre.purge()
    assert _subject._large[0] is None


def test_slot_2_is_paid_off_by_other_subjects_and_then_lets_the_text_go():
    hre.purge()
    huge = Text("x" * HUGE)
    alive = weakref.ref(huge)
    probe = hre.compile("x")
    assert probe.search(huge, 5).start() == 5
    assert _subject._large[0] is None          # one call never pins a huge text
    assert probe.search(huge, 6).start() == 6
    del huge
    gc.collect()
    assert alive() is not None                 # used twice in a row: kept while it is the text in use
    other = "y" * (HUGE // 4)
    for _ in range(3):                         # three quarters of its length in other work
        probe.search(other + "!")
    gc.collect()
    assert alive() is not None
    probe.search(other + "?")                  # the charge reaches the text's own length
    probe.search(other + "#")
    gc.collect()
    assert alive() is None and _subject._large[0] is None


def test_using_the_huge_subject_resets_the_charge_so_interleaved_work_never_thrashes():
    hre.purge()
    huge = "x" * HUGE
    probe = hre.compile("x")
    probe.search(huge)
    first = probe.search(huge)._data
    for round_number in range(12):             # in total 3 times its length in other work
        probe.search("y" * (HUGE // 4) + str(round_number))
        assert probe.search(huge, 10 + round_number)._data is first
    short = [f"s{k}" for k in range(HUGE // 1024 - 10)]   # short subjects count 1,024 characters each
    for item in short:
        probe.search(item)
    assert _subject._large[0] is huge
    for item in short[:20]:
        probe.search(item)
    assert _subject._large[0] is None          # paid off by many short calls as well
    hre.purge()


def test_a_second_huge_subject_and_purge_replace_the_first():
    hre.purge()
    one, two = Text("a" * HUGE), Text("b" * HUGE)
    gone = weakref.ref(one)
    assert hre.sub("a", "", one) == "" and _subject._large[0] is None   # a one-shot call leaves nothing behind
    del one
    gc.collect()
    assert gone() is None
    one = Text("a" * HUGE)
    gone = weakref.ref(one)
    for text, letter in ((one, "a"), (one, "a"), (two, "b"), (two, "b")):
        hre.search(letter, text)
    del one, text
    gc.collect()
    assert gone() is None and _subject._large[0] is two
    kept = weakref.ref(two)
    del two
    hre.purge()
    gc.collect()
    assert kept() is None


def test_slot_1_keeps_its_rule_and_its_use_charges_slot_2():
    hre.purge()
    mid, huge = "m" * 5000, "h" * HUGE
    hre.search("m", mid)
    hre.search("h", huge)
    hre.search("h", huge)
    assert _subject._last_subject[0] is mid and _subject._large[0] is huge   # one of each
    for _ in range(HUGE // 5000 + 2):
        hre.search("m", mid)                                                  # slot-1 hits are other work too
    assert _subject._last_subject[0] is mid and _subject._large[0] is None
    hre.search("q", "tiny")
    assert _subject._last_subject[0] is mid                                   # short subjects never evict slot 1
    hre.purge()
