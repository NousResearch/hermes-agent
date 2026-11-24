"""`pos` / `endpos` windows: a search confined to a short range encodes only that range.

A stream buffer is a new ``str`` on every tick, so the subject cache (keyed on the object) never
helps it. For a range of at most ``_subject.WINDOW`` characters of a subject that is not cached,
the facade encodes the range and the one character before it, instead of the whole subject.
Results must be stdlib's, whichever path a call takes.
"""
import re as std

import hermes_re as hre
from hermes_re import _subject
from tests.hermes_re._support import view

UNIT = "Za\u017c\u00f3\u0142\u0107 g\u0119\u015bl\u0105 ja\u017a\u0144 \u65e5\u672c\u8a9e key=abc\n"
PATTERNS = [r"\w+", r"^\w", r"(?m)^\w+$", r"\A.", r"a$", r"\b\u00e9\w*", r"x\b", r"(?<!\w)a", r"k(?=\s|$)", r"\s\S*$",
            r"(?i)KEY\s*[:=]", r"[^A-Za-z0-9_]", r"", r"(\w)(\d)?", r"\babc\b", r"(?<=\n)Z", r"\u65e5(?=\u672c)"]
RANGES = [(0, 3), (1, 1), (1, 9), (5, 40), (17, 18), (20, 400), (28, 29), (29, 60), (3, 10**9), (-5, 12)]


def _subjects():
    plain = "The quick brown fox, key=abc\nand more text x1\n"
    lone = "ab\ud83d " + UNIT + "\udc00 x"
    return [plain, UNIT * 3, lone, plain * 200, UNIT * 300, (UNIT * 150) + lone + (plain * 60)]


def _everything(pattern, subject, pos, endpos):
    spans = [(m.span(), m.pos, m.endpos, m.groups(), m.expand(r"<\g<0>>")) for m in pattern.finditer(subject, pos, endpos)]
    return (view(pattern.search(subject, pos, endpos)), view(pattern.match(subject, pos, endpos)),
            view(pattern.fullmatch(subject, pos, endpos)), pattern.findall(subject, pos, endpos), spans)


def test_every_operation_gives_stdlib_results_inside_a_window():
    checks = 0
    for source in PATTERNS:
        mine, ref = hre.compile(source), std.compile(source)
        for subject in _subjects():
            size = len(subject)
            ranges = RANGES + [(size - 12, size), (size - 300, size - 1), (size - 1, size), (size // 2, size // 2 + 700)]
            expected = [_everything(ref, subject, pos, endpos) for pos, endpos in ranges]
            hre.purge()  # nothing cached: short ranges take the window
            fresh = [_everything(mine, subject + "", pos, endpos) for pos, endpos in ranges]
            assert fresh == expected, (source, size)
            mine.search(subject)  # now the subject is the cached one (when it is long enough)
            assert [_everything(mine, subject, pos, endpos) for pos, endpos in ranges] == expected, (source, size)
            checks += 2 * len(ranges)
    assert checks > 2500


def test_a_window_reports_the_callers_string_and_positions():
    subject = UNIT * 400
    size = len(subject)
    found = hre.compile(r"(?P<word>\w+)=(\w+)").search(subject, size - 12, size - 1)
    assert found.string is subject and (found.pos, found.endpos) == (size - 12, size - 1)
    assert subject[found.start():found.end()] == found.group() == "key=abc"
    assert subject[found.start("word"):found.end("word")] == found["word"] == "key"
    assert found.span(2) == (size - 4, size - 1) and found.regs[0] == found.span()


def test_a_tail_window_of_a_new_string_does_not_prepare_the_whole_string():
    hre.purge()
    tail = hre.compile(r"\s\S*$")
    text = UNIT * 2000  # 64,000 characters, well inside the range the subject cache keeps
    for tick in range(5):
        text = text + "more \u017c"  # a new str object per tick, as a stream buffer is
        found = tail.search(text, len(text) - 512, len(text))
        assert found.span() == (len(text) - 2, len(text)) and len(found._data) < 4 * 513 + 4
        assert _subject._last_subject[0] is None, tick  # nothing of the buffer is kept either
    assert tail.search(text).span() == (len(text) - 2, len(text))  # the whole text: prepared and cached as before
    assert _subject._last_subject[0] is text


def test_a_long_range_or_a_cached_subject_keeps_using_the_subject_cache():
    hre.purge()
    word = hre.compile(r"[^ ]+")
    text = "ab\u00e9 " * 15000
    buffers, pos = set(), 4
    while pos < len(text) - 4:  # a `search(text, pos)` loop that does not begin at 0
        found = word.search(text, pos)
        buffers.add(id(found._data))
        pos = found.end()
    assert len(buffers) == 1 and _subject._last_subject[0] is text  # encoded once, down to the last few characters
