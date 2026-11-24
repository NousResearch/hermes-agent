"""Replacement text is the caller's Python text: surrogates, non-BMP characters and NUL come
through ``sub`` / ``subn`` / ``expand`` as stdlib gives them, on any subject."""
import itertools
import re as std

import pytest

import hermes_re as hre
from tests.hermes_re._support import outcome

_REPLACEMENTS = ["\ud800", "\udc00x", "x\udbff", "\ud83d\ude00", "\U0001F600", "\x00", "a\x00\U0001F600\ud800",
                 "\\1\ud800", "\\g<0>\U0001F600", "\\n\ud800"]
_SUBJECTS = ["abc", "a\u00e9b", "a\ud800b", "a\U0001F600b", "a\x00b", "\u00e9" * 600 + "a" + "\u00e9" * 600]
_PATTERNS = ["(a)", "(b)", "()", "(\u00e9)", "(\\w)", "(?<=a)()", "(.)"]


def _expand(m, template):
    return m and m.expand(template)


@pytest.mark.parametrize("pattern,subject", list(itertools.product(_PATTERNS, _SUBJECTS)),
                         ids=lambda value: ascii(value[:8]))
def test_replacement_text_with_surrogates_works_as_in_stdlib(pattern, subject):
    for repl in _REPLACEMENTS:
        calls = (
            lambda r: r.sub(pattern, repl, subject),
            lambda r: r.subn(pattern, repl, subject, count=1),
            lambda r: r.sub(pattern, lambda m: m.group() + repl, subject),
            lambda r: _expand(r.search(pattern, subject), repl),
        )
        for call in calls:
            assert call(hre) == call(std), (pattern, ascii(subject[:8]), ascii(repl))


@pytest.mark.parametrize("subject", _SUBJECTS, ids=lambda value: ascii(value[:8]))
@pytest.mark.parametrize("result", [b"x", 1, bytearray(b"x")])
def test_a_callback_returning_non_text_raises_type_error_as_in_stdlib(subject, result):
    assert outcome(lambda: std.sub("a", lambda m: result, subject)) == outcome(
        lambda: hre.sub("a", lambda m: result, subject))
