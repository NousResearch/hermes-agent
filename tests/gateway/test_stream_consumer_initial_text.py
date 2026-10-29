"""Initial previews depend on complete content, not token/delta boundaries."""

import pytest

from gateway.stream_consumer_initial_text import initial_text_ready


_LONG = "This first sentence contains enough useful information to make the initial notification readable"
_SHORT = "Yes. "
_SECOND = "The next sentence supplies the additional context needed for a useful notification"


@pytest.mark.parametrize(("prefix", "suffix", "expected"), [
    (_LONG, ". ", True),
    (_LONG + ".", " ", True),
    ("a" * 79, ". ", True),
    ("a" * 77, ". ", False),
    ("> " + "a" * 77, ". ", False),
    (_SHORT + _SECOND, ". More tokens may already follow", True),
    (_LONG + ' says “ready', '.” Next', True),
    (_LONG + " version 3.", "14 is ready. ", True),
    (_LONG + " from Dr. ", "Martin is ready. ", True),
    (_LONG + " from e.g. ", "a local configuration. ", True),
    (_LONG + " in the U.S. ", "is ready. ", True),
    (_LONG + " on https://example.com/path?q=1. ", "It is ready. ", True),
    (_LONG + " with `value. still code`", ". ", True),
    (_LONG + " with `value. still code", "` is ready. ", True),
    (_LONG + " with [details](https://example.com/foo.", ") available. ", True),
    ("# A heading. " * 10 + "\n\n" + _LONG, ". ", True),
    ("A heading\n===\n" + _LONG, ". ", True),
    (_LONG, "\n\n", True),
    ("- " + _LONG, "\n", True),
    ("1. " + _LONG, "\n", True),
    ("```python\n" + _LONG + ".\n\n", "```\n", True),
    ("~~~text\n" + _LONG + ".\n\n", "~~~\n", True),
    ("这是一段用于验证通知预览的完整中文说明" * 5, "。", True),
    (_LONG, "! ", True),
    (_LONG, "? ", True),
    (_LONG, "… ", True),
    ("Yes. ", "The rest of this sentence adds more than eighty characters but has not yet ended", False),
    ("# " + _LONG + ".\n\n", "", False),
    ("OK", "", False),
    ("\n\n", "", False),
], ids=[
    "sentence", "period-needs-lookahead", "exact-minimum", "below-minimum", "quote-marker-not-content",
    "short-plus-second", "quoted", "decimal-split", "title-abbreviation",
    "dotted-abbreviation", "initials", "url", "inline-code", "open-inline-code", "link-destination",
    "heading-before-prose", "setext-heading", "paragraph", "bullet", "numbered-item",
    "fenced-code", "tilde-fence", "cjk", "exclamation", "question", "ellipsis",
    "unfinished-second", "heading-only", "short-final-owned-by-caller", "empty",
])
def test_initial_text_requires_enough_completed_content(prefix, suffix, expected):
    assert all(not initial_text_ready(prefix[:end], 80) for end in range(len(prefix) + 1))
    assert initial_text_ready(prefix + suffix, 80) is expected
    # Chunking cannot make a decimal, abbreviation or code fragment look complete:
    # every prefix stays held until actual content crosses the release boundary.
    if expected:
        assert any(initial_text_ready((prefix + suffix)[:end], 80)
                   for end in range(len(prefix) + 1, len(prefix + suffix) + 1))
    assert initial_text_ready(prefix, 0)
