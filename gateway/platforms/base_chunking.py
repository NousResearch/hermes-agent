"""Choose link-safe boundaries without changing oversized-link fallbacks."""
import re
from collections.abc import Callable


# Code examples are not links. Backslash escapes belong to the span (notably
# MarkdownV2's escaped closing parentheses inside URL destinations).
_LINK_OR_CODE = re.compile(
    r"```[\s\S]*?```|`[^`\n]*`|"
    r"(?P<link>\[(?:\\.|[^\]\\\n])+\]"
    r"\((?:\\.|[^()\\\n]|\([^()\\\n]*\))*\))"
)


def link_safe_split(text: str, split_at: int, budget: int,
                    measure: Callable[[str], int]) -> int:
    """Keep a link together when it fits the next chunk's body budget.

    Oversized or malformed links retain the existing splitter's progress and
    size guarantees; we do not manufacture a new URL or discard its tail.
    """
    for match in _LINK_OR_CODE.finditer(text):
        start, end = match.span()
        if start >= split_at:
            break
        if not (match.group("link") and split_at < end):
            continue
        # An escaped '[' is literal text, not a link opener.
        backslashes = start - len(text[:start].rstrip("\\"))
        if backslashes % 2 or measure(match.group()) > budget:
            return split_at
        # At column zero, prefer the end instead: rewinding to zero would loop.
        return start if start else end
    return split_at
