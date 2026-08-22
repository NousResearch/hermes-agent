"""Deterministic pre-flight boundary check for cron delivery (TKT-0033 Phase A).

When a cron job declares ``text/markdown`` (the default) but its delivery
content contains raw HTML tags OUTSIDE code fences, the send must hard-fail
into a dead-letter record instead of leaking literal tags to the user.

Pure stdlib, ~zero latency, $0/message: a pair of regexes over the outgoing
text. HTML tags inside fenced ``` blocks or inline `code` spans are code
samples, not leaks — they are stripped before the tag scan. The tag regex
deliberately requires a letter immediately after ``<`` so comparisons
(``x < y``) and arrows (``a -> b``) are NOT treated as tags.
"""

from __future__ import annotations

import re
from typing import Optional

_FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^>]*)?>")


def strip_code_blocks(text: str) -> str:
    """Remove fenced ``` blocks and inline `code` spans from *text*."""
    text = _FENCED_BLOCK_RE.sub("", text)
    text = _INLINE_CODE_RE.sub("", text)
    return text


def find_html_leak(text: str) -> Optional[str]:
    """Return the first HTML tag found OUTSIDE code blocks, or None."""
    stripped = strip_code_blocks(text)
    match = _HTML_TAG_RE.search(stripped)
    if match is None:
        return None
    return match.group(0)


def should_deadletter(payload_type: str, content: str) -> bool:
    """True when markdown-claimed content leaks raw HTML into delivery.

    Declared ``text/html`` payloads skip the check entirely — HTML is allowed
    when the job says so.
    """
    if payload_type != "text/markdown":
        return False
    return find_html_leak(content) is not None
