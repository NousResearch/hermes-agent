"""Turn-size guard for what the Honcho plugin REMEMBERS (never what the model sees).

Why: on 2026-09-18 a 1 MB ``@file:`` attachment was mirrored into Honcho as 47 messages of
25k chars. One overflowed the deriver and parked the whole session (19 h of real conversation
underived), the summarizer hit the model context, and one chunk sank the embedding batch so the
session became unsearchable. The file body was agent tool output — worthless as memory about
Ron and a fabrication hazard for the deriver. Attachments and pastes are the ONLY way a user
turn gets that large; Ron's own words never do.

Two rules, applied to the expanded user turn before chunking:

1. The ``--- Attached Context ---`` block (built by ``agent/context_references.py``) is replaced
   by one stub line per attachment, so memory records THAT a file was shared (a real fact)
   without its contents. Binary/oversized references are already one-line ``📎`` stubs upstream
   and pass through unchanged.
2. Whatever remains is capped at ``turn_max_chars``: head kept, a trailing marker states how
   much was dropped. This is the "big paste" case, where no marker separates paste from prose.
"""
from __future__ import annotations

import re

ATTACHED_MARKER = "\n\n--- Attached Context ---\n\n"
WARNINGS_MARKER = "\n\n--- Context Warnings ---\n"
DEFAULT_TURN_MAX_CHARS = 8000

# "📄 @file:foo.json (12345 tokens)\n```json\n...\n```"  → name + token estimate
_INLINED_TEXT_RE = re.compile(r"^📄 (?P<name>\S+) \((?P<tokens>\d+) tokens\)\n```", re.MULTILINE)
# "🧾 git diff (123 tokens)\n```diff ..."
_INLINED_GIT_RE = re.compile(r"^🧾 (?P<label>[^\n(]+) \((?P<tokens>\d+) tokens\)\n```", re.MULTILINE)
# Already-stubbed references (binary, oversized text, folders, url) — keep the first line only.
_STUB_LINE_RE = re.compile(r"^(?P<line>📎 [^\n]+|📁 [^\n]+|🌐 [^\n]+)", re.MULTILINE)


def _stub_for_block(block: str) -> str:
    m = _INLINED_TEXT_RE.match(block)
    if m:
        return f"[attached: {m['name']}, ~{int(m['tokens']):,} tokens, contents not stored in memory]"
    m = _INLINED_GIT_RE.match(block)
    if m:
        return f"[attached: {m['label'].strip()}, ~{int(m['tokens']):,} tokens, contents not stored in memory]"
    m = _STUB_LINE_RE.match(block)
    if m:
        # Upstream already refused to inline it; keep its one-line description, drop the guidance.
        return m["line"].split(" — ")[0].strip()
    first = block.strip().splitlines()[0] if block.strip() else "attachment"
    return f"[attached: {first[:80]}, contents not stored in memory]"


def strip_attachments(user_content: str) -> str:
    """Replace the attached-context block with one stub line per attachment."""
    head, sep, tail = user_content.partition(ATTACHED_MARKER)
    if not sep:
        return user_content
    # Blocks are joined with a blank line; a fenced body may itself contain blank lines, so
    # split on the block HEADERS (emoji at line start), not on "\n\n".
    starts = [m.start() for m in re.finditer(r"^(?:📄|🧾|📎|📁|🌐) ", tail, re.MULTILINE)]
    if not starts:
        return head.rstrip()
    blocks = [tail[s:e] for s, e in zip(starts, starts[1:] + [len(tail)])]
    stubs = [_stub_for_block(b) for b in blocks]
    return (head.rstrip() + "\n\n" + "\n".join(stubs)).strip()


def cap_turn(text: str, limit: int) -> str:
    """Keep the first ``limit`` chars; say how much was left out."""
    if limit <= 0 or len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit].rstrip()}\n[… {dropped:,} more characters not stored in memory]"


def guard_user_turn(user_content: str, turn_max_chars: int = DEFAULT_TURN_MAX_CHARS) -> str:
    """What the plugin should persist for a user turn."""
    if not user_content:
        return user_content
    return cap_turn(strip_attachments(user_content), turn_max_chars)
