"""Last-reply context block for the external-editor composer.

``display.editor_include_last_reply`` puts the agent's previous reply under the draft as ``#``
comments when Ctrl+G / Alt+G / ``/prompt`` opens ``$EDITOR``. The block sits between two markers
and is removed again on save, so the user sends only what they kept.

Deliberately *not* "strip every line starting with #": markdown headings, shell comments and code
fences in a draft start with ``#`` too. Only the marked region goes.

The Ink composer mirrors this in ``ui-tui/src/domain/editorContext.ts``; both suites read
``tests/fixtures/editor_context_vectors.json``.
"""

from __future__ import annotations

MARKER_START = "# ── last reply (lines starting with # are removed) ──"
MARKER_END = "# ── end of last reply ──"


def _comment_line(line: str) -> str:
    """One reply line as a comment; a line spelling a marker is defanged so it cannot end the block."""
    if line.strip() in {MARKER_START, MARKER_END}:
        line = line.strip().replace("──", "--")
    return f"# {line}" if line else "#"


def build_editor_seed(draft: str, last_reply: str) -> str:
    """What the editor opens with: the draft, a blank line, then the reply block (if any)."""
    reply = (last_reply or "").strip("\n")
    if not reply.strip():
        return draft or ""
    body = "\n".join(_comment_line(line.rstrip("\r")) for line in reply.split("\n"))
    # Always a blank line above the block, even with no draft: editors open on line 1, and text
    # typed there must not land on the marker line (which would stop the block being found).
    return f"{(draft or '').rstrip(chr(10))}\n\n{MARKER_START}\n{body}\n{MARKER_END}"


def strip_last_reply_block(text: str) -> str:
    """Remove the marker block from an editor buffer (to the end, if the end marker was deleted).

    The blank separator above the block goes too, but only when the block runs to the end of the
    buffer: text typed below it keeps its spacing.
    """
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = next((i for i, line in enumerate(lines) if line.strip() == MARKER_START), None)
    if start is None:
        return text or ""
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == MARKER_END), len(lines) - 1)
    if end == len(lines) - 1 and start > 0 and not lines[start - 1].strip():
        start -= 1
    return "\n".join(lines[:start] + lines[end + 1:])
