"""Last-reply context block for the external-editor composer.

``display.editor_include_last_reply`` puts the agent's previous reply above the draft as ``#``
comments when Ctrl+G / Alt+G / ``/prompt`` opens ``$EDITOR``. The block is delimited by two
markers and removed again on save, so the user sends only what they kept.

Deliberately *not* "strip every line starting with #": markdown headings, shell comments and code
fences in a draft all start with ``#``, and eating them would quietly corrupt prompts. Only the
marked region goes; a ``#`` line the user typed above or below it survives untouched.

Both terminal interfaces share this decision table: the classic CLI calls these functions
directly, and the Ink composer mirrors them in ``ui-tui/src/domain/editorContext.ts`` against the
same vectors in ``tests/fixtures/editor_context_vectors.json``.
"""

from __future__ import annotations

MARKER_START = "# ── last reply (lines starting with # are removed) ──"
MARKER_END = "# ── end of last reply ──"

_TRUTHY = {"1", "true", "yes", "on"}


def _comment_line(line: str) -> str:
    """One reply line as a comment.

    Blank lines stay a bare ``#`` rather than an empty line, so the block can never be mistaken
    for part of the draft, and a reply line that happens to spell a marker is defanged rather
    than being able to close the block early.
    """
    if not line:
        return "#"

    stripped = line.strip()

    if stripped in {MARKER_START, MARKER_END}:
        return "# " + stripped.replace("──", "--")

    return "# " + line


def build_last_reply_block(last_reply: str) -> str:
    """The marker block for a reply, or ``""`` when there is nothing to show."""
    text = (last_reply or "").strip("\n")

    if not text.strip():
        return ""

    body = "\n".join(_comment_line(line.rstrip("\r")) for line in text.split("\n"))

    return MARKER_START + "\n" + body + "\n" + MARKER_END


def build_editor_seed(draft: str, last_reply: str = "") -> str:
    """What the editor opens with: the draft, then the reply block (when there is one)."""
    block = build_last_reply_block(last_reply)

    if not block:
        return draft or ""

    draft = draft or ""

    if not draft.strip():
        return block

    return draft.rstrip("\n") + "\n\n" + block


def strip_last_reply_block(text: str) -> str:
    """Remove the marker block (and the blank separator above it) from an editor buffer.

    A user who deleted the end marker still gets the block stripped — to the end of the buffer —
    because leaving it in would send the agent's own reply back to it as a prompt.
    """
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    start = next((i for i, line in enumerate(lines) if line.strip() == MARKER_START), None)

    if start is None:
        return text or ""

    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == MARKER_END), len(lines) - 1)

    # Drop the single blank separator line the builder inserted above the block — but only when
    # the block runs to the end of the buffer. Text the user typed *below* the block keeps its
    # spacing, so stripping never quietly re-flows a draft they rearranged.
    if end == len(lines) - 1 and start > 0 and not lines[start - 1].strip():
        start -= 1

    return "\n".join(lines[:start] + lines[end + 1 :])


def editor_seed(draft: str, last_reply: str, *, enabled: bool) -> str:
    """Seed text for `$EDITOR` under the `display.editor_include_last_reply` setting."""
    return build_editor_seed(draft, last_reply) if enabled else (draft or "")


def config_enabled(value: object) -> bool:
    """Read the setting the way the rest of the CLI reads YAML booleans."""
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY

    return False


__all__ = [
    "MARKER_END",
    "MARKER_START",
    "build_editor_seed",
    "build_last_reply_block",
    "config_enabled",
    "editor_seed",
    "strip_last_reply_block",
]
