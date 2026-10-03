"""Pure composition of Telegram group titles: ``<subject> · <model> · r<N>``.

The single place the string is built (spec §2). No I/O, no budget, no guards — callers
decide *whether* to rename; this module only decides *what the string is*.

Contract highlights (orchestrator-adjudicated):

* Separator is ``" \\u00b7 "`` (space, U+00B7, space) — 3 chars, charged to the cap.
* Model rendering happens HERE (lowercase, leading ``provider/`` segment stripped) so no
  caller can forget it.
* Cap is :data:`hermes_state.SessionDB.MAX_TITLE_LENGTH` (100), not Telegram's 128.
* Overflow truncates the SUBJECT only: the suffix is never truncated, never dropped.
"""

from __future__ import annotations

from hermes_constants import VALID_REASONING_EFFORTS
from hermes_state import SessionDB

__all__ = [
    "MAX_TITLE_LENGTH",
    "REASONING_ORDINALS",
    "SEPARATOR",
    "compose_group_title",
]

SEPARATOR = " \u00b7 "
_ELLIPSIS = "\u2026"

#: Charge for the ellipsis itself, on top of the 3-char separator.
_ELLIPSIS_LEN = len(_ELLIPSIS)

#: effort -> 1-based ordinal over VALID_REASONING_EFFORTS (minimal=r1 ... ultra=r7).
REASONING_ORDINALS: dict[str, int] = {
    effort: index for index, effort in enumerate(VALID_REASONING_EFFORTS, start=1)
}

MAX_TITLE_LENGTH: int = SessionDB.MAX_TITLE_LENGTH


def _render_model(model: str) -> str:
    """Display form of a model id: lowercase, leading provider segment stripped.

    One segment only: ``opencode-go/space-bunny-free`` -> ``space-bunny-free`` while
    ``provider/team/model-x`` -> ``team/model-x``. Deeper model ids are left intact. A
    bare ``provider/`` with no remainder is no model at all.
    """
    name = (model or "").strip().lower()
    head, sep, tail = name.partition("/")
    if sep and head:
        return tail.strip()
    return name


def _render_reasoning(reasoning: str | None) -> str:
    """``r<N>`` for a known effort, else empty. Never ``r0``, never a guess."""
    ordinal = REASONING_ORDINALS.get((reasoning or "").strip().lower())
    return f"r{ordinal}" if ordinal else ""


def compose_group_title(subject: str, model: str, reasoning: str | None) -> str:
    """Compose the group title for a session.

    Args:
        subject: session title from the store; only outer whitespace is stripped, casing
            is preserved. Empty or whitespace-only means "no subject".
        model: model id, rendered here (lowercase, ``provider/`` prefix stripped).
        reasoning: effort name from ``VALID_REASONING_EFFORTS``; anything else (None,
            empty, ``"disabled"``, unknown) yields no tag.

    Returns:
        The composed title, never longer than :data:`MAX_TITLE_LENGTH`.
    """
    head = (subject or "").strip()
    suffix_parts = [part for part in (_render_model(model), _render_reasoning(reasoning)) if part]
    suffix = SEPARATOR.join(suffix_parts)
    tail = f"{SEPARATOR}{suffix}" if suffix else ""

    if not head:
        return suffix
    if len(head) + len(tail) <= MAX_TITLE_LENGTH:
        return head + tail

    # Truncate the SUBJECT only; the suffix is never truncated, never dropped.
    trimmed = head[: MAX_TITLE_LENGTH - len(tail) - _ELLIPSIS_LEN].rstrip()
    return trimmed + _ELLIPSIS + tail
