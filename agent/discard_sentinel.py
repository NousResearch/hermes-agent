"""Shared ``[DISCARD]`` sentinel for suppressing background agent output.

Inspired by Poke's "wait" tool — the orchestrating agent silently drops a
background output that turned out to be redundant or irrelevant rather than
forcing it into the user-facing reply.

Hermes already has the inverse-direction precedent in the cron scheduler:
``cron/scheduler.py`` uses a ``[SILENT]`` marker so an agent job can run, do
its work, and suppress *delivery* of the final message ("save but don't
deliver"). This module generalizes that same idea for two other surfaces:

* The background self-improvement review — the review fork can persist
  memory/skill writes (the side effects already happened) yet decide its
  ``💾 Self-improvement review: ...`` notification is noise and suppress it.
* ``delegate_task`` children — a subagent can decide its result is redundant
  with a sibling's and ask to be dropped from the aggregated ``results[]``
  the parent ingests, keeping the parent's context clean.

The marker is matched the same permissive way as ``[SILENT]``: presence
anywhere in the (upper-cased) final text counts. This keeps the contract
trivial for a model to satisfy and impossible to half-emit.
"""

from __future__ import annotations

# Canonical marker. A model emits this verbatim in its FINAL response text to
# request that the harness drop the output. Side effects (memory/skill writes,
# tool calls already executed) are NOT undone — only the user-facing surfacing
# / parent-context entry is suppressed.
DISCARD_MARKER = "[DISCARD]"


def is_discard_marked(text: object) -> bool:
    """Return True if ``text`` carries the discard sentinel.

    Mirrors the cron ``[SILENT]`` check: case-insensitive, presence-based,
    tolerant of surrounding content. Non-string input is treated as not
    marked so callers never need to guard the type themselves.
    """
    if not isinstance(text, str):
        return False
    return DISCARD_MARKER in text.strip().upper()


def strip_discard_marker(text: object) -> str:
    """Remove the discard marker from ``text`` (case-insensitive).

    Used when a discarded payload is still retained in a slimmed-down form
    (e.g. a one-line stub) so the marker itself doesn't leak into context.
    Returns the input unchanged if it is not a string.
    """
    if not isinstance(text, str):
        return ""
    out = text
    for variant in (DISCARD_MARKER, DISCARD_MARKER.lower(), DISCARD_MARKER.upper()):
        out = out.replace(variant, "")
    return out.strip()
