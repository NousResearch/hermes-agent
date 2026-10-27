"""Card work class -> default reasoning effort.

What KIND of work a card is, independent of who runs it. The class picks the
worker's default reasoning effort so a mechanical card does not pay the assignee
profile's full thinking budget (a lane pinned at ``xhigh`` does not need ``xhigh``
to merge a branch).

Source of truth is the ``tasks.work_class`` column, written at create time by
``hermes kanban create --work-class`` and the ``kanban_create`` tool's
``work_class`` argument. It is NEVER inferred from the title or body: a guess the
author cannot audit would silently change a worker's thinking budget.

Resolution order for a dispatched worker (``effective_reasoning_effort``):

1. the card's explicit ``reasoning_effort`` pin (``"none"`` = thinking off)
2. the class default below, when the class is mechanical
3. ``None`` — inherit the assignee profile's ``agent.reasoning_effort`` (and its
   per-model ``reasoning_overrides``)

Classes with no mapping row — implementation, design, diagnosis, research — and
any unknown or NULL class land on (3) by design: substantive work inherits.

Kept out of ``kanban_db`` (4k-line facade, growth-ratcheted) so this policy has
one auditable home; ``kanban_db`` re-exports these names for its callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cycle
    from hermes_cli.kanban_db import Task

VALID_WORK_CLASSES = (
    "merge", "deploy", "approval", "qa-handoff", "dependency-bump", "formatting",
    "implementation", "design", "diagnosis", "research",
)

# Mechanical classes -> a working level. Everything else inherits the profile.
WORK_CLASS_REASONING_EFFORT = {
    "merge": "medium",         # conflicts and ordering calls need real judgement
    "deploy": "low",           # a gated script, not a design decision
    "approval": "low",         # sign-off against a stated contract
    "qa-handoff": "low",       # verify the criteria, quote the evidence
    "dependency-bump": "low",  # bump, lock, run the gate
    "formatting": "low",       # mechanical edits
}


def normalize_work_class(work_class: Optional[str]) -> Optional[str]:
    """``VALID_WORK_CLASSES``, case/space-insensitive (``qa_handoff`` == ``qa-handoff``).
    Empty/None = no class, which inherits the profile. Anything else raises: a
    mistyped class must not silently fall back to inherit — that is exactly the
    thinking budget the author tried to lower."""
    value = str(work_class or "").strip().lower().replace("_", "-")
    if not value:
        return None
    if value in VALID_WORK_CLASSES:
        return value
    raise ValueError(f"work_class must be one of {', '.join(VALID_WORK_CLASSES)}, got {work_class!r}")


def resolve_work_class_effort(work_class: Optional[str]) -> Optional[str]:
    """Class-based default effort; ``None`` means the class inherits the profile."""
    value = str(work_class or "").strip().lower().replace("_", "-")
    return WORK_CLASS_REASONING_EFFORT.get(value)


def effective_reasoning_effort(task: Task) -> Optional[str]:
    """The effort a dispatched worker actually runs at: explicit card pin, else the
    work-class default, else ``None`` (inherit). ``hermes kanban show`` and the
    gateway's ``kanban.run`` policy both call this, so an audit reads the same
    answer the worker was given."""
    return task.reasoning_effort or resolve_work_class_effort(task.work_class)
