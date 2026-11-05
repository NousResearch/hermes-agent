"""kanban_assignee_gate.py — refuse an assignee the dispatcher cannot spawn, at WRITE time.

WHY THIS EXISTS (defect class: reserved-assignee dead letter)
-------------------------------------------------------------
A card filed to a reserved profile name is structurally undispatchable. The
dispatcher refuses the spawn (`hermes -p hermes` exits before the worker
starts), the card burns its failure budget, and lands in ``blocked`` with
"No worker ever ran this". The board reads as "work queued"; the truth is
"work queued for a lane that cannot start". Left to a nightly audit, the class
regenerates every time an agent types a reserved name into a filing call — and
it did: three cards carry the ``Profile name 'hermes' is reserved`` failure, and
the two P0 assignments stranded this way had to be re-routed by hand
(decisions/2026-09-29-reserved-assignee-dead-letter-reroute.md).

The cure is to refuse at the point of writing, where the caller still has a
choice of name, rather than audit after the fact. ``~/.hermes/scripts/
kanban_assignee_lint.py`` is the after-the-fact sweep and stays (it catches
names that became unspawnable later, e.g. a profile deleted after filing); this
module is the file-time rejection the 09-29 decision committed to.

THE PREDICATE: ``resolve_profile_env``, NOT list membership
-----------------------------------------------------------
``hermes_cli.profiles._RESERVED_NAMES`` is the authority for WHAT IS RESERVED.
It is not, by itself, the right refusal test: ``default`` is in that set yet
``resolve_profile_env("default")`` returns ``~/.hermes`` and the dispatcher
spawns it — cards ``t_e17c4c53`` and ``t_49eddcdf`` ran to ``done`` under
profile ``default`` on this board. A predicate of "name is in the reserved
list" would refuse a working lane, and a guard that refuses working lanes gets
switched off.

So the refusal is decided the way the dispatcher decides it: by asking
``resolve_profile_env`` — the call whose ``ValueError`` IS the spawn refusal.
A name is refused only when it is BOTH reserved AND unspawnable.

``default`` therefore passes; ``hermes``/``test``/``tmp``/``root``/``sudo`` do
not. This matches ``assignment.py::classify_assignee``, which the fleet's
dispatch path already imports as its single source of truth — a gate that
disagreed with it is exactly how work strands silently.

``FileNotFoundError`` (a profile that does not exist) is deliberately NOT this
gate's business. A missing profile can be created, it is home-relative (unlike
a reserved name, which raises under any ``HERMES_HOME``), and the dispatcher
already writes a per-card ``skipped_nonspawnable`` diagnostic for it
(#122422). Refusing it here would also break every test that files a card to a
placeholder assignee under an isolated home. Reserved names are the class that
can NEVER work; that is the class refused at write time.
"""
from __future__ import annotations

from typing import Optional


def reserved_and_unspawnable(assignee: Optional[str]) -> bool:
    """True when *assignee* is a reserved name the dispatcher cannot spawn."""
    name = (assignee or "").strip().lower()
    if not name:
        return False
    try:
        from hermes_cli.profiles import _RESERVED_NAMES, resolve_profile_env
    except Exception:  # pragma: no cover - profiles is always importable here
        return False
    if name not in {str(n).lower() for n in _RESERVED_NAMES}:
        return False
    try:
        resolve_profile_env(name)
    except ValueError:
        return True
    except Exception:
        # Cannot tell -> do not refuse. A false refusal blocks real work.
        return False
    return False


def is_reserved_name(assignee: Optional[str]) -> bool:
    """True when *assignee* is in ``_RESERVED_NAMES``, spawnable or not.

    For roster/prompt filtering, where the question is "would offering this
    name tempt a caller into a lane the fleet treats as not-a-profile" — a
    different question from :func:`reserved_and_unspawnable`, which is the
    refusal. Kept separate so the two never get conflated.
    """
    name = (assignee or "").strip().lower()
    if not name:
        return False
    try:
        from hermes_cli.profiles import _RESERVED_NAMES
    except Exception:  # pragma: no cover
        return False
    return name in {str(n).lower() for n in _RESERVED_NAMES}


def require_spawnable_children(children, *, surface: str) -> None:
    """Raise ``ValueError`` when any child spec names a reserved, unspawnable
    assignee. Called before the fan-out txn so a bad name cannot half-build a
    graph that then has to be torn down.
    """
    for idx, child in enumerate(children or ()):
        if isinstance(child, dict):
            require_spawnable_assignee(
                child.get("assignee"), surface=f"{surface} child[{idx}]",
            )


def refusal_message(assignee: str) -> str:
    """The loud, actionable refusal naming the reserved set and the escape."""
    name = (assignee or "").strip().lower()
    try:
        from hermes_cli.profiles import _RESERVED_NAMES
        reserved = sorted(n for n in (str(x).lower() for x in _RESERVED_NAMES)
                          if reserved_and_unspawnable(n))
    except Exception:  # pragma: no cover
        reserved = ["hermes", "root", "sudo", "test", "tmp"]
    return (
        f"{name!r} is a RESERVED profile name — `hermes -p {name}` refuses to start, "
        f"so a card filed to it dies spawn_failed/gave_up and no worker ever runs it. "
        f"Reserved and unspawnable: {', '.join(reserved)}. "
        f"Pick a spawnable profile (see `hermes kanban assignees`)."
    )


def require_spawnable_assignee(assignee: Optional[str], *, surface: str) -> None:
    """Raise ``ValueError`` when *assignee* is a reserved, unspawnable name.

    Called from every DB write path that can stamp an assignee, so the refusal
    lands wherever the name was typed (CLI, tool handler, decomposition) instead
    of on a card that can never run. No-op for ``None``/empty (callers use those
    to mean "leave unchanged" or "unassigned").
    """
    if assignee and reserved_and_unspawnable(assignee):
        raise ValueError(f"{surface}: {refusal_message(assignee)}")
