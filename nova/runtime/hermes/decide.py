"""Acting on work that is already on the board.

The control plane was read-only until now, and the audit that prompted this said why the
gap mattered: *"an approval with no identity is not an approval."* Identity arrived with the
principals store, so this is what it was for.

**Four verbs, not "set status".** A control plane that can write any state can write an
inconsistent one — a task marked ready whose parents never finished, a review closed with a
run still claimed. The runtime's own transitions keep that accounting straight, so NOVA asks
for the transition it wants and lets the runtime decide whether it is legal. The answer
"no, that task moved on while you were reading it" is a normal answer here, not an error.

**Through the runtime's own API, never raw SQL** — the same rule, and the same reasons, as
``submit.py``: ``kanban_db`` writes the event rows the dashboard and the notifier read, and
holds whatever locking the current schema requires. Imported lazily, inside the functions,
because ``nova`` must load with only the standard library and PyYAML present.

**The actor is carried, not implied.** ``promote_task`` takes one and writes it into the
runtime's own event log, so the runtime's record and NOVA's audit name the same human.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nova.errors import RuntimeAdapterError
from nova.runtime.base import WORK_ACTIONS, WorkDecision

#: Prefix on a note NOVA attaches, so a board carrying both worker chatter and operator
#: instruction says which is which without a join. The worker reads the whole comment.
NOTE_PREFIX = "OPERATOR"

#: Prefix on a rejection, matching the word the runtime's own operator command writes, so a
#: board shows one vocabulary whether the reason arrived through NOVA or through the CLI.
REJECT_PREFIX = "CHANGES REQUESTED"


def decide(
    home: Path,
    task_id: str,
    action: str,
    *,
    actor: str,
    reason: str = "",
    note: str = "",
) -> WorkDecision:
    """Apply one decision. Returns what happened rather than raising on a refusal."""
    if action not in WORK_ACTIONS:
        raise RuntimeAdapterError(
            f"{action!r} is not a work action; expected one of {', '.join(WORK_ACTIONS)}"
        )
    if not actor:
        # Refused rather than defaulted. A decision recorded against "someone" is the exact
        # failure the identity work existed to prevent, and a default would hide it.
        raise RuntimeAdapterError("a work decision needs an actor; refusing to record an anonymous one")

    kb, kbc = _runtime_modules()

    with kbc.connect_closing() as connection:
        task = kb.get_task(connection, task_id)
        if task is None:
            return WorkDecision(
                action=action,
                task_id=task_id,
                applied=False,
                reason="no such work item on this board",
            )
        before = getattr(task, "status", "") or ""

        if action == "release":
            ok, error = kb.promote_task(connection, task_id, actor=actor, reason=reason or None)
            if not ok:
                return WorkDecision(
                    action, task_id, False,
                    reason=error or f"cannot be released from {before!r}",
                    resulting_status=before,
                )

        elif action == "reject":
            if not reason:
                # The reason is the entire point: it is what the worker reads and re-runs
                # against. A rejection with no reason is a task bounced into a loop.
                return WorkDecision(
                    action, task_id, False,
                    reason="a rejection needs a reason — it is what the worker reads and acts on",
                    resulting_status=before,
                )
            # Two different rejections wear the same word, and only a live board shows it.
            #
            # ``request_changes`` closes an *active reviewer run* — a reviewer agent claimed
            # the task out of ``review`` and is handing it back. A human looking at a queue
            # of tasks awaiting review has claimed nothing, so that call refuses with "not
            # in an active review run", which is correct and unhelpful.
            #
            # The operator's rejection is ``reopen_review_task`` plus the comment carrying
            # the reason, which is exactly what the runtime's own operator command does.
            # Try the reviewer path first so a genuine reviewer run is closed properly, and
            # fall back to the operator path rather than reporting a refusal to a human who
            # did nothing wrong.
            ok, detail = kb.request_changes(connection, task_id, reason=reason)
            if not ok:
                if not kb.reopen_review_task(connection, task_id):
                    return WorkDecision(
                        action, task_id, False,
                        reason=detail or f"is {before!r}, not awaiting review",
                        resulting_status=before,
                    )
                # Redacted the runtime's own way: a rejection reason is durable, a human
                # wrote it in a hurry, and secrets end up in exactly that kind of text.
                safe = str(kb.redact_review_value(reason)).strip() or reason
                kb.add_comment(connection, task_id, actor, f"{REJECT_PREFIX}: {safe}")

        elif action == "resume":
            if not kb.unblock_task(connection, task_id):
                return WorkDecision(
                    action, task_id, False,
                    reason=f"cannot be resumed from {before!r}",
                    resulting_status=before,
                )

        elif action == "annotate":
            if not note:
                return WorkDecision(
                    action, task_id, False, reason="an empty note is not worth recording",
                    resulting_status=before,
                )
            kb.add_comment(connection, task_id, actor, f"{NOTE_PREFIX}: {note}")

        after = kb.get_task(connection, task_id)
        status = (getattr(after, "status", "") if after is not None else "") or before

    return WorkDecision(action=action, task_id=task_id, applied=True, resulting_status=status)


def _runtime_modules() -> tuple[Any, Any]:
    """The runtime's work-store API. See this module's docstring for why it is imported here."""
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
    except ImportError as exc:  # pragma: no cover — requires the runtime to be absent
        raise RuntimeAdapterError(
            "the Hermes runtime is not importable from this process, so NOVA cannot act on "
            f"its board ({exc}). Run the control plane where the runtime is installed"
        ) from exc
    return kb, kbc
