"""Orchestrator reconciliation cards for outcomes no worker will ever resolve.

Two dispatcher dead ends leave work that is terminal but not successful, on a
graph that can no longer move by itself:

* the failure breaker trips on a child (``gave_up``): the child parks in
  ``blocked``, and :func:`~hermes_cli.kanban_db.recompute_ready` promotes a
  waiter only when every parent is ``done``/``archived``, so the orchestrator
  root waits in ``todo`` forever, a lane no dispatcher reads;
* a card reaches ``review`` whose only candidate reviewer is its own
  implementer: spawning it would be self-review, so the review lane refuses,
  and nothing else ever claims a ``review`` row.

Both need the same thing. The orchestrator gets exactly one durable,
schedulable card naming the failure, while the unsuccessful work stays
unsuccessful. That is the ``one_failed`` failure handler of a DAG engine
(Airflow trigger rules, Argo ``depends: x.Failed``) expressed with the
primitives Hermes already has: a card, an assignee, the dispatcher. No edge,
status or gate semantics change, so a failed child still never satisfies a
success dependency -- it just stops being invisible.

Bounded, and bounded in both directions:

* **At most one OPEN card per task per kind.** Repeated ticks, and repeated
  failure generations while the first card is still unresolved, converge on
  that one card. Keying on the failure generation instead would mint a fresh
  card on every unblock/re-exhaust cycle, which is exactly the unblock loop
  ``_rule_block_unblock_cycling`` exists to detect.
* **A resolved card can be re-filed, up to** :data:`MAX_CARDS_PER_TASK`.
  Completing the card without acting must not consume the orchestrator's only
  opportunity while the condition persists; ``status != 'archived'`` alone
  would make a no-op completion silently permanent. The cap stops an
  orchestrator that keeps closing cards without fixing anything from being an
  unbounded card generator.

Concurrency: dispatcher callers hold the board's single-writer
``_dispatch_tick_lock`` for the whole tick, so ticks cannot race each other.
``agent/turn_finalizer.py`` is NOT under that lock -- it records a budget
exhaustion from the worker process on its own connection. The check-then-create
is not atomic and ``idx_tasks_idempotency`` is not UNIQUE, so that one caller
can race a dispatcher tick and produce a second card. Measured worst case: two
cards, both naming the same failure, both closable. It cannot lose a
requirement or turn a failure into a success, which is what the invariants
protect; a UNIQUE index would close it properly but cannot be added to live
boards that may already hold duplicates.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

#: ``idempotency_key`` prefix. Also the loop guard: a reconciliation card that
#: fails must never spawn a reconciliation card for itself.
KEY_PREFIX = "hermes-reconcile"

#: Cap on cards filed for one task and kind over its whole life. Reached only
#: when the orchestrator keeps resolving cards without resolving the condition.
MAX_CARDS_PER_TASK = 3


def orchestrator_profile() -> Optional[str]:
    """``kanban.orchestrator_profile``, or None when no orchestrator owns this
    board. Read like :func:`kanban_db_dispatch.review_dispatch_enabled` reads
    its flag, so it follows the dispatcher's config, not a worker's.

    Canonicalized: ``create_task`` stores assignees through
    ``normalize_profile_name``, so a config saying ``Foreman`` would otherwise
    never match a root stored as ``foreman`` and the root would strand exactly
    as it did before this module existed.
    """
    try:
        from hermes_cli.config import load_config
        from hermes_cli.kanban_db import _canonical_assignee

        value = (load_config() or {}).get("kanban", {}).get("orchestrator_profile")
        value = _canonical_assignee((value or "").strip() or None)
    except Exception as exc:  # pragma: no cover - config I/O is best effort
        logger.debug("kanban reconcile: could not read orchestrator_profile: %s", exc)
        return None
    return value or None


def is_reconciliation_card(conn: sqlite3.Connection, task_id: str) -> bool:
    """Whether this card is itself a reconciliation card. A reconciliation card
    that exhausts its retries must not produce another one, or a failing
    orchestrator would fan out an unbounded chain."""
    row = conn.execute(
        "SELECT idempotency_key FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    key = (row["idempotency_key"] or "") if row is not None else ""
    return key.startswith(KEY_PREFIX + ":")


def waiting_orchestrator_roots(
    conn: sqlite3.Connection, task_id: str, orchestrator: str,
) -> list[str]:
    """Cards parked in ``todo`` behind ``task_id`` that the orchestrator owns.

    These are exactly the cards ``recompute_ready`` can never promote while this
    parent is unsuccessful, and ``promote_task`` refuses by design.
    """
    return [
        row["id"] for row in conn.execute(
            "SELECT t.id FROM task_links l JOIN tasks t ON t.id = l.child_id "
            "WHERE l.parent_id = ? AND t.status = 'todo' AND t.assignee = ? "
            "ORDER BY t.id",
            (task_id, orchestrator),
        ).fetchall()
    ]


def review_implementer(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Implementer recorded on the newest ``review_requested`` event, the
    durable provenance ``request_changes`` already routes rework by."""
    from hermes_cli.kanban_db import _json_dict, _latest_event, _row_get

    event = _latest_event(conn, task_id, "review_requested")
    value = _json_dict(_row_get(event, "payload")).get("implementer")
    return value.strip() if isinstance(value, str) and value.strip() else None


#: Returned instead of a profile name when the card is in ``review`` but its
#: implementer provenance is missing or malformed. Independence cannot be
#: established, so the lane refuses rather than assuming the two differ.
IMPLEMENTER_UNKNOWN = "implementer_unknown"


def self_review_conflict(
    conn: sqlite3.Connection, task_id: str, assignee: Optional[str],
) -> Optional[str]:
    """Why ``assignee`` must not be spawned as this card's reviewer, else None.

    ``request_review`` leaves ``assignee`` untouched when no reviewer is named,
    so the default handoff parks the card in ``review`` still owned by its
    implementer; the review lane would then spawn that same profile with the
    ``sdlc-review`` skill to certify its own work.

    Fails closed on missing provenance too: a card sitting in ``review`` with no
    readable ``review_requested`` implementer cannot be shown to have an
    independent reviewer, and "cannot be shown" is not "is".
    """
    from hermes_cli.kanban_db import _canonical_assignee

    if not assignee:
        return None
    implementer = review_implementer(conn, task_id)
    if implementer is None:
        return IMPLEMENTER_UNKNOWN
    if _canonical_assignee(implementer) != _canonical_assignee(assignee):
        return None
    return implementer


def _key_like(task_id: str, kind: str) -> str:
    return f"{KEY_PREFIX}:{kind}:{task_id}:%"


def _open_card(conn: sqlite3.Connection, task_id: str, kind: str) -> Optional[str]:
    """An unresolved reconciliation card of this kind for this task."""
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key LIKE ? "
        "AND status NOT IN ('done', 'archived') ORDER BY created_at DESC LIMIT 1",
        (_key_like(task_id, kind),),
    ).fetchone()
    return row["id"] if row is not None else None


def _cards_filed(conn: sqlite3.Connection, task_id: str, kind: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key LIKE ?",
        (_key_like(task_id, kind),),
    ).fetchone()
    return int((row["n"] if row is not None else 0) or 0)


def _create(
    conn: sqlite3.Connection, *, kind: str, title: str, body: str, orchestrator: str,
    task_id: str, roots: list[str], board: Optional[str],
) -> Optional[str]:
    """File the card unless one is already open or the cap is reached."""
    from hermes_cli import kanban_db as kb

    if _open_card(conn, task_id, kind) is not None:
        return None
    filed = _cards_filed(conn, task_id, kind)
    if filed >= MAX_CARDS_PER_TASK:
        logger.warning(
            "kanban reconcile: %s already had %d %s card(s) resolved without resolving "
            "the condition; not filing another", task_id, filed, kind,
        )
        return None
    card_id = kb.create_task(
        conn, title=title, body=body, assignee=orchestrator,
        created_by="kanban-reconcile", board=board,
        idempotency_key=f"{KEY_PREFIX}:{kind}:{task_id}:{filed}",
    )
    payload = {"reason": kind, "card": card_id, "orchestrator": orchestrator}
    try:
        with kb.write_txn(conn):
            for owner in (task_id, *roots):
                kb._append_event(conn, owner, "reconcile_requested", payload)
    except Exception as exc:  # pragma: no cover - the card is what matters
        logger.debug("kanban reconcile: could not record the event for %s: %s", task_id, exc)
    return card_id


def _body(header: str, lines: list[str]) -> str:
    return "\n".join([header, "", *lines]).strip() + "\n"


def reconcile_gave_up(
    conn: sqlite3.Connection, task_id: str, *, failures: int, error: Optional[str],
    run_id: Optional[int] = None, board: Optional[str] = None,
) -> Optional[str]:
    """One card for the orchestrator after the breaker abandoned ``task_id``.

    Fires only when a card the orchestrator owns is actually parked behind this
    one: without a waiting root nothing is stranded, and the failed card already
    raises ``repeated_failures`` on its own. Never raises -- reconciliation must
    not be what takes a dispatcher tick down.
    """
    try:
        if is_reconciliation_card(conn, task_id):
            return None
        orchestrator = orchestrator_profile()
        if orchestrator is None:
            return None
        roots = waiting_orchestrator_roots(conn, task_id, orchestrator)
        if not roots:
            return None

        from hermes_cli.kanban_db import get_task

        task = get_task(conn, task_id)
        title = (task.title if task is not None else task_id) or task_id
        roots_line = ", ".join(roots)
        body = _body(
            f"The failure breaker abandoned `{task_id}` ({title!r}) after "
            f"{failures} consecutive failed attempts. It is `blocked` and will "
            f"not be retried. It is NOT done, so it does not satisfy the "
            f"dependency: `{roots_line}` stays in `todo`, out of every dispatch "
            f"lane, until you decide what happens to the requirement it was "
            f"carrying.",
            [
                "## Last error",
                "",
                f"```\n{(error or '(none recorded)')[:1000]}\n```",
                "",
                "## Decide the requirement",
                "",
                f"1. Read `{task_id}` with `kanban_show` (status, runs, events, "
                f"`last_failure_error`) and the root `{roots_line}` with its "
                f"acceptance-criteria comments.",
                "2. Decide exactly one outcome for the acceptance criterion this "
                "card was carrying:",
                "   - **Gap**: cut a replacement card with something material "
                "changed (tighter scope, missing context, a different "
                "specialist, a prerequisite card), link it as a parent of the "
                "root, then `kanban_unlink` the abandoned card from the root so "
                "the root is no longer waiting on work that will never finish.",
                "   - **Blocked**: a human decision or an external prerequisite "
                "is genuinely required. `kanban_block` the root with "
                "`kind=needs_input` and one precise question.",
                "   - **Cancelled**: the goal no longer wants it. Record the "
                "requirement, the reason and who decided on the root, then "
                "`kanban_unlink` the abandoned card.",
                "3. Do not complete or archive the abandoned card to release the "
                "root. It failed; making it read as success is the one outcome "
                "that is always wrong.",
                "",
                "Complete this card once the root can move again or is "
                "explicitly blocked. Completing it without acting does not make "
                "the problem go away: the board files this card again, up to "
                f"{MAX_CARDS_PER_TASK} times, and then stops asking.",
            ],
        )
        return _create(
            conn, kind="gave_up", title=f"Reconcile abandoned card {task_id}",
            body=body, orchestrator=orchestrator, task_id=task_id, roots=roots,
            board=board,
        )
    except Exception as exc:
        logger.warning("kanban reconcile: gave_up handling failed for %s: %s", task_id, exc)
        return None


def reconcile_self_review(
    conn: sqlite3.Connection, task_id: str, implementer: str, *,
    board: Optional[str] = None,
) -> Optional[str]:
    """One card for the orchestrator after the review lane refused to spawn a
    card's implementer as its own reviewer. The card stays in ``review``: it
    needs a reviewer, not a rerun. Never raises."""
    try:
        if is_reconciliation_card(conn, task_id):
            return None
        orchestrator = orchestrator_profile()
        if orchestrator is None:
            return None

        from hermes_cli.kanban_db import get_task

        task = get_task(conn, task_id)
        title = (task.title if task is not None else task_id) or task_id
        if implementer == IMPLEMENTER_UNKNOWN:
            header = (
                f"`{task_id}` ({title!r}) is waiting in `review`, but its "
                f"implementer cannot be read from the board: the "
                f"`review_requested` provenance is missing or malformed. "
                f"Nothing can show that its current assignee is independent of "
                f"whoever wrote the artifact, so the review lane refused to "
                f"dispatch it rather than assume."
            )
            last_step = (
                f"3. If you cannot establish who implemented it, treat the "
                f"review as unverified: `kanban_block` `{task_id}` with "
                f"`kind=needs_input`, or cut a fresh review card naming the "
                f"artifact and the reference it must conform to."
            )
        else:
            header = (
                f"`{task_id}` ({title!r}) is waiting in `review`, but the only "
                f"candidate reviewer is `{implementer}`, the profile that "
                f"implemented it. The review lane refused to dispatch it: a "
                f"profile cannot certify its own work, so the card has no "
                f"independent reviewer and will not be spawned until it gets "
                f"one."
            )
            last_step = (
                f"3. If no installed profile can review it independently, "
                f"`kanban_block` `{task_id}` with `kind=needs_input` and say "
                f"which reviewer is missing. Do not review it yourself and do "
                f"not send it back to `{implementer}`."
            )
        body = _body(
            header,
            [
                "## Route it",
                "",
                f"1. Read `{task_id}` with `kanban_show`: what was implemented, "
                f"and against which acceptance criteria.",
                "2. Assign a reviewer that did not produce the artifact, with "
                "`kanban_reassign` (code and diffs to the code reviewer, "
                "documents to the document reviewer). The card is already in "
                "`review`, so the review lane picks it up on the next tick.",
                last_step,
                "",
                "Complete this card once an independent reviewer owns it. "
                "Completing it without assigning one re-files it, up to "
                f"{MAX_CARDS_PER_TASK} times.",
            ],
        )
        return _create(
            conn, kind="self_review",
            title=f"Route an independent reviewer for {task_id}", body=body,
            orchestrator=orchestrator, task_id=task_id, roots=[], board=board,
        )
    except Exception as exc:
        logger.warning("kanban reconcile: self-review handling failed for %s: %s", task_id, exc)
        return None
