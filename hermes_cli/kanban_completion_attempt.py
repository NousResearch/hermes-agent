"""Identity of ONE ``complete_task`` attempt, and the receipt it left behind.

Several attempts on the same card are ordinary: a worker retries, an operator
completes by hand, a dispatcher and a human race for the same gate. Both opt-in
completion gates persist their receipt in the very transaction that refuses the
transition, so "why did MY attempt fail?" can only be answered by reading back
the receipt THIS attempt wrote.

An event-id floor alone cannot do that. The floor is read before the attempt
starts and the gates verify GitHub/git with no transaction open, so a receipt
another connection writes meanwhile lands above the floor too — and reporting
it tells the operator their completion failed for a condition some other
attempt hit. Each attempt therefore carries its own opaque id, stamped onto
every receipt it persists, and a reader matches that id exactly. The floor
survives only as a cheap bound on how far back the lookup has to scan.

Every surface that completes a card reads its refusal back through this one
module — the CLI and the ``kanban_complete`` tool both — so one piece of SQL
does the correlation and one rendering answers "why did MY attempt fail?".
"""
from __future__ import annotations

from typing import Optional

import uuid

#: Receipt kinds the two opt-in completion gates append, and how to name one of
#: them to an operator.
GATE_LABELS = {
    "integration_acceptance": "integration gate",
    "pr_acceptance": "PR acceptance",
}
COMPLETION_GATE_KINDS = tuple(GATE_LABELS)
#: The receipt field carrying the attempt that wrote it.
ATTEMPT_ID_KEY = "completion_attempt_id"


def new_completion_attempt_id() -> str:
    """An id for one completion attempt.

    Opaque on purpose: it is only ever compared for equality, never parsed,
    ordered or shown, so nothing can come to depend on its shape.
    """
    return uuid.uuid4().hex


def stamp_attempt(receipt: dict, attempt_id: str | None) -> dict:
    """Record which attempt a receipt belongs to, returning that receipt."""
    if isinstance(receipt, dict) and attempt_id:
        receipt[ATTEMPT_ID_KEY] = attempt_id
    return receipt


def latest_event_id(conn, task_id: str) -> int:
    """Highest event id on the task right now — a cheap floor bounding how far
    back a refusal lookup has to scan.

    Read BEFORE the attempt starts: every receipt the attempt could write lands
    above it, and the id match is what makes the answer exact.
    """
    row = conn.execute(
        "SELECT MAX(id) AS id FROM task_events WHERE task_id = ?", (task_id,)).fetchone()
    return int(row["id"] or 0) if row is not None else 0


def refusal_receipt(conn, task_id: str, event_floor: int, attempt_id: Optional[str]):
    """``(kind, receipt)`` of the gate receipt ``attempt_id`` persisted, else None.

    Only an exact id match counts, so a receipt from a concurrent attempt — or
    one left behind by an earlier attempt below the floor — is never reported as
    this attempt's reason.
    """
    if not attempt_id:
        return None
    from hermes_cli.kanban_db import _json_dict

    for row in conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? AND id > ? "
        f"AND kind IN ({', '.join('?' * len(COMPLETION_GATE_KINDS))}) ORDER BY id DESC",
        (task_id, event_floor, *COMPLETION_GATE_KINDS),
    ):
        receipt = _json_dict(row["payload"])
        if receipt.get(ATTEMPT_ID_KEY) == attempt_id:
            return row["kind"], receipt
    return None


def completion_refusal(conn, task_id: str, event_floor: int,
                       attempt_id: Optional[str]) -> Optional[str]:
    """Why ``complete_task`` actually refused THIS attempt, when it left a
    receipt behind; ``None`` when it left none.

    Both opt-in gates persist their receipt in the very transaction that
    refuses the transition, stamped with the attempt that wrote it — and only an
    exact ``attempt_id`` match is rendered here. The event floor alone was not
    enough: the gates verify GitHub/git with no transaction open, so a receipt
    another connection writes meanwhile also sits above this attempt's floor,
    and reporting it blames this caller for a condition another attempt hit.

    The card's ``last_failure_error`` is deliberately NOT a fallback. It is one
    shared column whose last writer wins, so reading it would re-introduce
    exactly the cross-attempt blame this id exists to prevent; both gates always
    carry their own ``recovery`` line, so an attempt with a receipt has prose of
    its own anyway.

    ``None`` leaves the caller its generic message: the refusal really was an
    unknown id, a terminal/ineligible status, unsatisfied parents or a lost run
    race, none of which a gate explains.
    """
    found = refusal_receipt(conn, task_id, event_floor, attempt_id)
    if found is None:
        return None
    kind, receipt = found
    if receipt.get("ok"):
        return None
    phase = str(receipt.get("phase") or "").strip()
    verdict = str(receipt.get("classification") or "").strip()
    if verdict and verdict != phase:
        # PR acceptance: the verdict is the answer and ``phase`` only records how
        # far collection got (a red required check leaves it at the harmless
        # ``stale_recheck``), so neither alone tells the operator what happened.
        stop = verdict + (f" at the {phase} phase" if phase else "")
    else:
        stop = phase or "refused"
    prose = [str(receipt.get(key) or "").strip() for key in ("detail", "recovery")]
    return " ".join([f"cannot complete {task_id}: {GATE_LABELS[kind]} refused — {stop}.",
                     *(part if part.endswith((".", "!", "?")) else f"{part}."
                       for part in prose if part)])
