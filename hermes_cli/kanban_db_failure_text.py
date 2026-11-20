"""``tasks.last_failure_error``: who writes it, and when it still blocks a respawn.

The column has two readers — the board/diagnostics, and
:func:`kanban_db_dispatch.check_respawn_guard`, which parks a card ``blocker_auth``
while the text matches a quota/auth pattern (:func:`blocks_respawn`). So the text
may only describe the card's CURRENT condition: a run that ends in a successful
handoff has worked past whatever stamped it, and leaving the old text behind
parked cards in ``review`` forever once the dispatcher's rate-limit requeue
stamped it and the next run handed off for review (hermes-agent board,
2026-11-13/14).

The write (:func:`retire_superseded_failure`, called from
:func:`kanban_db._end_run`), the read (:func:`blocks_respawn`), and the set both
share (:data:`SUPERSEDING_RUN_OUTCOMES`) live here so they cannot drift apart.
Reclaim closes (``stale``/``reclaimed``) are the other non-owner: the close
never worked the card, so it neither produced the text nor owns a stamp an
earlier run left behind — the automatic non-booking closes clear the column
on the close itself, and :data:`RECLAIM_RUN_OUTCOMES` keeps the read side
honest on rows an older install never cleared. Every other outcome keeps the
text live, so a real quota / auth wall still parks the card. Neither caller
loses the failure: the run row and the event log keep it.
"""

from __future__ import annotations

import re
import sqlite3

# Outcomes that mean the run FINISHED ITS JOB and handed the card on: the card
# stands in ``review``/``ready`` for the next profile, so a failure text older
# than this run is history rather than the card's condition.
SUPERSEDING_RUN_OUTCOMES = frozenset({"completed", "review_requested", "changes_requested"})

# Outcomes of reclaim closes (TTL-expired claim, orphan reconcile,
# heartbeat-stale, operator reclaim): the close never worked the card, so it
# produced no failure text of its own and does not OWN a stamp an earlier run
# left behind. The automatic non-booking closes (reconcile, heartbeat-stale)
# also clear the column on the close itself; this set is the read-side half,
# so a card already parked on such a row by an older install unsticks on the
# next tick (t_f7f7eef4).
RECLAIM_RUN_OUTCOMES = frozenset({"stale", "reclaimed"})

# Patterns in last_failure_error that indicate a quota / auth blocker.
# These errors won't resolve by retrying immediately — auto-block instead.
# The auth family is a curated list, not an open `auth\w*` stem: that stem
# also matched ordinary English words like "author"/"authored"/"authoring"/
# "authoritative" in worker progress prose, parking a healthy card forever
# (#117009).
_RESPAWN_BLOCKER_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit|429|403|"
    r"auth|authenticat(?:e|es|ed|ing|ion)|authoriz(?:e|es|ed|ing|ation)|"
    r"authoris(?:e|es|ed|ing|ation)|authz|"
    r"unauthorized|forbidden|billing|subscription|"
    r"access[\s_]denied|permission[\s_]denied|"
    r"invalid[\s_]api[\s_]key)\b",
    re.IGNORECASE,
)


def blocks_respawn(error_text: str, latest_outcome: str | None) -> bool:
    """True when the persisted failure text must park the card ``blocker_auth``.

    The text is blocker evidence only while the LATEST ended run still owns it:
    a successful handoff superseded it, a reclaim close (``stale``/``reclaimed``)
    never worked the card and so owns no earlier stamp, and a plain crash's
    persisted error folds in the worker's own last output — context rather than
    a diagnosis, which may quote benign commands such as ``claude auth status``
    (#117097).

    ``error_text`` must already be connection-decoded (``_lossy_text``); a legacy
    BLOB row reaches here as bytes and the pattern cannot scan it.
    """
    if (
        not error_text
        or latest_outcome in SUPERSEDING_RUN_OUTCOMES
        or latest_outcome in RECLAIM_RUN_OUTCOMES
    ):
        return False
    if latest_outcome == "crashed":
        return False
    return bool(_RESPAWN_BLOCKER_RE.search(error_text))


def retire_superseded_failure(conn: sqlite3.Connection, task_id: str, outcome: str) -> None:
    """Release the closed run's hold on the card.

    Clears ``current_run_id`` always, and on a successful handoff also
    ``last_failure_error`` — the text that handoff superseded. Caller owns the
    transaction (this is :func:`kanban_db._end_run`'s tail, inside the
    transition's own txn, so no reader can see the card in its new lane with the
    old failure still on it).

    ``consecutive_failures`` is deliberately untouched: only ``complete_task``
    treats a success as proof the pathology cleared, so a crash streak still
    accumulates across review rounds.
    """
    conn.execute("UPDATE tasks SET current_run_id = NULL WHERE id = ?", (task_id,))
    if outcome in SUPERSEDING_RUN_OUTCOMES:
        conn.execute("UPDATE tasks SET last_failure_error = NULL WHERE id = ?", (task_id,))
