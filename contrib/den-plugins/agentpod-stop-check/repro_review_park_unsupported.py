"""Reproduction: there is no supported way to park a card that is in the
``review`` phase and waiting on a known external schedule.

Two scenarios, both on an isolated temporary board with no real card touched:

(a) ``schedule_task`` refuses a ``review`` card, and the next dispatcher pass
    claims it again -> the same unchanged review work is re-spawned.
(b) a reviewed card that legitimately waits twice on the same external gate
    hits ``BLOCK_RECURRENCE_LIMIT`` and is routed to ``triage`` -- the real
    wake is lost and a human is pulled in for a wait that was never ambiguous.

Run it directly:

    python contrib/den-plugins/agentpod-stop-check/repro_review_park_unsupported.py

Exit 0 == both defects reproduced. This is evidence for the carve-out card,
NOT part of the plugin test suite.
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _board(tmp: Path):
    os.environ["HERMES_HOME"] = str(tmp)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as _kbc

    return kb, _kbc.connect(tmp / "board.db")


def scenario_a(kb, conn) -> bool:
    """review -> schedule_task refused; dispatcher re-claims the same work."""
    tid = kb.create_task(conn, title="repro-a", assignee="software-engineer")
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    assert kb.request_review(
        conn, tid, summary="impl done", reviewer="reviewer",
        expected_run_id=run_id,
    ), "request_review failed"
    assert kb.get_task(conn, tid).status == "review"

    parked = kb.schedule_task(conn, tid, reason="waiting on nightly 03:29Z")
    status_after = kb.get_task(conn, tid).status
    reclaimed = kb.claim_review_task(conn, tid)

    print("(a) schedule_task(review) returned:      ", parked)
    print("(a) status after park attempt:           ", status_after)
    print("(a) review re-claimed (respawn):         ", reclaimed is not None)
    return (parked is False) and status_after == "review" and reclaimed is not None


def scenario_b(kb, conn) -> bool:
    """A twice-repeated legitimate external wait degrades to human triage."""
    tid = kb.create_task(conn, title="repro-b", assignee="software-engineer")
    kb.claim_task(conn, tid)
    kb.block_task(conn, tid, reason="waiting on nightly 03:29Z", kind="needs_input")
    assert kb.get_task(conn, tid).status == "blocked"
    kb.unblock_task(conn, tid)
    kb.claim_task(conn, tid)
    kb.block_task(conn, tid, reason="still waiting on nightly 03:29Z", kind="needs_input")
    final = kb.get_task(conn, tid).status

    print("(b) status after 2nd identical wait:     ", final)
    print("(b) BLOCK_RECURRENCE_LIMIT:              ", kb.BLOCK_RECURRENCE_LIMIT)
    return final == "triage"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="review-park-repro-"))
    kb, conn = _board(tmp)
    try:
        a = scenario_a(kb, conn)
        print()
        b = scenario_b(kb, conn)
        print()
        print("RESULT (a) review park unsupported / respawn:", "REPRODUCED" if a else "not reproduced")
        print("RESULT (b) external wait -> triage:          ", "REPRODUCED" if b else "not reproduced")
        return 0 if (a and b) else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
