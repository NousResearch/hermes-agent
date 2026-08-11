"""Cell 2 — consume-once under cross-process concurrent delivery.

Contract clause (arXiv:2608.03836): a parked handoff is claimed by exactly
one consumer, no matter how many independent processes race for it. This is
the cell the paper found failing at saturation 1.0 in 36/40 cells across
deployed frameworks; Hermes' ``claim_handoff`` ships the paper's own repair
shape (single UPDATE with a state predicate, rowcount-checked), which the
tracking-issue probe confirmed. This cell pins it permanently.

8 independent OS processes are released from a file barrier simultaneously;
each calls ``SessionDB.claim_handoff()`` on the same pending session.
"""

from __future__ import annotations

from pathlib import Path

from tests.conformance.persistence._harness import (
    on_disk_journal_mode,
    reap,
    spawn_child,
    wait_for,
)

CLAIMANT = r"""
import sys, time
from pathlib import Path
from hermes_state import SessionDB

db_path = Path({db_path!r})
barrier = Path({barrier!r})
ready = Path({ready!r})

ready.touch()
deadline = time.monotonic() + 55
while not barrier.exists():
    if time.monotonic() > deadline:
        sys.exit(3)
    time.sleep(0.005)

db = SessionDB(db_path=db_path)
won = db.claim_handoff("cell2")
sys.exit(0 if won else 1)
"""

N_CLAIMANTS = 8


def test_exactly_one_claimant_wins(tmp_path):
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    barrier = tmp_path / "go"

    db = SessionDB(db_path=db_path)
    db.create_session("cell2", source="conformance")
    assert db.request_handoff("cell2", "telegram") is not False

    children = []
    ready_files = []
    for i in range(N_CLAIMANTS):
        ready = tmp_path / f"ready-{i}"
        ready_files.append(ready)
        children.append(
            spawn_child(
                CLAIMANT.format(
                    db_path=str(db_path), barrier=str(barrier), ready=str(ready)
                )
            )
        )

    # Barrier: release only when every process is up and polling.
    wait_for(
        lambda: all(r.exists() for r in ready_files),
        what="all claimants at the barrier",
    )
    barrier.touch()

    results = [reap(c) for c in children]
    mode = on_disk_journal_mode(db_path)

    codes = [rc for rc, _, _ in results]
    assert all(rc in (0, 1) for rc in codes), (
        f"[journal_mode={mode}] claimant crashed/timed out: {codes}; "
        f"stderr: {[e[-300:] for _, _, e in results if e]}"
    )
    winners = codes.count(0)
    assert winners == 1, (
        f"[journal_mode={mode}] consume-once violated: {winners} of "
        f"{N_CLAIMANTS} concurrent claimants won"
    )
