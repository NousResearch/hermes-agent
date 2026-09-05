"""Cell 3 — crash mid-compression-rotation: the atomic-publication contract.

Contract clause: a compression rotation (close parent + publish continuation
child + write compacted handoff) is visible **entirely or not at all**. A hard
crash at any point must never yield an orphan — a parent ended with
``end_reason='compression'`` and no continuation child — because that is the
state every reader treats as unreachable (the P1 in #80337; the recovery
merged in #80487 exists to drain the pre-atomicity population of exactly
these rows).

The forensics on #80337 established the contract holds architecturally on
current main (``publish_compression_child`` runs parent-close + child-insert
+ handoff in one transaction on one handle). This cell pins it empirically:
a writer process performs rotations in a tight loop through the REAL leased
path (``try_acquire_compression_lock`` → ``publish_compression_child``) and
is SIGKILLed mid-loop; recovery then audits every rotation for atomicity.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tests.conformance.persistence._harness import (
    effective_mode_or_skip,
    integrity_ok,
    kill9_and_reap,
    make_hermes_home,
    spawn_child,
    wait_for,
)

ROTATOR = r"""
import json, sys
from pathlib import Path
from hermes_state import SessionDB

db_path = Path({db_path!r})
journal = Path({journal!r})
db = SessionDB(db_path=db_path)

i = 0
with journal.open("a", buffering=1) as j:
    while True:
        parent = f"parent-{{i}}"
        child = f"child-{{i}}"
        db.create_session(parent, source="conformance")
        db.append_message(parent, "user", content="pre-rotation turn")
        holder = f"cell3:pid={{__import__('os').getpid()}}"
        assert db.try_acquire_compression_lock(parent, holder)
        db.publish_compression_child(
            parent_session_id=parent,
            child_session_id=child,
            source="conformance",
            messages=[{{"role": "user", "content": "compacted handoff"}}],
            compression_lock_holder=holder,
        )
        j.write(json.dumps({{"i": i}}) + "\n")
        j.flush()
        i += 1
"""


def _completed_rotations(journal: Path) -> int:
    if not journal.exists():
        return 0
    n = 0
    for line in journal.read_text().splitlines():
        try:
            json.loads(line)
            n += 1
        except json.JSONDecodeError:
            continue  # torn final line under SIGKILL — unacknowledged
    return n


@pytest.mark.parametrize("requested_mode", [None, "DELETE", "WAL"])
def test_rotation_is_atomic_under_sigkill(tmp_path, requested_mode):
    db_path = tmp_path / "state.db"
    journal = tmp_path / "rotations.jsonl"

    child_env = {}
    if requested_mode is not None:
        # Steer the CHILD's own resolver via an isolated HERMES_HOME config:
        # pre-seeding the file alone is not enough — SessionDB.__init__ runs
        # apply_wal_with_fallback(), which upgrades a non-WAL file to WAL
        # whenever the configured mode says so (the resolver's downgrade
        # gates may still refuse; effective_mode_or_skip audits post-run).
        child_env["HERMES_HOME"] = str(
            make_hermes_home(tmp_path, requested_mode.lower())
        )

    child = spawn_child(
        ROTATOR.format(db_path=str(db_path), journal=str(journal)), env=child_env
    )
    try:
        wait_for(
            lambda: _completed_rotations(journal) >= 50,
            what=">=50 completed rotations",
            child=child,
        )
        assert child.poll() is None, (
            f"rotator exited early (rc={child.returncode}): "
            f"{(child.stderr.read() if child.stderr else b'').decode(errors='replace')[-1500:]}"
        )
    finally:
        kill9_and_reap(child)

    # A leg that ran in a different mode than requested is no evidence for
    # the requested mode — skip it rather than silently double-counting WAL.
    mode = effective_mode_or_skip(db_path, requested_mode)
    acked = _completed_rotations(journal)
    assert acked >= 50, (
        f"[journal_mode={mode}] only {acked} rotations acknowledged before "
        "the kill — harness window too small"
    )

    conn = sqlite3.connect(str(db_path))
    try:
        # THE contract: no compression-ended parent may lack a child row.
        orphans = conn.execute(
            """
            SELECT p.id FROM sessions p
            WHERE p.end_reason = 'compression'
              AND NOT EXISTS (
                SELECT 1 FROM sessions c WHERE c.parent_session_id = p.id
              )
            """
        ).fetchall()
        assert not orphans, (
            f"[journal_mode={mode}] atomicity violated: {len(orphans)} "
            f"compression-ended parent(s) with no continuation "
            f"({[o[0] for o in orphans[:5]]}) — the #80337 orphan shape"
        )

        # Completeness the other way: every acknowledged rotation is fully
        # visible (parent ended AND child present).
        for i in range(acked):
            parent_row = conn.execute(
                "SELECT end_reason FROM sessions WHERE id = ?",
                (f"parent-{i}",),
            ).fetchone()
            child_row = conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (f"child-{i}",)
            ).fetchone()
            assert parent_row and parent_row[0] == "compression" and child_row, (
                f"[journal_mode={mode}] acknowledged rotation {i} not fully "
                f"visible after recovery (parent={parent_row}, child={child_row})"
            )

        # The interrupted trailing rotation (if any) must be all-or-nothing:
        # either invisible or complete — checked by the orphan query above.
    finally:
        conn.close()

    assert integrity_ok(db_path), f"[journal_mode={mode}] integrity_check failed"
