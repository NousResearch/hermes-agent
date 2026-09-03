"""Cell 1 — prefix continuation + recovery determinism under real SIGKILL.

Contract clause (arXiv:2608.03836): every append the store *acknowledged*
survives a hard crash; recovery yields a contiguous prefix (no holes, no
duplicates); two independent recovery passes see the identical transcript.

Adapted from the spot-probe in the tracking issue (#80921), which ran ~29.5K
messages to a SIGKILL with zero lost-after-return. Scaled down for CI: the
parent kills the writer once >= 200 acknowledged appends are journaled — the
property assertions are identical, only the exposure window is shorter.
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

WRITER = r"""
import json, sys
from pathlib import Path
from hermes_state import SessionDB

db_path = Path({db_path!r})
journal = Path({journal!r})
db = SessionDB(db_path=db_path)
db.create_session("cell1", source="conformance")
i = 0
with journal.open("a", buffering=1) as j:
    while True:
        rowid = db.append_message("cell1", "user", content=f"m{{i}}")
        # fsync-journal the acknowledged index AFTER append returns —
        # exactly the probe's definition of "claimed durable".
        j.write(json.dumps({{"i": i, "rowid": rowid}}) + "\n")
        j.flush()
        i += 1
"""


def _acknowledged(journal: Path) -> list[dict]:
    if not journal.exists():
        return []
    out = []
    for line in journal.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A torn final line is expected under SIGKILL: the write to the
            # journal itself was interrupted. That index was never fully
            # acknowledged to the harness, so it is out of scope.
            continue
    return out


def _recover(db_path: Path) -> list[tuple]:
    """One independent recovery pass in a fresh connection/process context."""
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT id, content FROM messages WHERE session_id='cell1' ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


@pytest.mark.parametrize("requested_mode", [None, "DELETE", "WAL"])
def test_acknowledged_appends_survive_sigkill(tmp_path, requested_mode):
    db_path = tmp_path / "state.db"
    journal = tmp_path / "acked.jsonl"

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
        WRITER.format(db_path=str(db_path), journal=str(journal)), env=child_env
    )
    try:
        wait_for(
            lambda: len(_acknowledged(journal)) >= 200,
            what=">=200 acknowledged appends",
            child=child,
        )
        # The kill must interrupt a LIVE writer — a child that already
        # exited would turn this into a clean-shutdown test.
        assert child.poll() is None, (
            f"writer exited early (rc={child.returncode}): "
            f"{(child.stderr.read() if child.stderr else b'').decode(errors='replace')[-1500:]}"
        )
    finally:
        kill9_and_reap(child)

    # A leg that ran in a different mode than requested is no evidence for
    # the requested mode — skip it rather than silently double-counting WAL.
    mode = effective_mode_or_skip(db_path, requested_mode)
    acked = _acknowledged(journal)
    assert len(acked) >= 200, (
        f"[journal_mode={mode}] only {len(acked)} acknowledged appends "
        "journaled before the kill — harness window too small"
    )

    pass1 = _recover(db_path)
    pass2 = _recover(db_path)

    # Zero lost-after-return: every acknowledged index is present.
    recovered_contents = {row[1] for row in pass1}
    lost = [a for a in acked if f"m{a['i']}" not in recovered_contents]
    assert not lost, (
        f"[journal_mode={mode}] {len(lost)} acknowledged appends lost after "
        f"SIGKILL (first: {lost[:3]})"
    )

    # Contiguous prefix: indices 0..N-1 with no holes and no duplicates.
    indices = sorted(int(row[1][1:]) for row in pass1)
    assert indices == list(range(len(indices))), (
        f"[journal_mode={mode}] recovered transcript is not a contiguous "
        "prefix (holes or duplicates present)"
    )

    # Recovery determinism: two independent passes identical.
    assert pass1 == pass2, f"[journal_mode={mode}] recovery passes diverge"

    assert integrity_ok(db_path), f"[journal_mode={mode}] integrity_check failed"
