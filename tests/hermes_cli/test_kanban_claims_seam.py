"""Seam + behavior tests for the kanban_db.py claim/lock lifecycle slice.

The six claim/lock functions (``claim_task``, ``claim_review_task``,
``heartbeat_claim``, ``release_stale_claims``, ``reclaim_task``,
``reassign_task``) were extracted byte-for-byte from ``hermes_cli/kanban_db.py``
(window 4226-4697 at pin 01a1037d1e) into ``hermes_cli.kanban_claims`` as part
of the god-file decomposition (epic #78647, target #78632).  ``kanban_db``
re-exports the six names at its bottom, so ``kanban_db.<name> is
kanban_claims.<name>`` must hold and every call site, dispatcher path, and
monkeypatch target keeps resolving.

These tests pin the seam (identity + import order in both directions) and the
claim/lock behavioral contracts the move must preserve: claim CAS rowcount
discipline, heartbeat ownership, stale-claim extension-vs-reclaim,
manual-reclaim + failure-counter reset, reassign reclaim-first semantics,
review-claim CAS, and the ``claim_rejected`` parent gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_claims as kc
from hermes_cli import kanban_db as kb

CLAIM_NAMES = [
    "claim_task",
    "claim_review_task",
    "heartbeat_claim",
    "release_stale_claims",
    "reclaim_task",
    "reassign_task",
]


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _event_payloads(conn, task_id, kind):
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
        (task_id, kind),
    ).fetchall()
    return [json.loads(r["payload"]) for r in rows]


# ---------------------------------------------------------------------------
# Seam: identity + import order
# ---------------------------------------------------------------------------


def test_reexport_identity():
    """kanban_db re-exports the six moved names as the SAME objects."""
    for name in CLAIM_NAMES:
        assert getattr(kb, name) is getattr(kc, name), name
        assert getattr(kc, name).__module__ == "hermes_cli.kanban_claims", name


def _import_order_probe(claims_first: bool) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[2]
    first, second = (
        ("kanban_claims", "kanban_db") if claims_first
        else ("kanban_db", "kanban_claims")
    )
    code = (
        f"import hermes_cli.{first} as m1\n"
        f"import hermes_cli.{second} as m2\n"
        f"names = {CLAIM_NAMES!r}\n"
        "assert all(getattr(m2, n) is getattr(m1, n) for n in names), (\n"
        "    'identity broken for import order: '\n"
        f"    + repr(names)\n"
        ")\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )


def test_import_order_claims_first_then_db():
    """Importing kanban_claims before kanban_db must not break the seam."""
    r = _import_order_probe(claims_first=True)
    assert r.returncode == 0, r.stderr


def test_import_order_db_first_then_claims():
    """Importing kanban_db before kanban_claims must not break the seam."""
    r = _import_order_probe(claims_first=False)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# Claim acquire / release lifecycle
# ---------------------------------------------------------------------------


def test_claim_task_acquires_and_reclaim_task_releases(conn):
    """claim_task CAS-acquires ready->running; reclaim_task releases to ready."""
    tid = kb.create_task(conn, title="slice", assignee="w")
    lock = kb._claimer_id()

    claimed = kb.claim_task(conn, tid, claimer=lock)
    assert claimed is not None
    row = conn.execute(
        "SELECT status, claim_lock, claim_expires FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == lock
    assert row["claim_expires"] is not None

    # CAS discipline: a second claimer must NOT win (rowcount != 1 -> None).
    assert kb.claim_task(conn, tid, claimer=f"{lock}x") is None

    # Manual reclaim releases the claim and closes the run as 'reclaimed'.
    assert kb.reclaim_task(conn, tid, reason="test") is True
    row = conn.execute("SELECT status, claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    assert _event_payloads(conn, tid, "reclaimed")


def test_heartbeat_claim_ownership(conn):
    """heartbeat_claim extends only for the claim owner (claim_lock match)."""
    tid = kb.create_task(conn, title="hb", assignee="w")
    lock = kb._claimer_id()
    kb.claim_task(conn, tid, claimer=lock)

    assert kb.heartbeat_claim(conn, tid, claimer=lock) is True
    assert kb.heartbeat_claim(conn, tid, claimer=f"{lock}x") is False

    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "running"


def test_release_stale_claims_extends_live_pid(conn):
    """A stale-by-TTL claim whose worker PID is alive is EXTENDED, not reclaimed."""
    tid = kb.create_task(conn, title="live", assignee="w")
    lock = kb._claimer_id()
    kb.claim_task(conn, tid, claimer=lock)

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        kb._set_worker_pid(conn, tid, proc.pid)
        conn.execute(
            "UPDATE tasks SET claim_expires = claim_expires - 9999 WHERE id = ?",
            (tid,),
        )
        conn.commit()
        old = conn.execute(
            "SELECT claim_expires FROM tasks WHERE id = ?", (tid,)
        ).fetchone()["claim_expires"]

        # Live pid -> extension, not reclaim: count stays 0.
        assert kb.release_stale_claims(conn) == 0

        row = conn.execute(
            "SELECT status, claim_lock, claim_expires FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["claim_lock"] == lock
        assert row["claim_expires"] > old
        assert _event_payloads(conn, tid, "claim_extended")
    finally:
        proc.terminate()


def test_release_stale_claims_reclaims_expired(conn, monkeypatch):
    """An expired claim whose worker is dead is reclaimed back to ready."""
    tid = kb.create_task(conn, title="stale", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    kb._set_worker_pid(conn, tid, 12345)
    conn.execute(
        "UPDATE tasks SET claim_expires = claim_expires - 9999 WHERE id = ?",
        (tid,),
    )
    conn.commit()

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    assert kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None) == 1

    row = conn.execute("SELECT status, claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    assert _event_payloads(conn, tid, "reclaimed")


def test_reclaim_task_resets_failure_counter(conn):
    """Manual reclaim gives the next retry a fresh failure budget."""
    tid = kb.create_task(conn, title="stuck", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    conn.execute("UPDATE tasks SET consecutive_failures = 5 WHERE id = ?", (tid,))
    conn.commit()

    assert kb.reclaim_task(
        conn, tid, reason="operator", signal_fn=lambda _p, _s: None
    ) is True

    row = conn.execute(
        "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 0


def test_reassign_task_reclaim_first_semantics(conn):
    """reassign refuses a running task unless reclaim_first releases the claim."""
    tid = kb.create_task(conn, title="reassign", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())

    # Running + reclaim_first=False -> RuntimeError from assign_task -> False.
    assert kb.reassign_task(conn, tid, "w2") is False
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "running"  # untouched

    # reclaim_first releases the claim, then the assign lands.
    assert kb.reassign_task(conn, tid, "w2", reclaim_first=True) is True
    row = conn.execute("SELECT status, assignee FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "ready"
    assert row["assignee"] == "w2"


def test_claim_review_task_transitions_review_to_running(conn):
    """claim_review_task CAS-acquires review->running with its own run."""
    tid = kb.create_task(conn, title="review", assignee="w")
    conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
    conn.commit()

    claimed = kb.claim_review_task(conn, tid, claimer=kb._claimer_id())
    assert claimed is not None

    row = conn.execute("SELECT status, claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] is not None

    # Review claims are tracked with a source_status marker on the claimed event.
    evs = _event_payloads(conn, tid, "claimed")
    assert evs and evs[-1].get("source_status") == "review"

    # CAS: a second review claim must fail.
    assert kb.claim_review_task(conn, tid, claimer="someone-else") is None


def test_claim_task_rejects_when_parent_undone(conn):
    """claim_task demotes ready-with-undone-parent and emits claim_rejected."""
    parent = kb.create_task(conn, title="parent", assignee="w")
    child = kb.create_task(conn, title="child", assignee="w")
    kb.link_tasks(conn, parent, child)  # parent not done -> child demoted to todo
    # A racy writer promoted the child while the parent is still undone.
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
    conn.commit()

    assert kb.claim_task(conn, child) is None

    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (child,)).fetchone()
    assert row["status"] == "todo"  # demoted back by the enforcement point
    evs = _event_payloads(conn, child, "claim_rejected")
    assert evs and evs[-1].get("reason") == "parents_not_done"
