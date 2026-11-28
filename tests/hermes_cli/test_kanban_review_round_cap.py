"""Review round cap: ``request_changes`` cannot ping-pong forever.

``kanban_request_changes`` deliberately bypasses block-loop accounting, so a
``request_changes -> re-review -> request_changes`` cycle had **no** automatic
circuit breaker. The only bounds were incidental: the ``dispatch_profiles``
allowlist, ``dispatch_stale_timeout_seconds``, the per-run iteration budget,
and a human eventually noticing.

These tests pin the declared cap (``kanban.review_max_rounds``, default 3):

* below the cap the handback still routes the card to its implementer
  (``ready``) — the cap must never fire early;
* at the cap the handback is refused and the card is **blocked
  ``needs_input``**, because a blocked card waits for the human who owns the
  argument while a triage card waits for a decomposer;
* the block names the ping-pong (rounds, limit, last reason) so the operator
  does not have to reconstruct the loop from the event log;
* the cap deliberately does NOT touch ``consecutive_failures``: a review going
  round and round is not a spawn failure, and spending the implementer's
  work-failure budget on it is the confusion the sibling budget defect is
  about;
* ``review_max_rounds: 0`` disables the cap, so an install that wants the old
  unbounded behaviour can say so out loud.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _write_config(home: Path, body: str) -> None:
    (home / "config.yaml").write_text(body, encoding="utf-8")


def _row(conn, tid):
    return conn.execute(
        "SELECT status, block_kind, block_recurrences, consecutive_failures, "
        "current_run_id, assignee FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (tid,),
    ).fetchall()
    out = [(r["kind"], json.loads(r["payload"]) if r["payload"] else None) for r in rows]
    if kind is not None:
        out = [e for e in out if e[0] == kind]
    return out


def _round(conn, tid, *, reason: str = "off-by-one in the retry counter", **kw):
    """One full review round: implementer works, hands off, reviewer returns it."""
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None, "implementer could not claim the card"
    assert kb.request_review(conn, tid, summary="implementation done",
                             reviewer="reviewer") is True
    assert kb.claim_review_task(conn, tid) is not None, "reviewer could not claim"
    return kb.request_changes(conn, tid, reason=reason, **kw)


# ---------------------------------------------------------------------------
# Below the cap — unchanged behaviour
# ---------------------------------------------------------------------------

def test_handback_below_the_cap_still_routes_to_the_implementer(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        ok, detail = _round(conn, tid, max_rounds=3)

        assert ok is True
        assert detail == "worker", "the implementer must be restored from review_requested"
        row = _row(conn, tid)
        assert row["status"] == "ready"
        assert row["assignee"] == "worker"
        assert row["block_kind"] is None
        assert _events(conn, tid, kind="review_round_cap") == []


# ---------------------------------------------------------------------------
# At the cap — refused and blocked for a human
# ---------------------------------------------------------------------------

def test_cap_blocks_needs_input_and_names_the_ping_pong(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")

        assert _round(conn, tid, max_rounds=2)[0] is True   # round 1
        assert _round(conn, tid, max_rounds=2)[0] is True   # round 2
        assert _row(conn, tid)["status"] == "ready"

        # Round 3 would be the third handback; the cap is 2.
        ok, detail = _round(conn, tid, max_rounds=2, reason="still wrong in the same place")

        assert ok is True, "the call succeeds — it lands the card in blocked, not an error"
        row = _row(conn, tid)
        assert row["status"] == "blocked"
        assert row["block_kind"] == "needs_input"
        assert row["current_run_id"] is None

        # A dedicated event names the loop, so the board can show it without
        # re-deriving rounds from prose.
        (capped,) = _events(conn, tid, kind="review_round_cap")
        payload = capped[1]
        assert payload["rounds"] == 2
        assert payload["limit"] == 2
        assert "still wrong in the same place" in payload["last_reason"]

        # The block reason itself has to make sense to a human on the board.
        blocks = _events(conn, tid, kind="blocked")
        assert blocks, "the cap must record a blocked event"
        reason = blocks[-1][1]["reason"]
        assert "review round cap" in reason
        assert "2" in reason


def test_cap_does_not_spend_the_work_failure_budget(kanban_home: Path) -> None:
    """A ping-pong review is not a spawn failure — it must not arm the breaker."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        _round(conn, tid, max_rounds=1)          # round 1 allowed
        _round(conn, tid, max_rounds=1)          # round 2 -> capped

        row = _row(conn, tid)
        assert row["status"] == "blocked"
        assert (row["consecutive_failures"] or 0) == 0, (
            "review rounds must not draw on the implementer's work-failure budget"
        )
        assert _events(conn, tid, kind="gave_up") == []


# ---------------------------------------------------------------------------
# The declared limit itself
# ---------------------------------------------------------------------------

def test_default_cap_is_three_rounds(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        for _ in range(3):
            assert _round(conn, tid)[0] is True
            assert _row(conn, tid)["status"] == "ready"

        assert _round(conn, tid)[0] is True
        assert _row(conn, tid)["status"] == "blocked"
        (capped,) = _events(conn, tid, kind="review_round_cap")
        assert capped[1]["limit"] == kb.DEFAULT_REVIEW_MAX_ROUNDS == 3


def test_config_key_is_honoured(kanban_home: Path) -> None:
    _write_config(kanban_home, "kanban:\n  review_max_rounds: 1\n")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        assert _round(conn, tid)[0] is True
        assert _row(conn, tid)["status"] == "ready"

        assert _round(conn, tid)[0] is True
        assert _row(conn, tid)["status"] == "blocked"
        (capped,) = _events(conn, tid, kind="review_round_cap")
        assert capped[1]["limit"] == 1


def test_zero_disables_the_cap(kanban_home: Path) -> None:
    """An install that wants the old unbounded loop can say so out loud."""
    _write_config(kanban_home, "kanban:\n  review_max_rounds: 0\n")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        for _ in range(5):
            assert _round(conn, tid)[0] is True
            assert _row(conn, tid)["status"] == "ready"
        assert _events(conn, tid, kind="review_round_cap") == []


# ---------------------------------------------------------------------------
# A second cap is no longer "who is right" — it escalates
# ---------------------------------------------------------------------------

def test_a_second_cap_after_an_unblock_escalates_to_triage(kanban_home: Path) -> None:
    """The cap blocks once for the human who owns the argument; the *repeat* cap
    takes the standard recurrence ladder to ``triage``, because a pair that
    resumed the same disagreement after a human said "go on" needs a
    re-specification, not another opinion."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        assert _round(conn, tid, max_rounds=1)[0] is True   # round 1 lands
        assert _round(conn, tid, max_rounds=1)[0] is True   # round 2 -> capped
        assert _row(conn, tid)["status"] == "blocked"
        assert _row(conn, tid)["block_kind"] == "needs_input"

        assert kb.unblock_task(conn, tid) is True

        # Whoever holds the card now (review, when that is where it left off) sends
        # the same disagreement back one more time.
        resumed = _row(conn, tid)["status"]
        if resumed == "review":
            assert kb.claim_review_task(conn, tid) is not None
            ok, detail = kb.request_changes(conn, tid, reason="same objection", max_rounds=1)
        else:
            assert _round(conn, tid, max_rounds=1, reason="same objection")[0] is True
            ok, detail = True, None
        assert ok is True, detail

        row = _row(conn, tid)
        assert row["status"] == "triage", (
            "a repeat cap must escalate through the recurrence ladder, not block again"
        )
        assert row["block_kind"] == "needs_input"
        assert len(_events(conn, tid, kind="review_round_cap")) == 2
        # The cap refuses the handback, so it never records a round of its own.
        assert len(_events(conn, tid, kind="changes_requested")) == 1
