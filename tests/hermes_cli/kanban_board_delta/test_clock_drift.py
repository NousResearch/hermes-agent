"""A card whose only change is the clock is not a change the board client shows.

A card the diagnostics rules read the clock for (``stranded_in_ready`` carries "Ready for 49m" and
``data.age_seconds``) is re-evaluated every 30 s, and its JSON differs every time ("49m" becomes
"50m"). Counted as a change, that put every stranded card in every delta: on a board with a few dozen
stranded cards, each "small" poll re-sent hundreds of kilobytes of the same cards.

What the board client shows of a diagnostic is the card badge (``warnings.count``); the drawer reads
the full diagnostics from ``GET /tasks/<id>``. So a card re-evaluated because the clock asked, from
inputs that did not move, is news only when something the client acts on moved: the set of
diagnostics, a severity, a count, an action. These tests count cards sent, not seconds:

* an unchanged board polled a minute later sends no card bodies;
* five real changes among forty stranded cards send those five and nothing else;
* a severity escalation and a diagnostic appearing ARE sent, each exactly when it happens;
* a full snapshot always carries the current clock text, and a client holding snapshot + deltas
  differs from it only in the clock-drifted diagnostic text.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tests.hermes_cli.kanban_board_delta.harness import (
    T0, cards_of, connect, diag_shape, get, merge_delta, poll, seed_done, sent_ids)

MIN = 60
N_DONE = 400


def seed_ready(conn, n: int, *, age_seconds: int, label: str) -> list[str]:
    """``n`` assigned, unclaimed ``ready`` cards that have been ready for ``age_seconds`` at T0."""
    ids = []
    for i in range(n):
        tid = kb.create_task(conn, title=f"{label} {i}", body="x" * 400, assignee="worker")
        ready_since = T0 - age_seconds
        with kbc.write_txn(conn):
            conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (ready_since, tid))
            conn.execute("UPDATE task_events SET created_at = ? WHERE task_id = ? AND kind = 'created'",
                         (ready_since, tid))
        ids.append(tid)
    return ids


@pytest.fixture
def board(env):
    """400 done cards, 40 stranded for 40 min, 30 ready for 27 min, 30 ready for 10 min, ten running."""
    conn = connect()
    done = seed_done(conn, N_DONE)
    a = seed_ready(conn, 40, age_seconds=40 * MIN, label="stranded")
    b = seed_ready(conn, 30, age_seconds=27 * MIN, label="almost")
    c = seed_ready(conn, 30, age_seconds=10 * MIN, label="fresh")
    running = []
    for i in range(10):
        tid = kb.create_task(conn, title=f"running {i}", body="x" * 400, assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        running.append(tid)
    return type("Board", (), {"conn": conn, "done": done, "stranded": a, "almost": b, "fresh": c,
                              "running": running})()


# ---------------------------------------------------------------------------
# the failure: the same stranded cards in every delta
# ---------------------------------------------------------------------------

def test_a_minute_of_clock_alone_sends_no_card_bodies(env, board, clock):
    held = get(env).json()
    stranded = [cards_of(held)[t] for t in board.stranded]
    assert all(diag_shape(c) == [("stranded_in_ready", "warning", 1)] for c in stranded), "the fixture must strand them"

    for _ in range(2):
        clock.advance(MIN)          # the desktop polls about once a minute; nothing else happens
        r, delta = poll(env, held)
        assert delta["delta"] is True
        assert sent_ids(delta) == set(), f"clock alone re-sent {len(sent_ids(delta))} cards ({len(r.content):,} bytes)"
        assert len(r.content) < 600
        held = merge_delta(held, delta)


def test_five_changed_cards_among_stranded_cards_send_five(env, board, clock):
    snap = get(env)
    held = snap.json()
    full_bytes = len(snap.content)
    changed = board.done[100:105]
    for tid in changed:
        kb.add_comment(board.conn, tid, "tester", "a comment that changes the card")
    clock.advance(MIN)

    r, delta = poll(env, held)
    assert sent_ids(delta) == set(changed), f"sent {len(sent_ids(delta))} cards for 5 changes"
    assert len(r.content) < 0.05 * full_bytes


def test_running_cards_with_heartbeats_are_the_only_bounded_extra(env, board, clock):
    """Ten running cards that heartbeat every poll are real changes (the arc shows them); stranded cards add nothing."""
    held = get(env).json()
    clock.advance(MIN)
    for tid in board.running:
        with kbc.write_txn(board.conn):
            board.conn.execute("UPDATE tasks SET last_heartbeat_at = ? WHERE id = ?", (int(clock()), tid))
    r, delta = poll(env, held)
    assert sent_ids(delta) == set(board.running), "only the heartbeating cards may ride"
    assert len(r.content) < 100_000


# ---------------------------------------------------------------------------
# what IS news is still sent, at the poll where it happens
# ---------------------------------------------------------------------------

def test_a_severity_escalation_and_a_new_diagnostic_are_sent_when_they_happen(env, board, clock):
    held = get(env).json()
    # +4 min: "almost" cards (27 min) cross the 30 min threshold and gain a diagnostic. Nothing else moves.
    clock.advance(4 * MIN)
    _, delta = poll(env, held)
    assert sent_ids(delta) == set(board.almost)
    held = merge_delta(held, delta)
    assert all(diag_shape(cards_of(held)[t]) == [("stranded_in_ready", "warning", 1)] for t in board.almost)

    # +17 min: stranded cards are 61 min old, past twice the threshold: warning -> error. "fresh" cards
    # (10 + 21 = 31 min) gain a diagnostic. "almost" cards are 48 min old: still a warning, not news.
    clock.advance(17 * MIN)
    _, delta = poll(env, held)
    assert sent_ids(delta) == set(board.stranded) | set(board.fresh)
    held = merge_delta(held, delta)
    assert all(diag_shape(cards_of(held)[t]) == [("stranded_in_ready", "error", 1)] for t in board.stranded)

    # +2 h: past six times the threshold (3 h): error -> critical for the first group.
    clock.advance(120 * MIN)
    _, delta = poll(env, held)
    assert set(board.stranded) <= sent_ids(delta)
    held = merge_delta(held, delta)
    assert all(diag_shape(cards_of(held)[t]) == [("stranded_in_ready", "critical", 1)] for t in board.stranded)


# ---------------------------------------------------------------------------
# nothing is hidden: the snapshot is always current, and the drift is confined to the clock text
# ---------------------------------------------------------------------------

def test_a_full_snapshot_after_drift_carries_the_current_clock_text(env, board, clock):
    first = cards_of(get(env).json())[board.stranded[0]]["diagnostics"][0]["title"]
    assert first == "Ready for 40m with no worker"
    clock.advance(65 * MIN)
    now_card = cards_of(get(env).json())[board.stranded[0]]["diagnostics"][0]
    assert now_card["title"] == "Ready for 1.8h with no worker"
    assert now_card["data"]["age_seconds"] == 105 * MIN


def test_a_delta_client_differs_from_a_fresh_snapshot_only_in_clock_text(env, board, clock):
    held = get(env).json()
    for _ in range(5):
        clock.advance(MIN)
        _, delta = poll(env, held)
        held = merge_delta(held, delta)
    fresh = get(env).json()
    a, b = cards_of(held), cards_of(fresh)
    assert list(a) == list(b), "the same cards"
    drifted = 0
    for tid in a:
        x, y = dict(a[tid]), dict(b[tid])
        if x == y:
            continue
        keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
        assert keys == ["diagnostics"], f"{tid}: {keys}"
        assert diag_shape(x) == diag_shape(y), f"{tid}: a change the client shows was withheld"
        assert x["warnings"] == y["warnings"]
        drifted += 1
    assert 0 < drifted <= len(board.stranded) + len(board.almost) + len(board.fresh)
