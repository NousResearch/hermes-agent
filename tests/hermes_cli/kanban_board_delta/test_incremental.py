"""GET /board builds only what changed and sends only what changed.

A board with a long history is thousands of cards, nearly all of them done and never changing again.
The endpoint used to rebuild every card (diagnostics over all of them) and send all of it on every
poll. These tests count WORK DONE, not seconds:

* an unchanged poll is a few hundred bytes and rebuilds no card;
* a poll after N changed cards rebuilds N cards and sends well under 5% of the snapshot;
* snapshot + deltas equals a fresh snapshot after a scripted run of every kind of write,
  including writes that leave no event (a claim extension, a raw column update);
* the cards equal the board as it was built before the cache (``reference_board``) on the same
  database, diagnostics included.
"""

from __future__ import annotations

import copy
import time

import pytest

from hermes_cli import kanban_board_delta as kbdelta
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as kd
from tests.hermes_cli.kanban_board_delta.harness import (
    BOARD, T0, cards_of, client_for, connect, diff_boards, get, load_plugin, merge_delta, modulo_clock_text,
    poll, reference_board, seed_done, without_etag)

N_DONE = 2_000


def add_block_cycles(conn, tid: str, end_ts: int, cycles: int = 4) -> None:
    """``cycles`` blocked-after-unblocked cycles ending at ``end_ts``: what ``block_unblock_cycling``
    counts inside its 24-hour window, on any status (done included)."""
    kinds = ["blocked"] + ["unblocked", "blocked"] * cycles
    for i, kind in enumerate(kinds):
        conn.execute("INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?,?,?,?,?)",
                     (tid, None, kind, None, end_ts - (len(kinds) - 1 - i) * 60))


def seed_active(conn) -> dict[str, str]:
    """Ten cards in every non-terminal shape the board has."""
    a: dict[str, str] = {}
    for i in range(1, 7):
        a[f"r{i}"] = kb.create_task(conn, title=f"ready {i}", body="x" * 400, assignee="worker")
    for i in range(1, 5):
        a[f"t{i}"] = kb.create_task(conn, title=f"waiting {i}", body="x" * 400, assignee="worker",
                                    parents=[a["r1"]])
    assert kb.claim_task(conn, a["r2"]) is not None
    assert kb.claim_task(conn, a["r3"]) is not None
    assert kb.block_task(conn, a["r3"], reason="need an answer", kind="needs_input")
    return a


class DiagCounter:
    """Counts rule evaluations: one per card whose diagnostics were computed."""

    def __init__(self, monkeypatch) -> None:
        self.calls = 0
        real = kd.compute_task_diagnostics

        def counted(*a, **k):
            self.calls += 1
            return real(*a, **k)

        monkeypatch.setattr(kd, "compute_task_diagnostics", counted)

    def take(self) -> int:
        n, self.calls = self.calls, 0
        return n


# ---------------------------------------------------------------------------
# work done: an unchanged poll costs nothing, a small change costs a small poll
# ---------------------------------------------------------------------------

def test_unchanged_poll_is_a_few_bytes_and_rebuilds_nothing(env, monkeypatch):
    conn = connect()
    seed_done(conn, N_DONE)
    seed_active(conn)
    counter = DiagCounter(monkeypatch)

    first = get(env)
    assert first.status_code == 200
    assert counter.take() == N_DONE + 10, "the first build computes every card once"

    r, body = poll(env, first.json())
    assert body["unchanged"] is True and body["delta"] is True
    assert len(r.content) < 400
    assert counter.take() == 0, "an unchanged poll must rebuild no card"

    # The etag is a body/URL token only. No HTTP validator: a 304 would let a browser cache keep
    # serving a snapshot whose ``now`` and ``age`` are old.
    assert "etag" not in {k.lower() for k in first.headers}
    again = get(env, headers={"If-None-Match": f'"{first.json()["etag"]}"'})
    assert again.status_code == 200 and "columns" in again.json()


def test_poll_after_five_changed_cards_is_small_and_rebuilds_five(env, monkeypatch):
    conn = connect()
    done = seed_done(conn, N_DONE)
    seed_active(conn)
    counter = DiagCounter(monkeypatch)
    snap0 = get(env)
    full_bytes = len(snap0.content)
    held = snap0.json()
    counter.take()

    changed = done[100:105]
    for tid in changed:
        kb.add_comment(conn, tid, "tester", "a comment that changes the card")

    r, delta = poll(env, held)
    assert delta["delta"] is True and "unchanged" not in delta
    assert sorted(c["id"] for c in delta["cards"]) == sorted(changed)
    assert counter.take() == 5, "exactly the changed cards are rebuilt"
    assert len(r.content) < 0.05 * full_bytes, (len(r.content), full_bytes)

    merged = merge_delta(held, delta)
    fresh = get(env).json()
    assert diff_boards(without_etag(merged), without_etag(fresh)) == []
    assert diff_boards(without_etag(fresh), reference_board(env.plugin)) == []


def test_events_only_delta_without_a_token_still_merges_exactly(env):
    conn = connect()
    done = seed_done(conn, 300)
    seed_active(conn)
    held = get(env).json()
    kb.add_comment(conn, done[7], "tester", "later")
    delta = get(env, params=f"since={held['latest_event_id']}").json()
    assert delta["delta"] is True
    assert done[7] in {c["id"] for c in delta["cards"]}
    merged = merge_delta(held, delta)
    assert diff_boards(without_etag(merged), reference_board(env.plugin)) == []


def test_full_snapshot_equals_the_original_algorithm_and_keeps_the_old_shape(env):
    conn = connect()
    seed_done(conn, 200)
    seed_active(conn)
    snap = get(env).json()
    assert list(snap)[:5] == ["columns", "tenants", "assignees", "latest_event_id", "now"]
    assert isinstance(snap["etag"], str) and snap["etag"]
    assert "delta" not in snap
    assert diff_boards(without_etag(snap), reference_board(env.plugin)) == []
    conn.execute("UPDATE tasks SET status = 'done' WHERE id = (SELECT id FROM tasks WHERE status = 'todo' LIMIT 1)")
    conn.execute("UPDATE tasks SET status = 'archived' WHERE id = (SELECT id FROM tasks WHERE status = 'done' LIMIT 1)")
    assert all(t["status"] != "archived" for col in get(env).json()["columns"] for t in col["tasks"])
    assert diff_boards(without_etag(get(env).json()), reference_board(env.plugin)) == []
    assert diff_boards(without_etag(get(env, "include_archived=true").json()),
                       reference_board(env.plugin, include_archived=True)) == []


def test_the_in_process_dict_form_is_the_same_board(env):
    conn = connect()
    seed_done(conn, 30)
    seed_active(conn)
    assert diff_boards(without_etag(env.plugin.get_board(None, False, None, None, None)),
                       reference_board(env.plugin)) == []


def test_a_since_the_server_cannot_honour_gets_a_full_snapshot(env):
    conn = connect()
    seed_done(conn, 20)
    snap = get(env).json()
    for bad in ("since=999999999", "since=-3", "since=abc", "since="):
        r = get(env, params=bad)
        assert r.status_code == 200, bad
        assert "delta" not in r.json(), bad
        assert len(cards_of(r.json())) == len(cards_of(snap)), bad


def test_etag_changes_with_the_filters_and_with_a_write(env):
    conn = connect()
    kb.create_task(conn, title="a", tenant="t1")
    kb.create_task(conn, title="b", tenant="t2")
    e_all = get(env).json()["etag"]
    e_t1 = get(env, "tenant=t1").json()["etag"]
    assert e_all != e_t1
    assert "delta" not in get(env, f"tenant=t1&etag={e_all}").json(), "another view's etag is not this one's"
    assert get(env, f"tenant=t1&etag={e_t1}").json()["unchanged"] is True
    kb.create_task(conn, title="c", tenant="t1")
    assert "unchanged" not in get(env, f"tenant=t1&etag={e_t1}").json()
    assert diff_boards(without_etag(get(env, "tenant=t1").json()), reference_board(env.plugin, tenant="t1")) == []


# ---------------------------------------------------------------------------
# equivalence: snapshot + deltas == fresh snapshot, through every kind of write
# ---------------------------------------------------------------------------

def test_snapshot_plus_deltas_equals_a_fresh_snapshot_through_a_scripted_run(env, clock, monkeypatch):
    conn = connect()
    done = seed_done(conn, 60)
    a = seed_active(conn)
    held = get(env).json()
    held_events_only = copy.deepcopy(held)
    steps: list[tuple[str, object, bool]] = []  # (name, fn, leaves_an_event)

    def step(name, leaves_event=True):
        def deco(fn):
            steps.append((name, fn, leaves_event))
            return fn
        return deco

    new: dict[str, str] = {}

    @step("create")
    def _():
        new["a"] = kb.create_task(conn, title="fresh a", body="b" * 300, assignee="worker")
        new["b"] = kb.create_task(conn, title="fresh b", assignee="worker", parents=[new["a"]])

    @step("claim")
    def _():
        assert kb.claim_task(conn, new["a"]) is not None

    @step("heartbeat event")
    def _():
        from hermes_cli.kanban_db_dispatch import heartbeat_worker
        assert heartbeat_worker(conn, new["a"], note="still going")

    @step("claim extension leaves no event", leaves_event=False)
    def _():
        assert kb.heartbeat_claim(conn, new["a"], ttl_seconds=5000)

    @step("comment on a done card")
    def _():
        kb.add_comment(conn, done[3], "tester", "after the fact")

    @step("complete releases the child")
    def _():
        assert kb.complete_task(conn, new["a"], result="ok", summary="finished the work " * 20)

    @step("block")
    def _():
        assert kb.claim_task(conn, a["r4"]) is not None
        assert kb.block_task(conn, a["r4"], reason="wait", kind="needs_input")

    @step("link")
    def _():
        kb.link_tasks(conn, a["r5"], a["t3"])

    @step("archive a done card and an active one")
    def _():
        assert kb.archive_task(conn, done[10])
        assert kb.archive_task(conn, a["r5"])

    @step("a raw column write on a done card leaves no event", leaves_event=False)
    def _():
        conn.execute("UPDATE tasks SET workspace_path = '/somewhere/else' WHERE id = ?", (done[20],))

    @step("hard delete")
    def _():
        assert kb.delete_task(conn, done[30])

    @step("two hours pass: ready cards are stranded")
    def _():
        clock.advance(2 * 3600)

    @step("a done card cycles block -> unblock", leaves_event=True)
    def _():
        add_block_cycles(conn, done[40], int(time.time()))

    @step("a day and a half passes: the cycling leaves the window")
    def _():
        clock.advance(36 * 3600)

    for name, fn, leaves_event in steps:
        clock.advance(45)  # past the volatile re-evaluation period, so time-driven diagnostics refresh
        fn()
        _, body = poll(env, held)
        held = merge_delta(held, body) if body.get("delta") else body
        fresh = get(env).json()
        ref = reference_board(env.plugin)
        assert diff_boards(without_etag(fresh), ref) == [], f"full snapshot after '{name}'"
        assert diff_boards(modulo_clock_text(without_etag(held), ref), ref) == [], f"snapshot + token delta after '{name}'"
        if leaves_event:
            # No token: only the events cursor. Exact for every write that leaves an event.
            p2 = get(env, params=f"since={held_events_only['latest_event_id']}")
            held_events_only = merge_delta(held_events_only, p2.json()) if p2.json().get("delta") else p2.json()
            assert diff_boards(modulo_clock_text(without_etag(held_events_only), ref), ref) == [], \
                f"snapshot + events delta after '{name}'"
        else:
            held_events_only = copy.deepcopy(fresh)

    ids_done = {t["id"] for col in held["columns"] if col["name"] == "done" for t in col["tasks"]}
    assert done[30] not in cards_of(held)
    assert done[10] not in ids_done
    assert cards_of(held)[done[20]]["workspace_path"] == "/somewhere/else"


def test_a_done_card_keeps_a_windowed_diagnostic_until_it_ages_out(env, clock):
    """A done card can carry a diagnostic (recent block -> unblock cycling counts on any status), and the
    rule is windowed: the cached card must drop it when the window passes, with no write to rebuild it."""
    conn = connect()
    done = seed_done(conn, 30)
    add_block_cycles(conn, done[5], T0 - 60)
    board = get(env).json()
    assert "block_unblock_cycling" in [d["kind"] for d in cards_of(board)[done[5]].get("diagnostics", [])]
    assert diff_boards(without_etag(board), reference_board(env.plugin)) == []
    clock.advance(30 * 3600)
    later = get(env).json()
    assert "diagnostics" not in cards_of(later)[done[5]]
    assert diff_boards(without_etag(later), reference_board(env.plugin)) == []


def test_a_time_driven_diagnostic_stays_current_on_a_live_evaluated_card(env, clock):
    """A ready card's "stranded" diagnostic is a function of the clock, and no write rebuilds the card.
    It must still move with time, and the cached card must match the original algorithm."""
    conn = connect()
    seed_done(conn, 20)
    tid = kb.create_task(conn, title="nobody claims me", assignee="worker")
    first = cards_of(get(env).json())[tid]
    assert "diagnostics" not in first
    created_at = first["created_at"]
    clock.t = float(created_at) + 1920  # 32 minutes ready: past the 30-minute stranded threshold
    later = get(env).json()
    card = cards_of(later)[tid]
    assert card["diagnostics"][0]["kind"] == "stranded_in_ready"
    assert card["diagnostics"][0]["data"]["age_seconds"] == 1920
    assert diff_boards(without_etag(later), reference_board(env.plugin)) == []


def test_a_parent_card_is_rebuilt_when_one_of_its_children_changes(env):
    """A link, a child's status, a child's title: the parent's progress and graph diagnostics read them,
    and the parent gets no event of its own. The neighbours are part of what a card is keyed on."""
    conn = connect()
    parent = kb.create_task(conn, title="parent", assignee="worker")
    first = kb.create_task(conn, title="first child", assignee="worker")
    held = get(env).json()
    kb.link_tasks(conn, parent, first)
    _, delta = poll(env, held)
    merged = merge_delta(held, delta)
    assert cards_of(merged)[parent]["progress"] == {"done": 0, "total": 1}
    assert diff_boards(without_etag(merged), reference_board(env.plugin)) == []
    # The child is renamed through a plain UPDATE that leaves no event: only the row tuple sees it.
    conn.execute("UPDATE tasks SET title = 'renamed child' WHERE id = ?", (first,))
    _, delta = poll(env, merged)
    merged = merge_delta(merged, delta)
    assert diff_boards(without_etag(merged), reference_board(env.plugin)) == []


def test_a_build_that_fails_half_way_does_not_hide_the_cards_it_built(env, monkeypatch):
    """A build that dies after building 400 of 450 cards must not leave the first 400 newer than the token
    the client holds: nothing is stored, so the next answer still carries every change."""
    conn = connect()
    done = seed_done(conn, 1000)
    held = get(env).json()
    for i, tid in enumerate(done[-450:]):  # a plain UPDATE: no event, so only the token names these cards
        conn.execute("UPDATE tasks SET workspace_path = ? WHERE id = ?", (f"/moved/{i}", tid))
    real = env.plugin._board_card
    calls = {"n": 0}

    def explode_late(*a, **k):
        calls["n"] += 1
        if calls["n"] == 401:
            raise RuntimeError("disk gone")
        return real(*a, **k)

    monkeypatch.setattr(env.plugin, "_board_card", explode_late)
    with pytest.raises(RuntimeError):
        poll(env, held)
    assert calls["n"] == 401
    monkeypatch.setattr(env.plugin, "_board_card", real)
    _, delta = poll(env, held)
    merged = merge_delta(held, delta)
    assert cards_of(merged)[done[-450]]["workspace_path"] == "/moved/0"
    assert cards_of(merged)[done[-1]]["workspace_path"] == "/moved/449"
    assert diff_boards(without_etag(merged), reference_board(env.plugin)) == []


def test_a_process_restart_degrades_to_an_events_delta_not_to_a_wrong_one(env, monkeypatch):
    conn = connect()
    done = seed_done(conn, 80)
    seed_active(conn)
    held = get(env).json()
    kb.add_comment(conn, done[2], "tester", "while the server was down")
    monkeypatch.setattr(kbdelta, "_caches", {})  # a new server process holds no cards and a new epoch
    restarted = load_plugin("kanban_board_delta_test_restarted")
    r = client_for(restarted).get(BOARD, params={"since": held["latest_event_id"], "etag": held["etag"]})
    assert r.status_code == 200 and r.json()["delta"] is True and "unchanged" not in r.json()
    merged = merge_delta(held, r.json())
    assert diff_boards(without_etag(merged), reference_board(restarted)) == []


# ---------------------------------------------------------------------------
# the read contract the old get_board carried still holds
# ---------------------------------------------------------------------------

def test_board_build_runs_in_one_read_transaction(env, monkeypatch):
    """Every read of one build sees one snapshot: a write landing mid-build is not half-seen."""
    conn = connect()
    seed_done(conn, 40)
    seen = []
    real = env.plugin._compute_task_diagnostics

    def spy(c, *a, **k):
        seen.append(c.in_transaction)
        return real(c, *a, **k)

    monkeypatch.setattr(env.plugin, "_compute_task_diagnostics", spy)
    assert get(env).status_code == 200
    assert seen and all(seen)
