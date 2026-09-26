"""Stranded-card routing: the detection's acting half.

``stranded_in_ready`` inside a read-only rule is a log line. These tests cover
the control: WHY the card is not running (read from board state, never guessed),
which lane owns that cause, and the repair card the dispatcher files — once, at
error severity, routed by the fixed table, and never recursively.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_diagnostics as kd

THRESHOLD = 1800


def _live_kd():
    """The diagnostics module object the code under test will import.

    Two files in this suite purge ``hermes_cli.kanban_diagnostics`` from
    ``sys.modules`` to exercise import-time behaviour, so after them this file's
    module-level import is bound to a stale object while the dispatcher's own lazy
    import re-creates the module. A fake patched only onto the stale object is
    invisible to the dispatcher, which then reads the real host — patch through
    this, and both halves of the code under test see the same registry.
    """
    import importlib

    return importlib.import_module("hermes_cli.kanban_diagnostics")


def _patch_profile_registry(monkeypatch, fake) -> None:
    """Make ``_profile_exists`` answer ``fake`` in every copy of the module."""
    monkeypatch.setattr(kd, "_profile_exists", fake)
    monkeypatch.setattr(_live_kd(), "_profile_exists", fake)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    # The host's profile registry is not what these tests are about: pretend the
    # named lanes exist so the cause classifier reaches the board-state branches.
    _patch_profile_registry(monkeypatch, lambda name: bool(name))
    with kbc.connect() as c:
        yield c


def _stranded(conn, assignee: str, *, age_seconds: int = 4000, **kw) -> str:
    """A ready card with no worker, aged past the stranded threshold.

    Both the row and its events are aged: the rule takes the most recent
    ready-putting event as the start of the wait, so ageing only the row would
    leave a fresh ``created`` event claiming the card just appeared.
    """
    tid = kb.create_task(conn, title=f"stranded for {assignee}", assignee=assignee, **kw)
    aged = int(time.time()) - age_seconds
    conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (aged, tid))
    conn.execute("UPDATE task_events SET created_at = ? WHERE task_id = ?", (aged, tid))
    conn.commit()
    return tid


def _diags(conn, tid, *, config=None, facts=None):
    return kd.compute_task_diagnostics(
        kb.get_task(conn, tid), kb.list_events(conn, tid), kb.list_runs(conn, tid),
        config=config or {"stranded_threshold_seconds": THRESHOLD},
        board_facts=facts if facts is not None else kd.board_facts_for_ready_lane(
            conn, config=config or {"stranded_threshold_seconds": THRESHOLD}),
    )


def _stranded_diag(conn, tid, **kw):
    hits = [d for d in _diags(conn, tid, **kw) if d.kind == "stranded_in_ready"]
    return hits[0] if hits else None


def _route(diagnostic) -> dict:
    return next(a.payload for a in diagnostic.actions if a.kind == "route")


def _cards_with_key(conn, key):
    return conn.execute(
        "SELECT id, title, assignee, status, body FROM tasks WHERE idempotency_key = ?", (key,)
    ).fetchall()


# --- WHY, read from board state ----------------------------------------------

def test_queue_ahead_is_named_and_owned_by_the_lane_authority(conn):
    _stranded(conn, "demo-coder", age_seconds=9000)
    behind = _stranded(conn, "demo-coder", age_seconds=4000)
    diagnostic = _stranded_diag(conn, behind)
    route = _route(diagnostic)
    assert route["cause"] == kd.STRANDED_CAUSE_LANE_QUEUE_AHEAD
    assert route["owner"] == "demo-stl"
    assert "1 ready card(s) for 'demo-coder' rank ahead of it" in diagnostic.detail
    assert diagnostic.data["queue_ahead"] == 1
    assert diagnostic.data["cause"] == kd.STRANDED_CAUSE_LANE_QUEUE_AHEAD


def test_a_missing_profile_is_named_and_owned_by_the_ops_head(conn, monkeypatch):
    _patch_profile_registry(monkeypatch, lambda name: name == "default")
    tid = _stranded(conn, "ghost-lane")
    route = _route(_stranded_diag(conn, tid))
    assert route["cause"] == kd.STRANDED_CAUSE_ASSIGNEE_UNKNOWN
    assert route["owner"] == kd.DEFAULT_REPAIR_OWNER


def test_fleet_at_capacity_is_named(conn):
    stranded = _stranded(conn, "demo-coder")
    busy = kb.create_task(conn, title="busy", assignee="demo-coder")
    assert kb.claim_task(conn, busy, claimer=kb._claimer_id()) is not None
    cfg = {"stranded_threshold_seconds": THRESHOLD, "kanban": {"max_in_progress": 1}}
    diagnostic = _stranded_diag(conn, stranded, config=cfg)
    route = _route(diagnostic)
    assert route["cause"] == kd.STRANDED_CAUSE_FLEET_AT_CAPACITY
    assert route["owner"] == kd.DEFAULT_REPAIR_OWNER
    assert "concurrency cap (1/1 running)" in diagnostic.detail


def test_lane_without_a_worker_at_a_free_slot_is_named(conn):
    tid = _stranded(conn, "demo-coder")
    route = _route(_stranded_diag(conn, tid))
    assert route["cause"] == kd.STRANDED_CAUSE_LANE_NO_WORKER
    assert route["owner"] == kd.DEFAULT_REPAIR_OWNER
    assert "the spawn did not happen" in route["reason"]


def test_without_board_facts_no_cause_is_guessed(conn):
    tid = _stranded(conn, "demo-coder")
    diagnostic = _stranded_diag(conn, tid, facts={})
    assert diagnostic is not None
    assert not [a for a in diagnostic.actions if a.kind == "route"]
    assert "Common causes" in diagnostic.detail


def test_a_fresh_ready_card_is_not_stranded(conn):
    tid = kb.create_task(conn, title="fresh", assignee="demo-coder")
    assert _stranded_diag(conn, tid) is None


# --- the control: file the repair -------------------------------------------

def test_route_files_one_repair_card_routed_to_the_owner(conn):
    stranded = _stranded(conn, "demo-coder")
    routed = kd.route_stranded_cards(conn, config={"stranded_threshold_seconds": THRESHOLD})
    assert [r["outcome"] for r in routed] == ["filed"]
    assert routed[0]["cause"] == kd.STRANDED_CAUSE_LANE_NO_WORKER
    assert routed[0]["owner"] == kd.DEFAULT_REPAIR_OWNER
    rows = _cards_with_key(conn, kd.repair_idempotency_key(stranded, routed[0]["cause"]))
    assert len(rows) == 1
    card = rows[0]
    assert card["assignee"] == kd.DEFAULT_REPAIR_OWNER
    assert card["status"] == "ready"
    assert stranded in card["body"]
    assert kd.STRANDED_CAUSE_LANE_NO_WORKER in card["body"]
    assert "Board facts:" in card["body"]
    assert "Required fix:" in card["body"]
    # The routing itself is audited on the stranded card.
    events = [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (stranded,)).fetchall()]
    assert events.count("stranded_routed") == 1


def test_routing_is_idempotent_and_never_doubles(conn):
    stranded = _stranded(conn, "demo-coder")
    cfg = {"stranded_threshold_seconds": THRESHOLD}
    first = kd.route_stranded_cards(conn, config=cfg)
    second = kd.route_stranded_cards(conn, config=cfg)
    assert [r["outcome"] for r in first] == ["filed"]
    assert [r["outcome"] for r in second] == ["existing"]
    assert second[0]["repair_card"] == first[0]["repair_card"]
    key = kd.repair_idempotency_key(stranded, first[0]["cause"])
    assert len(_cards_with_key(conn, key)) == 1


def test_a_repair_card_is_never_itself_routed(conn):
    stranded = _stranded(conn, "demo-coder")
    cfg = {"stranded_threshold_seconds": THRESHOLD}
    repair = kd.route_stranded_cards(conn, config=cfg)[0]["repair_card"]
    aged = int(time.time()) - 9000
    conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (aged, repair))
    conn.execute("UPDATE task_events SET created_at = ? WHERE task_id = ?", (aged, repair))
    conn.commit()
    routed = kd.route_stranded_cards(conn, config=cfg)
    by_task = {r["task_id"]: r["outcome"] for r in routed}
    assert by_task[repair] == "skipped_repair_card"
    assert len(conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key LIKE ?",
        (kd.REPAIR_IDEMPOTENCY_PREFIX + "%",)).fetchall()) == 1


def test_nothing_is_filed_below_the_severity_floor(conn):
    """<2x the threshold is a warning: it reports, it does not file."""
    stranded = _stranded(conn, "demo-coder", age_seconds=THRESHOLD + 60)
    assert _stranded_diag(conn, stranded).severity == "warning"
    cfg = {"stranded_threshold_seconds": THRESHOLD}
    assert kd.route_stranded_cards(conn, config=cfg) == []
    assert kd.route_stranded_cards(conn, config=cfg, min_severity="warning") != []
    assert len(conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key LIKE ?",
        (kd.REPAIR_IDEMPOTENCY_PREFIX + "%",)).fetchall()) == 1


def test_dry_run_writes_nothing(conn):
    _stranded(conn, "demo-coder")
    cfg = {"stranded_threshold_seconds": THRESHOLD}
    routed = kd.route_stranded_cards(conn, config=cfg, dry_run=True)
    assert [r["outcome"] for r in routed] == ["dry_run"]
    assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1


def test_the_pass_is_capped(conn):
    for _ in range(3):
        _stranded(conn, "demo-coder")
    cfg = {"stranded_threshold_seconds": THRESHOLD}
    routed = kd.route_stranded_cards(conn, config=cfg, limit=2)
    assert [r["outcome"] for r in routed] == ["filed", "filed"]
    # Each stranded card has its own cause-specific key, so the third waits.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key LIKE ?",
        (kd.REPAIR_IDEMPOTENCY_PREFIX + "%",)).fetchone()["n"] == 2


def test_a_board_wide_stall_collapses_to_one_card_per_cause(conn):
    """A stalled board must not become a card storm: one card per cause.

    Six stuck cards of one cause would otherwise file six repair cards, each of
    which can itself strand -- the report would bury the board it reports on.
    """
    for _ in range(6):
        _stranded(conn, "demo-coder")
    cfg = {"stranded_threshold_seconds": THRESHOLD}
    routed = kd.route_stranded_cards(conn, config=cfg, limit=5, collapse_over=3)
    collapsed = [r for r in routed if r["collapsed"]]
    assert collapsed, "a stall of more than collapse_over cards of one cause must collapse"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key LIKE ?",
        (kd.REPAIR_IDEMPOTENCY_PREFIX + "bulk:%",)).fetchone()["n"] == len(collapsed)
    body = _cards_with_key(conn, kd.bulk_repair_key(collapsed[0]["cause"]))[0]["body"]
    assert "ready cards are stranded" in body
    assert "Required fix:" in body
    assert "not inference" in body
    # Idempotent across ticks: the population is already carded.
    before = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
    again = kd.route_stranded_cards(conn, config=cfg, limit=5, collapse_over=3)
    assert "filed" not in [r["outcome"] for r in again]
    assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == before


# --- the control is wired to the unattended actor ----------------------------

def test_the_dispatcher_tick_files_the_repair_card(conn):
    """The whole point: the control runs without a human invoking it.

    The card is one the tick will not claim (its lane is not spawnable from this
    home), which is exactly a ready card left with no worker.
    """
    stranded = _stranded(conn, "demo-coder")
    # A spawn that never happens — exactly the condition the rule reports.
    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: None)
    rows = conn.execute(
        "SELECT id, assignee, idempotency_key FROM tasks WHERE idempotency_key LIKE ?",
        (kd.REPAIR_IDEMPOTENCY_PREFIX + "%",)).fetchall()
    assert len(rows) == 1, [dict(r) for r in conn.execute(
        "SELECT id, status, assignee FROM tasks").fetchall()]
    assert rows[0]["assignee"] == kd.DEFAULT_REPAIR_OWNER
    assert rows[0]["idempotency_key"] == kd.repair_idempotency_key(
        stranded, kd.STRANDED_CAUSE_LANE_NO_WORKER)


def test_a_tick_that_spawns_files_nothing(conn, all_assignees_spawnable):
    kb.create_task(conn, title="live", assignee="alice")
    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    assert result.spawned, result
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key LIKE ?",
        (kd.REPAIR_IDEMPOTENCY_PREFIX + "%",)).fetchone()["n"] == 0


def test_a_broken_router_never_fails_the_tick(conn, monkeypatch, all_assignees_spawnable):
    reached: list = []

    def boom(*a, **k):
        reached.append(a)
        raise RuntimeError("boom")

    monkeypatch.setattr(_live_kd(), "route_stranded_cards", boom)
    kb.create_task(conn, title="live", assignee="alice")
    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    # The tick still ran to completion; the router's failure was logged, not raised.
    assert result is not None and not result.skipped_locked
    # ...and the pass really reached the broken router (a fake patched onto a
    # stale module copy would leave this test green while testing nothing).
    assert reached

