"""``kanban.max_active_per_board`` — the per-board ACTIVE-card ceiling — and the
board-level quota-exhaustion state.

Background: an external card creator running every 5 minutes flooded one board
with ~100 cards while the shared provider quota was already exhausted; the
dispatcher kept spawning (187 spawns in 10.6h, 86 killed by HTTP 429) because
nothing bounded *active* cards (only ``running``) and a rate-limited exit held
back only the card that died, never the profile. Invariants covered here:

1. the dispatcher never runs more than the ceiling on a board, and ``blocked``
   cards are cap-invisible;
2. a create against a full board is deferred with a visible result, never
   silently dropped, and is promoted once a slot frees;
3. a rate-limited worker exit records a durable, queryable board-level fact
   and holds the WHOLE profile for the cooldown; a completed run clears it.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_admission as kbcl
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB and real ``alpha``/``beta`` profiles."""
    home = tmp_path / ".hermes"
    home.mkdir()
    for prof in ("alpha", "beta"):
        pdir = home / "profiles" / prof
        pdir.mkdir(parents=True)
        (pdir / "config.yaml").write_text("{}\n")  # identity marker
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _set_status(conn: sqlite3.Connection, task_id: str, status: str) -> None:
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    conn.commit()


def _tick(conn: sqlite3.Connection, **kwargs) -> kbd.DispatchResult:
    """Dry-run dispatch; hand-set ``running`` rows carry no claim bookkeeping, so
    orphan reconciliation is off (it would requeue them mid-test)."""
    return kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, reconcile_orphans=False, **kwargs)


def _fake_spawn(task, workspace, board=None):
    return 4242


def _exited_status(code: int) -> int:
    return code << 8


# ---------------------------------------------------------------------------
# 1. Dispatcher honours the ceiling; blocked is cap-invisible
# ---------------------------------------------------------------------------


def test_dispatcher_refuses_to_spawn_past_the_active_ceiling(kanban_home):
    """Ceiling 2 with 2 cards already running: zero spawns, the hold is named on the
    result, and the ready cards stay queued (deferred, not dropped)."""
    with kbc.connect_closing() as conn:
        running = [kb.create_task(conn, title=f"run{i}", assignee="alpha", max_active=None)
                   for i in range(2)]
        for tid in running:
            _set_status(conn, tid, "running")
        queued = [kb.create_task(conn, title=f"q{i}", assignee="alpha", max_active=None)
                  for i in range(3)]
    with kbc.connect_closing() as conn:
        res = _tick(conn, max_active_per_board=2)
        assert res.spawned == []
        assert res.active_ceiling_held == (2, 2)
        assert {t.status for t in (kb.get_task(conn, q) for q in queued)} == {"ready"}


def test_ceiling_bounds_running_and_composes_with_other_caps(kanban_home):
    """Ceiling 3 with 1 running: at most 2 more spawn this tick. A looser
    ``max_spawn`` does not widen it; a tighter per-profile cap narrows it."""
    with kbc.connect_closing() as conn:
        busy = kb.create_task(conn, title="busy", assignee="alpha", max_active=None)
        _set_status(conn, busy, "running")
        for i in range(5):
            kb.create_task(conn, title=f"a{i}", assignee="alpha", max_active=None)
    with kbc.connect_closing() as conn:
        res = _tick(conn, max_active_per_board=3, max_spawn=10)
        assert len(res.spawned) == 2
    with kbc.connect_closing() as conn:
        res = _tick(conn, max_active_per_board=3, max_in_progress_per_profile=1)
        # alpha already has 1 running -> per-profile cap wins.
        assert res.spawned == []
        assert len(res.skipped_per_profile_capped) == 5


def test_blocked_cards_do_not_count_toward_the_ceiling(kanban_home):
    """Three blocked cards + ceiling 2 leave the board empty for admission: a
    create lands ``ready`` and the dispatcher spawns it."""
    with kbc.connect_closing() as conn:
        for i in range(3):
            # Operator parking (sticky): ``recompute_ready`` leaves these alone.
            kb.create_task(conn, title=f"blk{i}", assignee="alpha", initial_status="blocked", max_active=None)
        assert kbcl.count_active_tasks(conn) == 0
        fresh = kb.create_task(conn, title="fresh", assignee="alpha", max_active=2)
        assert kb.get_task(conn, fresh).status == "ready"
        assert kbcl.ceiling_deferral(conn, fresh) is None
    with kbc.connect_closing() as conn:
        res = _tick(conn, max_active_per_board=2)
        assert [s[0] for s in res.spawned] == [fresh]


# ---------------------------------------------------------------------------
# 2. Admission: create is deferred with a visible result; promotion drains it
# ---------------------------------------------------------------------------


def test_create_against_a_full_board_is_deferred_visibly(kanban_home):
    """The 3rd card under ceiling 2 is created (never dropped) in ``todo`` with a
    ``ceiling_wait`` event that every create surface can read."""
    with kbc.connect_closing() as conn:
        first, second = (kb.create_task(conn, title=t, assignee="alpha", max_active=2)
                         for t in ("one", "two"))
        third = kb.create_task(conn, title="three", assignee="alpha", max_active=2)
        assert [kb.get_task(conn, t).status for t in (first, second)] == ["ready", "ready"]
        parked = kb.get_task(conn, third)
        assert parked is not None and parked.status == "todo"
        deferral = kbcl.ceiling_deferral(conn, third)
        assert deferral == {
            "reason": "board_active_ceiling", "ceiling": 2, "active": 2, "intended_status": "ready",
        }
        kinds = [e.kind for e in kb.list_events(conn, third)]
        assert kinds.count("ceiling_wait") == 1 and "dependency_wait" not in kinds
        # A blocked creation is operator parking, not admission: it bypasses the ceiling.
        parked_by_hand = kb.create_task(
            conn, title="hold", assignee="alpha", initial_status="blocked", max_active=2,
        )
        assert kb.get_task(conn, parked_by_hand).status == "blocked"


def test_deferred_card_is_promoted_when_a_slot_frees(kanban_home):
    """``recompute_ready`` admits deferred cards in priority/age order only up to
    the ceiling, and a promoted ``triage`` create keeps its triage intent."""
    with kbc.connect_closing() as conn:
        a, b = (kb.create_task(conn, title=t, assignee="alpha", max_active=2) for t in ("a", "b"))
        low = kb.create_task(conn, title="low", assignee="alpha", max_active=2, priority=0)
        urgent = kb.create_task(conn, title="urgent", assignee="alpha", max_active=2, priority=5)
        tri = kb.create_task(conn, title="tri", assignee="alpha", max_active=2, triage=True)
        assert all(kb.get_task(conn, t).status == "todo" for t in (low, urgent, tri))

        # Board still full -> nothing moves.
        assert kb.recompute_ready(conn, max_active=2) == 0
        assert kb.get_task(conn, urgent).status == "todo"

        # One slot frees -> the highest-priority deferred card takes it.
        _set_status(conn, a, "done")
        assert kb.recompute_ready(conn, max_active=2) == 1
        assert kb.get_task(conn, urgent).status == "ready"
        assert kb.get_task(conn, low).status == "todo"
        assert kbcl.ceiling_deferral(conn, urgent) is None  # no longer parked

        # Two more slots -> the triage create resumes as triage, not ready.
        _set_status(conn, b, "done")
        _set_status(conn, urgent, "done")
        assert kb.recompute_ready(conn, max_active=2) == 2
        assert kb.get_task(conn, tri).status == "triage"
        assert kb.get_task(conn, low).status == "ready"


def test_ceiling_is_read_from_config_at_every_admission_point(kanban_home):
    """``kanban.max_active_per_board`` in config.yaml drives create, promotion and
    the dispatcher tick without any caller passing it explicitly."""
    (kanban_home / "config.yaml").write_text("kanban:\n  max_active_per_board: 1\n")
    assert kbd.configured_max_active_per_board() == 1
    with kbc.connect_closing() as conn:
        first = kb.create_task(conn, title="first", assignee="alpha")
        second = kb.create_task(conn, title="second", assignee="alpha")
        assert kb.get_task(conn, first).status == "ready"
        assert kb.get_task(conn, second).status == "todo"
        assert kb.recompute_ready(conn) == 0
        _set_status(conn, first, "running")
    with kbc.connect_closing() as conn:
        res = _tick(conn)
        assert res.active_ceiling_held == (1, 1) and res.spawned == []
    # Unsetting the knob restores the uncapped default.
    (kanban_home / "config.yaml").write_text("kanban: {}\n")
    assert kbd.configured_max_active_per_board() is None
    with kbc.connect_closing() as conn:
        assert kb.recompute_ready(conn) == 1


def test_kanban_create_tool_reports_the_deferral(kanban_home, monkeypatch):
    """The ``kanban_create`` tool result carries ``deferred=True`` + the ceiling numbers."""
    import json
    from tools import kanban_tools

    (kanban_home / "config.yaml").write_text("kanban:\n  max_active_per_board: 1\n")
    monkeypatch.setattr(kanban_tools, "_maybe_auto_subscribe", lambda conn, tid: False)
    out1 = json.loads(kanban_tools._handle_create({"title": "one", "assignee": "alpha"}))
    out2 = json.loads(kanban_tools._handle_create({"title": "two", "assignee": "alpha"}))
    assert out1["deferred"] is False and out1["status"] == "ready"
    assert out2["deferred"] is True
    assert out2["deferred_reason"] == "board_active_ceiling"
    assert (out2["ceiling"], out2["active"], out2["status"]) == (1, 1, "todo")
    assert "parked" in out2["note"]


# ---------------------------------------------------------------------------
# 3. Quota exhaustion is a board-level, queryable, profile-wide state
# ---------------------------------------------------------------------------


def test_rate_limited_exit_records_board_quota_state_and_holds_the_profile(kanban_home, monkeypatch):
    """One alpha worker dying rate-limited: ``quota_state`` answers "starved since T"
    for alpha, every OTHER alpha card is held for the cooldown (beta still spawns),
    and ``board_stats`` exposes it without reading ``task_events``."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "600")

    with kbc.connect_closing() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        victim = kb.create_task(conn, title="victim", assignee="alpha", max_active=None)
        others = [kb.create_task(conn, title=f"a{i}", assignee="alpha", max_active=None) for i in range(2)]
        bystander = kb.create_task(conn, title="b", assignee="beta", max_active=None)
        kb.claim_task(conn, victim, claimer=f"{host}:w1")
        conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (91001, victim))
        conn.commit()
        kbd._record_worker_exit(91001, _exited_status(kb.KANBAN_RATE_LIMIT_EXIT_CODE))

        assert kbcl.quota_state(conn) == {}
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
        assert res.rate_limited == [victim]

        state = kbcl.quota_state(conn)
        assert set(state) == {"alpha"}
        alpha = state["alpha"]
        assert alpha["last_task"] == victim and alpha["count"] == 1 and alpha["in_cooldown"]
        assert alpha["since"] == alpha["last_seen"] > 0
        assert alpha["cooldown_until"] == alpha["last_seen"] + 600
        # Durable + queryable: a plain SELECT on the board, no event grep.
        row = conn.execute("SELECT value FROM board_state WHERE key = 'quota_exhausted:alpha'").fetchone()
        assert row is not None
        stats = kb.board_stats(conn)
        assert stats["quota"]["alpha"]["since"] == alpha["since"]

        # The whole profile is held; the victim's own per-task guard still applies too.
        held = {tid for tid, who in res.skipped_quota_cooldown}
        assert held >= set(others) and all(who == "alpha" for _, who in res.skipped_quota_cooldown)
        assert [s[0] for s in res.spawned] == [bystander]
        assert "quota_cooldown=" in kbd.describe_suppression([res])

        # A repeat keeps ``since`` (start of starvation) and advances ``last_seen``/``count``.
        with kb.write_txn(conn):
            kbcl.note_quota_exhausted(conn, "alpha", others[0], now=alpha["last_seen"] + 50)
        again = kbcl.quota_state(conn)["alpha"]
        assert (again["since"], again["last_seen"], again["count"]) == (alpha["since"], alpha["last_seen"] + 50, 2)


def test_completed_run_clears_the_profile_quota_state(kanban_home, monkeypatch):
    """A terminal board call by an alpha worker proves quota is back: the row goes."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            kbcl.note_quota_exhausted(conn, "alpha", "t_dead")
        assert "alpha" in kbcl.quota_state(conn)
        # Cooldown 0 -> not held, but the fact stays until a run completes.
        assert kbcl.profiles_in_quota_cooldown(conn) == set()
        tid = kb.create_task(conn, title="back", assignee="alpha", max_active=None)
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        assert kb.complete_task(conn, tid, result="ok")
        assert kbcl.quota_state(conn) == {}


# ---------------------------------------------------------------------------
# 4. Every dispatch entry point forwards the knob
# ---------------------------------------------------------------------------


def test_gateway_dispatcher_settings_forward_the_ceiling(kanban_home, monkeypatch):
    """The embedded gateway dispatcher reads ``kanban.max_active_per_board`` at boot and
    passes it into every board tick (the one entry point that runs in production)."""
    from dataclasses import asdict
    from gateway import kanban_watchers_dispatcher as kwd

    settings = kwd._resolve_dispatcher_settings(
        {"max_active_per_board": 4, "max_in_progress_per_profile": 2}, kb,
    )
    assert settings.max_active_per_board == 4 and settings.max_in_progress_per_profile == 2
    captured: dict = {}
    monkeypatch.setattr(kbd, "dispatch_once", lambda conn, **kw: captured.update(kw) or kbd.DispatchResult())
    kwd._KanbanDispatcher(kb, settings).tick_once_for_board("default")
    assert captured["max_active_per_board"] == 4
    assert "interval" not in captured and set(asdict(settings)) - {"interval"} <= set(captured)
