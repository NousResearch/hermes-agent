"""ADR 0020 §2.3/§3.4/§4.1 — board-liveness invariant unit tests (build card t_68d5b5a4).

``assignee_live`` is injected in every classification test so "can this card actually be
spawned" is a stated input rather than an accident of which profiles exist on the test host.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_board_liveness as bl
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

BOARD = "default"
NEVER = lambda _name: False  # noqa: E731 - "no profile on this host can claim it"
ALWAYS = lambda _name: True  # noqa: E731


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _events(conn, task_id, kind):
    return [e for e in kb.list_events(conn, task_id) if e.kind == kind]


def _sentinel_events(conn, kind):
    return _events(conn, bl.sentinel_id(BOARD), kind)


def _check(conn, *, live=NEVER, board=BOARD, now=None, emit=True):
    return bl.check_and_emit_board_liveness(
        conn, board=board, now=now, assignee_live=live, emit=emit,
    )


# --- AC 3: open cards, zero running, zero promotable ⇒ DEAD, exactly once ---

def test_open_goals_with_nothing_runnable_is_dead_and_alerts_once(kanban_home):
    conn = kbc.connect()
    try:
        dead_ready = kb.create_task(conn, title="orphan ready", assignee="ghost-profile")
        result = _check(conn)
        assert result.state == bl.STATE_DEAD
        assert result.counts["open_goals"] == 1 and result.counts["promotable"] == 0
        assert dead_ready in result.suspects
        assert "nothing running or promotable" in result.reason

        events = _sentinel_events(conn, "board_dead_quiet")
        assert len(events) == 1
        payload = events[0].payload
        assert payload["state"] == "DEAD" and payload["open_goals"] == 1
        assert payload["running"] == 0 and payload["ready"] == 1
        assert payload["board"] == BOARD

        # Repeated checks on the same UTC day never repeat the alert.
        for _ in range(3):
            assert _check(conn).state == bl.STATE_DEAD
        assert len(_sentinel_events(conn, "board_dead_quiet")) == 1
        assert _sentinel_events(conn, "board_quiet_by_design") == []
    finally:
        conn.close()


def test_dead_alert_rearms_on_the_next_utc_day(kanban_home):
    conn = kbc.connect()
    try:
        kb.create_task(conn, title="orphan ready", assignee="ghost-profile")
        day_one = 1_800_000_000  # a fixed clock keeps the day boundary deterministic
        assert _check(conn, now=day_one).state == bl.STATE_DEAD
        assert _check(conn, now=day_one + 3600) is not None
        assert len(_sentinel_events(conn, "board_dead_quiet")) == 1
        assert bl._dead_dedup_key(BOARD, day_one) != bl._dead_dedup_key(BOARD, day_one + 86_400)
        assert bl._emit_board_liveness(
            conn, board=BOARD, state=bl.STATE_DEAD,
            result=bl.check_board_liveness(conn, board=BOARD, assignee_live=NEVER),
            now=day_one + 86_400,
        ) is not None
        assert len(_sentinel_events(conn, "board_dead_quiet")) == 2
    finally:
        conn.close()


# --- AC 4: everything legitimately parked ⇒ no DEAD alert, one quiet digest ---

def _park_blocked(conn, title, *, kind="needs_input"):
    tid = kb.create_task(conn, title=title, assignee="dev")
    assert kb.claim_task(conn, tid) is not None
    assert kb.block_task(conn, tid, reason="operator decision needed", kind=kind) is True
    return tid


def test_blocked_only_board_is_quiet_by_design_never_dead(kanban_home):
    conn = kbc.connect()
    try:
        first = _park_blocked(conn, "waiting on you")
        result = _check(conn, live=ALWAYS)
        assert result.state == bl.STATE_QUIET_BY_DESIGN
        assert first in result.suspects
        assert _sentinel_events(conn, "board_dead_quiet") == []
        quiet = _sentinel_events(conn, "board_quiet_by_design")
        assert len(quiet) == 1
        assert quiet[0].payload["state"] == "QUIET_BY_DESIGN"

        # Same open-goal set ⇒ still one digest, no matter how many checks run.
        _check(conn, live=ALWAYS)
        _check(conn, live=ALWAYS)
        assert len(_sentinel_events(conn, "board_quiet_by_design")) == 1

        # A changed open-goal set re-arms exactly one more.
        _park_blocked(conn, "also waiting")
        _check(conn, live=ALWAYS)
        assert len(_sentinel_events(conn, "board_quiet_by_design")) == 2
        assert _sentinel_events(conn, "board_dead_quiet") == []
    finally:
        conn.close()


def test_scheduled_only_board_stays_silent(kanban_home):
    """ADR §3.5 — the wake driver owns a scheduled card's clock; do not alert at all."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="parked till monday", assignee="dev")
        conn.execute("UPDATE tasks SET status = 'scheduled' WHERE id = ?", (tid,))
        result = _check(conn, live=ALWAYS)
        assert result.state == bl.STATE_HEALTHY
        assert result.counts["scheduled"] == 1
        assert _sentinel_events(conn, "board_dead_quiet") == []
        assert _sentinel_events(conn, "board_quiet_by_design") == []
    finally:
        conn.close()


def test_todo_gated_on_an_operator_blocked_parent_is_quiet_not_dead(kanban_home):
    conn = kbc.connect()
    try:
        parent = _park_blocked(conn, "needs operator answer")
        kb.create_task(conn, title="child waiting on parent", assignee="dev", parents=[parent])
        result = _check(conn, live=ALWAYS)
        assert result.state == bl.STATE_QUIET_BY_DESIGN
        assert _sentinel_events(conn, "board_dead_quiet") == []
    finally:
        conn.close()


def test_todo_gated_on_a_stuck_sibling_chain_is_dead(kanban_home):
    """The gargantua01 shape: the parent chain broke, so nothing downstream can promote."""
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="stuck upstream", assignee="ghost-profile")
        child = kb.create_task(conn, title="downstream work", assignee="dev", parents=[parent])
        result = _check(conn, live=NEVER)
        assert result.state == bl.STATE_DEAD
        assert child in result.suspects or parent in result.suspects
        assert len(_sentinel_events(conn, "board_dead_quiet")) == 1
    finally:
        conn.close()


# --- Healthy arms: silence is only reportable when it is wrong ---

def test_running_board_is_healthy(kanban_home):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="actually working", assignee="ghost-profile")
        assert kb.claim_task(conn, tid) is not None
        result = _check(conn)
        assert result.state == bl.STATE_HEALTHY
        assert result.counts["running"] == 1
        assert _sentinel_events(conn, "board_dead_quiet") == []
    finally:
        conn.close()


def test_spawnable_ready_board_is_healthy(kanban_home):
    conn = kbc.connect()
    try:
        kb.create_task(conn, title="queued for a live profile", assignee="dev")
        assert _check(conn, live=ALWAYS).state == bl.STATE_HEALTHY
        assert _sentinel_events(conn, "board_dead_quiet") == []
        # The same board with a dead assignee is DEAD — the only difference is promotability.
        assert _check(conn, live=NEVER).state == bl.STATE_DEAD
    finally:
        conn.close()


def test_done_board_is_healthy(kanban_home):
    conn = kbc.connect()
    try:
        assert _check(conn).state == bl.STATE_HEALTHY
        assert _sentinel_events(conn, "board_dead_quiet") == []
    finally:
        conn.close()


# --- The sentinel itself must never look like work ---

def test_sentinel_is_idempotent_inert_and_unlisted(kanban_home):
    conn = kbc.connect()
    try:
        first = bl.ensure_sentinel_task(conn, BOARD)
        second = bl.ensure_sentinel_task(conn, BOARD)
        assert first == second == f"board:{BOARD}:liveness"
        assert bl.ensure_sentinel_task(conn, "  ") is None

        task = kb.get_task(conn, first)
        assert task.status == "scheduled" and task.assignee is None
        assert bl.is_sentinel_id(first) and not bl.is_sentinel_id("t_deadbeef")

        # Invisible to normal listings, visible to the forensic opt-in.
        assert first not in [t.id for t in kb.list_tasks(conn, include_system=True if False else False)]
        assert first in [t.id for t in kb.list_tasks(conn, include_system=True)]

        # Never counted as an open goal (so a lone sentinel cannot make a board look dead).
        assert _check(conn).state == bl.STATE_HEALTHY
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE id = ?", (first,)
        ).fetchone()
        assert row["n"] == 1
    finally:
        conn.close()


# --- AC / FIX D routing: the alert reaches the board's existing notification destination ---

def test_dead_alert_is_deliverable_through_the_existing_cursor(kanban_home):
    conn = kbc.connect()
    try:
        from gateway.kanban_watchers_notifier import TERMINAL_KINDS, diagnostic_event

        card = kb.create_task(conn, title="a normal card", assignee="ghost-profile")
        kbn.add_notify_sub(
            conn, task_id=card, platform="telegram", chat_id="ops-chat",
            notifier_profile="swco-butler", delivery_mode="notify+wake",
        )
        kb.create_task(conn, title="orphan ready", assignee="ghost-profile")
        assert _check(conn).state == bl.STATE_DEAD

        sentinel = bl.sentinel_id(BOARD)
        subs = kbn.list_notify_subs(conn, sentinel)
        assert [s["chat_id"] for s in subs] == ["ops-chat"], "alert must inherit the board's route"

        _old, _new, events = kbn.claim_unseen_events_for_sub(
            conn, task_id=sentinel, platform="telegram", chat_id="ops-chat",
            kinds=TERMINAL_KINDS,
        )
        assert [e.kind for e in events] == ["board_dead_quiet"]
        assert not diagnostic_event(events[0]), "liveness alerts are primary, never diagnostic"
        # Cursor advanced: a second claim delivers nothing (per-subscription dedup).
        assert kbn.claim_unseen_events_for_sub(
            conn, task_id=sentinel, platform="telegram", chat_id="ops-chat",
            kinds=TERMINAL_KINDS,
        )[2] == []
    finally:
        conn.close()


# --- Config knobs ---

def test_knobs_gate_and_shape_the_beat(kanban_home, monkeypatch):
    import hermes_cli.config as cfg_mod

    conn = kbc.connect()
    try:
        kb.create_task(conn, title="orphan ready", assignee="ghost-profile")
        assert bl.board_liveness_interval_seconds() == bl.DEFAULT_INTERVAL_SECONDS == 300

        monkeypatch.setattr(cfg_mod, "load_config", lambda *a, **k: {
            "kanban": {"board_liveness": {"enabled": False}},
        })
        assert not bl.board_liveness_enabled()
        assert _check(conn).state == bl.STATE_DEAD  # verdict still computable for callers
        assert _sentinel_events(conn, "board_dead_quiet") == []  # but the dispatcher gate is closed

        monkeypatch.setattr(cfg_mod, "load_config", lambda *a, **k: {
            "kanban": {"board_liveness": {"enabled": True, "interval_seconds": 10}},
        })
        assert bl.board_liveness_enabled()
        assert bl.board_liveness_interval_seconds() == 10
    finally:
        conn.close()


def test_emit_false_reports_without_writing(kanban_home):
    conn = kbc.connect()
    try:
        kb.create_task(conn, title="orphan ready", assignee="ghost-profile")
        result = _check(conn, emit=False)
        assert result.state == bl.STATE_DEAD
        assert _sentinel_events(conn, "board_dead_quiet") == []
        # No sentinel row is created when nothing is reportable.
        assert kb.get_task(conn, bl.sentinel_id(BOARD)) is None
    finally:
        conn.close()
