"""Hold-visibility regression tests for the ready-lane dispatcher guards.

Field incident (2026-09-26 review of the ``active_pr`` respawn guard): a board
stalled for 6.9h (04:59 -> 11:51). Three READY cards were each deferred by the
``active_pr`` guard — every one of them legitimately, since each named its own
open PR — the queue drained to zero, and NOTHING said so: ``hermes kanban
status``/``stats`` and the dashboard rendered the starved board exactly like a
correctly idle one. The only per-tick evidence was the ``respawn_guarded`` event
that guard 4 wrote on EVERY tick: 8837 of them across 24 tasks, 412 for one card
in that single stall, which is why the signal was unreadable.

Five contracts, one per finding:

* ``active_pr`` lifts on a DELIBERATE re-queue (``status`` / ``promoted`` /
  ``unblocked``) as well as on a handoff (``assigned`` / ``changes_requested`` /
  ``review_reopened``) — the implementer and the closer alike already exist as
  separate kinds. ``reclaimed`` stays out of guard 4: crash recovery is not a
  decision, and the worker that opened the PR must not be re-spawned against it.
* ``describe_suppression`` names EVERY bucket that can hold a ready row, not
  just the respawn-guard one.
* ``board_health`` reports ``(ready_total, spawnable, suppressed_by_reason)`` and
  a ``state`` verdict, so a starved board is distinguishable from an idle one —
  whether ONE row is held or a hundred.
* six consecutive stalled ticks file ONE card, routed by the held rows' lane to
  that lane's ``<lane>-stl`` seat, with repeat spacing.
* a hold writes ONE ``respawn_guarded`` event per episode, not one per tick.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kcli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_parser as kbp

PR_COMMENT = "Opened https://github.com/example/repo/pull/77 for review."


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    # The process-local stall counter is module state, so clear it per test.
    # Looked up defensively: the fixture must never be the thing under test —
    # against the base commit the tracker does not exist yet, and the premise
    # check needs these tests to fail on their ASSERTIONS, not in setup.
    reset_tracker = getattr(kbd, "reset_stall_tracker", None)
    if reset_tracker is not None:
        reset_tracker()
    return home


@pytest.fixture
def spawnable_profiles(monkeypatch: pytest.MonkeyPatch):
    """Every assignee resolves to a real profile (the dispatcher's spawn gate)."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)


def _backdate_comments(conn, tid: str, seconds: int = 60) -> None:
    """Second-granularity timestamps: the lift must be strictly NEWER than the
    PR comment, so push the comment back before recording the re-queue."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = created_at - ? WHERE task_id = ?",
            (seconds, tid),
        )


def _guard_events(conn, tid: str) -> list[dict]:
    return [
        dict(row)
        for row in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'respawn_guarded' "
            "ORDER BY id",
            (tid,),
        )
    ]


def _pr_ready_task(conn, title: str = "pr held", assignee: str = "dev") -> str:
    """A ready card whose own PR URL is in a fresh comment — ``active_pr``."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    kb.add_comment(conn, tid, author=assignee, body=PR_COMMENT)
    _backdate_comments(conn, tid)
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"
    return tid


# ---------------------------------------------------------------------------
# R2 — the hold lifts on a deliberate re-queue, not only on a handoff
# ---------------------------------------------------------------------------


def test_active_pr_guard_lifts_on_a_status_requeue(
    kanban_home: Path, spawnable_profiles
) -> None:
    """A done→ready drag (the dashboard's direct status write) lifts the hold.

    ``status`` is the kind ``plugin_api._set_status_direct`` records for a
    drag-drop move without a structured verb — the operator saying "run this
    again" with no ownership change and no review verdict. Without it the only
    lifts available are ``assigned`` and ``changes_requested``, i.e. the
    operator would have to fabricate an ownership move or a review verdict to
    re-run a card.
    """
    with kbc.connect() as conn:
        tid = _pr_ready_task(conn)
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "status", {"from": "done", "to": "ready"})
        assert kbd.check_respawn_guard(conn, tid) is None


def test_active_pr_guard_lifts_on_parent_completion_promotion(
    kanban_home: Path, spawnable_profiles
) -> None:
    """``promoted`` (all parents done) lifts the hold — the real field case.

    The held card is usually a child whose PR mention is a PRECONDITION (the
    parent's merge, the merge it verifies). ``recompute_ready`` promotes it to
    ``ready`` precisely so it can run; the guard must not then hold it for 24h.
    """
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="dev")
        child = kb.create_task(conn, title="child", assignee="dev", parents=[parent])
        kb.add_comment(conn, child, author="dev", body=PR_COMMENT)
        _backdate_comments(conn, child)
        assert kbd.check_respawn_guard(conn, child) == "active_pr"

        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
        kb.recompute_ready(conn)

        assert conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (child,)
        ).fetchone()["status"] == "ready"
        assert kbd.check_respawn_guard(conn, child) is None


def test_active_pr_guard_lifts_on_unblock(kanban_home: Path, spawnable_profiles) -> None:
    """``unblocked`` lifts the hold: an operator unblock IS a re-run request."""
    with kbc.connect() as conn:
        tid = _pr_ready_task(conn)
        assert kb.block_task(conn, tid, reason="operator paused", kind="needs_input") is True
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        assert kb.unblock_task(conn, tid) is True
        assert conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (tid,)
        ).fetchone()["status"] == "ready"
        assert kbd.check_respawn_guard(conn, tid) is None


def test_reclaim_keeps_the_active_pr_guard(kanban_home: Path, spawnable_profiles) -> None:
    """``reclaimed`` is recovery, not a decision — guard 4 must NOT lift on it.

    Guard 3 (``recent_success``) accepts a reclaim because a crashed worker's
    completion is not evidence the work finished. Guard 4 must not: the PR is
    still open and unattended, so re-spawning the same worker re-creates the
    duplicate-work risk the guard exists to prevent. The divergence is
    deliberate and recorded in ``_RESPAWN_GUARD_RECOVERY_KINDS``.
    """
    with kbc.connect() as conn:
        tid = _pr_ready_task(conn)
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "reclaimed", {"note": "crash recovery"})
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_handoff_vocabulary_is_unchanged(kanban_home: Path, spawnable_profiles) -> None:
    """Regression guard on #111910: the handoff lifts still work."""
    with kbc.connect() as conn:
        tid = _pr_ready_task(conn, assignee="dev")
        assert kb.assign_task(conn, tid, "closer") is True
        assert kbd.check_respawn_guard(conn, tid) is None


# ---------------------------------------------------------------------------
# R1 — the 24h window is NOT the lever: it must stay 86400
# ---------------------------------------------------------------------------


def test_pr_window_stays_24h() -> None:
    """Ruling R1: the harm is the guard's PREDICATE, not its window length.

    All three cards in the 2026-09-25 stall cleared at their own 24h marks, and
    two of the six holds measured that day were the legitimate case the window
    protects (a worker's own PR, still open, still unattended). Shortening the
    window re-dispatches a worker against the PR it authored — the duplicate
    work guard 4 exists to prevent — so the number a later reader reaches for
    first is the one this assertion pins.
    """
    assert kbd._RESPAWN_GUARD_PR_WINDOW == 86400


# ---------------------------------------------------------------------------
# R4 — one event per hold EPISODE, not one per tick
# ---------------------------------------------------------------------------


def _record(conn, tid: str, reason: str = "active_pr") -> bool:
    with kb.write_txn(conn):
        return kbd._append_respawn_guard_event(conn, tid, reason)


def test_hold_records_one_event_across_many_ticks(
    kanban_home: Path, spawnable_profiles
) -> None:
    with kbc.connect() as conn:
        tid = _pr_ready_task(conn)
        assert _record(conn, tid) is True
        for _ in range(4):
            assert _record(conn, tid) is False
        assert len(_guard_events(conn, tid)) == 1


def test_hold_event_reemits_when_the_reason_changes(
    kanban_home: Path, spawnable_profiles
) -> None:
    with kbc.connect() as conn:
        tid = _pr_ready_task(conn)
        assert _record(conn, tid, "active_pr") is True
        assert _record(conn, tid, "rate_limit_cooldown") is True
        assert _record(conn, tid, "rate_limit_cooldown") is False
        assert [json.loads(e["payload"])["reason"] for e in _guard_events(conn, tid)] == [
            "active_pr", "rate_limit_cooldown",
        ]


def test_hold_event_reemits_when_the_episode_breaks(
    kanban_home: Path, spawnable_profiles
) -> None:
    """A card that leaves the queue and comes back starts a fresh episode."""
    with kbc.connect() as conn:
        tid = _pr_ready_task(conn)
        assert _record(conn, tid) is True
        assert _record(conn, tid) is False
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "claimed", {"profile": "dev"})
        assert _record(conn, tid) is True
        assert len(_guard_events(conn, tid)) == 2


def test_hold_event_reemits_after_the_repeat_interval(
    kanban_home: Path, spawnable_profiles
) -> None:
    """A hold that outlives the interval re-notifies once per interval."""
    with kbc.connect() as conn:
        tid = _pr_ready_task(conn)
        assert _record(conn, tid) is True
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET created_at = created_at - ? "
                "WHERE task_id = ? AND kind = 'respawn_guarded'",
                (kbd._RESPAWN_GUARD_EVENT_REPEAT_SECONDS + 1, tid),
            )
        assert _record(conn, tid) is True
        assert len(_guard_events(conn, tid)) == 2


def test_dispatch_ticks_write_one_guard_event(kanban_home: Path, spawnable_profiles) -> None:
    """The dispatcher path (not just the helper) writes one event per episode.

    This is the R4 regression: four real ticks over a held card used to write
    four events.
    """
    with kbc.connect() as conn:
        tid = _pr_ready_task(conn)
        for _ in range(4):
            res = kbd.dispatch_once(conn, dry_run=False)
            assert dict(res.respawn_guarded).get(tid) == "active_pr"
        assert len(_guard_events(conn, tid)) == 1


# ---------------------------------------------------------------------------
# R3(a) — the operator line names every hold bucket
# ---------------------------------------------------------------------------


def test_describe_suppression_names_every_hold_bucket() -> None:
    """``describe_suppression`` must not omit a bucket that can hold a row.

    A line reading ``active_pr=3`` while dozens of rows sat in
    ``skipped_per_profile_capped`` is not a report — it is the reason the stall
    was misread as a healthy queue.
    """
    res = kbd.DispatchResult(
        respawn_guarded=[("t1", "active_pr"), ("t2", "recent_success")],
        rate_limited=["t3"],
        skipped_locked=False,
        skipped_per_profile_capped=[("t4", "dev", 2), ("t5", "dev", 2)],
        skipped_nonspawnable=["lane-worker"],
        skipped_unassigned=["t6"],
    )
    line = kbd.describe_suppression([res])
    for expected in (
        "active_pr=1", "recent_success=1", "rate_limited=1",
        "skipped_per_profile_capped=2", "skipped_nonspawnable=1", "skipped_unassigned=1",
    ):
        assert expected in line, line


def test_describe_suppression_empty_tick_is_empty() -> None:
    assert kbd.describe_suppression([kbd.DispatchResult()]) == ""


# ---------------------------------------------------------------------------
# R3(b) — starved is distinguishable from idle, at any row count
# ---------------------------------------------------------------------------


def test_board_health_starved_for_a_single_held_row(
    kanban_home: Path, spawnable_profiles
) -> None:
    with kbc.connect() as conn:
        tid = _pr_ready_task(conn, assignee="dev")
        health = kbd.board_health(conn)
        assert health.ready_total == 1
        assert health.spawnable == 1
        assert health.suppressed_by_reason == {"active_pr": 1}
        assert health.starved is True
        assert health.state == "starved"
        assert tid  # the read is about this row
        assert "suppressed_by_reason=active_pr:1" in health.describe()
        assert health.as_dict()["state"] == "starved"


def test_board_health_starved_does_not_depend_on_row_count(
    kanban_home: Path, spawnable_profiles
) -> None:
    """One held row and twelve held rows must read the same condition."""
    with kbc.connect() as conn:
        for index in range(12):
            _pr_ready_task(conn, title=f"held-{index}", assignee="dev")
        health = kbd.board_health(conn)
        assert (health.ready_total, health.spawnable) == (12, 12)
        assert health.suppressed_by_reason == {"active_pr": 12}
        assert health.starved is True
        assert health.state == "starved"

        # A single healthy row alongside them makes the board dispatchable again.
        kb.create_task(conn, title="free", assignee="dev")
        health = kbd.board_health(conn)
        assert (health.ready_total, health.spawnable) == (13, 13)
        assert health.suppressed_by_reason == {"active_pr": 12}
        assert health.starved is False
        assert health.state == "dispatchable"


def test_board_health_idle_board_is_not_starved(
    kanban_home: Path, spawnable_profiles
) -> None:
    with kbc.connect() as conn:
        health = kbd.board_health(conn)
        assert (health.ready_total, health.spawnable) == (0, 0)
        assert health.suppressed_by_reason == {}
        assert health.starved is False
        assert health.state == "idle"


def test_board_health_lane_owned_rows_are_idle_not_starved(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows the dispatcher would never spawn are named, not counted as a stall.

    An assignee that is not a profile is a control-plane lane that pulls the
    card itself via ``claim_task`` — correctly idle, never starvation.
    """
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: name == "dev")
    with kbc.connect() as conn:
        kb.create_task(conn, title="unrouted")
        kb.create_task(conn, title="lane-owned", assignee="ops-checks")
        health = kbd.board_health(conn)
        assert health.ready_total == 2
        assert health.spawnable == 0
        assert health.starved is False
        assert health.state == "idle"
        assert health.unavailable_by_reason == {"unassigned": 1, "not_a_profile": 1}


def test_board_health_capped_rows_are_busy_not_starved(
    kanban_home: Path, spawnable_profiles, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-profile cap hold is backpressure, not starvation."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(
        cfgmod, "load_config_readonly", lambda *a, **k: {"kanban": {"max_in_progress": 1}}
    )
    with kbc.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="dev")
        assert kb.claim_task(conn, running) is not None
        kb.create_task(conn, title="ready-1", assignee="dev")
        kb.create_task(conn, title="ready-2", assignee="dev")
        health = kbd.board_health(conn)
        assert (health.ready_total, health.spawnable) == (2, 2)
        assert health.unavailable_by_reason == {"per_profile_capped": 2}
        assert health.suppressed_by_reason == {}
        assert health.starved is False
        assert health.state == "idle"


# ---------------------------------------------------------------------------
# R3(c) — a persistent stall escalates as a routed CARD, not a log line
# ---------------------------------------------------------------------------


def _escalation_cards(conn) -> list:
    return conn.execute(
        "SELECT id, title, body, assignee FROM tasks "
        "WHERE idempotency_key LIKE 'kanban-dispatch-stall:%' ORDER BY created_at"
    ).fetchall()


def test_stall_escalation_files_one_card_routed_to_the_lane_stl(
    kanban_home: Path, spawnable_profiles, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {"kanban": {}})
    with kbc.connect() as conn:
        held = _pr_ready_task(conn, assignee="platform-coder")

        # The window is 6 consecutive stalled ticks; five must not escalate.
        for _ in range(kbd._STALL_ESCALATION_WINDOW - 1):
            res = kbd.dispatch_once(conn, dry_run=False)
            assert dict(res.respawn_guarded).get(held) == "active_pr"
        assert _escalation_cards(conn) == []

        res = kbd.dispatch_once(conn, dry_run=False)
        assert dict(res.respawn_guarded).get(held) == "active_pr"
        cards = _escalation_cards(conn)
        assert len(cards) == 1
        card = cards[0]
        assert card["assignee"] == "platform-stl"
        assert "dispatcher stall" in card["title"]
        assert "active_pr=1" in card["title"]
        assert held in card["body"]
        assert "state=starved" in card["body"]

        # Repeat spacing: further stalled ticks in the same window add no card.
        for _ in range(4):
            kbd.dispatch_once(conn, dry_run=False)
        assert len(_escalation_cards(conn)) == 1


def test_stall_escalation_routes_by_the_majority_lane(
    kanban_home: Path, spawnable_profiles, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The routing is the held rows' lane, not the board or a default."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {"kanban": {}})
    with kbc.connect() as conn:
        _pr_ready_task(conn, title="r1", assignee="research-coder")
        _pr_ready_task(conn, title="r2", assignee="research-worker")
        _pr_ready_task(conn, title="p1", assignee="platform-coder")
        for _ in range(kbd._STALL_ESCALATION_WINDOW):
            kbd.dispatch_once(conn, dry_run=False)
        cards = _escalation_cards(conn)
        assert len(cards) == 1
        assert cards[0]["assignee"] == "research-stl"


def test_stall_escalation_falls_back_to_the_orchestrator_profile(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lane seat this home does not have must not produce an unroutable card."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    existing = {"platform-coder", "default"}
    monkeypatch.setattr(profmod, "profile_exists", lambda name: name in existing)
    monkeypatch.setattr(
        cfgmod, "load_config", lambda *a, **k: {"kanban": {"orchestrator_profile": "default"}}
    )
    monkeypatch.setattr(cfgmod, "load_config_readonly", lambda *a, **k: {"kanban": {}})
    with kbc.connect() as conn:
        _pr_ready_task(conn, assignee="platform-coder")
        for _ in range(kbd._STALL_ESCALATION_WINDOW):
            kbd.dispatch_once(conn, dry_run=False)
        cards = _escalation_cards(conn)
        assert len(cards) == 1
        assert cards[0]["assignee"] == "default"


def test_stall_counter_resets_when_a_tick_spawns(
    kanban_home: Path, spawnable_profiles, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A healthy tick clears the streak: 5 stalled + 1 healthy + 5 stalled
    must not add up to an escalation."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {"kanban": {}})
    with kbc.connect() as conn:
        _pr_ready_task(conn, assignee="platform-coder")
        for _ in range(kbd._STALL_ESCALATION_WINDOW - 1):
            kbd.dispatch_once(conn, dry_run=False)

        spawned: list = []
        kb.create_task(conn, title="free", assignee="platform-stl")
        res = kbd.dispatch_once(
            conn, spawn_fn=lambda task, workspace, board=None: spawned.append(task.id) or 42,
        )
        assert res.spawned and spawned

        # Five more stalled ticks: still under the window after the reset.
        for _ in range(kbd._STALL_ESCALATION_WINDOW - 1):
            kbd.dispatch_once(conn, dry_run=False)
        assert _escalation_cards(conn) == []


def test_idle_tick_never_escalates(kanban_home: Path, spawnable_profiles) -> None:
    """A quiet board is not a stall, however many ticks it stays quiet."""
    with kbc.connect() as conn:
        for _ in range(kbd._STALL_ESCALATION_WINDOW * 3):
            kbd.dispatch_once(conn, dry_run=False)
        assert _escalation_cards(conn) == []


# ---------------------------------------------------------------------------
# R3(b) — the CLI read exists and reports the same tuple
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> int:
    import argparse

    root = argparse.ArgumentParser(prog="hermes")
    kcli.build_parser(root.add_subparsers())
    return kcli._HANDLERS["health"](root.parse_args(["kanban", *argv]))


def test_cli_health_reports_starved_board(
    kanban_home: Path, spawnable_profiles, capsys: pytest.CaptureFixture
) -> None:
    with kbc.connect() as conn:
        _pr_ready_task(conn, assignee="dev")
    assert _run_cli(["health", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "starved"
    assert payload["ready_total"] == 1
    assert payload["spawnable"] == 1
    assert payload["suppressed_by_reason"] == {"active_pr": 1}


def test_cli_health_renders_a_line_for_an_idle_board(
    kanban_home: Path, spawnable_profiles, capsys: pytest.CaptureFixture
) -> None:
    assert _run_cli(["health"]) == 0
    line = capsys.readouterr().out
    assert "state=idle" in line
    assert "ready_total=0" in line
