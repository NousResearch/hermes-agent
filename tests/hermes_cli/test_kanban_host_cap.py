"""Host-level concurrency accounting + review-lane fairness (OOF-30 review).

Three gaps found in review of the original memory-guard PR:

1. The standalone daemon path (``hermes kanban daemon --force`` /
   :func:`hermes_cli.kanban_db_dispatch.run_daemon`) never resolved
   ``kanban.max_in_progress`` at all — the one shipped entry point that
   could still fan out an entire backlog in a single tick.
2. ``max_in_progress`` was enforced per-board while the gateway dispatcher
   ticks every active board — N boards multiplied the host budget by N.
3. The ready loop consumed the entire shared spawn budget before the
   review loop ran, so a sustained ready backlog starved autonomous
   reviews indefinitely.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _set_task_status(conn: sqlite3.Connection, task_id: str, status: str) -> None:
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


# ---------------------------------------------------------------------------
# 1. Standalone daemon resolves max_in_progress (P1a)
# ---------------------------------------------------------------------------


def test_run_daemon_resolves_and_passes_max_in_progress(
    kanban_home, monkeypatch,
):
    """The daemon tick must pass a resolved cap into dispatch_once.

    Regression guard for the OOF-30 review finding: ``run_daemon`` only
    forwarded ``max_spawn`` — with no explicit ``--max`` (the shipped
    systemd shape) nothing capped the tick even though the gateway and
    ``hermes kanban dispatch`` paths both resolved the memory-derived
    default.
    """
    captured: dict = {}
    stop = threading.Event()

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kb.DispatchResult()

    monkeypatch.setattr(kbd, "dispatch_once", fake_dispatch_once)
    # No explicit config → the derived default must flow through.
    monkeypatch.setattr(kbd, "configured_max_in_progress", lambda: None)
    monkeypatch.setattr(kbd, "derive_default_max_in_progress", lambda sample=None: 3)

    def on_tick(res):
        stop.set()

    kbd.run_daemon(interval=0.01, stop_event=stop, on_tick=on_tick)

    assert captured.get("max_in_progress") == 3




def test_configured_max_in_progress_parsing(monkeypatch):
    import hermes_cli.config as cfgmod

    cases = [
        ({"kanban": {"max_in_progress": 4}}, 4),
        ({"kanban": {"max_in_progress": "5"}}, 5),
        ({"kanban": {"max_in_progress": 0}}, None),
        ({"kanban": {"max_in_progress": -2}}, None),
        ({"kanban": {"max_in_progress": "lots"}}, None),
        ({"kanban": {}}, None),
        ({}, None),
    ]
    for config, expected in cases:
        monkeypatch.setattr(
            cfgmod, "load_config_readonly", lambda c=config: c
        )
        assert kbd.configured_max_in_progress() == expected, config


# ---------------------------------------------------------------------------
# 2. max_in_progress counts running work on ALL boards (P1b)
# ---------------------------------------------------------------------------


def test_max_in_progress_counts_other_boards(
    kanban_home, all_assignees_spawnable,
):
    """Workers running on another board consume the same host budget."""
    kb.create_board("second")

    # Two workers already running on the second board.
    with kbc.connect(board="second") as conn:
        for title in ("busy-1", "busy-2"):
            tid = kb.create_task(conn, title=title, assignee="alice")
            assert kb.claim_task(conn, tid) is not None

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # Host budget (2) already consumed by the second board → nothing spawns.
    assert not spawns
    assert not res.spawned


def test_max_in_progress_partial_budget_across_boards(
    kanban_home, all_assignees_spawnable,
):
    kb.create_board("second")

    with kbc.connect(board="second") as conn:
        tid = kb.create_task(conn, title="busy", assignee="alice")
        assert kb.claim_task(conn, tid) is not None

    spawns: list = []
    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # 1 running elsewhere + budget 2 → exactly one new spawn here.
    assert len(spawns) == 1
    assert len(res.spawned) == 1


def test_count_running_tasks_other_boards_fails_open(
    kanban_home, monkeypatch,
):
    """A broken board enumeration must not brick dispatch (returns 0)."""
    monkeypatch.setattr(
        kb, "list_boards",
        lambda **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert kbd.count_running_tasks_other_boards() == 0


def test_max_spawn_stays_per_board(kanban_home, all_assignees_spawnable):
    """``max_spawn`` keeps its historical per-board semantics."""
    kb.create_board("second")
    with kbc.connect(board="second") as conn:
        tid = kb.create_task(conn, title="busy", assignee="alice")
        assert kb.claim_task(conn, tid) is not None

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_spawn=1,
        )

    # The other board's worker does NOT count against max_spawn.
    assert len(spawns) == 1
    assert len(res.spawned) == 1


# ---------------------------------------------------------------------------
# 3. Review lane cannot be starved by a sustained ready backlog (P2)
# ---------------------------------------------------------------------------


def _park_in_review(conn: sqlite3.Connection, title: str, assignee: str) -> str:
    tid = kb.create_task(conn, title=title, assignee=assignee)
    _set_task_status(conn, tid, "review")
    return tid


def test_review_lane_gets_reserved_slot_under_ready_backlog(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kbc.connect() as conn:
        for title in ("ready-1", "ready-2", "ready-3"):
            kb.create_task(conn, title=title, assignee="alice")
        review_id = _park_in_review(conn, "review-me", "reviewer")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    spawned_ids = [s[0] for s in res.spawned]
    # Budget 2: one ready + the reserved review slot — never 2×ready.
    assert len(spawned_ids) == 2
    assert review_id in spawned_ids


def _guard_review_row(conn: sqlite3.Connection, review_id: str) -> dict:
    """Latest run ``rate_limited`` → ``check_respawn_guard`` returns a cooldown."""
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, "
            "started_at, ended_at) VALUES (?, 'reviewer', 'rate_limited', "
            "'rate_limited', ?, ?)",
            (review_id, now, now),
        )
    assert kbd.check_respawn_guard(conn, review_id, lane="review") == "rate_limit_cooldown"
    return {"max_in_progress": 1}


def _cap_review_row(conn: sqlite3.Connection, review_id: str) -> dict:
    """``reviewer`` already has one running worker → the review row is per-profile capped."""
    busy_id = kb.create_task(conn, title="busy", assignee="reviewer")
    assert kb.claim_task(conn, busy_id) is not None
    return {"max_in_progress": 2, "max_in_progress_per_profile": 1}


@pytest.mark.parametrize("make_unspawnable", [_guard_review_row, _cap_review_row])
def test_unspawnable_review_does_not_reserve_the_only_ready_slot(
    kanban_home, all_assignees_spawnable, monkeypatch, make_unspawnable,
):
    """A review card the review loop would refuse this tick (respawn guard,
    per-profile cap) must not consume the fairness reservation — otherwise the
    ready lane starves every tick while the reserved slot goes unused."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kbc.connect() as conn:
        ready_id = kb.create_task(conn, title="ready-now", assignee="alice")
        review_id = _park_in_review(conn, "review-unspawnable", "reviewer")
        caps = make_unspawnable(conn, review_id)
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), **caps)

    assert [task_id for task_id, *_ in res.spawned] == [ready_id]


def test_unguarded_review_reserves_the_only_ready_slot(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A dispatchable review card still receives the single shared slot."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="ready-now", assignee="alice")
        review_id = _park_in_review(conn, "review-now", "reviewer")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    assert [task_id for task_id, *_ in res.spawned] == [review_id]


def test_review_reservation_released_when_no_review_work(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kbc.connect() as conn:
        for title in ("ready-1", "ready-2", "ready-3"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # No review work → ready lane keeps the full budget.
    assert len(res.spawned) == 2


def test_nonspawnable_review_does_not_tax_ready_budget(
    kanban_home, monkeypatch,
):
    """Review tasks parked for humans (no real profile) release the slot."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    # Only 'alice' is a real profile; the review assignee is a human lane.
    monkeypatch.setattr(
        profmod, "profile_exists", lambda name: name == "alice"
    )

    spawns: list = []
    with kbc.connect() as conn:
        for title in ("ready-1", "ready-2"):
            kb.create_task(conn, title=title, assignee="alice")
        _park_in_review(conn, "human-review", "some-human")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # Human-lane review is not spawnable → no reservation, ready gets both.
    assert len(res.spawned) == 2


def test_review_budget_still_bounded_by_shared_cap(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The reservation caps the ready lane; it grants review no extra slots."""
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="ready-1", assignee="alice")
        for i in range(3):
            _park_in_review(conn, f"review-{i}", "reviewer")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # Budget 2 total across both lanes, reservation notwithstanding.
    assert len(res.spawned) == 2


# ---------------------------------------------------------------------------
# 4. The host cap names what it held back (#124392)
# ---------------------------------------------------------------------------


def test_host_cap_buckets_deferred_ready_tasks(
    kanban_home, all_assignees_spawnable,
):
    """A tick blocked by the host cap reports the deferred task ids instead
    of returning a bare zero-spawn result.

    The cap binds before the ready rows are enumerated, so it cannot
    attribute a deferral per task at the gate the way the per-profile cap
    does — the caller fills ``skipped_host_capped`` from the held-back rows
    on its behalf. Deferred, not dropped: the tasks stay ``ready``.
    """
    spawns: list = []
    with kbc.connect() as conn:
        claimed = kb.create_task(conn, title="running", assignee="alice")
        assert kb.claim_task(conn, claimed) is not None
        ready_ids = [
            kb.create_task(conn, title=f"ready-{i}", assignee="alice")
            for i in range(3)
        ]
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    assert not spawns
    assert sorted(res.skipped_host_capped) == sorted(ready_ids)
    assert res.skipped_host_capped_deferred is True
    with kbc.connect() as conn:
        for task_id in ready_ids:
            row = kb.get_task(conn, task_id)
            assert row is not None and row.status == "ready"


def test_host_cap_deferral_is_per_tick_not_permanent(
    kanban_home, all_assignees_spawnable,
):
    """Once the running worker finishes, the deferred task dispatches."""
    spawns: list = []
    with kbc.connect() as conn:
        claimed = kb.create_task(conn, title="running", assignee="alice")
        assert kb.claim_task(conn, claimed) is not None
        ready_id = kb.create_task(conn, title="ready-0", assignee="alice")

        res1 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )
        assert res1.skipped_host_capped == [ready_id]

        _set_task_status(conn, claimed, "done")
        conn.execute(
            "UPDATE tasks SET claim_lock = NULL WHERE id = ?", (claimed,)
        )
        res2 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    assert [task_id for task_id, *_ in res2.spawned] == [ready_id]
    assert res2.skipped_host_capped == []
    assert res2.skipped_host_capped_deferred is False


def test_describe_suppression_names_host_cap(
    kanban_home, all_assignees_spawnable,
):
    """``describe_suppression`` feeds the "dispatcher stuck" warnings; the
    host cap must appear there as ``host_capped=N`` so the operator is not
    sent to check profile health for a host-wide concurrency hold."""
    spawns: list = []
    with kbc.connect() as conn:
        claimed = kb.create_task(conn, title="running", assignee="alice")
        assert kb.claim_task(conn, claimed) is not None
        for i in range(2):
            kb.create_task(conn, title=f"ready-{i}", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    assert kbd.describe_suppression([res]) == "host_capped=2"


def test_describe_suppression_silent_when_host_cap_not_binding(
    kanban_home, all_assignees_spawnable,
):
    """A free tick contributes no entry — the stuck warnings must not grow
    a spurious ``host_capped=0`` line."""
    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="only", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=5,
        )

    assert len(res.spawned) == 1
    assert res.skipped_host_capped == []
    assert kbd.describe_suppression([res]) == ""


# ---------------------------------------------------------------------------
# 4. #124489 follow-up: the per-tick max_spawn cap must defer VISIBLY, and the
#    host_capped bucket must stay disjoint from the per-task buckets.
# ---------------------------------------------------------------------------


def test_max_spawn_cap_defers_visibly(kanban_home, all_assignees_spawnable, caplog):
    """``--max N`` binding must bucket + log, like the host cap does.

    The host-level deferral logs and buckets (#124392); the per-tick cap four
    lines up returned a bare ``(False, None)`` with no bucket, no log line,
    while the host branch's comment claimed to have closed the last silent
    deferral. An operator running ``hermes kanban dispatch --max 1`` with one
    task already running saw a zero-spawn tick with nothing to point at.
    """
    import logging

    spawns: list = []
    with kbc.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        assert kb.claim_task(conn, running) is not None
        ready_ids = [
            kb.create_task(conn, title=f"ready-{i}", assignee="alice")
            for i in range(2)
        ]
        with caplog.at_level(logging.WARNING, logger="gateway.run"):
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn_factory(spawns), max_spawn=1,
            )

    assert not spawns
    assert res.skipped_max_spawn_deferred is True
    # Rows are named: the bucket and the describe_suppression line both exist.
    assert sorted(res.skipped_max_spawn) == sorted(ready_ids)
    assert kbd.describe_suppression([res]) == "max_spawn_capped=2"
    assert any("max_spawn=1" in rec.getMessage() for rec in caplog.records)
    # Deferred, not dropped.
    with kbc.connect() as conn:
        for task_id in ready_ids:
            assert kb.get_task(conn, task_id).status == "ready"


def test_describe_suppression_reports_one_cap_not_both(
    kanban_home, all_assignees_spawnable,
):
    """When the per-tick cap binds, suppress reporting is ``max_spawn_capped``
    only — the same rows sit in ``skipped_host_capped`` and counting them under
    both names would double-report one hold."""
    spawns: list = []
    with kbc.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        assert kb.claim_task(conn, running) is not None
        for _ in range(2):
            kb.create_task(conn, title="ready", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_spawn=1,
            max_in_progress=5,  # host cap NOT binding
        )

    out = kbd.describe_suppression([res])
    assert "max_spawn_capped=2" in out
    assert "host_capped" not in out


def _kbd_prepare_probe(conn, *, unassigned=0, ghost=0, spawnable=0):
    """Create ready tasks of the three per-task-bucket kinds plus plain ones.

    Returns ``(unassigned_ids, ghost_ids, spawnable_ids)``.
    """
    unassigned_ids = [
        kb.create_task(conn, title=f"unassigned-{i}") for i in range(unassigned)
    ]
    # A ghost profile: all_assignees_spawnable patches profile_exists to True,
    # so this test opts out of that fixture instead (see caller) — here we just
    # create the rows with a profile name the patched predicate will reject
    # when the caller does NOT use the fixture.
    ghost_ids = [
        kb.create_task(conn, title=f"ghost-{i}", assignee="ghost-profile")
        for i in range(ghost)
    ]
    spawnable_ids = [
        kb.create_task(conn, title=f"ready-{i}", assignee="alice")
        for i in range(spawnable)
    ]
    return unassigned_ids, ghost_ids, spawnable_ids


def test_host_cap_bucket_excludes_unassigned_and_ghost_rows(
    kanban_home, monkeypatch,
):
    """The host_capped bucket must not absorb rows held for other reasons.

    Sweeping every ready/review row into ``skipped_host_capped`` mislabels a
    task that needs routing as "host cap busy" — the exact misdirection
    #124392 exists to remove. This test does NOT use
    ``all_assignees_spawnable`` so the profile-exists guard is live: the ghost
    profile row must land in ``skipped_nonspawnable``, the unassigned row in
    ``skipped_unassigned``, and only the plain row in ``skipped_host_capped``.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(
        profiles, "profile_exists",
        lambda name: name == "alice",  # ghost-profile does not exist
    )
    spawns: list = []
    with kbc.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        assert kb.claim_task(conn, running) is not None
        unassigned_ids, ghost_ids, spawnable_ids = _kbd_prepare_probe(
            conn, unassigned=1, ghost=1, spawnable=1,
        )
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    assert not spawns
    assert res.skipped_unassigned == unassigned_ids
    assert res.skipped_nonspawnable == ghost_ids
    assert res.skipped_host_capped == spawnable_ids
    out = kbd.describe_suppression([res])
    assert "host_capped=1" in out


def test_host_cap_bucket_excludes_guarded_rows(kanban_home, all_assignees_spawnable):
    """A row the respawn guard would refuse still reports its guard reason
    while the host cap holds the tick — not ``host_capped``."""
    spawns: list = []
    with kbc.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        assert kb.claim_task(conn, running) is not None
        guarded = kb.create_task(conn, title="guarded", assignee="alice")
        plain = kb.create_task(conn, title="plain", assignee="alice")
        # A completed run inside the guard window makes check_respawn_guard
        # return "recent_success" for the ready lane. claim_task creates the
        # task_runs row; release it back to ready so the host cap sees it.
        assert kb.claim_task(conn, guarded) is not None
        guarded_run_id = kb.get_task(conn, guarded).current_run_id
        conn.execute(
            "UPDATE task_runs SET status='completed', outcome='completed', "
            "ended_at=? WHERE id=?",
            (int(time.time()), guarded_run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (guarded,),
        )
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    assert not spawns
    assert [tid for tid, _ in res.respawn_guarded] == [guarded]
    assert res.skipped_host_capped == [plain]
    out = kbd.describe_suppression([res])
    assert "recent_success=1" in out and "host_capped=1" in out
