"""Respawn guard containment and recovery.

A quota stamp left on ``last_failure_error`` by a rate-limited run outlived a
later worker ``kanban_block`` and the dependency promotion that followed. On
every dispatcher tick ``blocker_auth`` matched it, so the card sat in ``ready``
forever with zero runs, and each tick appended a ``respawn_guarded`` row (420
rows in two days on one live board). These tests pin down six behaviours:

* a dependency promotion is a recovery point, so stale failure text must not
  re-trap the card;
* an unresolved ``blocker_auth`` parks the card ``blocked`` exactly once and
  does not skip it forever while it shows as ``ready``;
* rate-limit retries back off exponentially and are bounded, also with
  cooldown 0 and in the review lane;
* an unblock with the blocker still present buys one run, then re-parks to triage;
* every respawn claim records who initiated it and which run it retries;
* repeated identical guard ticks are recorded once.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

QUOTA_STAMP = "pid 45103 exited rate-limited (quota wall) — requeued without counting a failure"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def spawns():
    calls: list[str] = []

    def fake_spawn(task, workspace, board=None):
        calls.append(task.id)
        return 4242

    fake_spawn.calls = calls
    return fake_spawn


def _end_run(conn, tid, outcome, ended_at, *, status="ready"):
    """Seed a finished run with ``outcome`` at ``ended_at`` and leave the card in ``status``."""
    with kb.write_txn(conn):
        run_id = conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'worker', ?, ?, ?, ?)", (tid, outcome, outcome, ended_at - 1, ended_at),
        ).lastrowid
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))
    return run_id


def _events(conn, tid, kind):
    return [e.payload for e in kb.list_events(conn, tid) if e.kind == kind]


def test_dependency_promotion_clears_stale_quota_stamp(kanban_home, all_assignees_spawnable, spawns, monkeypatch):
    """The production sequence: a rate-limited run, then a worker block on
    an open parent, then the parent finishes. Promotion must not leave the
    quota stamp behind for ``blocker_auth`` to match, and the card must spawn.
    The failure counter is preserved, so the breaker still bounds retries."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        child = kb.create_task(conn, title="wave-2", assignee="worker")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        _end_run(conn, child, "rate_limited", 1_000)
        _end_run(conn, child, "blocked", 2_000, status="todo")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error=?, consecutive_failures=1 WHERE id=?",
                         (QUOTA_STAMP, child))
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))

        assert kb.recompute_ready(conn) >= 1
        task = kb.get_task(conn, child)
        assert task.status == "ready"
        assert task.last_failure_error is None
        assert task.consecutive_failures == 1
        assert _events(conn, child, "promoted")[-1] == {"actor": "system", "reason": "parents_done"}
        assert kbd.check_respawn_guard(conn, child) is None

        kbd.dispatch_once(conn, spawn_fn=spawns)
    assert spawns.calls == [child]


def test_unresolved_blocker_auth_parks_blocked_once_and_never_spawns(kanban_home, all_assignees_spawnable, spawns):
    """Containment: a real auth blocker must not leave the card looking
    ``ready`` while every tick skips it. It parks ``blocked`` (sticky, for a
    human) once; further ticks and promotion passes leave it there."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="auth", assignee="worker")
        _end_run(conn, tid, "timed_out", 1_000)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error=?, consecutive_failures=1 WHERE id=?",
                         ("403 forbidden: invalid api key", tid))
        for _ in range(4):
            kbd.dispatch_once(conn, spawn_fn=spawns)
            kb.recompute_ready(conn)
        task = kb.get_task(conn, tid)
        blocked = _events(conn, tid, "blocked")
        guarded = _events(conn, tid, "respawn_guarded")
    assert spawns.calls == []
    assert (task.status, task.block_kind) == ("blocked", "needs_input")
    assert task.consecutive_failures == 1, "history and retry counters are preserved"
    assert len(blocked) == 1 and "blocker_auth" in blocked[0]["reason"]
    assert (blocked[0]["actor"], blocked[0]["source"]) == ("dispatcher", "dispatcher")
    assert guarded == [{"reason": "blocker_auth", "actor": "dispatcher"}]


def test_unblock_after_auth_fix_recovers_with_one_spawn(kanban_home, all_assignees_spawnable, spawns):
    """Recovery: an authorized unblock clears the blocker and the next tick
    spawns exactly one worker, whose claim names the run it retries."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="auth", assignee="worker")
        prior = _end_run(conn, tid, "timed_out", 1_000)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", ("401 unauthorized", tid))
        kbd.dispatch_once(conn, spawn_fn=spawns)
        assert kb.get_task(conn, tid).status == "blocked"

        assert kb.unblock_task(conn, tid)
        kbd.dispatch_once(conn, spawn_fn=spawns)
        kbd.dispatch_once(conn, spawn_fn=spawns)
        claimed = _events(conn, tid, "claimed")[-1]
    assert spawns.calls == [tid]
    assert claimed["actor"] == "dispatcher"
    # The block synthesizes a "blocked" run; the retry is of that latest run.
    assert claimed["retry_of"]["run_id"] > prior
    assert claimed["retry_of"]["outcome"] == "blocked"


def test_unblock_with_auth_still_broken_gets_one_attempt_then_reparks(kanban_home, all_assignees_spawnable, spawns):
    """An unblock while the credential is still bad buys exactly one worker
    run. Its auth failure parks the card again (escalated to ``triage``)
    instead of starting a new loop."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="auth", assignee="worker")
        _end_run(conn, tid, "timed_out", 1_000)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", ("401 unauthorized", tid))
        kbd.dispatch_once(conn, spawn_fn=spawns)
        assert kb.unblock_task(conn, tid, actor="ops", source="cli")

        kbd.dispatch_once(conn, spawn_fn=spawns)
        kbd._record_task_failure(conn, tid, "401 unauthorized: token revoked", outcome="timed_out",
                                 failure_limit=5, release_claim=True, end_run=True)
        for _ in range(3):
            kbd.dispatch_once(conn, spawn_fn=spawns)
        task = kb.get_task(conn, tid)
        loop = _events(conn, tid, "block_loop_detected")
        unblocked = _events(conn, tid, "unblocked")
    assert spawns.calls == [tid]
    # The same park reason twice is upstream's block loop: routed to triage for a human.
    assert task.status == "triage" and len(loop) == 1
    assert unblocked == [{"actor": "ops", "source": "cli"}]


def test_crash_with_auth_words_is_not_parked(kanban_home):
    """Only a failure classified as needing intervention parks. A crash whose
    output mentions auth is transient here and stays with the failure breaker."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="flaky", assignee="worker")
        _end_run(conn, tid, "crashed", 1_000)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", ("401 unauthorized", tid))
        assert kbd.check_respawn_guard(conn, tid) is None


def test_rate_limit_backoff_grows_and_is_bounded(kanban_home, monkeypatch):
    """Each consecutive rate-limited run doubles the wait up to the cap; once
    the streak reaches the limit the guard reports exhaustion, not another
    retry. A non-rate-limited run resets the streak."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_COOLDOWN_SECONDS", "1000")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_CONSECUTIVE", "4")
    clock = {"now": 0}
    monkeypatch.setattr(kb.time, "time", lambda: clock["now"])
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl", assignee="worker")
        end = 10_000
        for streak, wait in ((1, 300), (2, 600), (3, 1000)):
            _end_run(conn, tid, "rate_limited", end)
            clock["now"] = end + wait - 1
            assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown", streak
            clock["now"] = end + wait + 1
            assert kbd.check_respawn_guard(conn, tid) is None, streak
            end += 5_000
        _end_run(conn, tid, "rate_limited", end)
        clock["now"] = end + 10**6
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_exhausted"

        _end_run(conn, tid, "completed", end + 1)
        _end_run(conn, tid, "rate_limited", end + 2)
        clock["now"] = end + 2 + 301
        assert kbd.check_respawn_guard(conn, tid) is None


def test_rate_limit_backoff_reads_text_timestamps(kanban_home, monkeypatch):
    """A run stamped with an ISO ``ended_at`` (written outside the mutators)
    is timed by value, not compared as text."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    clock = {"now": 1_000_000}
    monkeypatch.setattr(kb.time, "time", lambda: clock["now"])
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl", assignee="worker")
        run_id = _end_run(conn, tid, "rate_limited", clock["now"])
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET ended_at=? WHERE id=?",
                         (kb.time.strftime("%Y-%m-%dT%H:%M:%S+00:00", kb.time.gmtime(clock["now"])), run_id))
        clock["now"] += 299
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
        clock["now"] += 2
        assert kbd.check_respawn_guard(conn, tid) is None


def test_rate_limit_exhaustion_parks_then_unblock_recovers(kanban_home, all_assignees_spawnable, spawns, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "1")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_CONSECUTIVE", "2")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl", assignee="worker")
        _end_run(conn, tid, "rate_limited", 1_000)
        _end_run(conn, tid, "rate_limited", 2_000)
        kbd.dispatch_once(conn, spawn_fn=spawns)
        task = kb.get_task(conn, tid)
        assert (task.status, task.block_kind) == ("blocked", "needs_input")
        assert "rate_limit_exhausted" in _events(conn, tid, "blocked")[-1]["reason"]
        assert spawns.calls == []

        assert kb.unblock_task(conn, tid)
        kbd.dispatch_once(conn, spawn_fn=spawns)
    assert spawns.calls == [tid]


def test_zero_cooldown_disables_waiting_but_not_the_exhaustion_ceiling(kanban_home, monkeypatch):
    """Cooldown 0 means "retry without waiting". It must not also remove the
    separate cap on consecutive rate-limited runs."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_CONSECUTIVE", "3")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl", assignee="worker")
        for i in range(2):
            _end_run(conn, tid, "rate_limited", 1_000 + i)
            assert kbd.check_respawn_guard(conn, tid) is None
        _end_run(conn, tid, "rate_limited", 1_002)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_exhausted"


def test_review_lane_exhaustion_parks_and_unblock_returns_to_review(kanban_home, all_assignees_spawnable, spawns,
                                                                    monkeypatch):
    """A review card that keeps hitting the quota wall parks ``blocked`` like a
    ready card would. It must not sit in ``review`` refused on every tick. The
    unblock sends it back to ``review``, not to the implementer."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "1")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_CONSECUTIVE", "2")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        impl = kb.claim_task(conn, tid)
        assert kb.request_review(conn, tid, summary="ready", expected_run_id=impl.current_run_id)
        now = int(kb.time.time())  # after the implementer's run, which request_review ended now
        _end_run(conn, tid, "rate_limited", now + 1, status="review")
        _end_run(conn, tid, "rate_limited", now + 2, status="review")
        for _ in range(3):
            kbd.dispatch_once(conn, spawn_fn=spawns)
        task = kb.get_task(conn, tid)
        blocked = _events(conn, tid, "blocked")
        assert (task.status, task.block_kind) == ("blocked", "needs_input")
        assert len(blocked) == 1 and blocked[0]["source_status"] == "review"
        assert spawns.calls == []

        assert kb.unblock_task(conn, tid, actor="ops", source="cli")
        assert kb.get_task(conn, tid).status == "review"


def test_review_lane_stale_quota_stamp_is_not_parked(kanban_home, all_assignees_spawnable, spawns):
    """A review card whose ``blocker_auth`` comes from the implementer's stale
    quota stamp (#119070) is not escalated to an operator; it is recorded once
    and stays in ``review``."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        impl = kb.claim_task(conn, tid)
        assert kb.request_review(conn, tid, summary="ready", expected_run_id=impl.current_run_id)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (QUOTA_STAMP, tid))
        assert kbd.check_respawn_guard(conn, tid, lane="review") == "blocker_auth"
        for _ in range(3):
            kbd.dispatch_once(conn, spawn_fn=spawns)
        assert kb.get_task(conn, tid).status == "review"
        assert _events(conn, tid, "blocked") == []
        assert _events(conn, tid, "respawn_guarded") == [{"reason": "blocker_auth", "actor": "dispatcher"}]


def test_repeated_guard_ticks_record_one_event_per_reason(kanban_home, all_assignees_spawnable, spawns, monkeypatch):
    """History stays auditable without one row per tick: an unchanged guard
    reason is written once; a different reason is written again."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "100000")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl", assignee="worker")
        _end_run(conn, tid, "rate_limited", int(kb.time.time()))
        kbd.dispatch_once(conn, spawn_fn=spawns, dry_run=True)
        assert _events(conn, tid, "respawn_guarded") == []
        for _ in range(5):
            kbd.dispatch_once(conn, spawn_fn=spawns)
        assert _events(conn, tid, "respawn_guarded") == [{"reason": "rate_limit_cooldown", "actor": "dispatcher"}]
    assert spawns.calls == []
