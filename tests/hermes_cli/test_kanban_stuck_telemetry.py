"""Stuck-telemetry contracts for the kanban dispatcher (#80513).

The dispatcher health telemetry warns "stuck" when a tick finds ready
work but spawns nothing. Tasks deliberately deferred by the respawn
guard (recent completed run / open PR / rate-limit cooldown / auth
blocker) are idle BY DESIGN, not stuck — counting them as "spawnable
work waiting" produced hundreds of consecutive false-alarm ticks.

Contracts pinned here:

* ``has_spawnable_ready`` / ``has_spawnable_review`` report False when
  every candidate task in the column is guard-deferred, True again once
  the deferral window expires, and True when at least one genuinely
  spawnable task coexists with guarded ones (a mixed board can still be
  stuck).
* The CLI daemon (--force standalone loop) does NOT emit the stuck
  warning while its only ready task is guard-deferred — and still does
  not attempt any spawn (the change is telemetry-only).
* The CLI daemon DOES emit the warning when a spawnable task truly
  exists but nothing spawned, annotating the message with any
  respawn-guard deferrals observed that tick.
* The gateway-embedded dispatcher watcher agrees with the daemon on
  both counts (same probe, same suppression, same annotation).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

SUCCESS_WINDOW = kb._RESPAWN_GUARD_SUCCESS_WINDOW
PR_WINDOW = kb._RESPAWN_GUARD_PR_WINDOW


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def all_profiles_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every assignee name maps to an existing Hermes profile."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)


def _seed_recent_success(conn, tid: str, ended_at: int | None = None) -> None:
    if ended_at is None:
        ended_at = int(time.time()) - 30
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, "
            "started_at, ended_at) VALUES (?, 'worker', 'completed', "
            "'completed', ?, ?)",
            (tid, ended_at - 30, ended_at),
        )


# ---------------------------------------------------------------------------
# DB-level probe contract
# ---------------------------------------------------------------------------


def test_ready_probe_counts_unguarded_spawnable_task(
    kanban_home, all_profiles_real
) -> None:
    conn = kb.connect()
    try:
        kb.create_task(conn, title="real work", assignee="worker")
        assert kb.has_spawnable_ready(conn) is True
    finally:
        conn.close()


def test_ready_probe_excludes_recent_success_deferred_task(
    kanban_home, all_profiles_real
) -> None:
    """A completed run inside the guard window is idle-by-design, and the
    probe must say so — this is the reporter's ~986-false-tick scenario.
    """
    now = int(time.time())
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="just finished", assignee="worker")
        _seed_recent_success(conn, tid, ended_at=now - 60)
        assert kb.check_respawn_guard(conn, tid) == "recent_success"
        assert kb.has_spawnable_ready(conn) is False

        # Once the window elapses the deferral lifts and the task reads
        # as spawnable again.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET started_at = ?, ended_at = ? "
                "WHERE task_id = ?",
                (now - SUCCESS_WINDOW - 40, now - SUCCESS_WINDOW - 10, tid),
            )
        assert kb.has_spawnable_ready(conn) is True
    finally:
        conn.close()


def test_ready_probe_excludes_active_pr_deferred_task(
    kanban_home, all_profiles_real
) -> None:
    now = int(time.time())
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="pr open", assignee="worker")
        kb.add_comment(
            conn, tid, author="worker",
            body="Opened https://github.com/example/repo/pull/123.",
        )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        assert kb.has_spawnable_ready(conn) is False

        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_comments SET created_at = ? WHERE task_id = ?",
                (now - PR_WINDOW - 10, tid),
            )
        assert kb.has_spawnable_ready(conn) is True
    finally:
        conn.close()


def test_ready_probe_excludes_rate_limit_cooldown_deferred_task(
    kanban_home, all_profiles_real
) -> None:
    now = int(time.time())
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="quota wall", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at) VALUES (?, 'worker', 'rate_limited', "
                "'rate_limited', ?, ?)",
                (tid, now - 10, now),
            )
        assert kb.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
        assert kb.has_spawnable_ready(conn) is False
    finally:
        conn.close()


def test_ready_probe_true_when_guarded_and_clean_tasks_coexist(
    kanban_home, all_profiles_real
) -> None:
    """A mixed board still has genuinely spawnable work — the probe must
    keep reporting True so a real stuck condition stays visible.
    """
    now = int(time.time())
    conn = kb.connect()
    try:
        guarded = kb.create_task(conn, title="parked", assignee="worker")
        _seed_recent_success(conn, guarded, ended_at=now - 60)
        kb.create_task(conn, title="fresh work", assignee="worker2")
        assert kb.has_spawnable_ready(conn) is True
    finally:
        conn.close()


def test_review_probe_excludes_rate_limit_cooldown_deferred_task(
    kanban_home, all_profiles_real
) -> None:
    """Mirror contract for the review column (cooldown/auth are the only
    guard rules that apply there)."""
    now = int(time.time())
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.request_review(
            conn, tid, summary="PR ready",
            expected_run_id=claimed.current_run_id,
        )
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at) VALUES (?, 'reviewer', 'rate_limited', "
                "'rate_limited', ?, ?)",
                (tid, now, now + 5),
            )
        assert (
            kb.check_respawn_guard(conn, tid, lane="review")
            == "rate_limit_cooldown"
        )
        assert kb.has_spawnable_review(conn) is False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI standalone daemon (--force) health accounting
# ---------------------------------------------------------------------------


def _run_daemon_ticks(monkeypatch, ticks: int, dispatch=None, result=None):
    """Drive ``_cmd_daemon`` through ``ticks`` health ticks without the
    real sleep loop.

    ``dispatch(conn, kwargs) -> DispatchResult`` replaces the per-tick
    dispatch call (default: the real ``dispatch_once``); ``result``
    supplies a fixed canned result instead. Returns (returncode, stderr).
    """
    import hermes_cli.kanban as kanban_cli

    def fake_run_daemon(**kwargs):
        on_tick = kwargs["on_tick"]
        with kb.connect_closing() as conn:
            for _ in range(ticks):
                if result is not None:
                    res = result
                else:
                    res = (dispatch or (lambda c, kw: kb.dispatch_once(c)))(
                        conn, kwargs
                    )
                on_tick(res)

    monkeypatch.setattr(kb, "run_daemon", fake_run_daemon)
    args = argparse.Namespace(
        force=True, pidfile=None, verbose=False, interval=5, max=None,
    )
    rc = kanban_cli._cmd_daemon(args)
    return rc


def test_daemon_no_stuck_warning_when_only_ready_task_is_guard_deferred(
    kanban_home, all_profiles_real, monkeypatch, capsys
) -> None:
    """The reporter's incident: every ready task deferred by the guard,
    zero spawns, board idle BY DESIGN — no stuck alarm may fire.

    Uses the real dispatch pass each tick; the spawn hook records any
    attempt to prove the telemetry fix did not need to touch dispatch.
    """
    now = int(time.time())
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="already done", assignee="worker")
        _seed_recent_success(conn, tid, ended_at=now - 60)
    finally:
        conn.close()

    spawn_attempts: list[str] = []

    def _no_spawn(task, workspace, **kwargs):
        spawn_attempts.append(task.id)
        return 424242

    def dispatch(conn, kwargs):
        return kb.dispatch_once(conn, spawn_fn=_no_spawn)

    rc = _run_daemon_ticks(monkeypatch, ticks=8, dispatch=dispatch)
    err = capsys.readouterr().err

    assert rc == 0
    assert spawn_attempts == [], "guard must defer before any spawn attempt"
    assert "dispatcher stuck" not in err, (
        "guard-deferred idle board must not be reported stuck"
    )


def test_daemon_warns_when_spawnable_task_truly_exists_but_nothing_spawns(
    kanban_home, all_profiles_real, monkeypatch, capsys
) -> None:
    """The warning must survive for a genuinely spawnable task that never
    spawns — suppressing on guard activity alone would mask real stalls.
    """
    conn = kb.connect()
    try:
        kb.create_task(conn, title="should spawn", assignee="worker")
    finally:
        conn.close()

    rc = _run_daemon_ticks(monkeypatch, ticks=8, result=kb.DispatchResult())
    err = capsys.readouterr().err

    assert rc == 0
    warnings = [ln for ln in err.splitlines() if "dispatcher stuck" in ln]
    assert len(warnings) == 1, err
    assert "Respawn-guard deferrals" not in warnings[0]


def test_daemon_mixed_board_still_warns_and_lists_deferrals(
    kanban_home, all_profiles_real, monkeypatch, capsys
) -> None:
    """Guard-deferred + genuinely-stuck tasks together: the warning fires
    AND names the deferrals so operators can split idle-by-design from
    broken.
    """
    now = int(time.time())
    conn = kb.connect()
    try:
        guarded_tid = kb.create_task(conn, title="parked", assignee="worker")
        _seed_recent_success(conn, guarded_tid, ended_at=now - 60)
        kb.create_task(conn, title="broken spawn target", assignee="worker2")
    finally:
        conn.close()

    canned = kb.DispatchResult(
        respawn_guarded=[(guarded_tid, "recent_success")],
    )
    rc = _run_daemon_ticks(monkeypatch, ticks=8, result=canned)
    err = capsys.readouterr().err

    assert rc == 0
    warnings = [ln for ln in err.splitlines() if "dispatcher stuck" in ln]
    assert len(warnings) == 1, err
    assert f"{guarded_tid} (recent_success)" in warnings[0]


# ---------------------------------------------------------------------------
# Gateway-embedded dispatcher watcher
# ---------------------------------------------------------------------------


def _gateway_config():
    import hermes_cli.config as cfgmod

    return {
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "auto_decompose": False,
        }
    }


def _run_gateway_dispatcher(monkeypatch, runner, min_ticks: int) -> list:
    """Run ``_kanban_dispatcher_watcher`` inline for ``min_ticks`` ticks.

    Mirrors the harness from test_kanban_core_functionality: fake
    ``asyncio.to_thread`` executes synchronously and stops the loop after
    enough reap calls (one reaper per tick); sleeps are no-ops.
    """
    import hermes_cli.kanban_db as _kb
    import hermes_cli.config as cfg_mod

    ticks = {"n": 0}

    async def _to_thread(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        if getattr(fn, "__name__", "") == "reap_worker_zombies":
            ticks["n"] += 1
            if ticks["n"] >= min_ticks:
                runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr(cfg_mod, "load_config", _gateway_config)
    monkeypatch.setattr(
        _kb, "list_boards",
        lambda include_archived=False: [{"slug": _kb.DEFAULT_BOARD}],
    )
    monkeypatch.setattr("asyncio.to_thread", _to_thread)
    monkeypatch.setattr("asyncio.sleep", _sleep)

    asyncio.run(
        asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=10.0)
    )
    return ticks


def test_gateway_watcher_no_stuck_warning_when_all_ready_guard_deferred(
    kanban_home, all_profiles_real, monkeypatch, caplog
) -> None:
    from gateway.run import GatewayRunner

    now = int(time.time())
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="pr merged", assignee="worker")
        _seed_recent_success(conn, tid, ended_at=now - 60)
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _run_gateway_dispatcher(monkeypatch, runner, min_ticks=8)

    stuck = [
        r.getMessage() for r in caplog.records
        if "kanban dispatcher stuck" in r.getMessage()
    ]
    assert stuck == [], (
        "guard-deferred idle board must not be reported stuck by the "
        "gateway watcher either"
    )


def test_gateway_watcher_warns_when_spawnable_task_truly_unspawned(
    kanban_home, all_profiles_real, monkeypatch, caplog
) -> None:
    """Aggregate telemetry must still fire when a spawnable ready task
    exists but the dispatch pass spawned nothing (dispatch stubbed to the
    empty outcome; the probe reads real DB state)."""
    import hermes_cli.kanban_db as _kb
    from gateway.run import GatewayRunner

    conn = kb.connect()
    try:
        kb.create_task(conn, title="never spawns", assignee="worker")
    finally:
        conn.close()

    monkeypatch.setattr(
        _kb, "dispatch_once", lambda conn, **kw: kb.DispatchResult()
    )

    runner = object.__new__(GatewayRunner)
    runner._running = True

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _run_gateway_dispatcher(monkeypatch, runner, min_ticks=8)

    stuck = [
        r.getMessage() for r in caplog.records
        if "kanban dispatcher stuck" in r.getMessage()
    ]
    assert len(stuck) == 1, caplog.records
