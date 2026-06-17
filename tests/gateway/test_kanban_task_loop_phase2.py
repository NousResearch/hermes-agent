"""Phase 2 task-loop callback conditional trigger tests.

The orchestrator task-loop callback used to fire on every tick that had
any "interesting" state (ready cards, terminal events, etc.). Phase 2
narrows the trigger so:

  Rule 1 — ready queue non-empty AND no terminal events → skip LLM
            injection (dispatcher will pick up the ready card itself;
            the orchestrator LLM only needs to react to terminal events
            or failure pathology).

  Rule 2 — ready empty AND terminal events → fire normally (this is
            the legacy behavior — explicitly preserved).

  Rule 3 — any card with consecutive_failures >= threshold → force
            injection with [URGENT] marker, even when the board has
            other ready cards waiting (failure loops don't fix
            themselves).

These tests drive ``_kanban_orchestrator_callback`` directly via
mocks; they do not need real SQLite for the rule-coverage tests. The
DB-backed tests under ``test_has_force_failure_*`` exercise the
SQLite path with a real tmp DB.

Run:
  cd /home/zml/workspace/hermes-agent
  venv/bin/python -m pytest tests/gateway/test_kanban_task_loop_phase2.py -v
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# --- helpers ---------------------------------------------------------------


def _make_runner_with_mocks():
    """Build a FakeRunner shaped like the production gateway runner,
    but with adapter / source / handlers stubbed out."""
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    class FakeRunner(GatewayKanbanWatchersMixin):
        def __init__(self):
            self.adapters = {}
            self._running = True
            self._kanban_last_user_source = {
                "test-board": ("feishu", "chat-abc"),
            }
            self.injected_events = []
            self.send_calls = []
            # Captured call to _inject_task_loop (called by callback when
            # trigger condition is met) so we can assert the trigger
            # fired or skipped without running the actual agent.
            self.inject_task_loop_calls = []

        async def _handle_message(self, ev):
            self.injected_events.append(ev)
            return None

        def _inject_task_loop(self, msg_text, slug, stats):
            self.inject_task_loop_calls.append({
                "msg_text": msg_text,
                "slug": slug,
                "stats": stats,
            })

    return FakeRunner()


def _stats(
    *,
    ready_count: int = 0,
    in_progress_count: int = 0,
    in_progress_names=None,
    event_details=None,
    has_terminal_events: bool = False,
    current_loop: int = 1,
    MAX_LOOPS: int = 10,
    blocked_count: int = 0,
):
    """Build the stats dict shape that ``_detect_task_loop`` returns."""
    return {
        "in_progress_count": in_progress_count,
        "in_progress_names": list(in_progress_names or []),
        "ready_count": ready_count,
        "blocked_count": blocked_count,
        "event_details": list(event_details or []),
        "has_terminal_events": has_terminal_events,
        "current_loop": current_loop,
        "MAX_LOOPS": MAX_LOOPS,
        "max_eid": 0,
    }


# --- core rules ------------------------------------------------------------


@pytest.mark.asyncio
async def test_rule1_skips_when_ready_nonempty_and_no_terminal_events():
    """Phase 2 rule 1: ready cards waiting + no terminal events →
    skip LLM injection entirely."""
    runner = _make_runner_with_mocks()
    stats = _stats(
        ready_count=2,
        has_terminal_events=False,
        event_details=[],
    )

    with patch.object(
        runner, "_scan_candidate_boards", return_value=["test-board"],
    ), patch.object(
        runner, "_detect_task_loop", return_value=stats,
    ), patch.object(
        runner, "_has_force_failure", return_value=False,
    ), patch.object(
        runner, "_auto_complete_parents", return_value=[],
    ):
        await runner._kanban_orchestrator_callback([], {"orchestrator_notify": True})

    assert runner.inject_task_loop_calls == [], (
        "rule 1: ready non-empty + no terminal + no force failure "
        "must skip injection"
    )


@pytest.mark.asyncio
async def test_rule2_fires_when_ready_empty_and_terminal_events():
    """Phase 2 rule 2 (legacy): empty ready + terminal events → fire
    normally. This is the path Phase 1 already used."""
    runner = _make_runner_with_mocks()
    stats = _stats(
        ready_count=0,
        has_terminal_events=True,
        event_details=[{
            "task_id": "t_done_001",
            "kind": "completed",
            "title": "Phase 1 work",
            "assignee": "coder",
            "summary": "shipped",
        }],
    )

    with patch.object(
        runner, "_scan_candidate_boards", return_value=["test-board"],
    ), patch.object(
        runner, "_detect_task_loop", return_value=stats,
    ), patch.object(
        runner, "_has_force_failure", return_value=False,
    ), patch.object(
        runner, "_auto_complete_parents", return_value=[],
    ):
        await runner._kanban_orchestrator_callback([], {"orchestrator_notify": True})

    assert len(runner.inject_task_loop_calls) == 1, (
        "rule 2: empty ready + terminal events must fire"
    )
    call = runner.inject_task_loop_calls[0]
    assert call["slug"] == "test-board"
    assert call["stats"]["force_urgent"] is False
    assert "[URGENT]" not in call["msg_text"]


@pytest.mark.asyncio
async def test_rule3_force_fires_when_consecutive_failures_exceeded():
    """Phase 2 rule 3: a stuck task with consecutive_failures >= 3
    must force the trigger even when ready cards are waiting."""
    runner = _make_runner_with_mocks()
    stats = _stats(
        ready_count=3,  # would normally block (rule 1)
        has_terminal_events=False,
        event_details=[],
    )

    with patch.object(
        runner, "_scan_candidate_boards", return_value=["test-board"],
    ), patch.object(
        runner, "_detect_task_loop", return_value=stats,
    ), patch.object(
        runner, "_has_force_failure", return_value=True,
    ), patch.object(
        runner, "_failing_task_ids",
        return_value=["t_stuck_aaa", "t_stuck_bbb"],
    ), patch.object(
        runner, "_auto_complete_parents", return_value=[],
    ):
        await runner._kanban_orchestrator_callback(
            [],
            {
                "orchestrator_notify": True,
                "orchestrator_force_failure_threshold": 3,
            },
        )

    assert len(runner.inject_task_loop_calls) == 1, (
        "rule 3: force-failure must trigger even when ready queue is full"
    )
    call = runner.inject_task_loop_calls[0]
    assert call["slug"] == "test-board"
    assert call["stats"]["force_urgent"] is True
    assert call["stats"]["force_failure_threshold"] == 3
    assert call["stats"]["failing_task_ids"] == ["t_stuck_aaa", "t_stuck_bbb"]
    msg = call["msg_text"]
    assert msg.startswith("[URGENT]"), (
        f"forced trigger must carry [URGENT] prefix; got: {msg[:120]!r}"
    )
    assert "Stuck tasks" in msg
    assert "t_stuck_aaa" in msg
    assert "t_stuck_bbb" in msg
    assert "Triage the stuck tasks first" in msg


@pytest.mark.asyncio
async def test_rule3_force_fires_with_terminal_events_too():
    """Sanity: rule 3 fires whether or not the board also has terminal
    events. We don't want force-urgent to be suppressed by concurrent
    activity — the failing cards still need eyes on them."""
    runner = _make_runner_with_mocks()
    stats = _stats(
        ready_count=1,
        has_terminal_events=True,
        event_details=[{
            "task_id": "t_blocked_xyz",
            "kind": "blocked",
            "title": "needs decision",
            "assignee": "pm",
            "summary": "waiting for user",
        }],
    )

    with patch.object(
        runner, "_scan_candidate_boards", return_value=["test-board"],
    ), patch.object(
        runner, "_detect_task_loop", return_value=stats,
    ), patch.object(
        runner, "_has_force_failure", return_value=True,
    ), patch.object(
        runner, "_failing_task_ids",
        return_value=["t_loop_loop"],
    ), patch.object(
        runner, "_auto_complete_parents", return_value=[],
    ):
        await runner._kanban_orchestrator_callback([], {"orchestrator_notify": True})

    assert len(runner.inject_task_loop_calls) == 1
    call = runner.inject_task_loop_calls[0]
    assert call["stats"]["force_urgent"] is True
    assert call["stats"]["failing_task_ids"] == ["t_loop_loop"]


@pytest.mark.asyncio
async def test_no_force_no_terminal_and_no_ready_returns_none_skip():
    """A board that is truly idle (no ready, no terminal events, no
    failing tasks) must still skip — _detect_task_loop returns None in
    that case and we never even reach the force-failure probe."""
    runner = _make_runner_with_mocks()

    with patch.object(runner, "_detect_task_loop", return_value=None):
        await runner._kanban_orchestrator_callback([], {"orchestrator_notify": True})

    assert runner.inject_task_loop_calls == []


@pytest.mark.asyncio
async def test_force_failure_threshold_from_config():
    """orchestrator_force_failure_threshold in config must override
    the hard-coded default (3)."""
    runner = _make_runner_with_mocks()
    stats = _stats(
        ready_count=1,
        has_terminal_events=False,
        event_details=[],
    )

    with patch.object(
        runner, "_scan_candidate_boards", return_value=["test-board"],
    ), patch.object(
        runner, "_detect_task_loop", return_value=stats,
    ), patch.object(
        runner, "_has_force_failure", return_value=True,
    ), patch.object(
        runner, "_failing_task_ids", return_value=["t_x"],
    ), patch.object(
        runner, "_auto_complete_parents", return_value=[],
    ):
        await runner._kanban_orchestrator_callback(
            [],
            {
                "orchestrator_notify": True,
                "orchestrator_force_failure_threshold": 5,
            },
        )

    assert runner.inject_task_loop_calls[0]["stats"]["force_failure_threshold"] == 5


# --- _has_force_failure unit tests (real DB) -------------------------------


def _seed_task(db_path: Path, task_id: str, status: str, failures: int):
    """Insert a task directly via SQL so we can control every column.

    create_task() in kanban_db.py doesn't expose consecutive_failures
    as a kwarg (it's set later by the failure paths), so we go
    straight to SQL to seed deterministic state.
    """
    import time
    from hermes_cli import kanban_db as _kb

    conn = _kb.connect(db_path=db_path)
    try:
        now = int(time.time())
        conn.execute(
            "INSERT INTO tasks "
            "(id, title, status, assignee, created_at, "
            " consecutive_failures) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, f"task-{task_id}", status, "coder", now, failures),
        )
        conn.commit()
    finally:
        conn.close()


def _force_failure_with_db(board_slug: str, threshold: int, db_path: Path):
    """Run ``_has_force_failure`` against a controlled tmp DB."""
    from hermes_cli import kanban_db as _kb
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    real_connect = _kb.connect
    with patch.object(
        _kb, "connect",
        side_effect=lambda board=None, db_path_arg=None, **kw: real_connect(
            db_path=db_path,
        ),
    ):
        class Stub(GatewayKanbanWatchersMixin):
            pass

        stub = Stub()
        return stub._has_force_failure(board_slug, threshold)


def _failing_ids_with_db(board_slug: str, threshold: int, db_path: Path):
    from hermes_cli import kanban_db as _kb
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    real_connect = _kb.connect
    with patch.object(
        _kb, "connect",
        side_effect=lambda board=None, db_path_arg=None, **kw: real_connect(
            db_path=db_path,
        ),
    ):
        class Stub(GatewayKanbanWatchersMixin):
            pass

        stub = Stub()
        return stub._failing_task_ids(board_slug, threshold)


def test_has_force_failure_returns_true_when_task_above_threshold(tmp_path):
    """End-to-end DB read: a task with consecutive_failures >= threshold
    in a non-terminal status must be detected."""
    db_path = tmp_path / "kanban.db"
    _seed_task(db_path, "t_stuck", "blocked", failures=4)

    assert _force_failure_with_db("phase2-test", threshold=3, db_path=db_path) is True


def test_has_force_failure_returns_false_when_below_threshold(tmp_path):
    db_path = tmp_path / "kanban.db"
    _seed_task(db_path, "t_ok", "blocked", failures=2)

    assert _force_failure_with_db("phase2-test", threshold=3, db_path=db_path) is False


def test_has_force_failure_ignores_done_tasks(tmp_path):
    """A high failure count on a *done* task is historical — must not
    trigger force-urgent (defense-in-depth: status filter catches
    legacy rows even if the success path didn't clear the counter)."""
    db_path = tmp_path / "kanban.db"
    _seed_task(db_path, "t_done_with_history", "done", failures=9)

    assert _force_failure_with_db("phase2-test", threshold=3, db_path=db_path) is False


def test_has_force_failure_returns_false_on_db_error():
    """A transient DB error must NOT cause a false-positive force
    trigger (that would inject LLM work into a broken board)."""
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    class Stub(GatewayKanbanWatchersMixin):
        pass

    with patch("hermes_cli.kanban_db.connect", side_effect=RuntimeError("db down")):
        assert Stub()._has_force_failure("any-board", threshold=3) is False


def test_failing_task_ids_returns_only_above_threshold_sorted_by_failures(tmp_path):
    """Most-broken cards must come first so the orchestrator message
    surfaces the worst pathology first. Below-threshold rows are
    excluded entirely."""
    db_path = tmp_path / "kanban.db"
    _seed_task(db_path, "t_low", "blocked", failures=3)
    _seed_task(db_path, "t_high", "blocked", failures=7)
    _seed_task(db_path, "t_mid", "ready", failures=5)
    _seed_task(db_path, "t_ok", "blocked", failures=2)  # below threshold

    ids = _failing_ids_with_db("phase2-test", threshold=3, db_path=db_path)

    assert len(ids) == 3
    # The below-threshold row must not appear.
    assert "t_ok" not in ids
    # First id must be the highest-failure row.
    assert ids[0] == "t_high"


def test_failing_task_ids_returns_empty_on_db_error():
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    class Stub(GatewayKanbanWatchersMixin):
        pass

    with patch("hermes_cli.kanban_db.connect", side_effect=RuntimeError("db down")):
        assert Stub()._failing_task_ids("any-board", threshold=3) == []


# --- message builder -------------------------------------------------------


def test_build_task_loop_message_omits_urgent_when_normal():
    """Non-urgent task-loop cycles must NOT carry the [URGENT] prefix or the
    Stuck tasks line."""
    runner = _make_runner_with_mocks()
    stats = _stats(
        ready_count=0,
        has_terminal_events=True,
        event_details=[{
            "task_id": "t_normal",
            "kind": "completed",
            "title": "ok",
            "assignee": "coder",
            "summary": "shipped",
        }],
    )
    stats["auto_completed"] = []
    stats["force_urgent"] = False
    msg = runner._build_task_loop_message("test-board", stats["event_details"], stats)
    assert "[URGENT]" not in msg
    assert "Stuck tasks" not in msg


def test_build_task_loop_message_includes_urgent_when_force_triggered():
    runner = _make_runner_with_mocks()
    stats = _stats(
        ready_count=2,
        has_terminal_events=False,
        event_details=[],
    )
    stats["auto_completed"] = []
    stats["force_urgent"] = True
    stats["force_failure_threshold"] = 3
    stats["failing_task_ids"] = ["t_loop_001", "t_loop_002"]
    msg = runner._build_task_loop_message("test-board", stats["event_details"], stats)
    assert msg.startswith("[URGENT]")
    assert "Stuck tasks (>= 3 failures)" in msg
    assert "t_loop_001" in msg and "t_loop_002" in msg
    # Orchestrator instructions must include the triage-first prompt.
    assert "Triage the stuck tasks first" in msg


def test_build_task_loop_message_truncates_long_failing_lists():
    """The stuck-tasks preview is bounded to 8 ids to keep the message
    under control — anything past that gets a `(+N more)` suffix."""
    runner = _make_runner_with_mocks()
    failing = [f"t_failing_{i:03d}" for i in range(15)]
    stats = _stats(ready_count=0, has_terminal_events=False, event_details=[])
    stats["auto_completed"] = []
    stats["force_urgent"] = True
    stats["force_failure_threshold"] = 3
    stats["failing_task_ids"] = failing
    msg = runner._build_task_loop_message("test-board", stats["event_details"], stats)
    for tid in failing[:8]:
        assert tid in msg
    assert "(+7 more)" in msg
    for tid in failing[8:]:
        assert tid not in msg


# --- convergence injection (T0: t_aed127cf) -------------------------------


def _fake_converged_metrics(
    resolved: int = 5,
    total: int = 5,
    blocked_ratio: float = 0.0,
    resolve_rate: float = 1.0,
    vf: int = 0,
    new_tasks: int = 0,
) -> dict:
    """Build a fake ``compute_board_convergence`` metrics dict with
    ``converged=True``.  Matches the real function's output shape so
    the message builder renders metrics correctly.
    """
    return {
        "total_tasks": total,
        "resolved": resolved,
        "blocked": 0,
        "running": 0,
        "ready": 0,
        "blocked_ratio": blocked_ratio,
        "resolve_rate": resolve_rate,
        "new_tasks_created": new_tasks,
        "verification_failed": vf,
        "converged": True,
    }


@pytest.mark.asyncio
async def test_convergence_injects_final_summary_when_board_fully_done():
    """When the board is fully converged (all tasks done, no pending,
    no recent failures), the orchestrator callback must inject a
    final summary message — even when ``_detect_task_loop`` returns
    ``None`` because there's no recent activity.  This is the bug
    T0/t_aed127cf fixes: previously convergence wrote a DB event but
    never told the coordinator.
    """
    runner = _make_runner_with_mocks()
    # _detect_task_loop returns None for the "all-done quiescent" case.
    metrics = _fake_converged_metrics(resolved=5, total=5)

    with patch.object(
        runner, "_scan_candidate_boards", return_value=["test-board"],
    ), patch.object(
        runner, "_detect_task_loop", return_value=None,
    ), patch.object(
        runner, "_board_converged", return_value=metrics,
    ), patch.object(
        runner, "_auto_complete_parents", return_value=[],
    ):
        await runner._kanban_orchestrator_callback(
            [], {"orchestrator_notify": True},
        )

    assert len(runner.inject_task_loop_calls) == 1, (
        "convergence must trigger exactly one _inject_task_loop call"
    )
    call = runner.inject_task_loop_calls[0]
    assert call["slug"] == "test-board"
    # Convergence marker must be propagated through stats so the
    # message builder uses the dedicated convergence branch.
    assert call["stats"]["converged"] is True
    assert call["stats"]["convergence_metrics"] == metrics
    # The injected message must include convergence content.
    msg = call["msg_text"]
    assert "📋" in msg, f"convergence msg must have 📋 header; got: {msg!r}"
    assert "5/5" in msg, (
        f"convergence msg must report resolved/total; got: {msg!r}"
    )
    assert "resolve_rate=100%" in msg
    # The wrap-up instruction must be present.
    assert "所有任务已完成" in msg
    assert "complete 自己的 orchestrator 任务" in msg


@pytest.mark.asyncio
async def test_convergence_injects_even_when_detect_task_loop_returns_stats():
    """If ``_detect_task_loop`` already produced a stats dict (e.g. recent
    terminal events from final tasks), convergence must still fire
    and override Rule 1 / Rule 3 — the converged message is more
    important than normal loop bookkeeping.
    """
    runner = _make_runner_with_mocks()
    stats = _stats(
        ready_count=0,
        has_terminal_events=True,
        event_details=[{
            "task_id": "t_last_one",
            "kind": "completed",
            "title": "final task",
            "assignee": "coder",
            "summary": "shipped",
        }],
    )
    metrics = _fake_converged_metrics(resolved=3, total=3)

    with patch.object(
        runner, "_scan_candidate_boards", return_value=["test-board"],
    ), patch.object(
        runner, "_detect_task_loop", return_value=stats,
    ), patch.object(
        runner, "_board_converged", return_value=metrics,
    ), patch.object(
        runner, "_auto_complete_parents", return_value=[],
    ):
        await runner._kanban_orchestrator_callback(
            [], {"orchestrator_notify": True},
        )

    assert len(runner.inject_task_loop_calls) == 1
    call = runner.inject_task_loop_calls[0]
    assert call["stats"]["converged"] is True
    msg = call["msg_text"]
    # Convergence path uses the dedicated branch — no [URGENT] prefix,
    # no "Workers idle" header.
    assert "[URGENT]" not in msg
    assert "Workers idle" not in msg
    assert "📋" in msg
    assert "3/3" in msg


@pytest.mark.asyncio
async def test_convergence_skips_when_board_not_converged():
    """If ``_board_converged`` returns ``None``, the convergence branch
    must NOT inject — the regular rules (Rule 1, Rule 2, Rule 3)
    decide whether to fire.  This is the negative-control case: a
    board that still has work in progress or recent verification
    failures must not be told "all done" prematurely.
    """
    runner = _make_runner_with_mocks()
    stats = _stats(ready_count=1, has_terminal_events=True, event_details=[])

    with patch.object(
        runner, "_scan_candidate_boards", return_value=["test-board"],
    ), patch.object(
        runner, "_detect_task_loop", return_value=stats,
    ), patch.object(
        runner, "_board_converged", return_value=None,
    ), patch.object(
        runner, "_has_force_failure", return_value=False,
    ), patch.object(
        runner, "_auto_complete_parents", return_value=[],
    ):
        await runner._kanban_orchestrator_callback(
            [], {"orchestrator_notify": True},
        )

    # The regular Rule 2 path should still fire (terminal events
    # present, no ready cards blocking the trigger).  Crucially, the
    # message must NOT carry the convergence marker.
    assert len(runner.inject_task_loop_calls) == 1
    call = runner.inject_task_loop_calls[0]
    assert call["stats"].get("converged") is None or call["stats"].get("converged") is False
    msg = call["msg_text"]
    assert "📋" not in msg
    assert "所有任务已完成" not in msg


def test_build_task_loop_message_convergence_branch_includes_metrics():
    """The convergence branch in ``_build_task_loop_message`` must render
    the metrics in a way the coordinator can act on: resolved/total
    ratio, blocked_ratio, resolve_rate, recent activity counts, and
    the wrap-up instruction.
    """
    runner = _make_runner_with_mocks()
    stats = _stats(ready_count=0, has_terminal_events=False, event_details=[])
    stats["converged"] = True
    stats["convergence_metrics"] = _fake_converged_metrics(
        resolved=7, total=10, blocked_ratio=0.1, resolve_rate=0.7, vf=1, new_tasks=0,
    )
    stats["auto_completed"] = []
    stats["force_urgent"] = False
    stats["failing_task_ids"] = []

    msg = runner._build_task_loop_message("alpha-board", stats["event_details"], stats)
    # Header
    assert msg.startswith("📋 Kanban Board \"alpha-board\" — 全部完成")
    # Metrics line
    assert "7/10" in msg
    assert "resolve_rate=70%" in msg
    assert "blocked_ratio=10%" in msg
    # Recent activity only when nonzero
    assert "verification_failed=1" in msg
    # Wrap-up instructions
    assert "--- Orchestrator Instructions ---" in msg
    assert "所有任务已完成" in msg
    assert "complete 自己的 orchestrator 任务" in msg
    # Convergence path must not include loop-counter or stuck-task
    # formatting from the normal branch.
    assert "[Kanban Task Loop" not in msg
    assert "Stuck tasks" not in msg