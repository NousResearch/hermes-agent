"""Worker turn/phase budget + handoff telemetry (#111303).

Covers the six acceptance criteria: a running worker's budget is queryable without
logs (1), a completed worker leaves the same normalized record (2), parent / subagent
/ retry accounting stay distinguishable (3), unavailable telemetry is null rather than
zero (4), serialization is versioned + deterministic + secret-safe (5), and the normal,
hard-exhaustion, provider-retry/fallback, reserved-handoff and restart/recovery paths
all behave (6).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_worker_budget as wb

TASK_ID = "t_111303"
PROFILE = "alice"
RUN_ID = 7
SECRET = "sk-live-DO-NOT-LEAK-111303"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def worker_env(kanban_home, monkeypatch):
    """A claimed kanban worker's process environment with telemetry opted IN."""
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="telem", assignee=PROFILE)
        kb.claim_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
    finally:
        conn.close()
    monkeypatch.setenv(wb.ENV_OPT_IN, "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_PROFILE", PROFILE)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.id))
    return SimpleNamespace(home=kanban_home, task_id=task_id, run_id=run.id)


def _agent(*, total=60, used=5, remaining=55, api_calls=None, messages=None, **attrs):
    """A minimal live-agent stand-in: only the attributes the producer reads."""
    agent = SimpleNamespace(
        iteration_budget=SimpleNamespace(max_total=total, used=used, remaining=remaining),
        session_api_calls=api_calls if api_calls is not None else used,
        session_input_tokens=1000,
        session_output_tokens=200,
        session_estimated_cost_usd=0.25,
        session_cost_status="estimated",
        session_id="sess-1",
        budget_warning_ratio=0.9,
        _iteration_budget_warning_injected=False,
        model="test-model",
        # Things that must NEVER reach the wire:
        api_key=SECRET,
        system_prompt=f"secret prompt {SECRET}",
        messages=[{"role": "user", "content": f"private {SECRET}"}],
    )
    agent.__dict__.update(attrs)
    return agent


def _messages(with_tool_call=True):
    msgs = [{"role": "user", "content": "do the thing"}]
    if with_tool_call:
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]})
        msgs.append({"role": "tool", "name": "read_file", "content": "file body"})
    msgs.append({"role": "assistant", "content": "done"})
    return msgs


def _events(task_id):
    conn = kbc.connect()
    try:
        return [e for e in kb.list_events(conn, task_id) if e.kind == wb.EVENT_KIND]
    finally:
        conn.close()


# --- AC 5: opt-in gate -------------------------------------------------------


def test_telemetry_is_off_by_default_and_opt_in(kanban_home, monkeypatch):
    """Nothing is recorded unless it is opted in (env or kanban config)."""
    monkeypatch.delenv(wb.ENV_OPT_IN, raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    assert wb.telemetry_enabled() is False

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {wb.CONFIG_KEY: True}},
    )
    assert wb.telemetry_enabled() is True

    # An explicit env '0' overrides an opted-in config.
    monkeypatch.setenv(wb.ENV_OPT_IN, "0")
    assert wb.telemetry_enabled() is False
    monkeypatch.setenv(wb.ENV_OPT_IN, "1")
    assert wb.telemetry_enabled() is True


def test_emit_is_a_noop_when_not_opted_in(worker_env, monkeypatch):
    monkeypatch.delenv(wb.ENV_OPT_IN, raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    assert wb.emit(_agent(), phase=wb.PHASE_TERMINAL, force=True) is None
    assert _events(worker_env.task_id) == []


def test_emit_is_a_noop_outside_a_kanban_worker(kanban_home, monkeypatch):
    monkeypatch.setenv(wb.ENV_OPT_IN, "1")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert wb.emit(_agent(), phase=wb.PHASE_TERMINAL, force=True) is None


# --- AC 1: a running worker is queryable ------------------------------------


def test_running_worker_budget_is_queryable_from_the_board(worker_env):
    """A checkpoint written mid-turn answers the whole budget question — no log parsing."""
    snap = wb.emit(_agent(total=60, used=12, remaining=48, api_calls=12), phase=wb.PHASE_TURN)
    assert snap is not None

    report = wb.collect(worker_env.task_id)
    assert report["schema"] == wb.SCHEMA
    assert report["schema_version"] == wb.SCHEMA_VERSION
    assert report["source"]["kind"] == "worker"
    assert report["phase"]["name"] == "turn"
    assert report["turns"] == {
        "total": 60,
        "used": 12,
        "remaining": 48,
        "reserved_handoff": wb.HANDOFF_RESERVE_TURNS,
        "usable_remaining": 47,
    }
    assert report["counts"]["provider_attempts"] == 12
    assert report["identity"] == {
        "task_id": worker_env.task_id,
        "run_id": worker_env.run_id,
        "profile": PROFILE,
        "board": "default",
        "session_id": None,
        "parents": [],
        "children": [],
    }
    # Board-derived facts are present while the worker is still running.
    assert report["time"]["ended_at"] is None
    assert report["time"]["elapsed_seconds"] is not None
    assert report["attempts"]["index"] == 1
    assert report["handoff"]["available"] is False
    assert report["unavailable"] == sorted(report["unavailable"])


def test_checkpoint_emits_are_rate_limited_but_forced_ones_land(worker_env, monkeypatch):
    monkeypatch.setattr(wb, "_last_checkpoint_emit", 0.0)
    assert wb.emit(_agent(), phase=wb.PHASE_TURN) is not None
    assert wb.emit(_agent(), phase=wb.PHASE_TURN) is None  # inside the checkpoint window
    assert wb.emit(_agent(), phase=wb.PHASE_TURN, force=True) is not None
    assert len(_events(worker_env.task_id)) == 2


# --- AC 2: same fields after completion ------------------------------------


def test_completed_worker_keeps_the_same_normalized_fields(worker_env):
    wb.emit(
        _agent(total=60, used=60, remaining=0, api_calls=61, _iteration_budget_warning_injected=True),
        phase=wb.PHASE_HANDOFF,
        exit_reason="max_iterations_reached(60/60)",
        messages=_messages(),
    )
    conn = kbc.connect()
    try:
        assert kb.complete_task(conn, worker_env.task_id, result="ok", summary="handoff summary") is True
    finally:
        conn.close()

    report = wb.collect(worker_env.task_id)
    assert report["phase"]["name"] == "handoff"
    assert report["turns"]["used"] == 60
    assert report["turns"]["remaining"] == 0
    assert report["counts"]["tool_turns"] == 1
    assert report["counts"]["no_action_turns"] == 1
    assert report["usage"] == {
        "input_tokens": 1000, "output_tokens": 200, "cost_usd": 0.25, "cost_status": "estimated",
    }
    assert report["termination"]["reason"] == "max_iterations_reached(60/60)"
    assert report["termination"]["at"] is not None
    assert report["termination"]["warning_at"] is not None
    assert report["handoff"] == {
        "available": True,
        "run_status": "done",
        "outcome": "completed",
        "summary_chars": len("handoff summary"),
        "artifact_count": None,
    }
    assert report["source"]["kind"] == "worker"


# --- AC 3: parent / subagent / retry accounting stay apart ------------------


def test_parent_subagent_and_retry_accounting_are_distinguishable(worker_env, monkeypatch):
    from tools import delegate_tool_registry as registry

    conn = kbc.connect()
    try:
        # Retry: reclaim attempt 1, then claim attempt 2 (before the parent edge exists —
        # an unfinished parent would send the card back to ``todo``).
        assert kb.reclaim_task(conn, worker_env.task_id, reason="stale") is True
        assert kb.claim_task(conn, worker_env.task_id) is not None
        second = kb.latest_run(conn, worker_env.task_id)
        parent_id = kb.create_task(conn, title="parent", assignee="boss")
        kb.link_tasks(conn, parent_id, worker_env.task_id)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(second.id))

    live_child = SimpleNamespace(
        get_activity_summary=lambda: {"api_call_count": 4, "max_iterations": 50, "current_tool": "x"},
    )
    with registry._active_subagents_lock:
        registry._active_subagents.clear()
        registry._active_subagents["sa-1"] = {
            "subagent_id": "sa-1", "parent_id": None, "depth": 1, "status": "running",
            "tool_count": 3, "agent": live_child, "goal": f"secret {SECRET}",
        }
    try:
        snap = wb.emit(_agent(), phase=wb.PHASE_TURN, force=True)
        assert snap["subagents"] == [{
            "subagent_id": "sa-1",
            "parent_id": None,
            "depth": 1,
            "status": "running",
            "tool_count": 3,
            "turns": {"total": 50, "used": 4, "remaining": 46},
        }]
        report = wb.collect(worker_env.task_id)
    finally:
        with registry._active_subagents_lock:
            registry._active_subagents.clear()

    # Parent card vs this worker card vs the retry chain are all separate fields.
    assert report["identity"]["parents"] == [parent_id]
    assert report["identity"]["children"] == []
    assert report["identity"]["run_id"] == second.id
    assert report["attempts"]["index"] == 2
    assert report["attempts"]["total"] == 2
    assert report["subagents"][0]["subagent_id"] == "sa-1"
    # The child's goal text never reaches the record.
    assert SECRET not in json.dumps(report)


# --- AC 4: unavailable is null, never zero ---------------------------------


def test_unobserved_telemetry_is_null_and_named_not_zero(worker_env):
    """No snapshot (worker died before recording, or telemetry was off) -> nulls."""
    report = wb.collect(worker_env.task_id)
    assert report["source"] == {"kind": "store", "recorded_at": None}
    for leaf in ("total", "used", "remaining", "reserved_handoff", "usable_remaining"):
        assert report["turns"][leaf] is None
    assert report["counts"]["provider_attempts"] is None
    assert report["counts"]["tool_turns"] is None
    assert report["usage"]["input_tokens"] is None
    for path in ("turns.used", "turns.total", "counts.provider_attempts", "usage.cost_usd",
                 "phase.recorded_at", "termination.warning_ratio"):
        assert path in report["unavailable"]
    # Board-derived facts are still real values, not placeholders.
    assert report["attempts"] == {
        "index": 1, "total": 1, "consecutive_failures": 0, "protocol_violations": 0,
        "crashed": 0, "timed_out": 0, "rate_limited": 0,
    }


def test_cost_is_unavailable_when_pricing_is_unknown(worker_env):
    snap = wb.snapshot(
        _agent(session_cost_status="unknown", session_estimated_cost_usd=0.0),
        phase=wb.PHASE_TERMINAL, now=1,
    )
    assert snap["usage"]["cost_usd"] is None
    assert snap["usage"]["cost_status"] == "unknown"
    # A real zero (included in a subscription) stays zero.
    included = wb.snapshot(
        _agent(session_cost_status="included", session_estimated_cost_usd=0.0),
        phase=wb.PHASE_TERMINAL, now=1,
    )
    assert included["usage"]["cost_usd"] == 0.0


# --- AC 5: versioned, deterministic, secret-safe ---------------------------


def test_snapshot_serialization_is_deterministic_and_versioned(worker_env):
    first = wb.snapshot(_agent(), phase=wb.PHASE_TERMINAL, exit_reason="text_response(stop)", now=1_700_000_000)
    second = wb.snapshot(_agent(), phase=wb.PHASE_TERMINAL, exit_reason="text_response(stop)", now=1_700_000_000)
    assert json.dumps(first) == json.dumps(second)
    assert first["schema_version"] == 1
    assert first["schema"] == "hermes.kanban.worker_budget/1"


def test_snapshot_carries_no_prompts_credentials_or_tool_content(worker_env):
    agent = _agent()
    agent.iteration_budget.used = 3
    snap = wb.snapshot(
        agent, phase=wb.PHASE_TERMINAL, messages=_messages(), now=1,
    )
    wire = json.dumps(snap)
    for leaked in (SECRET, "secret prompt", "private", "file body", "do the thing"):
        assert leaked not in wire
    # Only whitelisted keys make it into the record.
    assert set(snap) == {
        "schema", "schema_version", "recorded_at", "source", "identity", "phase",
        "turns", "counts", "usage", "termination", "subagents",
    }


# --- AC 6: exhaustion, provider retry/fallback, reserved capacity, restart --


def test_hard_exhaustion_reports_zero_remaining_plus_reserved_handoff(worker_env):
    agent = _agent(total=30, used=30, remaining=0, api_calls=32)
    snap = wb.emit(agent, phase=wb.PHASE_HANDOFF, exit_reason="budget_exhausted", force=True)
    assert snap["turns"] == {
        "total": 30, "used": 30, "remaining": 0, "reserved_handoff": 1, "usable_remaining": 0,
    }
    report = wb.collect(worker_env.task_id)
    assert report["phase"]["name"] == wb.PHASE_HANDOFF
    assert report["termination"]["reason"] == "budget_exhausted"
    assert report["turns"]["reserved_handoff"] == wb.HANDOFF_RESERVE_TURNS
    assert report["turns"]["usable_remaining"] == 0


def test_reserved_handoff_capacity_is_reported_before_exhaustion(worker_env):
    """With budget left, the reserved call is still withheld from usable capacity."""
    snap = wb.snapshot(_agent(total=10, used=8, remaining=2), phase=wb.PHASE_TURN, now=1)
    assert snap["turns"]["remaining"] == 2
    assert snap["turns"]["usable_remaining"] == 1


def test_provider_retry_and_fallback_signals_are_reported(worker_env):
    wb.emit(
        _agent(
            used=4, remaining=56, api_calls=9,
            _rate_limited_until=time.monotonic() + 60,
            _provider_fallback_active=True,
            _fallback_index=2,
        ),
        phase=wb.PHASE_TURN,
        force=True,
    )
    report = wb.collect(worker_env.task_id)
    assert report["counts"] == {
        "provider_attempts": 9,
        "tool_turns": None,
        "no_action_turns": None,
        "provider_rate_limited": True,
        "provider_fallback_active": True,
        "fallback_index": 2,
    }

    # Post-run, the retry/failure classes come from the authoritative run history.
    conn = kbc.connect()
    try:
        conn.execute(
            "UPDATE task_runs SET status='rate_limited', outcome='rate_limited', ended_at=? "
            "WHERE task_id = ?", (int(time.time()), worker_env.task_id),
        )
        conn.execute("UPDATE tasks SET consecutive_failures = 2 WHERE id = ?", (worker_env.task_id,))
        conn.commit()
    finally:
        conn.close()
    attempts = wb.collect(worker_env.task_id)["attempts"]
    assert attempts["rate_limited"] == 1
    assert attempts["consecutive_failures"] == 2
    assert attempts["crashed"] == 0 and attempts["timed_out"] == 0


def test_restart_recovery_keeps_the_schema_and_the_prior_attempt(worker_env):
    """After a reclaim + re-claim the new attempt reports nulls, not the old attempt's numbers."""
    wb.emit(_agent(total=40, used=40, remaining=0), phase=wb.PHASE_TERMINAL,
            exit_reason="crashed", force=True)
    first_run = worker_env.run_id
    conn = kbc.connect()
    try:
        assert kb.reclaim_task(conn, worker_env.task_id, reason="stalled") is True
        assert kb.claim_task(conn, worker_env.task_id) is not None
        second_run = kb.latest_run(conn, worker_env.task_id)
    finally:
        conn.close()

    current = wb.collect(worker_env.task_id)
    assert current["identity"]["run_id"] == second_run.id
    assert current["source"]["kind"] == "store"
    assert current["turns"]["used"] is None
    assert current["attempts"] == {
        "index": 2, "total": 2, "consecutive_failures": 0, "protocol_violations": 0,
        "crashed": 0, "timed_out": 0, "rate_limited": 0,
    }
    # The prior attempt's record is still addressable by run id.
    prior = wb.collect(worker_env.task_id, run_id=first_run)
    assert prior["source"]["kind"] == "worker"
    assert prior["turns"]["used"] == 40
    assert prior["termination"]["reason"] == "crashed"


def test_unknown_run_id_is_an_error_not_a_mismatched_record(worker_env):
    assert wb.collect(worker_env.task_id, run_id=99999) is None
    assert wb.collect("t_missing") is None


def test_review_lane_runs_report_the_reviewer_role(worker_env, monkeypatch):
    """A review-lane attempt is distinguishable from the implementer's."""
    conn = kbc.connect()
    try:
        assert kb.request_review(
            conn, worker_env.task_id, summary="ready for review",
            expected_run_id=worker_env.run_id,
        ) is True
        assert kb.claim_review_task(conn, worker_env.task_id) is not None
        review_run = kb.latest_run(conn, worker_env.task_id)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review_run.id))

    report = wb.collect(worker_env.task_id)
    assert report["identity"]["run_id"] == review_run.id
    assert report["phase"]["role"] == "reviewer"
    assert report["attempts"]["index"] == 2


def test_emit_never_raises_when_the_board_write_fails(worker_env, monkeypatch):
    def _boom(*_a, **_kw):
        raise RuntimeError("board is gone")

    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect_closing", _boom)
    assert wb.emit(_agent(), phase=wb.PHASE_TERMINAL, force=True) is None


# --- producer call sites (turn_finalizer / activity bridge) ----------------


def test_turn_finalizer_records_terminal_snapshot_with_the_right_phase(kanban_home, monkeypatch):
    import agent.turn_finalizer as tf

    seen: list[dict] = []
    monkeypatch.setenv("HERMES_KANBAN_TASK", TASK_ID)
    monkeypatch.setattr(wb, "emit", lambda agent, **kw: seen.append(kw))

    tf._record_worker_budget(_agent(), _messages(), "text_response(stop)",
                             handoff_used=False, logger=logging.getLogger("t"))
    tf._record_worker_budget(_agent(), _messages(), "max_iterations_reached(60/60)",
                             handoff_used=True, logger=logging.getLogger("t"))
    assert [entry["phase"] for entry in seen] == [wb.PHASE_TERMINAL, wb.PHASE_HANDOFF]
    assert seen[1]["exit_reason"] == "max_iterations_reached(60/60)"

    # Outside a kanban worker the hook is inert.
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    tf._record_worker_budget(_agent(), [], "text_response(stop)",
                             handoff_used=False, logger=logging.getLogger("t"))
    assert len(seen) == 2


def test_activity_bridge_checkpoints_the_running_worker(worker_env, monkeypatch):
    """A liveness touch publishes a checkpoint snapshot for the live worker."""
    import agent.activity_tracking as at

    monkeypatch.setattr(wb, "_last_checkpoint_emit", 0.0)
    agent = _agent()
    agent._touch_activity = at.ActivityTrackingMixin._touch_activity.__get__(agent)
    agent._persist_session_activity_if_due = lambda: None

    agent._touch_activity("starting API call #2")
    events = _events(worker_env.task_id)
    assert len(events) == 1
    payload = events[0].payload
    assert payload["phase"]["name"] == wb.PHASE_TURN
    assert payload["turns"]["used"] == 5
    assert payload["identity"]["task_id"] == worker_env.task_id


# --- CLI wiring -------------------------------------------------------------


def test_cli_worker_budget_prints_the_json_record(worker_env):
    wb.emit(_agent(total=60, used=6, remaining=54), phase=wb.PHASE_TURN, force=True)
    payload = json.loads(kc.run_slash(f"worker-budget {worker_env.task_id}"))
    assert payload["schema_version"] == wb.SCHEMA_VERSION
    assert payload["identity"]["task_id"] == worker_env.task_id
    assert payload["turns"]["used"] == 6

    per_run = json.loads(kc.run_slash(f"worker-budget {worker_env.task_id} --run {worker_env.run_id}"))
    assert per_run["identity"]["run_id"] == worker_env.run_id

    unknown = kc.run_slash(f"worker-budget {worker_env.task_id} --run 99999")
    assert "no such task or run" in unknown
