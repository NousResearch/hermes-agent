from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_usage import task_usage
from tools import kanban_tools as kt


@pytest.fixture
def worker_task(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kbc._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="usage", assignee="builder")
        claimed = kb.claim_task(conn, task_id)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    return task_id


def _usage(session_id: str, *, scale: int, actual_cost_usd=None):
    return {
        "session_id": session_id,
        "input_tokens": 100 * scale,
        "output_tokens": 20 * scale,
        "cache_read_tokens": 50 * scale,
        "cache_write_tokens": 5 * scale,
        "reasoning_tokens": 3 * scale,
        "api_call_count": scale,
        "turns": 1,
        "estimated_cost_usd": 0.1 * scale,
        "auxiliary_estimated_cost_usd": 0.0,
        "actual_cost_usd": actual_cost_usd,
        "model": f"model-{scale}",
        "provider": "provider",
        "usage_recorded_at": 1_700_000_000 + scale,
    }


def test_lifecycle_usage_accumulates_across_retries_and_profiles(worker_task, monkeypatch):
    blocked = json.loads(kt._handle_block(
        {"reason": "retry", "kind": "transient"},
        session_usage=_usage("session-builder", scale=1),
    ))
    assert blocked["ok"] is True

    with kbc.connect_closing() as conn:
        assert kb.unblock_task(conn, worker_task)
        assert kb.assign_task(conn, worker_task, "tester")
        claimed = kb.claim_task(conn, worker_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))

    completed = json.loads(kt._handle_complete(
        {"summary": "done"},
        session_usage=_usage("session-tester", scale=2, actual_cost_usd=0.15),
    ))
    assert completed["ok"] is True

    with kbc.connect_closing() as conn:
        runs = kb.list_runs(conn, worker_task)
        usage = task_usage(conn, worker_task)
    assert [run.session_id for run in runs] == ["session-builder", "session-tester"]
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 60
    assert usage["turns"] == 2
    assert usage["estimated_cost_usd"] == pytest.approx(0.3)
    assert usage["cost_usd"] == pytest.approx(0.25)
    assert [(row["profile"], row["input_tokens"]) for row in usage["profiles"]] == [
        ("builder", 100),
        ("tester", 200),
    ]

    shown = json.loads(kc.run_slash(f"show {worker_task} --json"))
    assert shown["usage"] == usage
    assert shown["runs"][1]["model"] == "model-2"


def test_manual_completion_keeps_usage_unrecorded(worker_task):
    with kbc.connect_closing() as conn:
        assert kb.complete_task(conn, worker_task, summary="manual", expected_run_id=None)
        run = kb.latest_run(conn, worker_task)
    assert run.input_tokens == 0
    assert run.actual_cost_usd is None
    assert run.usage_recorded_at is None


@pytest.mark.parametrize(
    ("auxiliary", "expected_input", "expected_calls", "expected_cost"),
    [
        (False, 100, 1, 1.0),
        (True, 150, 2, 1.5),
    ],
)
def test_model_tools_lifecycle_snapshot_includes_auxiliary_without_double_counting(
    worker_task, auxiliary, expected_input, expected_calls, expected_cost,
):
    from agent.tool_executor import _kanban_session_usage
    from hermes_state import SessionDB
    import model_tools

    session_id = f"worker-{'aux' if auxiliary else 'main-only'}"
    session_db = SessionDB(Path(os.environ["HERMES_HOME"]) / "state.db")
    session_db.create_session(session_id, source="cli", model="main-model")
    session_db.update_token_counts(
        session_id,
        input_tokens=100,
        output_tokens=10,
        model="main-model",
        billing_provider="main-provider",
        estimated_cost_usd=1.0,
        api_call_count=1,
    )
    if auxiliary:
        session_db.record_auxiliary_usage(
            session_id,
            "compression",
            model="aux-model",
            billing_provider="aux-provider",
            input_tokens=50,
            output_tokens=5,
            estimated_cost_usd=0.5,
        )

    agent = SimpleNamespace(
        session_id=session_id,
        _session_db=session_db,
        session_input_tokens=100,
        session_output_tokens=10,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_api_calls=1,
        session_estimated_cost_usd=1.0,
        _user_turn_count=1,
        model="main-model",
        provider="main-provider",
    )
    snapshot = _kanban_session_usage(agent, "kanban_complete", {"summary": "done"})
    response = json.loads(model_tools.handle_function_call(
        "kanban_complete",
        {"summary": "done"},
        session_usage=snapshot,
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    ))
    assert response["ok"] is True

    with kbc.connect_closing() as conn:
        run = kb.latest_run(conn, worker_task)
        usage = task_usage(conn, worker_task)
    session_db.close()

    assert run.input_tokens == expected_input
    assert run.output_tokens == (15 if auxiliary else 10)
    assert run.api_call_count == expected_calls
    assert run.estimated_cost_usd == pytest.approx(expected_cost)
    assert run.auxiliary_estimated_cost_usd == pytest.approx(0.5 if auxiliary else 0.0)
    assert usage["input_tokens"] == expected_input
    assert usage["api_call_count"] == expected_calls
    assert usage["estimated_cost_usd"] == pytest.approx(expected_cost)


@pytest.mark.parametrize("lifecycle", ["kanban_complete", "kanban_block"])
def test_bridged_lifecycle_snapshot_keeps_worker_usage(worker_task, lifecycle, monkeypatch):
    from agent.tool_executor import _kanban_session_usage
    from tools import tool_search
    import model_tools

    monkeypatch.setattr(
        tool_search,
        "is_deferrable_tool_name",
        lambda name, defer_tools=None: name == lifecycle,
    )
    lifecycle_args = (
        {"summary": "done"}
        if lifecycle == "kanban_complete"
        else {"reason": "retry", "kind": "transient"}
    )
    bridge_args = {"name": lifecycle, "arguments": lifecycle_args}
    agent = SimpleNamespace(
        session_id="bridged-worker",
        _session_db=None,
        session_input_tokens=321,
        session_output_tokens=45,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_api_calls=2,
        session_estimated_cost_usd=0.25,
        _user_turn_count=1,
        model="main-model",
        provider="main-provider",
    )

    snapshot = _kanban_session_usage(agent, "tool_call", bridge_args)
    response = json.loads(model_tools.handle_function_call(
        "tool_call",
        bridge_args,
        session_usage=snapshot,
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    ))
    assert response["ok"] is True

    with kbc.connect_closing() as conn:
        run = kb.latest_run(conn, worker_task)
    assert run.session_id == "bridged-worker"
    assert run.input_tokens == 321
    assert run.output_tokens == 45
    assert run.api_call_count == 2


def test_task_usage_combines_main_actual_with_auxiliary_estimate(worker_task):
    usage = _usage("mixed-cost", scale=1, actual_cost_usd=0.8)
    usage["estimated_cost_usd"] = 1.5
    usage["auxiliary_estimated_cost_usd"] = 0.5
    completed = json.loads(kt._handle_complete({"summary": "done"}, session_usage=usage))
    assert completed["ok"] is True

    with kbc.connect_closing() as conn:
        run = kb.latest_run(conn, worker_task)
        aggregate = task_usage(conn, worker_task)

    assert run.estimated_cost_usd == pytest.approx(1.5)
    assert run.actual_cost_usd == pytest.approx(0.8)
    assert run.auxiliary_estimated_cost_usd == pytest.approx(0.5)
    assert aggregate["estimated_cost_usd"] == pytest.approx(1.5)
    assert aggregate["auxiliary_estimated_cost_usd"] == pytest.approx(0.5)
    assert aggregate["cost_usd"] == pytest.approx(1.3)

    shown = json.loads(kt._handle_show({}))
    assert shown["runs"][0]["auxiliary_estimated_cost_usd"] == pytest.approx(0.5)


def test_goal_gate_usage_is_captured_after_final_judge_call(worker_task, monkeypatch):
    from agent.tool_executor import _kanban_session_usage
    from hermes_state import SessionDB
    import model_tools

    session_id = "goal-mode-worker"
    session_db = SessionDB(Path(os.environ["HERMES_HOME"]) / "state.db")
    session_db.create_session(session_id, source="cli", model="main-model")
    session_db.update_token_counts(
        session_id,
        input_tokens=100,
        output_tokens=10,
        model="main-model",
        billing_provider="main-provider",
        estimated_cost_usd=1.0,
        api_call_count=1,
    )
    with kbc.connect_closing() as conn:
        conn.execute("UPDATE tasks SET goal_mode = 1 WHERE id = ?", (worker_task,))
        conn.commit()

    agent = SimpleNamespace(
        session_id=session_id,
        _session_db=session_db,
        session_input_tokens=100,
        session_output_tokens=10,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_api_calls=1,
        session_estimated_cost_usd=1.0,
        _user_turn_count=1,
        model="main-model",
        provider="main-provider",
    )

    def record_final_judge_usage(tool_name, task, tid, evidence):
        assert tool_name == "kanban_complete"
        assert task.goal_mode
        assert tid == worker_task
        assert evidence == "done"
        session_db.record_auxiliary_usage(
            session_id,
            "goal_judge",
            model="judge-model",
            billing_provider="judge-provider",
            input_tokens=25,
            output_tokens=5,
            estimated_cost_usd=0.25,
        )

    monkeypatch.setattr(kt, "_goal_gate", record_final_judge_usage)
    response = json.loads(model_tools.handle_function_call(
        "kanban_complete",
        {"summary": "done"},
        session_usage=lambda: _kanban_session_usage(agent, "kanban_complete"),
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    ))
    assert response["ok"] is True

    with kbc.connect_closing() as conn:
        run = kb.latest_run(conn, worker_task)
    session_db.close()

    assert run.input_tokens == 125
    assert run.output_tokens == 15
    assert run.api_call_count == 2
    assert run.estimated_cost_usd == pytest.approx(1.25)
    assert run.auxiliary_estimated_cost_usd == pytest.approx(0.25)


def test_request_review_preserves_implementation_usage(worker_task):
    from model_tools import handle_function_call

    out = json.loads(handle_function_call(
        "kanban_request_review",
        {"summary": "Implementation verified; ready for review."},
        session_usage=_usage("session-builder", scale=1),
    ))
    assert out["ok"] is True

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, worker_task)
        run = kb.latest_run(conn, worker_task)
        usage = task_usage(conn, worker_task)

    assert task is not None and run is not None
    assert task.status == "review"
    assert run.outcome == "review_requested"
    assert run.ended_at is not None
    assert run.session_id == "session-builder"
    assert usage["input_tokens"] == 100
    assert usage["estimated_cost_usd"] == pytest.approx(0.1)


def test_request_changes_preserves_reviewer_usage(worker_task, monkeypatch):
    from model_tools import handle_function_call

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, worker_task)
        assert task is not None
        assert kb.request_review(
            conn,
            worker_task,
            reviewer="reviewer",
            summary="Ready for review.",
            expected_run_id=task.current_run_id,
        )
        claimed = kb.claim_review_task(conn, worker_task)
    assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))

    out = json.loads(handle_function_call(
        "kanban_request_changes",
        {"reason": "Add coverage for the retry path."},
        session_usage=_usage("session-reviewer", scale=2, actual_cost_usd=0.15),
    ))
    assert out["ok"] is True

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, worker_task)
        run = kb.latest_run(conn, worker_task)
        usage = task_usage(conn, worker_task)

    assert task is not None and run is not None
    assert task.status == "ready"
    assert task.assignee == "builder"
    assert run.outcome == "changes_requested"
    assert run.ended_at is not None
    assert run.session_id == "session-reviewer"
    assert usage["input_tokens"] == 200
    assert usage["estimated_cost_usd"] == pytest.approx(0.2)
    assert usage["cost_usd"] == pytest.approx(0.15)
    assert [(row["profile"], row["input_tokens"]) for row in usage["profiles"]] == [
        ("builder", 0),
        ("reviewer", 200),
    ]


@pytest.mark.parametrize("tool_name", ["kanban_request_review", "kanban_request_changes"])
def test_review_terminals_collect_worker_usage(worker_task, monkeypatch, tool_name):
    from agent.tool_executor import _kanban_session_usage

    agent = SimpleNamespace(session_id="session-builder", session_input_tokens=100)
    usage = _kanban_session_usage(agent, tool_name)
    assert usage is not None
    assert usage["session_id"] == "session-builder"
    assert usage["input_tokens"] == 100
    assert _kanban_session_usage(agent, "kanban_heartbeat") is None
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert _kanban_session_usage(agent, tool_name) is None
