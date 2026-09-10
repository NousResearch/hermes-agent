from __future__ import annotations

import json
from pathlib import Path

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
