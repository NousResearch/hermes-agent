"""Regression tests for Kanban finalization on Codex app-server failures."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.codex_runtime import run_codex_app_server_turn
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _timeout_turn():
    return SimpleNamespace(
        interrupted=True,
        error="turn timed out after 600.0s",
        thread_id="thread-timeout",
        turn_id="turn-timeout",
        projected_messages=[],
        tool_iterations=7,
        final_text="",
        should_retire=True,
        token_usage_last=None,
    )


def _agent_for_turn(turn):
    agent = MagicMock()
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.return_value = turn
    agent._codex_session.close = MagicMock()
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = None
    agent._session_db_created = True
    agent.session_id = "sess-kanban-timeout"
    agent.session_api_calls = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.session_estimated_cost_usd = 0.0
    agent.model = "codex"
    agent.provider = "openai"
    agent.base_url = ""
    return agent


def test_codex_app_server_timeout_records_kanban_failure(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="long codex task", assignee="coder")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    result = run_codex_app_server_turn(
        _agent_for_turn(_timeout_turn()),
        user_message="run a long task",
        original_user_message="run a long task",
        messages=[{"role": "user", "content": "run a long task"}],
        effective_task_id=task_id,
    )

    assert result["completed"] is False
    assert result["partial"] is True
    assert result["error"] == "turn timed out after 600.0s"

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert task.status == "ready"
    assert task.current_run_id is None
    assert task.consecutive_failures == 1
    assert run.status == "timed_out"
    assert run.outcome == "timed_out"
    assert run.error == "turn timed out after 600.0s"
    assert events[-1].kind == "timed_out"
    assert events[-1].payload["error"] == "turn timed out after 600.0s"


def test_codex_app_server_timeout_finalizer_noops_after_completion(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="already done", assignee="coder")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None
        assert kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    result = run_codex_app_server_turn(
        _agent_for_turn(_timeout_turn()),
        user_message="finish",
        original_user_message="finish",
        messages=[{"role": "user", "content": "finish"}],
        effective_task_id=task_id,
    )
    assert result["partial"] is True

    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "done"
        assert kb.latest_run(conn, task_id).outcome == "completed"
