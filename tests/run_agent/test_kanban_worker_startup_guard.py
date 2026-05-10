from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_cli import kanban_db as kb
from run_agent import AIAgent


def test_run_conversation_skips_stale_kanban_worker_before_api(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="stale worker", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = claimed.current_run_id
        lock = claimed.claim_lock
        assert run_id is not None and lock is not None
        assert kb.reclaim_task(conn, tid, reason="operator reclaim")
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", lock)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path()))

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_logging.setup_logging"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="cli",
        )
    agent.client = MagicMock()

    result = agent.run_conversation("work kanban task")

    assert result["completed"] is False
    assert result["api_calls"] == 0
    assert result["interrupted"] is False
    assert result["partial"] is False
    assert result["turn_exit_reason"] == "kanban_startup_guard:task_not_running:ready"
    assert "no longer belongs to this worker" in result["final_response"]
    assert not agent.client.chat.completions.create.called
