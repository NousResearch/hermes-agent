"""End-to-end test: write -> compress -> resume cycle."""
import json
import pytest
import yaml
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from agent.checkpoint_store import CheckpointStore
from agent.checkpoint_injection import build_checkpoint_system_prompt
from tools.checkpoint_tool import checkpoint_tool


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")


def test_full_cycle_write_compress_resume(store):
    """Simulate: agent writes checkpoint -> new session gets injection."""
    old_agent = MagicMock()
    old_agent.session_id = "20260422_225800_abc123"
    old_agent._todo_store = MagicMock()
    old_agent._todo_store.format_for_injection.return_value = ""

    checkpoint_tool(
        action="write",
        task="Red team QA on PR #14303",
        progress=[
            {"step": "Read code", "status": "completed"},
            {"step": "Write tests", "status": "in_progress"},
        ],
        state={
            "active_branch": "fix/delegation",
            "tests_status": "19/19 passing",
            "last_commit": "a90557d",
            "pushed": True,
            "working_directory": "/tmp/repo",
        },
        decisions=["SSRF is low-severity, doc note only"],
        blocked=[],
        unresolved=[],
        store=store,
        agent=old_agent,
    )

    # New session starts, checkpoint gets injected
    prompt = build_checkpoint_system_prompt(
        store=store,
        parent_session_id="20260422_225800_abc123",
    )

    assert "Red team QA on PR #14303" in prompt
    assert "Write tests" in prompt
    assert "fix/delegation" in prompt
    assert "SSRF is low-severity" in prompt


def test_gc_removes_stale_checkpoints(store):
    """GC removes checkpoints older than 7 days."""
    mock_agent = MagicMock()
    mock_agent.session_id = "old_sess"
    mock_agent._todo_store = MagicMock()
    mock_agent._todo_store.format_for_injection.return_value = ""

    checkpoint_tool(
        action="write",
        task="Old task",
        progress=[],
        state={},
        decisions=[],
        store=store,
        agent=mock_agent,
    )
    # Age the checkpoint by writing directly with an old timestamp
    data = store.read("old_sess")
    data["updated"] = (datetime.now() - timedelta(days=10)).isoformat()
    store._path_for("old_sess").write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    removed = store.garbage_collect(max_age_days=7)
    assert "old_sess" in removed