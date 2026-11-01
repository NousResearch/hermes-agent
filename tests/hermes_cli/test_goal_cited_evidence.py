"""Cited goal evidence travels through real SessionDB and the judge prompt."""
import json
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def goal_db(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals
    goals._DB_CACHE.clear()
    db = goals._get_session_db()
    db.create_session("goal-citations", source="cli")
    yield db, goals
    goals._DB_CACHE.clear()


def test_recorded_citations_reach_judge_across_earlier_rows(goal_db):
    db, goals = goal_db
    manager = goals.GoalManager("goal-citations")
    state = manager.set("Ship the repair")
    at = state.created_at + 1
    recorded_id = db.append_message("goal-citations", "tool", "8 passed; commit abc123def456", tool_name="terminal", timestamp=at)
    # Compaction archives the original row but it remains recorded evidence.
    db._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET active = 0, compacted = 1 WHERE id = ?", (recorded_id,)))
    for i in range(130):
        db.append_message("goal-citations", "assistant", f"unrelated progress {i}", timestamp=at)
    prompts = []

    def call_llm(**kwargs):
        prompts.append(kwargs["messages"][1]["content"])
        return MagicMock(choices=[MagicMock(message=MagicMock(content='{"verdict":"done","reason":"verified"}'))])

    with patch("agent.auxiliary_client.call_llm", side_effect=call_llm):
        decision = manager.evaluate_after_turn("Evidence: `abc123def456`, 8 passed")
    assert decision["verdict"] == "done"
    assert "terminal result #" in prompts[0]
    assert "8 passed; commit abc123def456" in prompts[0]


def test_command_citation_resolves_to_result_and_fabricated_claim_does_not(goal_db):
    db, goals = goal_db
    manager = goals.GoalManager("goal-citations")
    state = manager.set("Run tests")
    at = state.created_at + 1
    command = "pytest tests/goal_check.py"
    db.append_message("goal-citations", "assistant", tool_calls=[{
        "id": "call-a", "type": "function", "function": {"name": "terminal", "arguments": json.dumps({"command": command})},
    }], timestamp=at)
    db.append_message("goal-citations", "tool", "2 failed", tool_name="terminal", tool_call_id="call-a", timestamp=at)
    db.append_message("goal-citations", "assistant", "FAKE_RESULT_987654321 50 passed", timestamp=at)
    db.append_message("goal-citations", "tool", "50 passed", tool_name="todo_list", timestamp=at)
    text = f"`{command}` ran, 50 passed, `FAKE_RESULT_987654321`"
    block = goals._cited_evidence_block("goal-citations", text, state.created_at)
    assert "2 failed" in block
    assert "Citations not found" in block
    assert "50 passed" in block.split("Citations not found")[1]
    assert "FAKE_RESULT_987654321" in block.split("Citations not found")[1]


def test_old_or_unrelated_results_are_not_evidence(goal_db):
    db, goals = goal_db
    db.append_message("goal-citations", "tool", "old-sha-abcdef123456", tool_name="terminal", timestamp=time.time() - 100)
    manager = goals.GoalManager("goal-citations")
    state = manager.set("Check new evidence")
    db.append_message("goal-citations", "tool", "new-sha-fedcba654321", tool_name="terminal", timestamp=state.created_at + 1)
    block = goals._cited_evidence_block("goal-citations", "`old-sha-abcdef123456` and `new-sha-fedcba654321`", state.created_at)
    assert "new-sha-fedcba654321" in block.split("Citations not found")[0]
    assert "old-sha-abcdef123456" in block.split("Citations not found")[1]
