"""Behavior contracts for the canonical human + agent Hybrid Kanban."""

from __future__ import annotations

import json

import pytest

from hermes_cli import hybrid_kanban as hybrid
from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path):
    connection = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        yield connection
    finally:
        connection.close()


def test_hybrid_board_persists_order_activity_and_agentic_boundary(conn, tmp_path):
    board = hybrid.create_board(conn, name="Marketing", actor_type="human", actor_id="kevyn")
    ideas = hybrid.create_column(conn, board_id=board["id"], name="Ideas", actor_type="human", actor_id="kevyn")
    doing = hybrid.create_column(conn, board_id=board["id"], name="Doing", actor_type="human", actor_id="kevyn")
    done = hybrid.create_column(conn, board_id=board["id"], name="Done", actor_type="human", actor_id="kevyn")
    campaign = hybrid.create_card(conn, board_id=board["id"], column_id=ideas["id"], title="Campaign X", description="**Draft**", actor_type="human", actor_id="kevyn")
    second = hybrid.create_card(conn, board_id=board["id"], column_id=ideas["id"], title="Campaign Y", actor_type="agent", actor_id="hermes", session_id="s-1")

    # Semantic intent, not a client-owned rank: second moves before campaign.
    hybrid.move_card(conn, card_id=second["id"], target_column_id=ideas["id"], before_id=campaign["id"], actor_type="agent", actor_id="hermes", session_id="s-1")
    hybrid.move_card(conn, card_id=campaign["id"], target_column_id=done["id"], actor_type="human", actor_id="kevyn")
    edited = hybrid.update_card(conn, card_id=campaign["id"], description="**Approved**", expected_revision=campaign["revision"] + 1, actor_type="agent", actor_id="hermes", session_id="s-1")
    assert edited["description"] == "**Approved**"
    assert edited["column_id"] == done["id"]

    # A Hybrid Done column has no agentic lifecycle meaning.
    task_id = kb.create_task(conn, title="Agentic task", assignee="worker", initial_status="running")
    agentic_status = kb.get_task(conn, task_id).status
    assert agentic_status != "done"

    snapshot = hybrid.get_board(conn, board["id"])
    assert [column["name"] for column in snapshot["columns"]] == ["Ideas", "Doing", "Done"]
    assert [card["title"] for card in snapshot["columns"][0]["cards"]] == ["Campaign Y"]
    assert [card["position"] for card in snapshot["columns"][0]["cards"]] == [0]
    activity = hybrid.get_card(conn, campaign["id"])["activity"]
    assert {entry["actor_type"] for entry in activity} >= {"human", "agent"}

    # Re-open through a fresh connection: canonical persistence survives restart.
    path = tmp_path / "kanban.db"
    conn.close()
    reopened = kb.connect(db_path=path)
    try:
        assert hybrid.get_card(reopened, campaign["id"])["description"] == "**Approved**"
        assert kb.get_task(reopened, task_id).status == agentic_status
    finally:
        reopened.close()


def test_hybrid_moves_reject_stale_revision_and_invalid_destination(conn):
    board = hybrid.create_board(conn, name="Board")
    first = hybrid.create_column(conn, board_id=board["id"], name="First")
    second = hybrid.create_column(conn, board_id=board["id"], name="Second")
    card = hybrid.create_card(conn, board_id=board["id"], column_id=first["id"], title="One")
    hybrid.update_card(conn, card_id=card["id"], title="One updated")

    with pytest.raises(hybrid.HybridKanbanConflict):
        hybrid.move_card(conn, card_id=card["id"], target_column_id=second["id"], expected_revision=card["revision"])
    with pytest.raises(hybrid.HybridKanbanError):
        hybrid.move_card(conn, card_id=card["id"], target_column_id="missing")


def test_hybrid_column_reorder_repairs_dense_positions(conn):
    board = hybrid.create_board(conn, name="Board")
    first = hybrid.create_column(conn, board_id=board["id"], name="First")
    second = hybrid.create_column(conn, board_id=board["id"], name="Second")
    third = hybrid.create_column(conn, board_id=board["id"], name="Third")
    hybrid.move_column(conn, column_id=third["id"], before_id=first["id"])
    ordered = hybrid.get_board(conn, board["id"])["columns"]
    assert [column["id"] for column in ordered] == [third["id"], first["id"], second["id"]]
    assert [column["position"] for column in ordered] == [0, 1, 2]


def test_agent_tool_uses_the_same_hybrid_domain(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "agent-kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "planner")
    monkeypatch.setenv("HERMES_SESSION_ID", "desktop-session")
    from tools.kanban_tools import _handle_hybrid

    created = json.loads(_handle_hybrid({"action": "create_board", "name": "Marketing"}))
    assert created["ok"] is True
    board_id = created["board"]["id"]
    column = json.loads(_handle_hybrid({"action": "create_column", "board_id": board_id, "name": "Ideas"}))
    card = json.loads(_handle_hybrid({"action": "create_card", "board_id": board_id, "column_id": column["column"]["id"], "title": "Agent card"}))
    assert card["card"]["title"] == "Agent card"


def test_hybrid_deletion_lifecycle_and_activity_audit(conn):
    board = hybrid.create_board(conn, name="Lifecycle Board", actor_type="human", actor_id="user1")
    col1 = hybrid.create_column(conn, board_id=board["id"], name="Col 1")
    col2 = hybrid.create_column(conn, board_id=board["id"], name="Col 2")
    col3 = hybrid.create_column(conn, board_id=board["id"], name="Col 3")

    c1 = hybrid.create_card(conn, board_id=board["id"], column_id=col1["id"], title="Card 1")
    c2 = hybrid.create_card(conn, board_id=board["id"], column_id=col1["id"], title="Card 2")
    c3 = hybrid.create_card(conn, board_id=board["id"], column_id=col1["id"], title="Card 3")

    # Delete middle card; remaining positions in col1 should be repaired to 0, 1
    assert hybrid.delete_card(conn, card_id=c2["id"], actor_type="human", actor_id="user1") is True
    with pytest.raises(hybrid.HybridKanbanError):
        hybrid.get_card(conn, c2["id"])

    updated_col1_cards = hybrid.get_board(conn, board["id"])["columns"][0]["cards"]
    assert [c["id"] for c in updated_col1_cards] == [c1["id"], c3["id"]]
    assert [c["position"] for c in updated_col1_cards] == [0, 1]

    # Delete column 2; remaining columns should be col1, col3 with positions 0, 1
    assert hybrid.delete_column(conn, column_id=col2["id"], actor_type="human") is True
    with pytest.raises(hybrid.HybridKanbanError):
        hybrid._require_column(conn, col2["id"])
    board_after_col_del = hybrid.get_board(conn, board["id"])
    assert [col["id"] for col in board_after_col_del["columns"]] == [col1["id"], col3["id"]]
    assert [col["position"] for col in board_after_col_del["columns"]] == [0, 1]

    # Activity audit log retrieval
    activities = hybrid.get_board_activity(conn, board["id"], limit=50)
    assert len(activities) > 0
    kinds = [a["kind"] for a in activities]
    assert "board_created" in kinds
    assert "card_deleted" in kinds
    assert "column_deleted" in kinds

    # Delete board completely
    assert hybrid.delete_board(conn, board_id=board["id"]) is True
    with pytest.raises(hybrid.HybridKanbanError):
        hybrid.get_board(conn, board["id"])

