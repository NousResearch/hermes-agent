"""Behavior contracts for the canonical human + agent Hybrid Kanban."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_hybrid_checklists_persist_reorder_and_enforce_revisions(conn, tmp_path):
    board = hybrid.create_board(conn, name="Delivery")
    column = hybrid.create_column(conn, board_id=board["id"], name="Doing")
    card = hybrid.create_card(conn, board_id=board["id"], column_id=column["id"], title="Ship")
    acceptance = hybrid.create_checklist(conn, card_id=card["id"], title="Acceptance", actor_id="kevyn")
    release = hybrid.create_checklist(conn, card_id=card["id"], title="Release")
    first = hybrid.add_checklist_item(conn, checklist_id=acceptance["id"], body="Run tests")
    second = hybrid.add_checklist_item(conn, checklist_id=acceptance["id"], body="Publish notes")

    completed = hybrid.update_checklist_item(
        conn, item_id=first["id"], completed=True, expected_revision=first["revision"]
    )
    assert completed["completed"] == 1
    with pytest.raises(hybrid.HybridKanbanConflict):
        hybrid.update_checklist_item(
            conn, item_id=first["id"], completed=False, expected_revision=first["revision"]
        )

    moved = hybrid.move_checklist_item(
        conn, item_id=second["id"], before_id=first["id"], expected_revision=second["revision"]
    )
    assert moved["position"] == 0
    detail = hybrid.get_card(conn, card["id"])
    assert [item["title"] for item in detail["checklists"]] == ["Acceptance", "Release"]
    assert [item["body"] for item in detail["checklists"][0]["items"]] == ["Publish notes", "Run tests"]
    assert any(event["kind"] == "checklist_item_updated" for event in detail["activity"])

    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    reopened = kb.connect(db_path=Path(db_path))
    try:
        assert hybrid.get_card(reopened, card["id"])["checklists"][0]["items"][1]["completed"] == 1
    finally:
        reopened.close()

    assert hybrid.delete_checklist(conn, checklist_id=release["id"], expected_revision=release["revision"])
    assert hybrid.delete_card(conn, card_id=card["id"])
    assert conn.execute("SELECT COUNT(*) FROM hybrid_checklists WHERE card_id = ?", (card["id"],)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM hybrid_checklist_items").fetchone()[0] == 0


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


def _delegation_fixture(conn):
    board = hybrid.create_board(conn, name="Delegation Board")
    column = hybrid.create_column(conn, board_id=board["id"], name="Inbox")
    card = hybrid.create_card(
        conn,
        board_id=board["id"],
        column_id=column["id"],
        title="Prepare launch brief",
        description="Collect the source facts and draft the brief.",
        metadata={"project": "launch", "priority_note": "customer-facing"},
    )
    return board, column, card


def test_human_card_delegation_is_durable_idempotent_and_contextual(conn, tmp_path):
    board, column, card = _delegation_fixture(conn)

    delegated = hybrid.delegate_card(
        conn,
        card_id=card["id"],
        actor_id="kevyn",
        session_id="human-session",
        assignee="worker",
    )
    link = delegated["delegation"]
    assert link["state"] == "queued"
    task_id = link["agent_task_id"]
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.title == card["title"]
    assert card["description"] in task.body
    assert card["id"] in task.body
    assert "customer-facing" in task.body

    # A duplicate click returns the same attempt, including after a fresh
    # connection has reconstructed the canonical stores.
    duplicate = hybrid.delegate_card(conn, card_id=card["id"], actor_id="kevyn", session_id="human-session")
    assert duplicate["delegation"]["agent_task_id"] == task_id
    path = tmp_path / "kanban.db"
    conn.close()
    reopened = kb.connect(db_path=path)
    try:
        restored = hybrid.delegate_card(reopened, card_id=card["id"], actor_id="kevyn", session_id="human-session")
        assert restored["delegation"]["agent_task_id"] == task_id
        assert len(restored["delegations"]) == 1
    finally:
        reopened.close()


def test_agent_completion_projects_result_ref_and_evidence_without_payload_copy(conn):
    _, _, card = _delegation_fixture(conn)
    delegated = hybrid.delegate_card(conn, card_id=card["id"], session_id="s-1")
    task_id = delegated["delegation"]["agent_task_id"]
    from workstation.contracts import TaskOutcome, OutcomeStatus, AcceptanceContract, EvidenceRef
    from dataclasses import asdict
    outcome = TaskOutcome(task_id, "s-1", "Launch brief", OutcomeStatus.VERIFIED_COMPLETED,
                          "Launch brief is ready.", evidence_refs=[EvidenceRef("verification", "sha256:evidence-1")],
                          verifier_results=[{"verifier": "brief-review", "passed": True, "evidence_ref": "sha256:evidence-1"}])
    from agent.verification_evidence import record_outcome_verifiers
    verifier_ids = record_outcome_verifiers(task_id, "s-1", outcome.verifier_results, environment="test")
    assert kb.complete_task(
        conn,
        task_id,
        result="A very large canonical result remains owned by the task.",
        summary="Launch brief is ready.",
        metadata={"result_ref": "sha256:result-1", "evidence_refs": ["sha256:evidence-1"],
                  "workstation": {"outcome": outcome.to_dict(), "acceptance_approved": True,
                                  "verification_event_ids": verifier_ids,
                                  "acceptance_contract": asdict(AcceptanceContract())}},
    ) is True

    projected = hybrid.sync_card_delegations(conn, card_id=card["id"])
    assert projected["delegation"]["state"] == "completed"
    assert projected["delegation"]["summary"] == "Launch brief is ready."
    assert projected["delegation"]["result_ref"] == "sha256:result-1"
    assert projected["delegation"]["evidence_refs"] == ["sha256:evidence-1"]
    assert projected["delegation"]["agent_task_id"] == task_id
    assert projected["delegation"]["summary"] != "A very large canonical result remains owned by the task."
    assert any(item["kind"] == "delegation_completed" for item in projected["activity"])


def test_human_and_agent_lifecycles_move_independently_and_retry_same_task(conn):
    _, column, card = _delegation_fixture(conn)
    other = hybrid.create_column(conn, board_id=card["board_id"], name="Doing")
    delegated = hybrid.delegate_card(conn, card_id=card["id"], session_id="s-1")
    task_id = delegated["delegation"]["agent_task_id"]

    assert kb.block_task(conn, task_id, reason="needs human input", kind="needs_input") is True
    waiting = hybrid.get_card(conn, card["id"])
    assert waiting["delegation"]["state"] == "waiting"
    moved = hybrid.move_card(conn, card_id=card["id"], target_column_id=other["id"], expected_revision=card["revision"])
    assert moved["column_id"] == other["id"]
    assert kb.get_task(conn, task_id).status == "blocked"

    retried = hybrid.retry_card_delegation(conn, card_id=card["id"], session_id="s-1")
    assert retried["delegation"]["agent_task_id"] == task_id
    assert retried["delegation"]["state"] == "queued"
    assert hybrid.get_card(conn, card["id"])["column_id"] == other["id"]


def test_canonical_task_event_projection_emits_durable_hybrid_progress(conn):
    _, _, card = _delegation_fixture(conn)
    delegated = hybrid.delegate_card(conn, card_id=card["id"], session_id="s-1")
    task_id = delegated["delegation"]["agent_task_id"]

    assert hybrid.sync_delegations_for_agent_task(conn, agent_task_id=task_id) is True
    assert kb.block_task(conn, task_id, reason="awaiting approval", kind="needs_input") is True
    assert hybrid.sync_delegations_for_agent_task(conn, agent_task_id=task_id) is True

    projected = hybrid.get_card(conn, card["id"])
    assert projected["delegation"]["state"] == "waiting"
    progress = [item for item in projected["activity"] if item["kind"] == "delegation_progressed"]
    waiting_transition = next(item for item in progress if item["payload"]["state"] == "waiting")
    assert waiting_transition["payload"]["agent_task_id"] == task_id
    assert waiting_transition["payload"]["previous_state"] == "queued"


def test_failed_cancelled_and_redelegated_attempts_are_explicit(conn):
    _, _, failed_card = _delegation_fixture(conn)
    failed = hybrid.delegate_card(conn, card_id=failed_card["id"], session_id="s-1")
    failed_id = failed["delegation"]["agent_task_id"]
    failed_result = hybrid.fail_card_delegation(conn, card_id=failed_card["id"], summary="Browser capability was unavailable.")
    assert failed_result["delegation"]["state"] == "failed"
    assert failed_result["delegation"]["agent_task_id"] == failed_id
    assert kb.get_task(conn, failed_id).status == "archived"

    redelegated = hybrid.delegate_card(conn, card_id=failed_card["id"], new_attempt=True, session_id="s-2")
    assert redelegated["delegation"]["attempt"] == 2
    assert redelegated["delegation"]["agent_task_id"] != failed_id
    assert len(redelegated["delegations"]) == 2

    _, _, cancelled_card = _delegation_fixture(conn)
    cancelled = hybrid.delegate_card(conn, card_id=cancelled_card["id"], session_id="s-3")
    cancelled_result = hybrid.cancel_card_delegation(conn, card_id=cancelled_card["id"], session_id="s-3")
    assert cancelled_result["delegation"]["state"] == "cancelled"
    assert kb.get_task(conn, cancelled["delegation"]["agent_task_id"]).status == "archived"
    assert any(item["kind"] == "delegation_cancelled" for item in cancelled_result["activity"])


def test_hybrid_archive_restore_is_normal_and_hard_delete_remains_explicit(conn):
    board, column, card = _delegation_fixture(conn)
    second = hybrid.create_column(conn, board_id=board["id"], name="Second")
    second_card = hybrid.create_card(conn, board_id=board["id"], column_id=second["id"], title="Second card")

    assert hybrid.archive_card(conn, card_id=card["id"]) is True
    assert hybrid.get_board(conn, board["id"])["columns"][0]["cards"] == []
    archived_card = hybrid.get_card(conn, card["id"], include_archived=True)
    assert archived_card["archived"] == 1
    assert hybrid.restore_card(conn, card_id=card["id"]) is True
    assert hybrid.get_card(conn, card["id"])["archived"] == 0

    assert hybrid.archive_column(conn, column_id=column["id"]) is True
    assert [item["id"] for item in hybrid.get_board(conn, board["id"])["columns"]] == [second["id"]]
    assert hybrid.restore_column(conn, column_id=column["id"]) is True
    assert [item["id"] for item in hybrid.get_board(conn, board["id"])["columns"]] == [second["id"], column["id"]]

    assert hybrid.archive_board(conn, board_id=board["id"]) is True
    assert hybrid.list_boards(conn) == []
    assert hybrid.restore_board(conn, board_id=board["id"]) is True
    assert hybrid.list_boards(conn)[0]["id"] == board["id"]

    # Hard deletion is still available as a separate, irreversible operation.
    assert hybrid.delete_card(conn, card_id=second_card["id"]) is True
    with pytest.raises(hybrid.HybridKanbanError):
        hybrid.get_card(conn, second_card["id"])
