"""Behavior tests for the per-board Kanban single-writer boundary."""

from __future__ import annotations

import concurrent.futures
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_safety as safety
from hermes_cli import kanban_writer as writer


pytestmark = pytest.mark.real_kanban_writer


@pytest.fixture
def board(tmp_path: Path) -> Path:
    path = tmp_path / "kanban.db"
    with writer.privileged_maintenance(path):
        kb.init_db(path)
    return path


@pytest.fixture
def service(board: Path):
    instance = writer.KanbanWriterService(board)
    instance.start()
    try:
        yield instance
    finally:
        instance.stop()


def test_non_owner_write_transaction_is_rejected(
    service: writer.KanbanWriterService,
) -> None:
    with kb.connect(service.db_path) as conn:
        with pytest.raises(writer.WriterOwnershipError, match="single-writer"):
            with kb.write_txn(conn):
                conn.execute("DELETE FROM tasks")


def test_non_owner_connection_is_query_only(
    service: writer.KanbanWriterService,
) -> None:
    with kb.connect(service.db_path) as conn:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            conn.execute("DELETE FROM tasks")


def test_public_mutation_routes_through_writer(
    service: writer.KanbanWriterService,
) -> None:
    with kb.connect(service.db_path) as conn:
        task_id = kb.create_task(conn, title="routed public API")
        assert kb.get_task(conn, task_id).title == "routed public API"


def test_public_mutation_fails_closed_after_service_death(
    service: writer.KanbanWriterService,
) -> None:
    service.stop()
    with kb.connect(service.db_path) as conn:
        with pytest.raises(writer.WriterUnavailableError):
            kb.create_task(conn, title="no fallback")


def test_multiple_clients_are_serialized(service: writer.KanbanWriterService) -> None:
    # This verifies serialization correctness, not a two-second latency SLA.
    # On constrained CI hosts the last of twelve queued SQLite mutations can
    # legitimately wait longer than two seconds behind the earlier clients.
    clients = [writer.KanbanWriterClient(service.db_path, timeout_seconds=10) for _ in range(12)]

    def create(index: int) -> str:
        return clients[index].mutate(
            "create_task",
            {"title": f"task-{index}"},
            request_id=f"create-{index}",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as pool:
        task_ids = list(pool.map(create, range(len(clients))))

    assert len(set(task_ids)) == len(clients)
    with kb.connect(service.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == len(clients)


def test_duplicate_request_replays_committed_response(service: writer.KanbanWriterService) -> None:
    client = writer.KanbanWriterClient(service.db_path, timeout_seconds=2)
    first = client.mutate("create_task", {"title": "once"}, request_id="same-request")
    second = client.mutate("create_task", {"title": "once"}, request_id="same-request")

    assert second == first
    with kb.connect(service.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_service_death_fails_closed(service: writer.KanbanWriterService) -> None:
    client = writer.KanbanWriterClient(service.db_path, timeout_seconds=0.05)
    service.stop()

    with pytest.raises(writer.WriterUnavailableError):
        client.mutate("create_task", {"title": "must not fallback"}, request_id="dead")

    with kb.connect(service.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_bad_authentication_is_rejected(service: writer.KanbanWriterService) -> None:
    client = writer.KanbanWriterClient(
        service.db_path,
        timeout_seconds=1,
        authentication_token="not-the-board-token",
    )
    with pytest.raises(writer.WriterAuthenticationError):
        client.mutate("create_task", {"title": "denied"}, request_id="bad-auth")


def test_quarantine_blocks_writer_mutation(service: writer.KanbanWriterService) -> None:
    safety.quarantine_board(service.db_path, reason="test fence", source="pytest")
    client = writer.KanbanWriterClient(service.db_path, timeout_seconds=1)

    with pytest.raises(safety.BoardQuarantinedError):
        client.mutate("create_task", {"title": "denied"}, request_id="quarantine")


def test_generation_mismatch_blocks_writer_mutation(service: writer.KanbanWriterService) -> None:
    client = writer.KanbanWriterClient(service.db_path, timeout_seconds=1)
    generations = safety.read_generations(service.db_path)
    safety.bump_board_generation(service.db_path)

    with pytest.raises(safety.GenerationFencedError):
        client.mutate(
            "create_task",
            {"title": "stale"},
            request_id="stale-generation",
            expected_service_generation=generations.service_generation,
            expected_board_generation=generations.board_generation,
        )


def test_runtime_mutation_registry_covers_public_write_transactions() -> None:
    expected = {
        "create_task", "assign_task", "assign_default_assignee", "link_tasks", "unlink_tasks", "add_comment",
        "add_attachment", "delete_attachment", "recompute_ready", "claim_task",
        "claim_review_task", "heartbeat_claim",
        "complete_task", "edit_completed_task_result", "block_task", "promote_task",
        "unblock_task", "specify_triage_task", "decompose_triage_task", "archive_task",
        "delete_archived_task", "delete_task", "edit_task_fields", "set_task_status_direct",
        "set_workspace_path", "set_branch_name",
        "schedule_task", "heartbeat_worker", "detect_crashed_workers",
        "_extend_stale_claim", "_defer_reclaim_for_live_worker", "_finalize_stale_reclaim",
        "_finalize_manual_reclaim", "_finalize_max_runtime", "_finalize_stale_running",
        "_record_spawn_failure", "_set_worker_pid",
        "record_respawn_guarded", "emit_scratch_tip_event", "add_notify_sub", "remove_notify_sub",
        "claim_unseen_events_for_sub", "advance_notify_cursor", "rewind_notify_cursor",
        "gc_events",
    }
    assert writer.RUNTIME_MUTATIONS == expected
