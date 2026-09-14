"""Protected contract tests for Kanban claim rollback and terminal handoff.

These tests use only temporary files and synthetic SQLite databases.  The
module under test is intentionally inactive and has no dispatcher imports.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli.kanban_launch_protocol import (
    LaunchProtocolError,
    acquire_dispatcher_owner,
    canonical_board_identity,
    claim_not_spawned,
    freeze_ack_blockers,
    get_launch_record,
    install_launch_schema,
    record_run_outcome,
    record_spawn_intent,
    record_spawned,
    release_claim,
)

BOARD_UUID = "11111111-1111-4111-8111-111111111111"
TASK_ID = "t_0123456789abcdef"
CLAIM_TOKEN = "claim_token_0123456789abcdef"
OWNER_GENERATION = 7
POLICY_GENERATION = 12
ROUTE_GENERATION = 3


def _connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None)
    install_launch_schema(conn)
    conn.execute(
        "INSERT INTO kanban_policy_pointer "
        "(singleton, schema_version, active_generation) VALUES (1, 1, ?)",
        (POLICY_GENERATION,),
    )
    return conn


def _claim(conn: sqlite3.Connection, board) -> None:
    claim_not_spawned(
        conn,
        board=board,
        task_id=TASK_ID,
        run_generation=1,
        dispatcher_owner_generation=OWNER_GENERATION,
        policy_generation=POLICY_GENERATION,
        route_generation=ROUTE_GENERATION,
        claim_token=CLAIM_TOKEN,
    )


def _board(path: Path):
    return canonical_board_identity(path, BOARD_UUID)


def test_release_claim_removes_pre_spawn_row_under_valid_lease(tmp_path):
    home = tmp_path / "home"
    run_dir = home / "run"
    run_dir.mkdir(parents=True, mode=0o700)
    db = tmp_path / "board.db"
    db.touch()
    board = _board(db)
    with acquire_dispatcher_owner(home, OWNER_GENERATION) as lease:
        conn = _connection(db)
        _claim(conn, board)
        record_spawn_intent(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            owner_lease=lease,
        )
        outcome = release_claim(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            owner_lease=lease,
        )
        assert outcome.previous_state == "spawn_intent"
        assert get_launch_record(conn, board, TASK_ID, 1) is None
        assert freeze_ack_blockers(conn) == []


def test_release_claim_rejects_spawned_rows(tmp_path):
    home = tmp_path / "home"
    (home / "run").mkdir(parents=True, mode=0o700)
    db = tmp_path / "board.db"
    db.touch()
    board = _board(db)
    with acquire_dispatcher_owner(home, OWNER_GENERATION) as lease:
        conn = _connection(db)
        _claim(conn, board)
        record_spawn_intent(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            owner_lease=lease,
        )
        record_spawned(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            pid=4242,
        )
        with pytest.raises(LaunchProtocolError):
            release_claim(
                conn,
                board=board,
                task_id=TASK_ID,
                run_generation=1,
                dispatcher_owner_generation=OWNER_GENERATION,
                policy_generation=POLICY_GENERATION,
                route_generation=ROUTE_GENERATION,
                claim_token=CLAIM_TOKEN,
                owner_lease=lease,
            )
        # The spawned row survives a rejected release.
        assert get_launch_record(conn, board, TASK_ID, 1) is not None


def test_release_claim_fails_closed_on_stale_identity_or_lost_lease(tmp_path):
    home = tmp_path / "home"
    (home / "run").mkdir(parents=True, mode=0o700)
    db = tmp_path / "board.db"
    db.touch()
    board = _board(db)
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    conn = _connection(db)
    _claim(conn, board)
    with lease:
        for overrides in (
            {"claim_token": "stale_token_0123456789"},
            {"policy_generation": POLICY_GENERATION + 1},
            {"dispatcher_owner_generation": OWNER_GENERATION + 1},
        ):
            values: dict[str, object] = {
                "board": board,
                "task_id": TASK_ID,
                "run_generation": 1,
                "dispatcher_owner_generation": OWNER_GENERATION,
                "policy_generation": POLICY_GENERATION,
                "route_generation": ROUTE_GENERATION,
                "claim_token": CLAIM_TOKEN,
                "owner_lease": lease,
            }
            values.update(overrides)
            with pytest.raises(LaunchProtocolError):
                release_claim(conn, **values)
            with pytest.raises(LaunchProtocolError):
                release_claim(conn, **values)
        assert get_launch_record(conn, board, TASK_ID, 1) is not None
    # Lease released: release must now fail closed and leave the row.
    with pytest.raises(LaunchProtocolError):
        release_claim(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            owner_lease=lease,
        )
    assert get_launch_record(conn, board, TASK_ID, 1) is not None


def test_release_claim_rolls_back_on_policy_flip(tmp_path):
    home = tmp_path / "home"
    (home / "run").mkdir(parents=True, mode=0o700)
    db = tmp_path / "board.db"
    db.touch()
    board = _board(db)
    with acquire_dispatcher_owner(home, OWNER_GENERATION) as lease:
        conn = _connection(db)
        _claim(conn, board)
        conn.execute(
            "UPDATE kanban_policy_pointer SET active_generation = ?",
            (POLICY_GENERATION + 5,),
        )
        with pytest.raises(LaunchProtocolError):
            release_claim(
                conn,
                board=board,
                task_id=TASK_ID,
                run_generation=1,
                dispatcher_owner_generation=OWNER_GENERATION,
                policy_generation=POLICY_GENERATION,
                route_generation=ROUTE_GENERATION,
                claim_token=CLAIM_TOKEN,
                owner_lease=lease,
            )
        assert get_launch_record(conn, board, TASK_ID, 1) is not None


def test_record_run_outcome_retires_spawned_row_with_bounded_evidence(tmp_path):
    home = tmp_path / "home"
    (home / "run").mkdir(parents=True, mode=0o700)
    db = tmp_path / "board.db"
    db.touch()
    board = _board(db)
    with acquire_dispatcher_owner(home, OWNER_GENERATION) as lease:
        conn = _connection(db)
        _claim(conn, board)
        record_spawn_intent(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            owner_lease=lease,
        )
        record_spawned(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            remote_execution_id="exec-1",
        )
        outcome = record_run_outcome(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            owner_lease=lease,
            exit_status=0,
            detail="clean exit",
        )
        assert outcome.previous_state == "spawned"
        assert outcome.exit_status == 0
        assert outcome.detail == "clean exit"
        assert get_launch_record(conn, board, TASK_ID, 1) is None


def test_record_run_outcome_rejects_invalid_evidence_and_states(tmp_path):
    home = tmp_path / "home"
    (home / "run").mkdir(parents=True, mode=0o700)
    db = tmp_path / "board.db"
    db.touch()
    board = _board(db)
    with acquire_dispatcher_owner(home, OWNER_GENERATION) as lease:
        conn = _connection(db)
        _claim(conn, board)
        base: dict[str, object] = {
            "board": board,
            "task_id": TASK_ID,
            "run_generation": 1,
            "dispatcher_owner_generation": OWNER_GENERATION,
            "policy_generation": POLICY_GENERATION,
            "route_generation": ROUTE_GENERATION,
            "claim_token": CLAIM_TOKEN,
            "owner_lease": lease,
            "exit_status": 0,
            "detail": "",
        }
        # Pre-spawn rows cannot receive a terminal outcome.
        with pytest.raises(LaunchProtocolError):
            record_run_outcome(conn, **base)
        record_spawn_intent(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            owner_lease=lease,
        )
        with pytest.raises(LaunchProtocolError):
            record_run_outcome(conn, **base)
        record_spawned(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            pid=99,
        )
        # Invalid evidence values fail closed without mutation.
        for bad in (
            {"exit_status": True},
            {"exit_status": "0"},
            {"exit_status": None},
            {"detail": "x" * 257},
            {"detail": " padded "},
            {"detail": 7},
        ):
            values = dict(base)
            values.update(bad)
            with pytest.raises(LaunchProtocolError):
                record_run_outcome(conn, **values)
        assert get_launch_record(conn, board, TASK_ID, 1) is not None


def _spawned_fixture(tmp_path):
    home = tmp_path / "home"
    (home / "run").mkdir(parents=True, mode=0o700)
    db = tmp_path / "board.db"
    db.touch()
    board = canonical_board_identity(db, BOARD_UUID)
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    conn = _connection(db)
    _claim(conn, board)
    record_spawn_intent(
        conn,
        board=board,
        task_id=TASK_ID,
        run_generation=1,
        dispatcher_owner_generation=OWNER_GENERATION,
        policy_generation=POLICY_GENERATION,
        route_generation=ROUTE_GENERATION,
        claim_token=CLAIM_TOKEN,
        owner_lease=lease,
    )
    record_spawned(
        conn,
        board=board,
        task_id=TASK_ID,
        run_generation=1,
        dispatcher_owner_generation=OWNER_GENERATION,
        policy_generation=POLICY_GENERATION,
        route_generation=ROUTE_GENERATION,
        claim_token=CLAIM_TOKEN,
        pid=99,
    )
    return conn, board, lease


_OUTCOME_BASE: dict[str, object] = {}


def _outcome_kwargs(**overrides):
    values: dict[str, object] = {
        "task_id": TASK_ID,
        "run_generation": 1,
        "dispatcher_owner_generation": OWNER_GENERATION,
        "policy_generation": POLICY_GENERATION,
        "route_generation": ROUTE_GENERATION,
        "claim_token": CLAIM_TOKEN,
        "exit_status": 0,
        "detail": "",
    }
    values.update(overrides)
    return values


def test_record_run_outcome_requires_held_owner_lease(tmp_path):
    conn, board, lease = _spawned_fixture(tmp_path)
    assert get_launch_record(conn, board, TASK_ID, 1) is not None
    # Released lease must fail closed even on a spawned row.
    lease.release()
    with pytest.raises(LaunchProtocolError):
        record_run_outcome(
            conn, board=board, owner_lease=lease, **_outcome_kwargs()
        )
    # Non-lease objects are rejected.
    with pytest.raises(LaunchProtocolError):
        record_run_outcome(
            conn, board=board, owner_lease=object(), **_outcome_kwargs()
        )
    assert get_launch_record(conn, board, TASK_ID, 1) is not None


def test_record_run_outcome_rejects_stale_identity(tmp_path):
    home = tmp_path / "home"
    (home / "run").mkdir(parents=True, mode=0o700)
    db = tmp_path / "board.db"
    db.touch()
    board = _board(db)
    with acquire_dispatcher_owner(home, OWNER_GENERATION) as lease:
        conn = _connection(db)
        _claim(conn, board)
        record_spawn_intent(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            owner_lease=lease,
        )
        record_spawned(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            pid=99,
        )
        for overrides in (
            {"claim_token": "stale_token_0123456789"},
            {"policy_generation": POLICY_GENERATION + 1},
            {"dispatcher_owner_generation": OWNER_GENERATION + 1},
            {"route_generation": ROUTE_GENERATION + 1},
        ):
            with pytest.raises(LaunchProtocolError):
                record_run_outcome(
                    conn, board=board, owner_lease=lease, **_outcome_kwargs(**overrides)
                )
        assert get_launch_record(conn, board, TASK_ID, 1) is not None


def test_record_run_outcome_rolls_back_on_policy_flip(tmp_path):
    conn, board, lease = _spawned_fixture(tmp_path)
    with lease:
        conn.execute(
            "UPDATE kanban_policy_pointer SET active_generation = ?",
            (POLICY_GENERATION + 9,),
        )
        with pytest.raises(LaunchProtocolError):
            record_run_outcome(conn, board=board, owner_lease=lease, **_outcome_kwargs())
        assert get_launch_record(conn, board, TASK_ID, 1) is not None
    lease.release()


def test_record_run_outcome_rejects_int32_boundaries(tmp_path):
    conn, board, lease = _spawned_fixture(tmp_path)
    with lease:
        for bad_status in (2**31, -(2**31) - 1, 2**63):
            with pytest.raises(LaunchProtocolError):
                record_run_outcome(
                    conn,
                    board=board,
                    owner_lease=lease,
                    **_outcome_kwargs(exit_status=bad_status),
                )
        # Exact int32 bounds are accepted; the row is re-launched each time.
        for ok_status in (2**31 - 1, -(2**31)):
            record_run_outcome(
                conn,
                board=board,
                owner_lease=lease,
                **_outcome_kwargs(exit_status=ok_status),
            )
            assert get_launch_record(conn, board, TASK_ID, 1) is None
            _claim(conn, board)
            record_spawn_intent(
                conn,
                board=board,
                task_id=TASK_ID,
                run_generation=1,
                dispatcher_owner_generation=OWNER_GENERATION,
                policy_generation=POLICY_GENERATION,
                route_generation=ROUTE_GENERATION,
                claim_token=CLAIM_TOKEN,
                owner_lease=lease,
            )
            record_spawned(
                conn,
                board=board,
                task_id=TASK_ID,
                run_generation=1,
                dispatcher_owner_generation=OWNER_GENERATION,
                policy_generation=POLICY_GENERATION,
                route_generation=ROUTE_GENERATION,
                claim_token=CLAIM_TOKEN,
                pid=99,
            )
        assert get_launch_record(conn, board, TASK_ID, 1) is not None
    lease.release()


def test_delete_failure_rolls_back_and_preserves_row(tmp_path):
    conn, board, lease = _spawned_fixture(tmp_path)
    with lease:
        # A BEFORE DELETE trigger that RAISE(IGNORE) makes the DELETE a no-op
        # (rowcount 0) without failing the statement: the rowcount guard must
        # catch it and roll the transaction back.
        conn.executescript(
            "CREATE TRIGGER block_delete BEFORE DELETE ON kanban_launch_protocol "
            "BEGIN SELECT RAISE(IGNORE); END;"
        )
        try:
            with pytest.raises(LaunchProtocolError):
                record_run_outcome(
                    conn, board=board, owner_lease=lease, **_outcome_kwargs()
                )
        finally:
            conn.execute("DROP TRIGGER block_delete")
        assert get_launch_record(conn, board, TASK_ID, 1) is not None
        # With the trigger gone, the same handoff succeeds.
        record_run_outcome(conn, board=board, owner_lease=lease, **_outcome_kwargs())
        assert get_launch_record(conn, board, TASK_ID, 1) is None
    lease.release()
