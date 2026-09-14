"""Protected contract tests for Kanban owner and launch primitives.

These tests use only temporary files and synthetic SQLite databases.  The
module under test is intentionally inactive and has no dispatcher imports.
"""

from __future__ import annotations

import importlib
import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_launch_protocol as launch_protocol
from hermes_cli.kanban_launch_protocol import (
    BoardIdentityError,
    DispatcherOwnerError,
    LaunchProtocolError,
    acquire_dispatcher_owner,
    canonical_board_identity,
    claim_not_spawned,
    dispatcher_owner_lock_path,
    freeze_ack_blockers,
    get_launch_record,
    install_launch_schema,
    record_spawn_intent,
    record_spawned,
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


def test_board_aliases_collapse_to_device_inode_and_uuid(tmp_path: Path) -> None:
    database = tmp_path / "kanban.db"
    database.touch()
    alias = tmp_path / "alias.db"
    alias.symlink_to(database)

    direct = canonical_board_identity(database, BOARD_UUID)
    through_alias = canonical_board_identity(alias, BOARD_UUID.upper())

    assert direct == through_alias
    assert direct.board_uuid == BOARD_UUID
    assert direct.device == os.stat(database).st_dev
    assert direct.inode == os.stat(database).st_ino


def test_board_identity_rejects_invalid_or_non_regular_targets(tmp_path: Path) -> None:
    with pytest.raises(BoardIdentityError):
        canonical_board_identity(tmp_path / "missing.db", BOARD_UUID)
    with pytest.raises(BoardIdentityError):
        canonical_board_identity(tmp_path, BOARD_UUID)
    database = tmp_path / "kanban.db"
    database.touch()
    with pytest.raises(BoardIdentityError):
        canonical_board_identity(database, "not-a-uuid")


def test_owner_lock_path_is_canonical_across_home_aliases(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    alias = tmp_path / "home-alias"
    alias.symlink_to(home, target_is_directory=True)

    assert dispatcher_owner_lock_path(home) == dispatcher_owner_lock_path(alias)


def test_owner_lock_is_exclusive_and_release_allows_reacquire(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    try:
        assert lease.validate() is True
        with pytest.raises(DispatcherOwnerError, match="contended"):
            acquire_dispatcher_owner(home, OWNER_GENERATION + 1)
    finally:
        lease.release()

    replacement = acquire_dispatcher_owner(home, OWNER_GENERATION + 1)
    assert replacement.validate() is True
    replacement.release()


def test_replaced_owner_lock_file_invalidates_lease(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    lock_path = dispatcher_owner_lock_path(home)
    lock_path.unlink()
    lock_path.touch(mode=0o600)
    try:
        assert lease.validate() is False
    finally:
        lease.release()


def test_owner_generation_requires_exact_positive_integer(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    for generation in (True, 1.0, 0, -1):
        with pytest.raises(DispatcherOwnerError):
            acquire_dispatcher_owner(home, generation)  # type: ignore[arg-type]


def test_claim_persists_complete_exact_identity(tmp_path: Path) -> None:
    database = tmp_path / "board.db"
    database.touch()
    board = canonical_board_identity(database, BOARD_UUID)
    conn = _connection(tmp_path / "launch.db")
    try:
        _claim(conn, board)
        record = get_launch_record(conn, board, TASK_ID, 1)
        assert record is not None
        assert record.board == board
        assert record.state == "claimed_not_spawned"
        assert record.dispatcher_owner_generation == OWNER_GENERATION
        assert record.policy_generation == POLICY_GENERATION
        assert record.route_generation == ROUTE_GENERATION
        assert record.claim_token == CLAIM_TOKEN
        assert record.pid is None
        assert record.remote_execution_id is None
    finally:
        conn.close()


def test_duplicate_claim_and_active_transaction_fail_without_partial_write(
    tmp_path: Path,
) -> None:
    database = tmp_path / "board.db"
    database.touch()
    board = canonical_board_identity(database, BOARD_UUID)
    conn = _connection(tmp_path / "launch.db")
    try:
        _claim(conn, board)
        with pytest.raises(LaunchProtocolError, match="duplicate"):
            _claim(conn, board)
        assert len(freeze_ack_blockers(conn)) == 1

        conn.execute("BEGIN")
        with pytest.raises(LaunchProtocolError, match="active transaction"):
            claim_not_spawned(
                conn,
                board=board,
                task_id="t_1111111111111111",
                run_generation=1,
                dispatcher_owner_generation=OWNER_GENERATION,
                policy_generation=POLICY_GENERATION,
                route_generation=ROUTE_GENERATION,
                claim_token="claim_token_1111111111111111",
            )
        conn.rollback()
        assert len(freeze_ack_blockers(conn)) == 1
    finally:
        conn.close()


def _intent(conn: sqlite3.Connection, board, lease) -> None:
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


def test_spawned_rejects_lost_owner_lease(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    database = tmp_path / "board.db"
    database.touch()
    board = canonical_board_identity(database, BOARD_UUID)
    conn = _connection(tmp_path / "launch.db")
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    try:
        _intent(conn, board, lease)
        lease.release()
        with pytest.raises(LaunchProtocolError, match="owner lease"):
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
                owner_lease=lease,
            )
        record = get_launch_record(conn, board, TASK_ID, 1)
        assert record is not None
        assert record.state == "spawn_intent"  # CAS never ran
    finally:
        conn.close()


def test_spawned_rejects_policy_flip_inside_transaction(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    database = tmp_path / "board.db"
    database.touch()
    board = canonical_board_identity(database, BOARD_UUID)
    conn = _connection(tmp_path / "launch.db")
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    try:
        _intent(conn, board, lease)
        conn.execute(
            "UPDATE kanban_policy_pointer SET active_generation = ? "
            "WHERE singleton = 1",
            (POLICY_GENERATION + 1,),
        )
        with pytest.raises(LaunchProtocolError, match="policy generation"):
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
                owner_lease=lease,
            )
        record = get_launch_record(conn, board, TASK_ID, 1)
        assert record is not None
        assert record.state == "spawn_intent"
    finally:
        lease.release()
        conn.close()


def test_spawned_with_lease_succeeds_when_generations_match(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    database = tmp_path / "board.db"
    database.touch()
    board = canonical_board_identity(database, BOARD_UUID)
    conn = _connection(tmp_path / "launch.db")
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    try:
        _intent(conn, board, lease)
        record = record_spawned(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
            pid=4242,
            owner_lease=lease,
        )
        assert record.state == "spawned"
        assert record.pid == 4242
    finally:
        lease.release()
        conn.close()




def test_spawned_reentrant_release_during_cas_is_deferred(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    database = tmp_path / "board.db"
    database.touch()
    board = canonical_board_identity(database, BOARD_UUID)
    conn = _connection(tmp_path / "launch.db")
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    try:
        _intent(conn, board, lease)
        module = importlib.import_module("hermes_cli.kanban_launch_protocol")
        real_cas = module._cas_record

        def cas_then_release(cconn, **kw):
            result = real_cas(cconn, **kw)
            try:
                lease.release()  # same-thread release attempt after CAS
            except DispatcherOwnerError:
                pass  # deferred: serialized op active — hardened behavior
            return result

        module._cas_record = cas_then_release
        try:
            record = record_spawned(
                conn,
                board=board,
                task_id=TASK_ID,
                run_generation=1,
                dispatcher_owner_generation=OWNER_GENERATION,
                policy_generation=POLICY_GENERATION,
                route_generation=ROUTE_GENERATION,
                claim_token=CLAIM_TOKEN,
                pid=4242,
                owner_lease=lease,
            )
        finally:
            module._cas_record = real_cas
        # The release was deferred, not honored: the op commits atomically
        # with the lease intact, and the caller can release afterwards.
        assert record.state == "spawned"
        assert lease.validate()
        lease.release()
        row = get_launch_record(conn, board, TASK_ID, 1)
        assert row is not None and row.state == "spawned"
    finally:
        try:
            lease.release()
        except Exception:
            pass
        conn.close()




def test_spawned_lease_released_during_commit_stays_spawned(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    database = tmp_path / "board.db"
    database.touch()
    board = canonical_board_identity(database, BOARD_UUID)
    conn = _connection(tmp_path / "launch.db")
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    try:
        _intent(conn, board, lease)

        hook_calls: list[str] = []

        def commit_hook(statement: str) -> None:
            if statement.upper().startswith("COMMIT"):
                hook_calls.append(statement)
                try:
                    lease.release()  # same-thread reentrant release at COMMIT
                except DispatcherOwnerError:
                    pass  # deferred: serialized op active — the hardened path

        conn.set_trace_callback(commit_hook)
        try:
            record = record_spawned(
                conn,
                board=board,
                task_id=TASK_ID,
                run_generation=1,
                dispatcher_owner_generation=OWNER_GENERATION,
                policy_generation=POLICY_GENERATION,
                route_generation=ROUTE_GENERATION,
                claim_token=CLAIM_TOKEN,
                pid=4242,
                owner_lease=lease,
            )
        finally:
            conn.set_trace_callback(None)
        assert record.state == "spawned"
        assert hook_calls, "COMMIT trace hook never fired"
        # Hardened behavior: the release was deferred, so the lease is STILL
        # VALID after the commit — a spawned row never exists with a dead lease.
        assert lease.validate()
        lease.release()
        assert not lease.validate()
        row = get_launch_record(conn, board, TASK_ID, 1)
        assert row is not None and row.state == "spawned"
    finally:
        conn.close()


def test_release_deferred_during_serialized_op() -> None:
    home_like = None
    # Unit-level: a serialized op in flight defers same-thread release.
    from hermes_cli.kanban_launch_protocol import DispatcherOwnerError

    lease = acquire_dispatcher_owner(_home_for_unit(), OWNER_GENERATION)
    try:
        with lease._serialized():
            with pytest.raises(DispatcherOwnerError):
                lease.release()
        assert lease.validate()
        lease.release()
        assert not lease.validate()
    finally:
        if lease.validate():
            lease.release()
        del home_like


def _home_for_unit() -> Path:
    import tempfile

    d = Path(tempfile.mkdtemp())
    (d / "run").mkdir()
    (d / "run").chmod(0o700)
    d.chmod(0o755)
    return d


def test_policy_flip_before_spawn_intent_rolls_back(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    database = tmp_path / "board.db"
    database.touch()
    board = canonical_board_identity(database, BOARD_UUID)
    conn = _connection(tmp_path / "launch.db")
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    try:
        _claim(conn, board)
        conn.execute(
            "UPDATE kanban_policy_pointer SET active_generation = ? WHERE singleton = 1",
            (POLICY_GENERATION + 1,),
        )
        with pytest.raises(LaunchProtocolError, match="policy generation"):
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
        record = get_launch_record(conn, board, TASK_ID, 1)
        assert record is not None
        assert record.state == "claimed_not_spawned"
    finally:
        lease.release()
        conn.close()


def test_lost_owner_or_stale_token_cannot_create_spawn_intent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    database = tmp_path / "board.db"
    database.touch()
    board = canonical_board_identity(database, BOARD_UUID)
    conn = _connection(tmp_path / "launch.db")
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    try:
        _claim(conn, board)
        lease.release()
        with pytest.raises(LaunchProtocolError, match="owner lease"):
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
        assert get_launch_record(conn, board, TASK_ID, 1).state == "claimed_not_spawned"  # type: ignore[union-attr]

        lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
        with pytest.raises(LaunchProtocolError, match="claim identity"):
            record_spawn_intent(
                conn,
                board=board,
                task_id=TASK_ID,
                run_generation=1,
                dispatcher_owner_generation=OWNER_GENERATION,
                policy_generation=POLICY_GENERATION,
                route_generation=ROUTE_GENERATION,
                claim_token="stale_claim_token_0123456789",
                owner_lease=lease,
            )
        assert get_launch_record(conn, board, TASK_ID, 1).state == "claimed_not_spawned"  # type: ignore[union-attr]
    finally:
        lease.release()
        conn.close()


def test_spawn_requires_durable_intent_and_one_execution_identity(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    database = tmp_path / "board.db"
    database.touch()
    board = canonical_board_identity(database, BOARD_UUID)
    conn = _connection(tmp_path / "launch.db")
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    try:
        _claim(conn, board)
        with pytest.raises(LaunchProtocolError, match="transition"):
            record_spawned(
                conn,
                board=board,
                task_id=TASK_ID,
                run_generation=1,
                claim_token=CLAIM_TOKEN,
                dispatcher_owner_generation=OWNER_GENERATION,
                policy_generation=POLICY_GENERATION,
                route_generation=ROUTE_GENERATION,
                pid=1234,
            )

        intent = record_spawn_intent(
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
        assert intent.state == "spawn_intent"
        assert [row.state for row in freeze_ack_blockers(conn)] == ["spawn_intent"]

        for pid, remote_id in ((None, None), (1234, "remote-1"), (True, None)):
            with pytest.raises(LaunchProtocolError, match="execution identity"):
                record_spawned(
                    conn,
                    board=board,
                    task_id=TASK_ID,
                    run_generation=1,
                    claim_token=CLAIM_TOKEN,
                    dispatcher_owner_generation=OWNER_GENERATION,
                    policy_generation=POLICY_GENERATION,
                    route_generation=ROUTE_GENERATION,
                    pid=pid,  # type: ignore[arg-type]
                    remote_execution_id=remote_id,
                )

        spawned = record_spawned(
            conn,
            board=board,
            task_id=TASK_ID,
            run_generation=1,
            claim_token=CLAIM_TOKEN,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            pid=1234,
        )
        assert spawned.state == "spawned"
        assert spawned.pid == 1234
        assert freeze_ack_blockers(conn) == []
    finally:
        lease.release()
        conn.close()


def test_owner_lock_rejects_unsafe_run_directory_and_hardlink(tmp_path: Path) -> None:
    unsafe_home = tmp_path / "unsafe-home"
    unsafe_home.mkdir()
    unsafe_run = unsafe_home / "run"
    unsafe_run.mkdir(mode=0o777)
    unsafe_run.chmod(0o777)
    with pytest.raises(DispatcherOwnerError, match="permissions"):
        acquire_dispatcher_owner(unsafe_home, OWNER_GENERATION)

    home = tmp_path / "home"
    home.mkdir()
    run = home / "run"
    run.mkdir(mode=0o700)
    protected = tmp_path / "protected"
    protected.write_text("must-not-change", encoding="utf-8")
    (run / "kanban-dispatcher-owner.lock").hardlink_to(protected)
    with pytest.raises(DispatcherOwnerError, match="lock file"):
        acquire_dispatcher_owner(home, OWNER_GENERATION)
    assert protected.read_text(encoding="utf-8") == "must-not-change"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork contract")
def test_forked_child_release_does_not_unlock_parent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    child = os.fork()
    if child == 0:  # pragma: no cover - assertions run in parent
        lease.release()
        os._exit(0)
    os.waitpid(child, 0)
    try:
        assert lease.validate() is True
        with pytest.raises(DispatcherOwnerError, match="contended"):
            acquire_dispatcher_owner(home, OWNER_GENERATION + 1)
    finally:
        lease.release()


def test_incompatible_existing_schema_is_rejected(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "launch.db", isolation_level=None)
    conn.execute("CREATE TABLE kanban_launch_protocol (task_id TEXT)")
    try:
        with pytest.raises(LaunchProtocolError, match="schema"):
            install_launch_schema(conn)
    finally:
        conn.close()


def test_schema_rejects_execution_identity_before_spawn(tmp_path: Path) -> None:
    conn = _connection(tmp_path / "launch.db")
    values = (
        BOARD_UUID,
        1,
        1,
        TASK_ID,
        1,
        OWNER_GENERATION,
        POLICY_GENERATION,
        ROUTE_GENERATION,
        CLAIM_TOKEN,
        "claimed_not_spawned",
        1234,
    )
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO kanban_launch_protocol
                   (board_uuid, board_device, board_inode, task_id,
                    run_generation, dispatcher_owner_generation,
                    policy_generation, route_generation, claim_token, state, pid)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )
    finally:
        conn.close()


def test_schema_verification_is_case_sensitive_inside_literals(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "launch.db", isolation_level=None)
    incompatible = launch_protocol._SCHEMA.replace(
        "'claimed_not_spawned'", "'CLAIMED_NOT_SPAWNED'"
    )
    conn.execute(incompatible)
    try:
        with pytest.raises(LaunchProtocolError, match="schema"):
            install_launch_schema(conn)
    finally:
        conn.close()


def test_schema_verification_rejects_behavior_changing_trigger(tmp_path: Path) -> None:
    conn = _connection(tmp_path / "launch.db")
    conn.execute(
        "CREATE TRIGGER reject_launches BEFORE INSERT ON kanban_launch_protocol "
        "BEGIN SELECT RAISE(ABORT, 'rejected'); END"
    )
    try:
        with pytest.raises(LaunchProtocolError, match="schema objects"):
            install_launch_schema(conn)
    finally:
        conn.close()


def test_missing_policy_pointer_fails_closed_without_transition(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    database = tmp_path / "board.db"
    database.touch()
    board = canonical_board_identity(database, BOARD_UUID)
    conn = _connection(tmp_path / "launch.db")
    lease = acquire_dispatcher_owner(home, OWNER_GENERATION)
    try:
        _claim(conn, board)
        conn.execute("DELETE FROM kanban_policy_pointer")
        with pytest.raises(LaunchProtocolError, match="policy pointer"):
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
        record = get_launch_record(conn, board, TASK_ID, 1)
        assert record is not None and record.state == "claimed_not_spawned"
    finally:
        lease.release()
        conn.close()


def test_task_id_accepts_live_board_short_format():
    """RED: live board mints t_ + token_hex(4) (8 hex chars); the kernel
    launch protocol must accept the live format, not only 16-hex."""
    from hermes_cli.kanban_launch_protocol import _valid_task_id
    assert _valid_task_id("t_04d0aa77")
    assert _valid_task_id("t_" + "a" * 16)
    assert not _valid_task_id("t_short")      # non-hex
    assert not _valid_task_id("t_04d0aa7")    # 7 hex
    assert not _valid_task_id("t_04d0aa777")  # 9 hex
    assert not _valid_task_id("x_04d0aa77")


def test_claim_not_spawned_persists_live_board_short_task_id(tmp_path: Path) -> None:
    """End-to-end: an 8-hex live-format task ID must survive the full
    claim_not_spawned path — regex validation AND the SQLite schema CHECK
    (length(task_id) IN (10, 18)) — not just the in-memory validator."""
    home = tmp_path / "home"
    home.mkdir()
    database = tmp_path / "board.db"
    database.touch()
    live_task_id = "t_04d0aa77"
    conn = _connection(tmp_path / "launch.db")
    try:
        board = canonical_board_identity(database, BOARD_UUID)
        record = claim_not_spawned(
            conn,
            board=board,
            task_id=live_task_id,
            run_generation=1,
            dispatcher_owner_generation=OWNER_GENERATION,
            policy_generation=POLICY_GENERATION,
            route_generation=ROUTE_GENERATION,
            claim_token=CLAIM_TOKEN,
        )
        assert record.task_id == live_task_id
        assert record.state == "claimed_not_spawned"
        stored = get_launch_record(conn, board, live_task_id, 1)
        assert stored is not None and stored.state == "claimed_not_spawned"
        # Same board/task/generation again must fail-closed as a genuine
        # duplicate (not as a schema rejection).
        try:
            claim_not_spawned(
                conn,
                board=board,
                task_id=live_task_id,
                run_generation=1,
                dispatcher_owner_generation=OWNER_GENERATION,
                policy_generation=POLICY_GENERATION,
                route_generation=ROUTE_GENERATION,
                claim_token=CLAIM_TOKEN + "ff",
            )
            raise AssertionError("expected duplicate claim to be rejected")
        except LaunchProtocolError as exc:
            assert "duplicate or conflicting" in str(exc)
    finally:
        conn.close()
