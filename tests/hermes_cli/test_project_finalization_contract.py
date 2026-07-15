"""Focused tests for HOF-002 project finalization contract and schema.

Run only with the verified interpreter and -q -n 0 per task contract.
Covers:
- schema creation, indexes, uniqueness
- idempotent migration + marker
- partial migration repair/detection
- project identity (gen=1 start, board/root)
- all listed validations
- locking (acquire, renew, expire, cross-owner, release)
- checker verdict (only designated)
- artifacts / terminal / cleanup scheduling idempotency
- membership (kinds, idempotent, separate from task_links)
- no mutation of pre-existing kanban tables/rows
- migration identity queryable
"""

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from hermes_cli.project_finalization_contract import (
    ensure_project_finalization_schema,
    create_project_finalization,
    get_project_finalization,
    list_project_finalizations,
    acquire_finalization_lock,
    release_finalization_lock,
    record_checker_verdict,
    record_final_artifacts,
    record_terminal_outcome,
    schedule_project_cleanup,
    register_project_member,
    list_project_members,
    get_project_finalization_migration_marker,
    get_project_finalization_schema_version,
    PROJECT_FINALIZATION_STATES,
    TERMINAL_OUTCOMES,
    CHECKER_VERDICTS,
    NOTIFICATION_POLICIES,
    MEMBERSHIP_KINDS,
    ProjectFinalization,
    ProjectMember,
    # boundary recorders for schema coverage
    record_delivery_attempt,
    record_failure_envelope,
    record_cleanup_journal,
)


def _make_temp_db() -> tuple[sqlite3.Connection, Path]:
    """Create a fresh temp DB file (simulates board) and return (conn, path).
    Uses isolation_level=None to match kanban normal path (avoids implicit tx).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = Path(tmp.name)
    tmp.close()
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    # Normal connection path pragmas (as required by contract)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA wal_autocheckpoint=100")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn, path


def _close_and_unlink(conn: sqlite3.Connection, path: Path) -> None:
    try:
        conn.close()
    except Exception:
        pass
    try:
        path.unlink()
    except Exception:
        pass


def test_schema_creates_all_required_tables_and_indexes():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)

        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'project_%'"
        )}
        assert "project_finalizations" in tables
        assert "project_finalization_members" in tables
        assert "project_delivery_attempts" in tables
        assert "project_failure_envelopes" in tables
        assert "project_cleanup_journal" in tables
        assert "project_finalization_meta" in tables

        # indexes
        idx_names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_p%'"
        )}
        assert any("pfinal" in n for n in idx_names)
        assert any("pmembers" in n for n in idx_names)

        # uniqueness on identity
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO project_finalizations (board_id, root_task_id, generation, state, "
                "final_checker_task_id, notification_policy, retention_days, created_at, updated_at, version) "
                "VALUES ('b','r',1,'open','chk','project_summary',3,1,1,1)"
            )
            conn.execute(
                "INSERT INTO project_finalizations (board_id, root_task_id, generation, state, "
                "final_checker_task_id, notification_policy, retention_days, created_at, updated_at, version) "
                "VALUES ('b','r',1,'open','chk','project_summary',3,2,2,1)"
            )
    finally:
        _close_and_unlink(conn, path)


def test_migration_is_idempotent_and_marker_queryable():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        marker1 = get_project_finalization_migration_marker(conn)
        ver1 = get_project_finalization_schema_version(conn)
        assert marker1 == "hof002-v1"
        assert ver1 == "1"

        # repeated call
        ensure_project_finalization_schema(conn)
        assert get_project_finalization_migration_marker(conn) == marker1
        assert get_project_finalization_schema_version(conn) == ver1

        # meta table has rows
        rows = list(conn.execute("SELECT key, value FROM project_finalization_meta"))
        assert ("migration", "hof002-v1") in [(r[0], r[1]) for r in rows]
    finally:
        _close_and_unlink(conn, path)


def test_partial_migration_is_repaired_deterministically():
    conn, path = _make_temp_db()
    try:
        # simulate a partial prior create (missing columns)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS project_finalizations (
                board_id TEXT NOT NULL,
                root_task_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                state TEXT NOT NULL,
                final_checker_task_id TEXT NOT NULL,
                PRIMARY KEY (board_id, root_task_id, generation)
            );
        """)
        ensure_project_finalization_schema(conn)  # must repair via ALTER ADD

        cols = {r["name"] for r in conn.execute("PRAGMA table_info(project_finalizations)")}
        assert "notification_policy" in cols
        assert "retention_days" in cols
        assert "lock_owner" in cols
        assert "version" in cols
        assert get_project_finalization_migration_marker(conn) is not None
    finally:
        _close_and_unlink(conn, path)


def test_create_returns_idempotent_gen1_and_board_root_separation():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)

        p1 = create_project_finalization(
            conn,
            board_id="boardA",
            root_task_id="root123",
            final_checker_task_id="chk_t1",
        )
        assert p1.generation == 1
        assert p1.state == "open"
        assert p1.final_checker_task_id == "chk_t1"
        assert p1.board_id == "boardA"
        assert p1.root_task_id == "root123"

        p1_again = create_project_finalization(
            conn, board_id="boardA", root_task_id="root123", final_checker_task_id="chk_t1"
        )
        assert p1_again.generation == 1
        assert p1_again.created_at == p1.created_at  # idempotent

        p2 = create_project_finalization(
            conn, board_id="boardB", root_task_id="root123", final_checker_task_id="chk_t2"
        )
        assert p2.board_id == "boardB"
        assert p2.generation == 1
        assert p2 != p1

        # get latest
        got = get_project_finalization(conn, board_id="boardA", root_task_id="root123")
        assert got is not None and got.generation == 1

        # list filter
        lst = list_project_finalizations(conn, board_id="boardA")
        assert len(lst) == 1
    finally:
        _close_and_unlink(conn, path)


def test_validations_reject_bad_values():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)

        with pytest.raises(ValueError):
            create_project_finalization(conn, board_id="b", root_task_id="r", final_checker_task_id="c",
                                        notification_policy="bad")
        with pytest.raises(ValueError):
            create_project_finalization(conn, board_id="b", root_task_id="r", final_checker_task_id="c",
                                        retention_days=1)
        with pytest.raises(ValueError):
            create_project_finalization(conn, board_id="b", root_task_id="r", final_checker_task_id="c",
                                        repair_budget=5)

        p = create_project_finalization(conn, board_id="b", root_task_id="r", final_checker_task_id="c")
        with pytest.raises(ValueError):
            record_terminal_outcome(conn, board_id="b", root_task_id="r", generation=1, outcome="BAD")
        with pytest.raises(ValueError):
            record_checker_verdict(conn, board_id="b", root_task_id="r", generation=1, checker_task_id="c",
                                   verdict="NOPE")
        with pytest.raises(ValueError):
            record_final_artifacts(conn, board_id="b", root_task_id="r", generation=1,
                                   report_path="/r", report_sha256="short", manifest_path="/m",
                                   manifest_sha256="nothex")
    finally:
        _close_and_unlink(conn, path)


def test_locking_cas_renew_expire_cross_owner_and_release():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        p = create_project_finalization(conn, board_id="b", root_task_id="root", final_checker_task_id="chk")

        # first acquire
        assert acquire_finalization_lock(conn, board_id="b", root_task_id="root", generation=1,
                                         owner="worker1", lease_seconds=30)
        got = get_project_finalization(conn, board_id="b", root_task_id="root")
        assert got.lock_owner == "worker1"
        assert got.lock_expires_at is not None

        # same owner renews
        assert acquire_finalization_lock(conn, board_id="b", root_task_id="root", generation=1,
                                         owner="worker1", lease_seconds=60)

        # different owner blocked while unexpired
        assert not acquire_finalization_lock(conn, board_id="b", root_task_id="root", generation=1,
                                             owner="worker2", lease_seconds=10)

        # simulate expire by backdating (bypass for test)
        past = int(time.time()) - 10
        conn.execute(
            "UPDATE project_finalizations SET lock_expires_at=? WHERE board_id='b' AND root_task_id='root' AND generation=1",
            (past,)
        )
        # now worker2 can acquire expired
        assert acquire_finalization_lock(conn, board_id="b", root_task_id="root", generation=1,
                                         owner="worker2", lease_seconds=5)

        # release only by owner
        release_finalization_lock(conn, board_id="b", root_task_id="root", generation=1, owner="worker2")
        got2 = get_project_finalization(conn, board_id="b", root_task_id="root")
        assert got2.lock_owner is None

        # repeated release safe
        release_finalization_lock(conn, board_id="b", root_task_id="root", generation=1, owner="worker2")
    finally:
        _close_and_unlink(conn, path)


def test_checker_verdict_only_by_designated_and_idempotent():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        p = create_project_finalization(conn, board_id="b", root_task_id="rt", final_checker_task_id="thechk")

        with pytest.raises(ValueError):
            record_checker_verdict(conn, board_id="b", root_task_id="rt", generation=1,
                                   checker_task_id="wrong", verdict="PASS")

        rec = record_checker_verdict(conn, board_id="b", root_task_id="rt", generation=1,
                                     checker_task_id="thechk", verdict="PASS")
        assert rec.checker_verdict == "PASS"

        # idempotent
        rec2 = record_checker_verdict(conn, board_id="b", root_task_id="rt", generation=1,
                                      checker_task_id="thechk", verdict="PASS")
        assert rec2.evaluated_at == rec.evaluated_at
    finally:
        _close_and_unlink(conn, path)


def test_artifacts_terminal_cleanup_idempotent_and_validations():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(conn, board_id="b", root_task_id="rt", final_checker_task_id="chk")

        good_sha = "a" * 64
        rec = record_final_artifacts(
            conn, board_id="b", root_task_id="rt", generation=1,
            report_path="/tmp/report.md", report_sha256=good_sha,
            manifest_path="/tmp/manifest.json", manifest_sha256=good_sha,
        )
        assert rec.final_report_sha256 == good_sha

        # idempotent same
        rec2 = record_final_artifacts(
            conn, board_id="b", root_task_id="rt", generation=1,
            report_path="/tmp/report.md", report_sha256=good_sha,
            manifest_path="/tmp/manifest.json", manifest_sha256=good_sha,
        )
        assert rec2.finalized_at == rec.finalized_at

        term = record_terminal_outcome(conn, board_id="b", root_task_id="rt", generation=1,
                                       outcome="COMPLETE")
        assert term.terminal_outcome == "COMPLETE"
        assert term.state == "complete"

        sched = schedule_project_cleanup(conn, board_id="b", root_task_id="rt", generation=1,
                                         cleanup_after="2026-08-01T00:00:00Z")
        assert sched.state == "cleanup_scheduled"
        assert sched.cleanup_after is not None
    finally:
        _close_and_unlink(conn, path)


def test_membership_idempotent_and_separate_from_task_links():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(conn, board_id="b", root_task_id="rt", final_checker_task_id="chk")

        # also ensure basic task tables exist for separation test (they are not touched)
        conn.executescript("CREATE TABLE IF NOT EXISTS task_links (parent_id TEXT, child_id TEXT, PRIMARY KEY (parent_id, child_id));")

        register_project_member(conn, board_id="b", root_task_id="rt", generation=1,
                                task_id="supp1", membership_kind="support", required=False)
        register_project_member(conn, board_id="b", root_task_id="rt", generation=1,
                                task_id="chk_task", membership_kind="checker", required=True)

        # idempotent
        register_project_member(conn, board_id="b", root_task_id="rt", generation=1,
                                task_id="supp1", membership_kind="support", required=False)

        members = list_project_members(conn, board_id="b", root_task_id="rt", generation=1)
        kinds = {m.membership_kind for m in members}
        assert "support" in kinds
        assert "checker" in kinds
        assert len(members) == 2

        # project membership did not touch task_links
        link_count = conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0]
        assert link_count == 0
    finally:
        _close_and_unlink(conn, path)


def test_compatibility_existing_kanban_rows_untouched():
    conn, path = _make_temp_db()
    try:
        # seed some legacy kanban rows
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT, created_at INTEGER);
            CREATE TABLE IF NOT EXISTS task_links (parent_id TEXT, child_id TEXT, PRIMARY KEY(parent_id, child_id));
            INSERT INTO tasks (id, title, status, created_at) VALUES ('t1','old','todo',1);
            INSERT INTO task_links VALUES ('p','t1');
        """)

        ensure_project_finalization_schema(conn)
        create_project_finalization(conn, board_id="b", root_task_id="rt", final_checker_task_id="chk")

        # existing untouched
        t = conn.execute("SELECT * FROM tasks WHERE id='t1'").fetchone()
        assert t["title"] == "old"
        assert conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0] == 1

        # project tables coexist
        assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 1
    finally:
        _close_and_unlink(conn, path)


def test_delivery_failure_cleanup_boundaries_persist():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(conn, board_id="b", root_task_id="rt", final_checker_task_id="chk")

        da = record_delivery_attempt(
            conn, board_id="b", root_task_id="rt", generation=1,
            idempotency_key="del-1", platform="telegram", attempt_number=1,
            delivery_state="sent",
        )
        assert da.idempotency_key == "del-1"

        fe = record_failure_envelope(
            conn, board_id="b", root_task_id="rt", generation=1,
            task_id="t1", redacted_error="timeout", provider="openai",
        )
        assert fe.task_id == "t1"

        cj = record_cleanup_journal(
            conn, board_id="b", root_task_id="rt", generation=1,
            status="done", deleted_task_count=5,
        )
        assert cj.status == "done"
    finally:
        _close_and_unlink(conn, path)
