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

import hashlib
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from hermes_cli import project_runtime_registration as runtime
from hermes_cli.project_finalization_contract import (
    ensure_project_finalization_schema,
    create_project_finalization,
    reopen_project_finalization,
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
    PROJECT_SCHEMA_SQL,
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
        assert "project_finalization_notification_routes" in tables
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
        assert marker1 == "hermes-orch-finish-001-g4-r9-v3"
        assert ver1 == "3"

        # repeated call
        ensure_project_finalization_schema(conn)
        assert get_project_finalization_migration_marker(conn) == marker1
        assert get_project_finalization_schema_version(conn) == ver1

        # meta table has rows
        rows = list(conn.execute("SELECT key, value FROM project_finalization_meta"))
        assert ("migration", "hermes-orch-finish-001-g4-r9-v3") in [(r[0], r[1]) for r in rows]
    finally:
        _close_and_unlink(conn, path)


def test_v2_to_v3_route_authority_migration_is_additive_and_marked():
    conn, path = _make_temp_db()
    try:
        # Build the exact pre-route table set, then retain its v2 marker.
        # The v3 migration must create only the new route-authority table and
        # advance metadata transactionally.
        ensure_project_finalization_schema(conn)
        conn.execute(
            """
            CREATE TABLE kanban_notify_subs (
                task_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                thread_id TEXT NOT NULL DEFAULT '',
                user_id TEXT,
                notifier_profile TEXT
            )
            """
        )
        route_identity = "subscription:sha256:" + hashlib.sha256(
            json.dumps(
                ("telegram", "-100-v2-route", "77"),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        conn.execute(
            """
            INSERT INTO project_finalizations (
                board_id, root_task_id, generation, state, final_checker_task_id,
                admission_key, checker_profile, notification_route_identity,
                repair_budget, notification_policy, retention_days, created_at,
                updated_at, version
            ) VALUES ('v2-board', 'v2-root', 7, 'open', 'v2-checker',
                      'v2-admission', 'v2-checker-profile', ?, 1,
                      'project_summary', 3, 701, 702, 1)
            """,
            (route_identity,),
        )
        conn.execute(
            """
            INSERT INTO project_finalization_members
                (board_id, root_task_id, generation, task_id, membership_kind, required, created_at)
            VALUES ('v2-board', 'v2-root', 7, 'v2-required', 'required', 1, 701)
            """
        )
        conn.execute(
            """
            INSERT INTO kanban_notify_subs
                (task_id, platform, chat_id, thread_id, user_id, notifier_profile)
            VALUES ('v2-required', 'telegram', '-100-v2-route', '77', 'v2-user', 'v2-profile')
            """
        )
        conn.execute(
            "UPDATE project_finalization_meta SET value='2' WHERE key='version'"
        )
        conn.execute(
            "UPDATE project_finalization_meta SET value='hermes-orch-finish-001-g3-v2' "
            "WHERE key='migration'"
        )

        # This is the pre-migration runtime fallback: root GC has already
        # removed the source subscription, but the generation's required task
        # still carries the exact hash-bound route.
        legacy_destination = runtime.resolve_project_telegram_destination(
            conn, board_id="v2-board", root_task_id="v2-root", generation=7
        )
        assert legacy_destination.status == runtime.DESTINATION_FOUND
        assert legacy_destination.route_identity == route_identity
        assert len(
            runtime._resolve_route_rows(
                conn,
                board_id="v2-board",
                root_task_id="v2-root",
                generation=7,
                identities=(route_identity,),
            )
        ) == 1

        conn.execute("DROP TABLE project_finalization_notification_routes")
        ensure_project_finalization_schema(conn)

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(project_finalization_notification_routes)"
            )
        }
        assert {
            "board_id", "root_task_id", "generation", "platform", "chat_id",
            "thread_id", "route_identity", "created_at",
        } <= columns
        assert get_project_finalization_schema_version(conn) == "3"
        assert get_project_finalization_migration_marker(conn) == "hermes-orch-finish-001-g4-r9-v3"
        route = conn.execute(
            """
            SELECT generation, chat_id, thread_id, route_identity, created_at
              FROM project_finalization_notification_routes
             WHERE board_id='v2-board' AND root_task_id='v2-root'
            """
        ).fetchone()
        assert tuple(route) == (7, "-100-v2-route", "77", route_identity, 701)
        destination = runtime.resolve_project_telegram_destination(
            conn, board_id="v2-board", root_task_id="v2-root", generation=7
        )
        assert destination.status == runtime.DESTINATION_FOUND
        assert destination.route_identity == route_identity
        copier_rows = runtime._resolve_route_rows(
            conn,
            board_id="v2-board",
            root_task_id="v2-root",
            generation=7,
            identities=(route_identity,),
        )
        assert len(copier_rows) == 1
        assert copier_rows[0]["chat_id"] == "-100-v2-route"
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


def test_populated_partial_table_is_rejected_without_writing_marker():
    conn, path = _make_temp_db()
    try:
        conn.executescript("""
            CREATE TABLE project_finalizations (
                board_id TEXT NOT NULL,
                root_task_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                state TEXT NOT NULL,
                final_checker_task_id TEXT NOT NULL,
                PRIMARY KEY (board_id, root_task_id, generation)
            );
            INSERT INTO project_finalizations
                (board_id, root_task_id, generation, state, final_checker_task_id)
            VALUES ('board', 'root', 1, 'open', 'checker');
        """)

        with pytest.raises(ValueError, match="cannot safely repair populated table"):
            ensure_project_finalization_schema(conn)

        assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='project_finalization_meta'"
        ).fetchone()[0] == 0
    finally:
        _close_and_unlink(conn, path)


def test_empty_compatible_partial_table_is_additively_repaired():
    conn, path = _make_temp_db()
    try:
        # All non-defaulted required columns and the identity constraint are present.
        # Missing columns are nullable or have defaults, so ALTER ADD is safe.
        conn.executescript("""
            CREATE TABLE project_finalizations (
                board_id TEXT NOT NULL,
                root_task_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                state TEXT NOT NULL,
                final_checker_task_id TEXT NOT NULL,
                notification_policy TEXT NOT NULL,
                retention_days INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (board_id, root_task_id, generation)
            );
        """)

        ensure_project_finalization_schema(conn)

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(project_finalizations)")}
        assert {"checker_verdict", "repair_generation", "version"} <= columns
        assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 0
        assert get_project_finalization_migration_marker(conn) == "hermes-orch-finish-001-g4-r9-v3"
    finally:
        _close_and_unlink(conn, path)


def test_populated_v1_schema_migrates_additively_and_preserves_finalization_row():
    conn, path = _make_temp_db()
    try:
        # Build the exact prior project_finalizations shape, then populate it
        # before asking the v2 migration to add nullable G3 columns.
        legacy_finalizations = PROJECT_SCHEMA_SQL.split(
            "CREATE TABLE IF NOT EXISTS project_finalization_members", 1
        )[0]
        for column in (
            "    admission_key         TEXT,\n",
            "    checker_profile       TEXT,\n",
            "    notification_route_identity TEXT,\n",
            "    checker_candidate_snapshot_version TEXT,\n",
            "    checker_candidate_id  TEXT,\n",
            "    terminal_intent       TEXT,\n",
            "    terminal_candidate_snapshot_version TEXT,\n",
            "    artifact_candidate_snapshot_version TEXT,\n",
        ):
            legacy_finalizations = legacy_finalizations.replace(column, "")
        conn.executescript(legacy_finalizations)
        conn.executescript("""
            CREATE TABLE project_finalization_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO project_finalization_meta (key, value) VALUES
                ('version', '1'), ('migration', 'hof002-v1');
            INSERT INTO project_finalizations (
                board_id, root_task_id, generation, state, final_checker_task_id,
                notification_policy, retention_days, created_at, updated_at, version
            ) VALUES ('board', 'root', 1, 'open', 'checker', 'project_summary', 3, 11, 12, 1);
        """)

        ensure_project_finalization_schema(conn)

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(project_finalizations)")}
        assert {
            "admission_key",
            "checker_profile",
            "notification_route_identity",
            "checker_candidate_snapshot_version",
            "checker_candidate_id",
            "terminal_intent",
            "terminal_candidate_snapshot_version",
            "artifact_candidate_snapshot_version",
        } <= columns
        row = conn.execute("SELECT * FROM project_finalizations").fetchone()
        assert row["board_id"] == "board"
        assert row["root_task_id"] == "root"
        assert row["created_at"] == 11
        assert row["admission_key"] is None
        assert get_project_finalization_schema_version(conn) == "3"
        assert get_project_finalization_migration_marker(conn) == "hermes-orch-finish-001-g4-r9-v3"
    finally:
        _close_and_unlink(conn, path)


def test_mismatched_legacy_metadata_fails_closed_without_partial_v2_repair():
    conn, path = _make_temp_db()
    try:
        conn.executescript("""
            CREATE TABLE project_finalizations (
                board_id TEXT NOT NULL,
                root_task_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                state TEXT NOT NULL,
                final_checker_task_id TEXT NOT NULL,
                notification_policy TEXT NOT NULL,
                retention_days INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (board_id, root_task_id, generation)
            );
            CREATE TABLE project_finalization_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO project_finalization_meta (key, value) VALUES
                ('version', '1'), ('migration', 'wrong-marker');
        """)

        with pytest.raises(ValueError, match="unsupported migration marker"):
            ensure_project_finalization_schema(conn)

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(project_finalizations)")}
        assert "admission_key" not in columns
        assert dict(conn.execute("SELECT key, value FROM project_finalization_meta")) == {
            "version": "1",
            "migration": "wrong-marker",
        }
    finally:
        _close_and_unlink(conn, path)


def test_missing_required_index_is_recreated_without_rewriting_rows():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(conn, board_id="board", root_task_id="root", final_checker_task_id="checker")
        conn.execute("DROP INDEX idx_pfinal_board_state")

        ensure_project_finalization_schema(conn)

        indexes = {row["name"] for row in conn.execute("PRAGMA index_list(project_finalizations)")}
        assert "idx_pfinal_board_state" in indexes
        assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 1
    finally:
        _close_and_unlink(conn, path)


def test_drifted_primary_key_shape_is_rejected_before_marker_write():
    conn, path = _make_temp_db()
    try:
        conn.executescript("""
            CREATE TABLE project_finalizations (
                board_id TEXT NOT NULL,
                root_task_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                state TEXT NOT NULL,
                terminal_outcome TEXT,
                final_checker_task_id TEXT NOT NULL,
                checker_verdict TEXT,
                repair_generation INTEGER NOT NULL DEFAULT 0,
                repair_budget INTEGER NOT NULL DEFAULT 1,
                notification_policy TEXT NOT NULL,
                retention_days INTEGER NOT NULL,
                final_report_path TEXT,
                final_report_sha256 TEXT,
                manifest_path TEXT,
                manifest_sha256 TEXT,
                usage_summary_json TEXT,
                blocker_json TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                evaluated_at INTEGER,
                finalized_at INTEGER,
                cleanup_after TEXT,
                cleaned_at INTEGER,
                lock_owner TEXT,
                lock_expires_at INTEGER,
                version INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (board_id, root_task_id)
            );
        """)

        with pytest.raises(ValueError, match="incompatible primary key"):
            ensure_project_finalization_schema(conn)

        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='project_finalization_meta'"
        ).fetchone()[0] == 0
    finally:
        _close_and_unlink(conn, path)


def test_future_schema_version_is_rejected_without_changing_marker():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        conn.execute("UPDATE project_finalization_meta SET value='4' WHERE key='version'")

        with pytest.raises(ValueError, match="unsupported future schema version"):
            ensure_project_finalization_schema(conn)

        assert get_project_finalization_schema_version(conn) == "4"
        assert get_project_finalization_migration_marker(conn) == "hermes-orch-finish-001-g4-r9-v3"
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


def test_create_registers_root_and_final_checker_membership_atomically():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)

        create_project_finalization(
            conn,
            board_id="boardA",
            root_task_id="root123",
            final_checker_task_id="chk_t1",
        )

        members = list_project_members(
            conn,
            board_id="boardA",
            root_task_id="root123",
            generation=1,
        )
        assert {
            (member.task_id, member.membership_kind, member.required)
            for member in members
        } == {
            ("root123", "required", True),
            ("chk_t1", "checker", True),
        }
    finally:
        _close_and_unlink(conn, path)


def test_identical_create_retry_heals_missing_owned_members_without_touching_other_members():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(
            conn, board_id="boardA", root_task_id="root123", final_checker_task_id="chk_t1"
        )
        register_project_member(
            conn,
            board_id="boardA",
            root_task_id="root123",
            generation=1,
            task_id="support1",
            membership_kind="support",
            required=False,
        )
        conn.execute(
            "DELETE FROM project_finalization_members "
            "WHERE board_id='boardA' AND root_task_id='root123' AND generation=1 "
            "AND membership_kind IN ('required', 'checker')"
        )

        retry = create_project_finalization(
            conn, board_id="boardA", root_task_id="root123", final_checker_task_id="chk_t1"
        )

        members = list_project_members(
            conn, board_id="boardA", root_task_id="root123", generation=retry.generation
        )
        assert {
            (member.task_id, member.membership_kind, member.required)
            for member in members
        } == {
            ("root123", "required", True),
            ("chk_t1", "checker", True),
            ("support1", "support", False),
        }
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


def test_legacy_complete_still_requires_a_pass_checker_verdict():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(
            conn, board_id="legacy", root_task_id="root", final_checker_task_id="chk"
        )
        with pytest.raises(ValueError, match="COMPLETE requires a PASS verdict"):
            record_terminal_outcome(
                conn,
                board_id="legacy",
                root_task_id="root",
                generation=1,
                outcome="COMPLETE",
            )

        record_checker_verdict(
            conn,
            board_id="legacy",
            root_task_id="root",
            generation=1,
            checker_task_id="chk",
            verdict="PASS",
        )
        completed = record_terminal_outcome(
            conn,
            board_id="legacy",
            root_task_id="root",
            generation=1,
            outcome="COMPLETE",
        )
        assert completed.terminal_outcome == "COMPLETE"

        for outcome in ("BLOCKED", "FAILED"):
            root_task_id = outcome.lower()
            create_project_finalization(
                conn,
                board_id="legacy",
                root_task_id=root_task_id,
                final_checker_task_id=f"{root_task_id}-checker",
            )
            non_success = record_terminal_outcome(
                conn,
                board_id="legacy",
                root_task_id=root_task_id,
                generation=1,
                outcome=outcome,
            )
            assert non_success.terminal_outcome == outcome
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

        record_checker_verdict(
            conn, board_id="b", root_task_id="rt", generation=1,
            checker_task_id="chk", verdict="PASS",
        )
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


@pytest.mark.parametrize(
    "cleanup_after",
    ("", "not-a-timestamp", "2026-08-01T00:00:00"),
)
def test_cleanup_schedule_rejects_invalid_or_naive_timestamp_before_mutation(cleanup_after: str):
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        original = create_project_finalization(
            conn, board_id="b", root_task_id="rt", final_checker_task_id="chk"
        )

        with pytest.raises(ValueError, match="cleanup_after"):
            schedule_project_cleanup(
                conn, board_id="b", root_task_id="rt", generation=1, cleanup_after=cleanup_after
            )

        assert get_project_finalization(conn, board_id="b", root_task_id="rt", generation=1) == original
        assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 1
    finally:
        _close_and_unlink(conn, path)


def test_cleanup_schedule_rejects_open_generation_before_mutation():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        original = create_project_finalization(
            conn, board_id="b", root_task_id="rt", final_checker_task_id="chk"
        )

        with pytest.raises(ValueError, match="terminal_outcome"):
            schedule_project_cleanup(
                conn,
                board_id="b",
                root_task_id="rt",
                generation=1,
                cleanup_after="2026-08-01T00:00:00Z",
            )

        assert get_project_finalization(conn, board_id="b", root_task_id="rt", generation=1) == original
        assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 1
    finally:
        _close_and_unlink(conn, path)


@pytest.mark.parametrize("outcome", TERMINAL_OUTCOMES)
@pytest.mark.parametrize(
    "cleanup_after",
    ("2026-08-01T00:00:00Z", "2026-08-01T05:30:00+05:30"),
)
def test_cleanup_schedule_accepts_timezone_aware_iso8601_timestamp(
    outcome: str, cleanup_after: str
):
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(conn, board_id="b", root_task_id="rt", final_checker_task_id="chk")
        if outcome == "COMPLETE":
            record_checker_verdict(
                conn, board_id="b", root_task_id="rt", generation=1,
                checker_task_id="chk", verdict="PASS",
            )
        terminal = record_terminal_outcome(
            conn, board_id="b", root_task_id="rt", generation=1, outcome=outcome
        )

        scheduled = schedule_project_cleanup(
            conn, board_id="b", root_task_id="rt", generation=1, cleanup_after=cleanup_after
        )

        assert scheduled.cleanup_after == cleanup_after
        assert scheduled.state == "cleanup_scheduled"
        assert scheduled.terminal_outcome == terminal.terminal_outcome
    finally:
        _close_and_unlink(conn, path)


def test_persisted_identity_conflicts_reject_before_mutation_and_exact_repeats_return_same_row():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(conn, board_id="b", root_task_id="rt", final_checker_task_id="checker")
        good_sha = "a" * 64

        checker = record_checker_verdict(
            conn, board_id="b", root_task_id="rt", generation=1,
            checker_task_id="checker", verdict="PASS",
        )
        assert record_checker_verdict(
            conn, board_id="b", root_task_id="rt", generation=1,
            checker_task_id="checker", verdict="PASS",
        ) == checker
        with pytest.raises(ValueError, match="immutable checker_verdict conflict"):
            record_checker_verdict(
                conn, board_id="b", root_task_id="rt", generation=1,
                checker_task_id="checker", verdict="FAIL_REPAIRABLE",
            )
        assert get_project_finalization(conn, board_id="b", root_task_id="rt", generation=1) == checker
        assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 1

        artifacts = record_final_artifacts(
            conn, board_id="b", root_task_id="rt", generation=1,
            report_path="/report.md", report_sha256=good_sha,
            manifest_path="/manifest.json", manifest_sha256=good_sha,
        )
        assert record_final_artifacts(
            conn, board_id="b", root_task_id="rt", generation=1,
            report_path="/report.md", report_sha256=good_sha,
            manifest_path="/manifest.json", manifest_sha256=good_sha,
        ) == artifacts
        for field, changed_value in (
            ("report_path", "/other-report.md"),
            ("report_sha256", "b" * 64),
            ("manifest_path", "/other-manifest.json"),
            ("manifest_sha256", "c" * 64),
        ):
            values = {
                "report_path": "/report.md",
                "report_sha256": good_sha,
                "manifest_path": "/manifest.json",
                "manifest_sha256": good_sha,
            }
            values[field] = changed_value
            persisted_field = f"final_{field}" if field.startswith("report_") else field
            with pytest.raises(ValueError, match=f"immutable {persisted_field} conflict"):
                record_final_artifacts(conn, board_id="b", root_task_id="rt", generation=1, **values)
            assert get_project_finalization(conn, board_id="b", root_task_id="rt", generation=1) == artifacts
            assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 1

        terminal = record_terminal_outcome(
            conn, board_id="b", root_task_id="rt", generation=1, outcome="COMPLETE",
        )
        assert record_terminal_outcome(
            conn, board_id="b", root_task_id="rt", generation=1, outcome="COMPLETE",
        ) == terminal
        with pytest.raises(ValueError, match="immutable terminal_outcome conflict"):
            record_terminal_outcome(
                conn, board_id="b", root_task_id="rt", generation=1, outcome="FAILED",
            )
        assert get_project_finalization(conn, board_id="b", root_task_id="rt", generation=1) == terminal
        assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 1

        cleanup = schedule_project_cleanup(
            conn, board_id="b", root_task_id="rt", generation=1,
            cleanup_after="2026-08-01T00:00:00Z",
        )
        assert schedule_project_cleanup(
            conn, board_id="b", root_task_id="rt", generation=1,
            cleanup_after="2026-08-01T00:00:00Z",
        ) == cleanup
        with pytest.raises(ValueError, match="immutable cleanup_after conflict"):
            schedule_project_cleanup(
                conn, board_id="b", root_task_id="rt", generation=1,
                cleanup_after="2026-08-02T00:00:00Z",
            )
        assert get_project_finalization(conn, board_id="b", root_task_id="rt", generation=1) == cleanup
        assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 1
    finally:
        _close_and_unlink(conn, path)


def test_membership_all_kinds_idempotent_and_separate_from_task_links():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(conn, board_id="b", root_task_id="rt", final_checker_task_id="chk")

        # also ensure basic task tables exist for separation test (they are not touched)
        conn.executescript("CREATE TABLE IF NOT EXISTS task_links (parent_id TEXT, child_id TEXT, PRIMARY KEY (parent_id, child_id));")
        conn.execute("INSERT INTO task_links VALUES ('preexisting_parent', 'preexisting_child')")
        task_links_before = conn.execute("SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id").fetchall()

        expected_members = (
            ("rt", "required", True),
            ("chk", "checker", True),
            ("required_task", "required", True),
            ("support_task", "support", False),
            ("repair_task", "repair", False),
            ("checker_task", "checker", True),
        )
        for task_id, membership_kind, required in expected_members:
            register_project_member(
                conn, board_id="b", root_task_id="rt", generation=1,
                task_id=task_id, membership_kind=membership_kind, required=required,
            )

        # idempotent
        register_project_member(
            conn, board_id="b", root_task_id="rt", generation=1,
            task_id="support_task", membership_kind="support", required=False,
        )

        members = list_project_members(conn, board_id="b", root_task_id="rt", generation=1)
        assert {(m.task_id, m.membership_kind, m.required) for m in members} == set(expected_members)
        assert {m.membership_kind for m in members} == set(MEMBERSHIP_KINDS)
        assert len(members) == len(expected_members)

        # project membership did not touch task_links
        assert conn.execute("SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id").fetchall() == task_links_before
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


def test_reopen_rejects_a_latest_generation_without_terminal_outcome():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(
            conn, board_id="board", root_task_id="root", final_checker_task_id="checker"
        )

        with pytest.raises(ValueError, match="latest generation is not terminal"):
            reopen_project_finalization(conn, board_id="board", root_task_id="root")

        assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 1
    finally:
        _close_and_unlink(conn, path)


def test_reopen_creates_next_generation_and_preserves_completed_generation():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(
            conn,
            board_id="board",
            root_task_id="root",
            final_checker_task_id="checker",
            notification_policy="verbose",
            retention_days=7,
            repair_budget=0,
        )
        record_checker_verdict(
            conn, board_id="board", root_task_id="root", generation=1,
            checker_task_id="checker", verdict="PASS",
        )
        terminal = record_terminal_outcome(
            conn, board_id="board", root_task_id="root", generation=1, outcome="COMPLETE"
        )

        reopened = reopen_project_finalization(conn, board_id="board", root_task_id="root")

        generation_one_members = list_project_members(
            conn, board_id="board", root_task_id="root", generation=1
        )
        generation_two_members = list_project_members(
            conn, board_id="board", root_task_id="root", generation=2
        )
        assert {
            (member.task_id, member.membership_kind, member.required)
            for member in generation_one_members
        } == {
            ("root", "required", True),
            ("checker", "checker", True),
        }
        assert {
            (member.task_id, member.membership_kind, member.required)
            for member in generation_two_members
        } == {
            ("root", "required", True),
            ("checker", "checker", True),
        }

        assert reopened.generation == 2
        assert reopened.state == "open"
        assert reopened.terminal_outcome is None
        assert reopened.final_checker_task_id == "checker"
        assert reopened.notification_policy == "verbose"
        assert reopened.retention_days == 7
        assert reopened.repair_budget == 0
        assert get_project_finalization(
            conn, board_id="board", root_task_id="root", generation=1
        ) == terminal
    finally:
        _close_and_unlink(conn, path)


def test_reopen_is_deterministically_rejected_while_an_active_generation_exists():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(
            conn, board_id="board", root_task_id="root", final_checker_task_id="checker"
        )
        record_terminal_outcome(
            conn, board_id="board", root_task_id="root", generation=1, outcome="FAILED"
        )
        assert reopen_project_finalization(conn, board_id="board", root_task_id="root").generation == 2

        with pytest.raises(ValueError, match="latest generation is not terminal"):
            reopen_project_finalization(conn, board_id="board", root_task_id="root")

        assert conn.execute(
            "SELECT COUNT(*) FROM project_finalizations "
            "WHERE board_id='board' AND root_task_id='root' AND terminal_outcome IS NULL"
        ).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 2
    finally:
        _close_and_unlink(conn, path)


def test_reopen_preserves_board_and_root_isolation():
    conn, path = _make_temp_db()
    try:
        ensure_project_finalization_schema(conn)
        create_project_finalization(
            conn, board_id="board-a", root_task_id="root-a", final_checker_task_id="checker-a"
        )
        create_project_finalization(
            conn, board_id="board-b", root_task_id="root-a", final_checker_task_id="checker-b"
        )
        create_project_finalization(
            conn, board_id="board-a", root_task_id="root-b", final_checker_task_id="checker-c"
        )
        record_terminal_outcome(
            conn, board_id="board-a", root_task_id="root-a", generation=1, outcome="BLOCKED"
        )

        reopened = reopen_project_finalization(conn, board_id="board-a", root_task_id="root-a")

        assert reopened.generation == 2
        assert get_project_finalization(
            conn, board_id="board-b", root_task_id="root-a"
        ).generation == 1
        assert get_project_finalization(
            conn, board_id="board-a", root_task_id="root-b"
        ).generation == 1
        assert create_project_finalization(
            conn, board_id="board-a", root_task_id="root-a", final_checker_task_id="ignored"
        ).generation == 1
    finally:
        _close_and_unlink(conn, path)
