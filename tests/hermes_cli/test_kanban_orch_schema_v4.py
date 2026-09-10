"""
T03: Schema V4 compile + FK/trigger smoke test.
"""

import sqlite3
import pytest
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_schema_v4 import (
    EXPECTED_TABLES,
    apply_schema,
    get_table_names,
    get_trigger_names,
)


@pytest.fixture
def fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        # Compile + allow test writes through explicit open capability grant.
        apply_schema(conn, test_open_capability=True)
        conn.execute("PRAGMA foreign_keys=ON")
    except Exception:
        conn.close()
        if os.path.exists(path):
            os.unlink(path)
        raise
    yield conn
    conn.close()
    if os.path.exists(path):
        os.unlink(path)


class TestSchemaV4:

    def test_exact_schema_compile(self, fresh_db):
        """All DDL statements execute without error on fresh DB."""
        fk = fresh_db.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1, "foreign_keys must be ON"

    def test_foreign_key_check_empty(self, fresh_db):
        violations = fresh_db.execute("PRAGMA foreign_key_check").fetchall()
        assert violations == [], f"FK violations: {violations}"

    def test_all_expected_tables_exist(self, fresh_db):
        tables = set(get_table_names(fresh_db))
        missing = EXPECTED_TABLES - tables
        assert not missing, f"Missing tables: {missing}"

    def test_triggers_exist(self, fresh_db):
        triggers = get_trigger_names(fresh_db)
        assert len(triggers) >= 50, f"Only {len(triggers)} triggers created"

    def test_orphan_child_rejected_by_fk(self, fresh_db):
        """Insert into orch_requests with nonexistent origin_id is rejected."""
        # Raw SQLite lacks the runtime capability context.  Register a
        # fail-closed implementation so protected triggers reject the write as
        # sqlite3.IntegrityError before any row can be created.
        fresh_db.create_function("orch_capability_ok", 7, lambda *args: 0)
        digest = "a" * 64
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db.execute("""
                INSERT INTO orch_requests (
                    board_instance_id, tenant_scope, orch_id, lineage_id,
                    generation, selector_key, selector_ledger_revision,
                    request_key, request_schema_version, request_json, request_digest,
                    origin_id, parent_task_id,
                    lifecycle_state, lifecycle_revision, cancel_epoch,
                    delivery_epoch_revision, plan_epoch_revision, plan_version,
                    synthesis_strategy, max_retries, created_at, updated_at
                ) VALUES (
                    'board-orphan', '', 'orch-orphan', 'lineage-orphan',
                    1, :digest, 0, :digest, 4, '{}', :digest,
                    'nonexistent-origin', 'parent-orphan',
                    'submitted', 0, 0, 0, 0, 0,
                    'parent_owned', 0, 1, 1
                )
            """, {"digest": digest})

    def test_integrity_check_passes(self, fresh_db):
        result = fresh_db.execute("PRAGMA integrity_check").fetchone()[0]
        assert result == "ok", f"integrity_check: {result}"

    def test_full_request_plan_binding(self, fresh_db):
        """§13 T06: origin/request/plan full composite scope and header mismatch."""
        import json
        from hermes_cli.kanban_orch_canonical import request_digest
        from hermes_cli.kanban_orch_digest_udf import build_route_json_and_digest

        d2 = "b" * 64
        b = "board_0123456789abcdef"  # 22 chars, satisfies CHECK length 16-128
        sk = "a" * 64  # selector_key must be 64-hex (digest)
        route_json, route_d = build_route_json_and_digest(
            origin_kind="board_only",
            platform="telegram",
            adapter_instance_id="ad1",
            account_id="acc1",
            conversation_id="conv1",
            required_ack_family="none",
            required_ack_strength="none",
            route_revision=1,
        )
        req_obj = {
            "schema_version": 4,
            "kind": "orch_request",
            "selector_key": sk,
            "request_key": sk,
            "origin_id": "oid1",
            "lineage_id": "lin1",
            "generation": 1,
            "title": "parent",
            "synthesis_strategy": "parent_owned",
            "completion_policy": "board_only",
            "requirements": [],
        }
        req_json = json.dumps(req_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        req_d = request_digest(req_obj)

        fresh_db.execute(
            "INSERT INTO kanban_board_identity (singleton, board_instance_id, canonical_board_key, created_at)"
            " VALUES (1, ?, 'board1', 1)",
            (b,),
        )
        fresh_db.execute(
            "INSERT INTO orch_replay_selectors"
            " (selector_key, board_instance_id, tenant_scope, selector_kind, selector_value,"
            "  adapter_instance_id, conversation_id, lineage_id, current_generation,"
            "  ledger_revision, created_at, updated_at)"
            " VALUES (?, ?, '', 'event', 'evt1', 'ad1', 'conv1', 'lin1', 0, 0, 1, 1)",
            (sk, b),
        )
        fresh_db.execute(
            "INSERT INTO orch_origins"
            " (board_instance_id, tenant_scope, origin_id, schema_version, selector_key,"
            "  origin_kind, platform, adapter_instance_id, account_id, conversation_id,"
            "  selector_kind, selector_value, thread_id, reply_to_id, session_id,"
            "  notifier_profile, route_revision, route_json, route_digest,"
            "  required_ack_family, required_ack_strength, created_at)"
            " VALUES (?, '', 'oid1', 4, ?, 'board_only', 'telegram', 'ad1', 'acc1', 'conv1',"
            "  'event', 'evt1', '', '', '', '', 1, ?, ?, 'none', 'none', 1)",
            (b, sk, route_json, route_d),
        )
        fresh_db.execute(
            "INSERT INTO tasks (id, title, status, created_at) VALUES ('parent1', 'parent', 'pending', 1)"
        )
        fresh_db.execute(
            "INSERT INTO orch_requests"
            " (board_instance_id, tenant_scope, orch_id, lineage_id, generation,"
            "  selector_key, selector_ledger_revision, request_key, request_schema_version,"
            "  request_json, request_digest, origin_id, parent_task_id,"
            "  lifecycle_state, lifecycle_revision, cancel_epoch,"
            "  delivery_epoch_revision, plan_epoch_revision, plan_version,"
            "  synthesis_strategy, max_retries, created_at, updated_at)"
            " VALUES (?, '', 'orch1', 'lin1', 1, ?, 0, ?, 4, ?, ?, 'oid1', 'parent1',"
            "  'submitted', 0, 0, 0, 0, 0, 'parent_owned', 0, 1, 1)",
            (b, sk, sk, req_json, req_d),
        )

        # Forged route_digest must fail at SQL trigger (not just Python).
        with pytest.raises(sqlite3.IntegrityError, match="route_digest_mismatch"):
            fresh_db.execute(
                "INSERT INTO orch_origins"
                " (board_instance_id, tenant_scope, origin_id, schema_version, selector_key,"
                "  origin_kind, platform, adapter_instance_id, account_id, conversation_id,"
                "  selector_kind, selector_value, thread_id, reply_to_id, session_id,"
                "  notifier_profile, route_revision, route_json, route_digest,"
                "  required_ack_family, required_ack_strength, created_at)"
                " VALUES (?, '', 'oid-forged', 4, ?, 'board_only', 'telegram', 'ad1', 'acc1', 'conv1',"
                "  'event', 'evt1', '', '', '', '', 1, ?, ?, 'none', 'none', 1)",
                (b, sk, route_json, "c" * 64),
            )

        # Valid plan insert — all FK columns match the request
        fresh_db.execute(
            "INSERT INTO orch_plans"
            " (board_instance_id, tenant_scope, orch_id, plan_version, schema_version,"
            "  lineage_id, generation, request_key, request_digest, origin_id,"
            "  parent_task_id, synthesis_strategy, plan_json, plan_digest,"
            "  created_by_run_id, created_at)"
            " VALUES (?, '', 'orch1', 1, 4, 'lin1', 1, ?, ?, 'oid1', 'parent1',"
            "  'parent_owned', '{}', ?, 0, 1)",
            (b, sk, req_d, req_d),
        )

        # Mismatched request_key → FK violation
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db.execute(
                "INSERT INTO orch_plans"
                " (board_instance_id, tenant_scope, orch_id, plan_version, schema_version,"
                "  lineage_id, generation, request_key, request_digest, origin_id,"
                "  parent_task_id, synthesis_strategy, plan_json, plan_digest,"
                "  created_by_run_id, created_at)"
                " VALUES (?, '', 'orch1', 2, 4, 'lin1', 1, ?, ?, 'oid1', 'parent1',"
                "  'parent_owned', '{}', ?, 0, 1)",
                (b, d2, req_d, req_d),
            )

        # Mismatched origin_id → FK violation
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db.execute(
                "INSERT INTO orch_plans"
                " (board_instance_id, tenant_scope, orch_id, plan_version, schema_version,"
                "  lineage_id, generation, request_key, request_digest, origin_id,"
                "  parent_task_id, synthesis_strategy, plan_json, plan_digest,"
                "  created_by_run_id, created_at)"
                " VALUES (?, '', 'orch1', 3, 4, 'lin1', 1, ?, ?, 'wrong_origin', 'parent1',"
                "  'parent_owned', '{}', ?, 0, 1)",
                (b, sk, req_d, req_d),
            )

        # Mismatched request_digest → FK violation
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db.execute(
                "INSERT INTO orch_plans"
                " (board_instance_id, tenant_scope, orch_id, plan_version, schema_version,"
                "  lineage_id, generation, request_key, request_digest, origin_id,"
                "  parent_task_id, synthesis_strategy, plan_json, plan_digest,"
                "  created_by_run_id, created_at)"
                " VALUES (?, '', 'orch1', 4, 4, 'lin1', 1, ?, ?, 'oid1', 'parent1',"
                "  'parent_owned', '{}', ?, 0, 1)",
                (b, sk, d2, req_d),
            )
