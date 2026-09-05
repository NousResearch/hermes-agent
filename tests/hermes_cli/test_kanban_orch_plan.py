"""M2 T07-T09: executable plan grammar and closed-world tests."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_cli.kanban_orch_plan import (
    closed_world_closure,
    grammar_findings,
    materialization_findings,
)


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE orch_requests (
          board_instance_id TEXT, tenant_scope TEXT, orch_id TEXT,
          lineage_id TEXT, generation INTEGER, request_key TEXT,
          request_digest TEXT, origin_id TEXT, parent_task_id TEXT,
          synthesis_strategy TEXT, lifecycle_revision INTEGER, cancel_epoch INTEGER
        );
        CREATE TABLE orch_request_requirements (
          board_instance_id TEXT, tenant_scope TEXT, orch_id TEXT,
          requirement_id TEXT, required INTEGER
        );
        CREATE TABLE orch_plans (
          board_instance_id TEXT, tenant_scope TEXT, orch_id TEXT,
          plan_version INTEGER, request_key TEXT, request_digest TEXT,
          origin_id TEXT, parent_task_id TEXT, synthesis_strategy TEXT,
          lineage_id TEXT, generation INTEGER, plan_digest TEXT
        );
        CREATE TABLE orch_plan_nodes (
          board_instance_id TEXT, tenant_scope TEXT, orch_id TEXT,
          plan_version INTEGER, node_key TEXT, role TEXT, ordinal INTEGER,
          lane_label TEXT, required INTEGER
        );
        CREATE TABLE orch_plan_edges (
          board_instance_id TEXT, tenant_scope TEXT, orch_id TEXT,
          plan_version INTEGER, parent_node_key TEXT, child_node_key TEXT,
          edge_kind TEXT
        );
        CREATE TABLE orch_plan_coverage (
          board_instance_id TEXT, tenant_scope TEXT, orch_id TEXT,
          plan_version INTEGER, requirement_id TEXT, node_key TEXT
        );
        CREATE TABLE orch_plan_materializations (
          board_instance_id TEXT, tenant_scope TEXT, orch_id TEXT,
          plan_version INTEGER, plan_digest TEXT,
          request_lifecycle_revision INTEGER, cancel_epoch INTEGER
        );
        CREATE TABLE tasks (
          id TEXT PRIMARY KEY, orch_board_instance_id TEXT,
          orch_tenant_scope TEXT, orch_id TEXT
        );
        CREATE TABLE task_links (parent_id TEXT, child_id TEXT);
        CREATE TABLE orch_nodes (
          board_instance_id TEXT, tenant_scope TEXT, orch_id TEXT,
          task_id TEXT
        );
        CREATE TABLE orch_external_edges (
          board_instance_id TEXT, tenant_scope TEXT, orch_id TEXT,
          parent_task_id TEXT, child_task_id TEXT
        );
        """
    )
    return conn


def _base_plan(conn: sqlite3.Connection) -> None:
    conn.execute(
        """INSERT INTO orch_requests VALUES
        ('board', '', 'orch', 'lineage', 1, 'request-key', 'request-digest',
         'origin', 'parent-task', 'parent_owned', 4, 0)"""
    )
    conn.execute(
        """INSERT INTO orch_plans VALUES
        ('board', '', 'orch', 1, 'request-key', 'request-digest', 'origin',
         'parent-task', 'parent_owned', 'lineage', 1, 'plan-digest')"""
    )
    conn.executemany(
        """INSERT INTO orch_plan_nodes VALUES
        ('board', '', 'orch', 1, ?, ?, ?, ?, ?)""",
        [
            ("__parent__", "parent", 1, "", 1),
            ("lane-a", "lane", 2, "A", 1),
            ("lane-b", "lane", 3, "B", 1),
        ],
    )
    conn.executemany(
        """INSERT INTO orch_request_requirements VALUES
        ('board', '', 'orch', ?, 1)""",
        [("req-a",), ("req-b",)],
    )
    conn.executemany(
        """INSERT INTO orch_plan_edges VALUES
        ('board', '', 'orch', 1, ?, '__parent__', 'orch_required_for_synthesis')""",
        [("lane-a",), ("lane-b",)],
    )
    conn.executemany(
        """INSERT INTO orch_plan_coverage VALUES
        ('board', '', 'orch', 1, ?, ?)""",
        [("req-a", "lane-a"), ("req-b", "lane-b")],
    )


def test_nonvacuous_plan_grammar(db: sqlite3.Connection) -> None:
    """Zero requirements, <2 required lanes, and parent ordinal drift reject."""
    _base_plan(db)
    assert grammar_findings(db, board="board", tenant="", orch="orch", plan_version=1) == []

    db.execute("DELETE FROM orch_request_requirements")
    assert "missing_requirements" in grammar_findings(
        db, board="board", tenant="", orch="orch", plan_version=1
    )

    db.execute("INSERT INTO orch_request_requirements VALUES ('board', '', 'orch', 'req', 1)")
    db.execute("UPDATE orch_plan_nodes SET required=0 WHERE node_key='lane-b'")
    assert "bad_required_lane_count" in grammar_findings(
        db, board="board", tenant="", orch="orch", plan_version=1
    )

    db.execute("UPDATE orch_plan_nodes SET ordinal=9 WHERE node_key='__parent__'")
    assert "bad_parent_ordinal" in grammar_findings(
        db, board="board", tenant="", orch="orch", plan_version=1
    )


def test_materialization_provenance(db: sqlite3.Connection) -> None:
    """Plan headers bind to the request; materialization without a plan rejects."""
    db.execute(
        """INSERT INTO orch_requests VALUES
        ('board', '', 'orch', 'lineage', 1, 'request-key', 'request-digest',
         'origin', 'parent-task', 'parent_owned', 4, 0)"""
    )
    db.execute(
        """INSERT INTO orch_plan_materializations VALUES
        ('board', '', 'orch', 1, 'plan-digest', 4, 0)"""
    )
    findings = materialization_findings(
        db, board="board", tenant="", orch="orch", plan_version=1
    )
    assert "missing_plan" in findings
    assert "materialization_without_plan" in findings

    _base_plan(db)  # same request key is intentionally ignored by this assertion
    db.execute("UPDATE orch_plans SET request_digest='wrong-digest'")
    findings = materialization_findings(
        db, board="board", tenant="", orch="orch", plan_version=1
    )
    assert "plan_request_header_mismatch" in findings


def test_recursive_closed_world(db: sqlite3.Connection) -> None:
    """The recursive closure finds A→X→Y, not merely the first edge."""
    db.execute(
        """INSERT INTO orch_requests VALUES
        ('board', '', 'orch', 'lineage', 1, 'request-key', 'request-digest',
         'origin', 'A', 'parent_owned', 0, 0)"""
    )
    db.executemany("INSERT INTO tasks(id) VALUES (?)", [("A",), ("X",), ("Y",)])
    db.executemany("INSERT INTO task_links VALUES (?, ?)", [("A", "X"), ("X", "Y")])
    assert closed_world_closure(db, board="board", tenant="", orch="orch") == ["A", "X", "Y"]
