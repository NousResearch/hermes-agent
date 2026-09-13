"""ORCH V4 plan grammar and closed-world graph queries.

The validator is intentionally SQL-backed: plan rows are the source of truth,
and a valid plan produces no grammar findings.  The SQL is the executable
version of contract §5.3 rather than a Python reimplementation of it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable


PLAN_GRAMMAR_SQL = """
WITH p AS (
  SELECT * FROM orch_plans
   WHERE board_instance_id=:board AND tenant_scope=:tenant
     AND orch_id=:orch AND plan_version=:pv
), nodes AS (
  SELECT * FROM orch_plan_nodes
   WHERE board_instance_id=:board AND tenant_scope=:tenant
     AND orch_id=:orch AND plan_version=:pv
), target AS (
  SELECT node_key FROM nodes
   WHERE (role='parent' AND (SELECT synthesis_strategy FROM p)='parent_owned')
      OR (role='synthesis' AND (SELECT synthesis_strategy FROM p)='separate_node')
)
SELECT 'missing_or_duplicate_plan' WHERE (SELECT count(*) FROM p)!=1
UNION ALL SELECT 'missing_requirements' WHERE (
  SELECT count(*) FROM orch_request_requirements
   WHERE board_instance_id=:board AND tenant_scope=:tenant AND orch_id=:orch
)=0
UNION ALL SELECT 'bad_parent_count' WHERE (SELECT count(*) FROM nodes WHERE role='parent')!=1
UNION ALL SELECT 'bad_parent_key' WHERE (SELECT count(*) FROM nodes WHERE role='parent' AND node_key='__parent__')!=1
UNION ALL SELECT 'bad_parent_ordinal' WHERE (SELECT count(*) FROM nodes WHERE role='parent' AND ordinal=1)!=1
UNION ALL SELECT 'bad_lane_count' WHERE (SELECT count(*) FROM nodes WHERE role='lane') NOT BETWEEN 2 AND 16
UNION ALL SELECT 'bad_required_lane_count' WHERE (SELECT count(*) FROM nodes WHERE role='lane' AND required=1)<2
UNION ALL SELECT 'duplicate_lane_label' WHERE
  (SELECT count(*) FROM nodes WHERE role='lane')!=(SELECT count(DISTINCT lane_label) FROM nodes WHERE role='lane')
UNION ALL SELECT 'bad_synthesis_count' WHERE
  (SELECT synthesis_strategy FROM p)='parent_owned'
  AND (SELECT count(*) FROM nodes WHERE role='synthesis')!=0
UNION ALL SELECT 'bad_synthesis_count' WHERE
  (SELECT synthesis_strategy FROM p)='separate_node'
  AND (SELECT count(*) FROM nodes WHERE role='synthesis')!=1
UNION ALL SELECT 'bad_target_count' WHERE (SELECT count(*) FROM target)!=1
UNION ALL SELECT 'ordinal_gap' WHERE
  (SELECT min(ordinal) FROM nodes)!=1
  OR (SELECT max(ordinal) FROM nodes)!=(SELECT count(*) FROM nodes)
UNION ALL SELECT 'bad_required_edge_count' WHERE EXISTS (
  SELECT 1 FROM nodes n WHERE n.role='lane' AND n.required=1 AND
    (SELECT count(*) FROM orch_plan_edges e
      WHERE e.board_instance_id=:board AND e.tenant_scope=:tenant
        AND e.orch_id=:orch AND e.plan_version=:pv
        AND e.parent_node_key=n.node_key
        AND e.child_node_key=(SELECT node_key FROM target)
        AND e.edge_kind='orch_required_for_synthesis')!=1
)
UNION ALL SELECT 'bad_optional_edge' WHERE EXISTS (
  SELECT 1 FROM nodes n WHERE n.role='lane' AND n.required=0 AND
    ((SELECT count(*) FROM orch_plan_edges e
       WHERE e.board_instance_id=:board AND e.tenant_scope=:tenant
         AND e.orch_id=:orch AND e.plan_version=:pv
         AND e.parent_node_key=n.node_key)>1
     OR EXISTS (
       SELECT 1 FROM orch_plan_edges e
        WHERE e.board_instance_id=:board AND e.tenant_scope=:tenant
          AND e.orch_id=:orch AND e.plan_version=:pv
          AND e.parent_node_key=n.node_key
          AND (e.child_node_key!=(SELECT node_key FROM target)
               OR e.edge_kind!='orch_optional_context')
     ))
)
UNION ALL SELECT 'illegal_edge_shape' WHERE EXISTS (
  SELECT 1 FROM orch_plan_edges e JOIN nodes n ON n.node_key=e.parent_node_key
   WHERE e.board_instance_id=:board AND e.tenant_scope=:tenant
     AND e.orch_id=:orch AND e.plan_version=:pv
     AND (n.role!='lane' OR e.child_node_key!=(SELECT node_key FROM target)
       OR (n.required=1 AND e.edge_kind!='orch_required_for_synthesis')
       OR (n.required=0 AND e.edge_kind!='orch_optional_context'))
)
UNION ALL SELECT 'uncovered_requirement' WHERE EXISTS (
  SELECT 1 FROM orch_request_requirements r
   WHERE r.board_instance_id=:board AND r.tenant_scope=:tenant
     AND r.orch_id=:orch AND r.required=1 AND NOT EXISTS (
       SELECT 1 FROM orch_plan_coverage c JOIN nodes n ON n.node_key=c.node_key
        WHERE c.board_instance_id=:board AND c.tenant_scope=:tenant
          AND c.orch_id=:orch AND c.requirement_id=r.requirement_id
          AND c.plan_version=:pv AND n.role='lane'
     )
)
UNION ALL SELECT 'coverage_to_non_lane' WHERE EXISTS (
  SELECT 1 FROM orch_plan_coverage c JOIN nodes n ON n.node_key=c.node_key
   WHERE c.board_instance_id=:board AND c.tenant_scope=:tenant
     AND c.orch_id=:orch AND c.plan_version=:pv AND n.role!='lane'
)
"""

MATERIALIZATION_PROVENANCE_SQL = """
WITH p AS (
  SELECT * FROM orch_plans
   WHERE board_instance_id=:board AND tenant_scope=:tenant
     AND orch_id=:orch AND plan_version=:pv
), r AS (
  SELECT * FROM orch_requests
   WHERE board_instance_id=:board AND tenant_scope=:tenant AND orch_id=:orch
), m AS (
  SELECT * FROM orch_plan_materializations
   WHERE board_instance_id=:board AND tenant_scope=:tenant
     AND orch_id=:orch AND plan_version=:pv
)
SELECT 'missing_plan' WHERE (SELECT count(*) FROM p)!=1
UNION ALL SELECT 'materialization_without_plan' WHERE
  (SELECT count(*) FROM m)>0 AND (SELECT count(*) FROM p)=0
UNION ALL SELECT 'plan_request_header_mismatch' WHERE EXISTS (
  SELECT 1 FROM p CROSS JOIN r
   WHERE p.request_key!=r.request_key
      OR p.request_digest!=r.request_digest
      OR p.origin_id!=r.origin_id
      OR p.parent_task_id!=r.parent_task_id
      OR p.synthesis_strategy!=r.synthesis_strategy
      OR p.lineage_id!=r.lineage_id
      OR p.generation!=r.generation
)
UNION ALL SELECT 'materialization_plan_digest_mismatch' WHERE EXISTS (
  SELECT 1 FROM m LEFT JOIN p
    ON p.board_instance_id=m.board_instance_id
   AND p.tenant_scope=m.tenant_scope AND p.orch_id=m.orch_id
   AND p.plan_version=m.plan_version AND p.plan_digest=m.plan_digest
   WHERE p.orch_id IS NULL
)
UNION ALL SELECT 'materialization_request_revision_mismatch' WHERE EXISTS (
  SELECT 1 FROM m CROSS JOIN r
   WHERE m.request_lifecycle_revision!=r.lifecycle_revision
      OR m.cancel_epoch!=r.cancel_epoch
)
"""

CLOSED_WORLD_SQL = """
WITH RECURSIVE
seed(task_id) AS (
  SELECT parent_task_id FROM orch_requests
   WHERE board_instance_id=:board AND tenant_scope=:tenant AND orch_id=:orch
  UNION
  SELECT task_id FROM orch_nodes
   WHERE board_instance_id=:board AND tenant_scope=:tenant AND orch_id=:orch
  UNION
  SELECT id FROM tasks
   WHERE orch_board_instance_id=:board AND orch_tenant_scope=:tenant AND orch_id=:orch
  UNION
  SELECT parent_task_id FROM orch_external_edges
   WHERE board_instance_id=:board AND tenant_scope=:tenant AND orch_id=:orch
  UNION
  SELECT child_task_id FROM orch_external_edges
   WHERE board_instance_id=:board AND tenant_scope=:tenant AND orch_id=:orch
),
closure(task_id) AS (
  SELECT task_id FROM seed
  UNION
  SELECT l.child_id FROM task_links l JOIN closure c ON l.parent_id=c.task_id
  UNION
  SELECT l.parent_id FROM task_links l JOIN closure c ON l.child_id=c.task_id
)
SELECT task_id FROM closure ORDER BY task_id
"""


def _params(board: str, tenant: str, orch: str, plan_version: int) -> dict[str, object]:
    return {"board": board, "tenant": tenant, "orch": orch, "pv": plan_version}


def grammar_findings(
    conn: sqlite3.Connection, *, board: str, tenant: str, orch: str, plan_version: int
) -> list[str]:
    """Return contract §5.3 grammar findings; an accepted plan returns ``[]``."""
    rows = conn.execute(PLAN_GRAMMAR_SQL, _params(board, tenant, orch, plan_version)).fetchall()
    return [str(row[0]) for row in rows]


def validate_plan(
    conn: sqlite3.Connection, *, board: str, tenant: str, orch: str, plan_version: int
) -> None:
    """Raise ``ValueError`` if the plan grammar is not non-vacuously valid."""
    findings = grammar_findings(conn, board=board, tenant=tenant, orch=orch, plan_version=plan_version)
    if findings:
        raise ValueError("invalid_orch_plan:" + ",".join(findings))


def materialization_findings(
    conn: sqlite3.Connection, *, board: str, tenant: str, orch: str, plan_version: int
) -> list[str]:
    """Return plan-header/materialization provenance findings."""
    rows = conn.execute(
        MATERIALIZATION_PROVENANCE_SQL,
        _params(board, tenant, orch, plan_version),
    ).fetchall()
    return [str(row[0]) for row in rows]


def validate_materialization_provenance(
    conn: sqlite3.Connection, *, board: str, tenant: str, orch: str, plan_version: int
) -> None:
    findings = materialization_findings(
        conn, board=board, tenant=tenant, orch=orch, plan_version=plan_version
    )
    if findings:
        raise ValueError("invalid_materialization_provenance:" + ",".join(findings))


def closed_world_closure(
    conn: sqlite3.Connection, *, board: str, tenant: str, orch: str
) -> list[str]:
    """Compute the recursive bidirectional task closure from contract §4.5."""
    return [str(row[0]) for row in conn.execute(
        CLOSED_WORLD_SQL, {"board": board, "tenant": tenant, "orch": orch}
    ).fetchall()]


# Short aliases useful to callers that treat the validator as a query object.
validate_grammar = grammar_findings
recursive_closed_world = closed_world_closure

__all__ = [
    "PLAN_GRAMMAR_SQL",
    "MATERIALIZATION_PROVENANCE_SQL",
    "CLOSED_WORLD_SQL",
    "grammar_findings",
    "validate_grammar",
    "validate_plan",
    "materialization_findings",
    "validate_materialization_provenance",
    "closed_world_closure",
    "recursive_closed_world",
]
