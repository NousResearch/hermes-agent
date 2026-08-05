"""Additive sidecar patches required for plan materialization / lane accept.

Safe to re-run. Does not touch native kanban.db.
"""

from __future__ import annotations

import sqlite3

_SOFT_LINK_COLS = {
    "orch_board_instance_id": "TEXT",
    "orch_tenant_scope": "TEXT",
    "orch_id": "TEXT",
    "orch_plan_version": "INTEGER",
    "orch_edge_key": "TEXT",
    "orch_binding_revision": "INTEGER",
    "orch_cancel_epoch": "INTEGER",
}

_SOFT_RUN_COLS = {
    "cancellation_epoch": "INTEGER NOT NULL DEFAULT 0",
}

# Expanded bind update: allow unbound -> lane bind when orch_nodes exists.
TASKS_ORCH_BINDING_UPDATE_PATCH = """
DROP TRIGGER IF EXISTS tasks_orch_binding_update;
CREATE TRIGGER tasks_orch_binding_update BEFORE UPDATE OF
  orch_board_instance_id,orch_tenant_scope,orch_id,orch_plan_version,
  orch_node_key,orch_binding_revision,orch_cancel_epoch ON _soft_fk_tasks
WHEN OLD.orch_id IS NOT NULL OR NEW.orch_id IS NOT NULL
BEGIN
  SELECT CASE WHEN NOT (
    (NEW.orch_board_instance_id IS OLD.orch_board_instance_id
      AND NEW.orch_tenant_scope IS OLD.orch_tenant_scope
      AND NEW.orch_id IS OLD.orch_id
      AND NEW.orch_plan_version IS OLD.orch_plan_version
      AND NEW.orch_node_key IS OLD.orch_node_key
      AND NEW.orch_binding_revision IS OLD.orch_binding_revision
      AND NEW.orch_cancel_epoch IS OLD.orch_cancel_epoch)
    OR
    (OLD.orch_board_instance_id IS NULL AND OLD.orch_tenant_scope IS NULL
      AND OLD.orch_id IS NULL AND OLD.orch_plan_version IS NULL
      AND OLD.orch_node_key IS NULL AND OLD.orch_binding_revision IS NULL
      AND OLD.orch_cancel_epoch IS NULL
      AND NEW.orch_node_key='__parent__'
      AND EXISTS (SELECT 1 FROM orch_requests r
        WHERE r.board_instance_id=NEW.orch_board_instance_id
          AND r.tenant_scope=NEW.orch_tenant_scope AND r.orch_id=NEW.orch_id
          AND r.parent_task_id=NEW.id AND r.plan_version=NEW.orch_plan_version
          AND r.lifecycle_revision=NEW.orch_binding_revision
          AND r.cancel_epoch=NEW.orch_cancel_epoch)
      AND orch_capability_ok('task_bind',NEW.orch_board_instance_id,NEW.orch_tenant_scope,NEW.orch_id,
        NEW.orch_binding_revision,NEW.orch_cancel_epoch,NEW.id)=1)
    OR
    (OLD.orch_board_instance_id IS NULL AND OLD.orch_tenant_scope IS NULL
      AND OLD.orch_id IS NULL AND OLD.orch_plan_version IS NULL
      AND OLD.orch_node_key IS NULL AND OLD.orch_binding_revision IS NULL
      AND OLD.orch_cancel_epoch IS NULL
      AND EXISTS (
        SELECT 1 FROM orch_nodes n
        JOIN orch_requests r
          ON r.board_instance_id=n.board_instance_id AND r.tenant_scope=n.tenant_scope
         AND r.orch_id=n.orch_id
        WHERE n.board_instance_id=NEW.orch_board_instance_id
          AND n.tenant_scope=NEW.orch_tenant_scope AND n.orch_id=NEW.orch_id
          AND n.plan_version=NEW.orch_plan_version AND n.node_key=NEW.orch_node_key
          AND n.task_id=NEW.id
          AND r.lifecycle_state='decomposing'
          AND r.lifecycle_revision=NEW.orch_binding_revision
          AND r.cancel_epoch=NEW.orch_cancel_epoch
      )
      AND orch_capability_ok('task_bind',NEW.orch_board_instance_id,NEW.orch_tenant_scope,NEW.orch_id,
        NEW.orch_binding_revision,NEW.orch_cancel_epoch,NEW.id)=1)
    OR
    (NEW.orch_board_instance_id IS NULL AND NEW.orch_tenant_scope IS NULL
      AND NEW.orch_id IS NULL AND NEW.orch_plan_version IS NULL
      AND NEW.orch_node_key IS NULL AND NEW.orch_binding_revision IS NULL
      AND NEW.orch_cancel_epoch IS NULL
      AND EXISTS (SELECT 1 FROM orch_requests r
        WHERE r.board_instance_id=OLD.orch_board_instance_id
          AND r.tenant_scope=OLD.orch_tenant_scope AND r.orch_id=OLD.orch_id
          AND r.lifecycle_state IN ('cancelling','cancelled'))
      AND orch_capability_ok(
        'task_retire',OLD.orch_board_instance_id,OLD.orch_tenant_scope,OLD.orch_id,
        COALESCE((SELECT lifecycle_revision FROM orch_requests r
          WHERE r.board_instance_id=OLD.orch_board_instance_id AND r.tenant_scope=OLD.orch_tenant_scope
            AND r.orch_id=OLD.orch_id),-1),
        COALESCE((SELECT cancel_epoch FROM orch_requests r
          WHERE r.board_instance_id=OLD.orch_board_instance_id AND r.tenant_scope=OLD.orch_tenant_scope
            AND r.orch_id=OLD.orch_id),-1),OLD.id)=1)
  ) THEN RAISE(ABORT,'immutable_orch_task_binding') END;
END;
"""


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    names = {r[1] for r in rows}
    return col in names


def apply_multilane_soft_fk_patch(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Add missing soft-FK columns + multilane bind update trigger."""
    added: dict[str, list[str]] = {"_soft_fk_task_links": [], "_soft_fk_task_runs": [], "triggers": []}
    for col, decl in _SOFT_LINK_COLS.items():
        if not _has_column(conn, "_soft_fk_task_links", col):
            conn.execute(f"ALTER TABLE _soft_fk_task_links ADD COLUMN {col} {decl}")
            added["_soft_fk_task_links"].append(col)
    for col, decl in _SOFT_RUN_COLS.items():
        if not _has_column(conn, "_soft_fk_task_runs", col):
            conn.execute(f"ALTER TABLE _soft_fk_task_runs ADD COLUMN {col} {decl}")
            added["_soft_fk_task_runs"].append(col)
    conn.executescript(TASKS_ORCH_BINDING_UPDATE_PATCH)
    added["triggers"].append("tasks_orch_binding_update")
    conn.commit()
    return added


__all__ = ["apply_multilane_soft_fk_patch", "TASKS_ORCH_BINDING_UPDATE_PATCH"]
