"""Additive SQL patch: allow board_only parent-done → completed on orch_requests.

Keeps existing transition matrix and adds a narrow board_only short-path used by C-min.
Safe to run repeatedly (DROP IF EXISTS + CREATE).
"""

from __future__ import annotations

import sqlite3

CMIN_REQUEST_TRANSITION_TRIGGER = """
DROP TRIGGER IF EXISTS orch_request_transition_guard;
CREATE TRIGGER orch_request_transition_guard BEFORE UPDATE ON orch_requests BEGIN
  SELECT CASE WHEN NEW.lifecycle_revision!=OLD.lifecycle_revision+1 OR NEW.updated_at<OLD.updated_at
    OR orch_capability_ok('request_transition',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,NEW.lifecycle_revision,NEW.cancel_epoch,NEW.request_key)!=1
    OR NOT (
      (OLD.lifecycle_state='submitted' AND NEW.lifecycle_state='decomposing')
      OR (OLD.lifecycle_state='decomposing' AND NEW.lifecycle_state IN ('waiting_lanes','blocked','failed','cancelling'))
      OR (OLD.lifecycle_state='waiting_lanes' AND NEW.lifecycle_state IN ('waiting_lanes','synthesizing','blocked','failed','cancelling'))
      OR (OLD.lifecycle_state='synthesizing' AND NEW.lifecycle_state IN ('work_accepted','blocked','failed','cancelling'))
      OR (OLD.lifecycle_state='work_accepted' AND NEW.lifecycle_state IN ('delivering','completed','cancelling'))
      OR (OLD.lifecycle_state='delivering' AND NEW.lifecycle_state IN ('completed','delivery_blocked','cancelling'))
      OR (OLD.lifecycle_state='delivery_blocked' AND NEW.lifecycle_state IN ('delivering','cancelling'))
      OR (OLD.lifecycle_state='blocked' AND NEW.lifecycle_state=OLD.resume_state)
      OR (OLD.lifecycle_state='cancelling' AND NEW.lifecycle_state='cancelled')
      OR (
        -- C-min board_only short path: parent native terminal => completed
        NEW.lifecycle_state='completed'
        AND OLD.lifecycle_state IN ('submitted','decomposing','waiting_lanes','synthesizing','work_accepted','delivering')
        AND NEW.delivery_closed_at IS NOT NULL
        AND EXISTS (
          SELECT 1 FROM orch_origins o
           WHERE o.board_instance_id=NEW.board_instance_id
             AND o.tenant_scope=NEW.tenant_scope
             AND o.origin_id=NEW.origin_id
             AND o.origin_kind='board_only'
        )
      )
    )
    OR NOT ((NEW.lifecycle_state='cancelling' AND OLD.lifecycle_state!='cancelling' AND NEW.cancel_epoch=OLD.cancel_epoch+1)
      OR (NEW.lifecycle_state!='cancelling' AND NEW.cancel_epoch=OLD.cancel_epoch))
    OR NOT ((OLD.lifecycle_state='synthesizing' AND NEW.lifecycle_state='work_accepted'
              AND OLD.delivery_epoch_revision=0 AND NEW.delivery_epoch_revision=NEW.lifecycle_revision
              AND NEW.work_accepted_at IS NOT NULL)
      OR (NOT (OLD.lifecycle_state='synthesizing' AND NEW.lifecycle_state='work_accepted')
              AND NEW.delivery_epoch_revision=OLD.delivery_epoch_revision
              AND (
                NEW.work_accepted_at IS OLD.work_accepted_at
                OR (NEW.lifecycle_state='completed' AND NEW.work_accepted_at IS NOT NULL)
              )))
    OR NOT ((OLD.lifecycle_state='decomposing' AND NEW.lifecycle_state='waiting_lanes'
              AND NEW.plan_version=OLD.plan_version+1
              AND OLD.plan_epoch_revision=0 AND NEW.plan_epoch_revision=NEW.lifecycle_revision
              AND EXISTS (SELECT 1 FROM orch_plan_materializations m
                WHERE m.board_instance_id=NEW.board_instance_id AND m.tenant_scope=NEW.tenant_scope
                  AND m.orch_id=NEW.orch_id AND m.plan_version=NEW.plan_version
                  AND m.request_lifecycle_revision=OLD.lifecycle_revision
                  AND m.cancel_epoch=OLD.cancel_epoch))
      OR (NOT (OLD.lifecycle_state='decomposing' AND NEW.lifecycle_state='waiting_lanes')
              AND NEW.plan_version=OLD.plan_version
              AND NEW.plan_epoch_revision=OLD.plan_epoch_revision))
    OR (NEW.lifecycle_state='completed' AND NEW.delivery_closed_at IS NULL)
    OR (NEW.lifecycle_state!='completed' AND NEW.delivery_closed_at IS NOT OLD.delivery_closed_at
        AND NOT (NEW.lifecycle_state='completed'))
    THEN RAISE(ABORT,'invalid_orch_request_transition') END;
END;
"""


def apply_cmin_transition_patch(conn: sqlite3.Connection) -> None:
    conn.executescript(CMIN_REQUEST_TRANSITION_TRIGGER)
    conn.commit()


__all__ = ["apply_cmin_transition_patch", "CMIN_REQUEST_TRANSITION_TRIGGER"]
