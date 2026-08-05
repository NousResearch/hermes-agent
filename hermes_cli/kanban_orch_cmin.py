"""ORCH V4 C-min: sidecar board_only judge driven by native parent status.

N1–N4:
- Never mutates native kanban schema
- All lifecycle writes go to sidecar orch_v4.db
- Native is read-only observation of parent task status
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

from hermes_cli.kanban_orch_api import OrchAPIError, apply_lifecycle_transition_db
from hermes_cli.kanban_orch_bridge import BridgeError, OrchBridge
from hermes_cli.kanban_orch_lifecycle import LifecycleError
from hermes_cli.kanban_orch_writer_switch import open_live_bridge

ACTIVE_NATIVE = frozenset({"ready", "running", "todo", "blocked", "triage"})
DONE_NATIVE = frozenset({"done", "completed"})


class CMinError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class JudgeStep:
    event: str
    from_state: str
    to_state: str
    lifecycle_revision: int


@dataclass
class JudgeResult:
    orch_id: str
    parent_task_id: str
    native_status: str
    origin_kind: str
    before_state: str
    after_state: str
    steps: list[JudgeStep]
    skipped: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _load_request(conn: sqlite3.Connection, *, board: str, tenant: str, orch_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT r.orch_id, r.parent_task_id, r.lifecycle_state, r.lifecycle_revision, "
        "r.board_instance_id, r.tenant_scope, o.origin_kind "
        "FROM orch_requests r "
        "JOIN orch_origins o ON o.board_instance_id=r.board_instance_id "
        " AND o.tenant_scope=r.tenant_scope AND o.origin_id=r.origin_id "
        "WHERE r.board_instance_id=? AND r.tenant_scope=? AND r.orch_id=?",
        (board, tenant, orch_id),
    ).fetchone()
    if row is None:
        raise CMinError("request_not_found")
    return row


def _load_by_parent(conn: sqlite3.Connection, *, parent_task_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT r.orch_id, r.parent_task_id, r.lifecycle_state, r.lifecycle_revision, "
        "r.board_instance_id, r.tenant_scope, o.origin_kind "
        "FROM orch_requests r "
        "JOIN orch_origins o ON o.board_instance_id=r.board_instance_id "
        " AND o.tenant_scope=r.tenant_scope AND o.origin_id=r.origin_id "
        "WHERE r.parent_task_id=? ORDER BY r.updated_at DESC LIMIT 1",
        (parent_task_id,),
    ).fetchone()
    if row is None:
        raise CMinError("request_not_found_for_parent")
    return row


def judge_board_only_once(
    conn: sqlite3.Connection,
    *,
    board_instance_id: str,
    tenant_scope: str,
    orch_id: str,
    native_status: str,
) -> JudgeResult:
    """Apply at most one progressive transition based on native parent status."""
    tenant = "" if tenant_scope is None else str(tenant_scope)
    row = _load_request(conn, board=board_instance_id, tenant=tenant, orch_id=orch_id)
    origin = row["origin_kind"]
    state = row["lifecycle_state"]
    parent = row["parent_task_id"]
    status = (native_status or "").strip().lower()

    if origin != "board_only":
        return JudgeResult(
            orch_id=orch_id,
            parent_task_id=parent,
            native_status=status,
            origin_kind=origin,
            before_state=state,
            after_state=state,
            steps=[],
            skipped=True,
            reason="not_board_only",
        )
    if state in {"completed", "failed", "cancelled"}:
        return JudgeResult(
            orch_id=orch_id,
            parent_task_id=parent,
            native_status=status,
            origin_kind=origin,
            before_state=state,
            after_state=state,
            steps=[],
            skipped=True,
            reason="already_terminal",
        )

    try:
        if status in DONE_NATIVE:
            step = apply_lifecycle_transition_db(
                conn,
                board_instance_id=board_instance_id,
                tenant_scope=tenant,
                orch_id=orch_id,
                event="board_only_parent_done",
                to_state="completed",
                origin_kind="board_only",
                native_parent_done=True,
            )
        elif status in ACTIVE_NATIVE and state == "submitted":
            step = apply_lifecycle_transition_db(
                conn,
                board_instance_id=board_instance_id,
                tenant_scope=tenant,
                orch_id=orch_id,
                event="claim_decomposition",
                to_state="decomposing",
            )
        else:
            return JudgeResult(
                orch_id=orch_id,
                parent_task_id=parent,
                native_status=status,
                origin_kind=origin,
                before_state=state,
                after_state=state,
                steps=[],
                skipped=True,
                reason=f"no_rule_for:{state}:{status}",
            )
    except (OrchAPIError, LifecycleError) as exc:
        code = getattr(exc, "code", None) or str(exc)
        raise CMinError(f"transition_failed:{code}") from exc

    js = JudgeStep(
        event=str(step.get("event") or ""),
        from_state=str(step["from_state"]),
        to_state=str(step["to_state"]),
        lifecycle_revision=int(step["lifecycle_revision"]),
    )
    return JudgeResult(
        orch_id=orch_id,
        parent_task_id=parent,
        native_status=status,
        origin_kind=origin,
        before_state=js.from_state,
        after_state=js.to_state,
        steps=[js],
        skipped=False,
    )


def judge_board_only_to_fixed_point(
    conn: sqlite3.Connection,
    *,
    board_instance_id: str,
    tenant_scope: str,
    orch_id: str,
    native_status: str,
    max_steps: int = 8,
) -> JudgeResult:
    """Apply progressive transitions until skip or terminal."""
    tenant = "" if tenant_scope is None else str(tenant_scope)
    row = _load_request(conn, board=board_instance_id, tenant=tenant, orch_id=orch_id)
    before = row["lifecycle_state"]
    parent = row["parent_task_id"]
    origin = row["origin_kind"]
    all_steps: list[JudgeStep] = []
    last = JudgeResult(
        orch_id=orch_id,
        parent_task_id=parent,
        native_status=native_status,
        origin_kind=origin,
        before_state=before,
        after_state=before,
        steps=[],
    )
    for _ in range(max_steps):
        one = judge_board_only_once(
            conn,
            board_instance_id=board_instance_id,
            tenant_scope=tenant,
            orch_id=orch_id,
            native_status=native_status,
        )
        if one.skipped or not one.steps:
            last = JudgeResult(
                orch_id=orch_id,
                parent_task_id=parent,
                native_status=native_status,
                origin_kind=origin,
                before_state=before,
                after_state=one.after_state,
                steps=all_steps,
                skipped=len(all_steps) == 0,
                reason=one.reason if len(all_steps) == 0 else None,
            )
            break
        all_steps.extend(one.steps)
        last = JudgeResult(
            orch_id=orch_id,
            parent_task_id=parent,
            native_status=native_status,
            origin_kind=origin,
            before_state=before,
            after_state=one.after_state,
            steps=list(all_steps),
            skipped=False,
        )
        if one.after_state in {"completed", "failed", "cancelled"}:
            break
    return last


def live_judge_parent_task(parent_task_id: str) -> JudgeResult:
    """Open live bridge, RO-read native parent, write sidecar lifecycle only."""
    br: OrchBridge | None = None
    try:
        br = open_live_bridge()
        ref = br.read_native_task(parent_task_id)
        if ref is None:
            raise CMinError("native_parent_missing")
        row = _load_by_parent(br._sidecar, parent_task_id=parent_task_id)
        return judge_board_only_to_fixed_point(
            br._sidecar,
            board_instance_id=row["board_instance_id"],
            tenant_scope=row["tenant_scope"] or "",
            orch_id=row["orch_id"],
            native_status=ref.status,
        )
    except BridgeError as exc:
        raise CMinError(f"bridge:{exc.code}") from exc
    finally:
        if br is not None:
            br.close()


__all__ = [
    "CMinError",
    "JudgeResult",
    "JudgeStep",
    "judge_board_only_once",
    "judge_board_only_to_fixed_point",
    "live_judge_parent_task",
]
