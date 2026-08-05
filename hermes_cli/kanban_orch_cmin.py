"""ORCH V4 C-min: sidecar board_only judge driven by native parent/children.

N1–N4:
- Never mutates native kanban schema
- All lifecycle writes go to sidecar orch_v4.db
- Native is read-only observation of parent + child task status
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from hermes_cli.kanban_orch_api import OrchAPIError, apply_lifecycle_transition_db
from hermes_cli.kanban_orch_bridge import BridgeError, OrchBridge
from hermes_cli.kanban_orch_lifecycle import LifecycleError
from hermes_cli.kanban_orch_writer_switch import open_live_bridge

ACTIVE_NATIVE = frozenset({"ready", "running", "todo", "blocked", "triage", "scheduled"})
DONE_NATIVE = frozenset({"done", "completed"})
TERMINAL_ORCH = frozenset({"completed", "failed", "cancelled"})


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
    children_total: int = 0
    children_done: int = 0
    children_all_done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TickResult:
    scanned: int = 0
    advanced: int = 0
    completed: int = 0
    skipped: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    results: list[JudgeResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "advanced": self.advanced,
            "completed": self.completed,
            "skipped": self.skipped,
            "errors": self.errors,
            "results": [r.to_dict() for r in self.results],
        }


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


def _list_open_board_only(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT r.orch_id, r.parent_task_id, r.lifecycle_state, r.lifecycle_revision, "
            "r.board_instance_id, r.tenant_scope, o.origin_kind "
            "FROM orch_requests r "
            "JOIN orch_origins o ON o.board_instance_id=r.board_instance_id "
            " AND o.tenant_scope=r.tenant_scope AND o.origin_id=r.origin_id "
            "WHERE o.origin_kind='board_only' "
            "AND r.lifecycle_state NOT IN ('completed','failed','cancelled') "
            "ORDER BY r.updated_at ASC"
        )
    )


def read_native_children(native_conn: sqlite3.Connection, parent_task_id: str) -> list[dict[str, str]]:
    """RO read child tasks via task_links (depends + spawned)."""
    has_links = native_conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='task_links'"
    ).fetchone()[0]
    if not has_links:
        return []
    rows = native_conn.execute(
        "SELECT l.child_id, COALESCE(t.status,''), COALESCE(l.kind,'depends') "
        "FROM task_links l LEFT JOIN tasks t ON t.id=l.child_id "
        "WHERE l.parent_id=?",
        (parent_task_id,),
    ).fetchall()
    return [{"child_id": r[0], "status": (r[1] or "").lower(), "kind": r[2] or "depends"} for r in rows]


def children_progress(children: list[dict[str, str]]) -> tuple[int, int, bool]:
    total = len(children)
    if total == 0:
        return 0, 0, False
    done = sum(1 for c in children if c["status"] in DONE_NATIVE)
    return total, done, done == total


def judge_board_only_once(
    conn: sqlite3.Connection,
    *,
    board_instance_id: str,
    tenant_scope: str,
    orch_id: str,
    native_status: str,
    children_all_done: bool = False,
    children_total: int = 0,
    children_done: int = 0,
) -> JudgeResult:
    """Apply at most one progressive transition based on native parent/children."""
    tenant = "" if tenant_scope is None else str(tenant_scope)
    row = _load_request(conn, board=board_instance_id, tenant=tenant, orch_id=orch_id)
    origin = row["origin_kind"]
    state = row["lifecycle_state"]
    parent = row["parent_task_id"]
    status = (native_status or "").strip().lower()
    parent_done = status in DONE_NATIVE
    work_done = parent_done or children_all_done

    base = dict(
        orch_id=orch_id,
        parent_task_id=parent,
        native_status=status,
        origin_kind=origin,
        before_state=state,
        after_state=state,
        steps=[],
        children_total=children_total,
        children_done=children_done,
        children_all_done=children_all_done,
    )

    if origin != "board_only":
        return JudgeResult(**base, skipped=True, reason="not_board_only")
    if state in TERMINAL_ORCH:
        return JudgeResult(**base, skipped=True, reason="already_terminal")

    try:
        if work_done:
            step = apply_lifecycle_transition_db(
                conn,
                board_instance_id=board_instance_id,
                tenant_scope=tenant,
                orch_id=orch_id,
                event="board_only_parent_done",
                to_state="completed",
                origin_kind="board_only",
                native_parent_done=parent_done,
                children_all_done=children_all_done,
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
            return JudgeResult(**base, skipped=True, reason=f"no_rule_for:{state}:{status}:children={children_done}/{children_total}")
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
        children_total=children_total,
        children_done=children_done,
        children_all_done=children_all_done,
    )


def judge_board_only_to_fixed_point(
    conn: sqlite3.Connection,
    *,
    board_instance_id: str,
    tenant_scope: str,
    orch_id: str,
    native_status: str,
    children_all_done: bool = False,
    children_total: int = 0,
    children_done: int = 0,
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
        children_total=children_total,
        children_done=children_done,
        children_all_done=children_all_done,
    )
    for _ in range(max_steps):
        one = judge_board_only_once(
            conn,
            board_instance_id=board_instance_id,
            tenant_scope=tenant,
            orch_id=orch_id,
            native_status=native_status,
            children_all_done=children_all_done,
            children_total=children_total,
            children_done=children_done,
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
                children_total=children_total,
                children_done=children_done,
                children_all_done=children_all_done,
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
            children_total=children_total,
            children_done=children_done,
            children_all_done=children_all_done,
        )
        if one.after_state in TERMINAL_ORCH:
            break
    return last


def _judge_with_bridge(br: OrchBridge, parent_task_id: str) -> JudgeResult:
    ref = br.read_native_task(parent_task_id)
    if ref is None:
        raise CMinError("native_parent_missing")
    children = read_native_children(br._native, parent_task_id)
    total, done, all_done = children_progress(children)
    row = _load_by_parent(br._sidecar, parent_task_id=parent_task_id)
    return judge_board_only_to_fixed_point(
        br._sidecar,
        board_instance_id=row["board_instance_id"],
        tenant_scope=row["tenant_scope"] or "",
        orch_id=row["orch_id"],
        native_status=ref.status,
        children_all_done=all_done,
        children_total=total,
        children_done=done,
    )


def live_judge_parent_task(parent_task_id: str) -> JudgeResult:
    """Open live bridge, RO-read native parent/children, write sidecar lifecycle only."""
    br: OrchBridge | None = None
    try:
        br = open_live_bridge()
        return _judge_with_bridge(br, parent_task_id)
    except BridgeError as exc:
        raise CMinError(f"bridge:{exc.code}") from exc
    finally:
        if br is not None:
            br.close()


def live_tick_once(*, limit: int = 100) -> TickResult:
    """Scan open board_only sidecar requests and advance from native truth."""
    br: OrchBridge | None = None
    out = TickResult()
    try:
        br = open_live_bridge()
        rows = _list_open_board_only(br._sidecar)[: max(0, int(limit))]
        out.scanned = len(rows)
        for row in rows:
            parent = row["parent_task_id"]
            try:
                res = _judge_with_bridge(br, parent)
                out.results.append(res)
                if res.skipped:
                    out.skipped += 1
                else:
                    out.advanced += 1
                    if res.after_state == "completed":
                        out.completed += 1
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", None) or f"{type(exc).__name__}:{exc}"
                out.errors.append({"parent_task_id": parent, "error": str(code)})
        return out
    except BridgeError as exc:
        raise CMinError(f"bridge:{exc.code}") from exc
    finally:
        if br is not None:
            br.close()


__all__ = [
    "CMinError",
    "JudgeResult",
    "JudgeStep",
    "TickResult",
    "judge_board_only_once",
    "judge_board_only_to_fixed_point",
    "live_judge_parent_task",
    "live_tick_once",
    "read_native_children",
    "children_progress",
]
