"""ORCH V4 typed API surface (isolated DB / sidecar).

Provides durable request bootstrap + parent bind + DB-backed lifecycle CAS.
Does not touch live fleet DBs unless allow_live=True is explicit.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any

from hermes_cli.kanban_orch_capability import CapabilityGrant, get_context
from hermes_cli.kanban_orch_db import OrchDBError, begin_immediate, grant
from hermes_cli.kanban_orch_lifecycle import apply_transition, Request, LifecycleError


class OrchAPIError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _hex64(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _now() -> int:
    return int(time.time())


@dataclass(frozen=True)
class BoundParent:
    board_instance_id: str
    tenant_scope: str
    orch_id: str
    origin_id: str
    parent_task_id: str
    selector_key: str
    request_key: str
    request_digest: str
    lifecycle_state: str


def _require_ctx(conn: sqlite3.Connection):
    ctx = get_context(conn)
    if ctx is None:
        raise OrchAPIError("capability_context_missing")
    return ctx


def ensure_board_identity(
    conn: sqlite3.Connection,
    *,
    board_instance_id: str,
    canonical_board_key: str,
) -> None:
    grant(conn, CapabilityGrant(kind="maintenance_identity", board=board_instance_id, target_key=canonical_board_key))
    conn.execute(
        "INSERT OR IGNORE INTO kanban_board_identity "
        "(singleton, board_instance_id, canonical_board_key, created_at) "
        "VALUES (1, ?, ?, ?)",
        (board_instance_id, canonical_board_key, _now()),
    )


def bootstrap_board_only_request(
    conn: sqlite3.Connection,
    *,
    board_instance_id: str,
    tenant_scope: str,
    parent_task_id: str,
    orch_id: str | None = None,
    lineage_id: str | None = None,
    title: str = "orch-request",
) -> BoundParent:
    """Create selector+origin+request bound to parent_task_id (board_only)."""
    _require_ctx(conn)
    if type(parent_task_id) is not str or not parent_task_id:
        raise OrchAPIError("invalid_parent_task_id")
    tenant = "" if tenant_scope is None else str(tenant_scope)
    oid = orch_id or f"orch-{uuid.uuid4().hex[:16]}"
    lin = lineage_id or f"lin-{uuid.uuid4().hex[:16]}"
    origin_id = f"origin-{uuid.uuid4().hex[:12]}"
    selector_value = f"bind:{parent_task_id}:{oid}"
    selector_key = _hex64(board_instance_id, tenant, selector_value)
    from hermes_cli.kanban_orch_digest_udf import build_route_json_and_digest

    route_json, route_digest = build_route_json_and_digest(
        origin_kind="board_only",
        platform="local",
        adapter_instance_id="bridge",
        account_id="local",
        conversation_id="local",
        thread_id="",
        reply_to_id="",
        session_id="",
        notifier_profile="",
        required_ack_family="none",
        required_ack_strength="none",
        route_revision=1,
    )
    request_key = _hex64("request", board_instance_id, tenant, oid, parent_task_id)
    request_obj = {
        "schema_version": 4,
        "kind": "orch_request",
        "selector_key": selector_key,
        "request_key": request_key,
        "origin_id": origin_id,
        "lineage_id": lin,
        "generation": 1,
        "title": title,
        "synthesis_strategy": "parent_owned",
        "completion_policy": "board_only",
        "requirements": [],
    }
    request_json = json.dumps(request_obj, separators=(",", ":"), sort_keys=True)
    # Server recomputes request digest; caller cannot forge identity.
    from hermes_cli.kanban_orch_canonical import request_digest as recompute_request_digest

    request_digest = recompute_request_digest(request_obj)
    now = _now()

    begin_immediate(conn)
    try:
        ensure_board_identity(
            conn,
            board_instance_id=board_instance_id,
            canonical_board_key=f"orch-{board_instance_id[-12:]}",
        )
        grant(conn, CapabilityGrant(kind="selector_create", board=board_instance_id, tenant=tenant, object_id=lin, target_key=selector_key))
        grant(conn, CapabilityGrant(kind="selector_advance", board=board_instance_id, tenant=tenant, object_id=lin, revision=0, epoch=1, target_key=selector_key))
        grant(conn, CapabilityGrant(kind="origin_register", board=board_instance_id, tenant=tenant, object_id=origin_id, revision=1, target_key=route_digest))
        # Insert uses request_submit(kind,..., selector_ledger_revision, generation, request_key)
        grant(
            conn,
            CapabilityGrant(
                kind="request_submit",
                board=board_instance_id,
                tenant=tenant,
                object_id=oid,
                revision=0,
                epoch=1,
                target_key=request_key,
            ),
        )
        grant(conn, CapabilityGrant(kind="request_transition", board=board_instance_id, tenant=tenant, object_id=oid, target_key="*"))

        # Soft-FK mirror row required by deferred FK on orch_requests.parent_task_id.
        conn.execute(
            "INSERT OR IGNORE INTO _soft_fk_tasks (id, title, status, created_at) "
            "VALUES (?, ?, 'pending', ?)",
            (parent_task_id, f"soft:{parent_task_id}", now),
        )
        # Insert empty selector generation 0 (contract create gate).
        conn.execute(
            "INSERT INTO orch_replay_selectors ("
            " selector_key, board_instance_id, tenant_scope, selector_kind, selector_value,"
            " adapter_instance_id, conversation_id, lineage_id, current_generation,"
            " current_orch_id, current_request_digest, ledger_revision, created_at, updated_at"
            ") VALUES (?, ?, ?, 'event', ?, 'bridge', 'local', ?, 0, NULL, NULL, 0, ?, ?)",
            (selector_key, board_instance_id, tenant, selector_value, lin, now, now),
        )
        conn.execute(
            "INSERT INTO orch_origins ("
            " board_instance_id, tenant_scope, origin_id, schema_version, selector_key,"
            " origin_kind, platform, adapter_instance_id, account_id, conversation_id,"
            " selector_kind, selector_value, thread_id, reply_to_id, session_id,"
            " notifier_profile, route_revision, route_json, route_digest,"
            " required_ack_family, required_ack_strength, created_at"
            ") VALUES (?, ?, ?, 4, ?, 'board_only', 'local', 'bridge', 'local', 'local',"
            " 'event', ?, '', '', '', '', 1, ?, ?, 'none', 'none', ?)",
            (board_instance_id, tenant, origin_id, selector_key, selector_value, route_json, route_digest, now),
        )
        conn.execute(
            "INSERT INTO orch_requests ("
            " board_instance_id, tenant_scope, orch_id, lineage_id, generation,"
            " selector_key, selector_ledger_revision, request_key, request_schema_version,"
            " request_json, request_digest, origin_id, parent_task_id,"
            " lifecycle_state, lifecycle_revision, cancel_epoch,"
            " delivery_epoch_revision, plan_epoch_revision, plan_version,"
            " synthesis_strategy, max_retries, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, 1, ?, 0, ?, 4, ?, ?, ?, ?, 'submitted', 0, 0, 0, 0, 0,"
            " 'parent_owned', 0, ?, ?)",
            (
                board_instance_id,
                tenant,
                oid,
                lin,
                selector_key,
                request_key,
                request_json,
                request_digest,
                origin_id,
                parent_task_id,
                now,
                now,
            ),
        )
        # Advance selector to generation 1 bound to this request.
        conn.execute(
            "UPDATE orch_replay_selectors SET current_generation=1, current_orch_id=?, "
            "current_request_digest=?, ledger_revision=1, updated_at=? "
            "WHERE board_instance_id=? AND tenant_scope=? AND selector_key=?",
            (oid, request_digest, now, board_instance_id, tenant, selector_key),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return BoundParent(
        board_instance_id=board_instance_id,
        tenant_scope=tenant,
        orch_id=oid,
        origin_id=origin_id,
        parent_task_id=parent_task_id,
        selector_key=selector_key,
        request_key=request_key,
        request_digest=request_digest,
        lifecycle_state="submitted",
    )


def apply_lifecycle_transition_db(
    conn: sqlite3.Connection,
    *,
    board_instance_id: str,
    tenant_scope: str,
    orch_id: str,
    event: str,
    to_state: str,
    resume_state: str | None = None,
    require_evidence: bool = True,
    accepted_required_lanes: int | None = None,
    has_result: bool | None = None,
    delivery_satisfied: bool | None = None,
    origin_kind: str | None = None,
    retry_budget_remaining: int | None = None,
    native_parent_done: bool | None = None,
    children_all_done: bool | None = None,
) -> dict[str, Any]:
    """DB-backed lifecycle CAS using pure transition table + SQL revision fence."""
    tenant = "" if tenant_scope is None else str(tenant_scope)
    begin_immediate(conn)
    try:
        row = conn.execute(
            "SELECT lifecycle_state, lifecycle_revision, cancel_epoch, generation "
            "FROM orch_requests WHERE board_instance_id=? AND tenant_scope=? AND orch_id=?",
            (board_instance_id, tenant, orch_id),
        ).fetchone()
        if row is None:
            raise OrchAPIError("request_not_found")
        req = Request(
            board=board_instance_id,
            tenant=tenant,
            orch_id=orch_id,
            state=row["lifecycle_state"],
            lifecycle_revision=int(row["lifecycle_revision"]),
            cancel_epoch=int(row["cancel_epoch"]),
            generation=int(row["generation"]),
        )
        nxt = apply_transition(
            req,
            event,
            to_state,
            resume_state=resume_state,
            require_evidence=require_evidence,
            accepted_required_lanes=accepted_required_lanes,
            has_result=has_result,
            delivery_satisfied=delivery_satisfied,
            origin_kind=origin_kind,
            retry_budget_remaining=retry_budget_remaining,
            native_parent_done=native_parent_done,
            children_all_done=children_all_done,
        )
        grant(
            conn,
            CapabilityGrant(
                kind="request_transition",
                board=board_instance_id,
                tenant=tenant,
                object_id=orch_id,
                revision=nxt.lifecycle_revision,
                epoch=nxt.cancel_epoch,
                target_key="*",
            ),
        )
        # Build UPDATE carefully so multi-lane plan/delivery epochs satisfy SQL fences.
        now = _now()
        sets = [
            "lifecycle_state=?",
            "lifecycle_revision=?",
            "cancel_epoch=?",
            "resume_state=?",
            "blocked_from_state=?",
            "block_revision=?",
            "updated_at=?",
        ]
        vals: list[Any] = [
            nxt.state,
            nxt.lifecycle_revision,
            nxt.cancel_epoch,
            nxt.resume_state,
            nxt.blocked_from_state,
            nxt.block_revision,
            now,
        ]
        if event == "valid_plan_materialized" and to_state == "waiting_lanes":
            sets.append("plan_version=plan_version+1")
            sets.append("plan_epoch_revision=?")
            vals.append(nxt.lifecycle_revision)
        if event == "result_accepted" and to_state == "work_accepted":
            sets.append("delivery_epoch_revision=?")
            sets.append("work_accepted_at=COALESCE(work_accepted_at, ?)")
            vals.extend([nxt.lifecycle_revision, now])
        if nxt.state == "completed":
            sets.append("delivery_closed_at=COALESCE(delivery_closed_at, ?)")
            sets.append("work_accepted_at=COALESCE(work_accepted_at, ?)")
            sets.append("terminal_reason_code=COALESCE(terminal_reason_code, ?)")
            reason = "board_only_parent_done" if event == "board_only_parent_done" else "completed"
            vals.extend([now, now, reason])
        elif nxt.state in {"failed", "cancelled"}:
            sets.append("terminal_reason_code=COALESCE(terminal_reason_code, ?)")
            vals.append(event)

        sql = (
            "UPDATE orch_requests SET "
            + ", ".join(sets)
            + " WHERE board_instance_id=? AND tenant_scope=? AND orch_id=? AND lifecycle_revision=?"
        )
        vals.extend([board_instance_id, tenant, orch_id, req.lifecycle_revision])
        cur = conn.execute(sql, vals)
        if cur.rowcount != 1:
            raise OrchAPIError("lifecycle_cas_conflict")
        conn.commit()
        return {
            "orch_id": orch_id,
            "from_state": req.state,
            "to_state": nxt.state,
            "lifecycle_revision": nxt.lifecycle_revision,
            "cancel_epoch": nxt.cancel_epoch,
            "event": event,
        }
    except Exception:
        conn.rollback()
        raise


__all__ = [
    "OrchAPIError",
    "BoundParent",
    "ensure_board_identity",
    "bootstrap_board_only_request",
    "apply_lifecycle_transition_db",
]
