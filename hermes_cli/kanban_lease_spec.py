"""Atomic Kanban claims that return credential-free worker specifications."""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterable, Mapping
from typing import Any

from agent.crew_route_contract import Route, RouteContractError, resolve_route


class _LeaseReadbackError(RuntimeError):
    pass


def _error(code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"ok": False, "code": code, "error": message, **context}


def _capability_routes(
    values: Iterable[Mapping[str, Any] | Route],
) -> tuple[set[tuple[str, str, str]] | None, dict[str, Any] | None]:
    try:
        items = tuple(values)
    except TypeError:
        return None, _error(
            "invalid_worker_capabilities", "worker capabilities must be an iterable",
        )
    if not items:
        return None, _error(
            "missing_worker_capabilities",
            "at least one complete provider/model/effort capability is required",
        )
    routes: set[tuple[str, str, str]] = set()
    for index, value in enumerate(items):
        try:
            route = value if isinstance(value, Route) else resolve_route(value)
        except (RouteContractError, TypeError, AttributeError) as exc:
            return None, _error(
                "invalid_worker_capability",
                f"worker capability {index} is invalid: {exc}",
                capability_index=index,
            )
        routes.add(route.as_tuple())
    return routes, None


def _candidate_route(conn, row) -> tuple[Route | None, dict[str, Any] | None]:
    from hermes_cli.kanban_sunny import get_route_policy

    task_id = row["id"]
    required = (
        ("bucket_key", "missing_bucket_key", "ready task has no bucket key"),
        ("tenant", "missing_tenant", "ready task has no tenant/business binding"),
        ("workspace_path", "missing_workspace", "ready task has no workspace binding"),
    )
    for field, code, message in required:
        if not str(row[field] or "").strip():
            return None, _error(code, f"{message}; set it before leasing", task_id=task_id)
    if not row["route_policy_ref"] or row["route_policy_version"] is None:
        return None, _error(
            "missing_route_policy",
            "ready task has no complete route-policy reference; set ref and version before leasing",
            task_id=task_id,
        )
    policy = get_route_policy(
        conn, row["route_policy_ref"], row["route_policy_version"],
    )
    if policy is None:
        return None, _error(
            "unknown_route_policy", "task route-policy reference does not exist",
            task_id=task_id, route_policy_ref=row["route_policy_ref"],
            route_policy_version=row["route_policy_version"],
        )
    try:
        route = resolve_route(
            {"provider": policy.provider_ref, "model_id": policy.model_ref,
             "effort": policy.effort_ref}
        )
    except RouteContractError as exc:
        return None, _error(
            "invalid_route_policy", f"route policy is invalid: {exc}",
            task_id=task_id, route_policy_ref=policy.policy_ref,
            route_policy_version=policy.policy_version,
        )
    return route, None


def _verify_claim_readback(task, *, run_id: int, claim_token: str, candidate) -> None:
    fields = (
        "tenant", "workspace_path", "bucket_key", "route_policy_ref",
        "route_policy_version",
    )
    if (
        task is None
        or task.status != "running"
        or task.current_run_id != int(run_id)
        or task.claim_lock != claim_token
        or any(getattr(task, field) != candidate[field] for field in fields)
    ):
        raise _LeaseReadbackError("claimed lease identity could not be read back exactly")


def claim_lease_exec_spec(
    conn,
    *,
    worker_identity: str,
    worker_capabilities: Iterable[Mapping[str, Any] | Route],
    bucket_key: str | None = None,
    ttl_seconds: int | None = None,
    board: str | None = None,
) -> dict[str, Any]:
    """Claim at most one compatible ready task and return its exact lease identity."""
    from hermes_cli import kanban_db as kb

    worker = str(worker_identity or "").strip()
    if not worker:
        return _error("missing_worker_identity", "worker identity is required")
    capabilities, capability_error = _capability_routes(worker_capabilities)
    if capability_error is not None:
        return capability_error
    assert capabilities is not None
    bucket = str(bucket_key).strip() if bucket_key is not None else None
    if bucket_key is not None and not bucket:
        return _error("invalid_bucket_key", "bucket key cannot be empty")

    resolved_board = str(board or kb.get_current_board()).strip()
    if not resolved_board:
        return _error("missing_board", "board identity is required")
    now = int(time.time())
    ttl = kb._resolve_claim_ttl_seconds(ttl_seconds)
    expires = now + ttl
    claim_token = f"{worker}:{secrets.token_urlsafe(18)}"
    selected = selected_route = selected_run_id = None
    parents: list[str] = []
    attempt_number = 0
    try:
        with kb.write_txn(conn):
            query = "SELECT * FROM tasks WHERE status = 'ready' AND claim_lock IS NULL"
            params: list[Any] = []
            if bucket is not None:
                query += " AND bucket_key = ?"
                params.append(bucket)
            query += " ORDER BY priority DESC, created_at ASC, id ASC"
            for row in conn.execute(query, params).fetchall():
                route, route_error = _candidate_route(conn, row)
                if route_error is not None:
                    return route_error
                assert route is not None
                if route.as_tuple() not in capabilities:
                    continue
                if not kb._parents_satisfied(conn, row["id"]):
                    continue
                run_id = kb._claim_and_open_run(
                    conn, row["id"], "ready", claim_token, expires, now,
                    event_extra={"worker_identity": worker},
                )
                if run_id is None:
                    continue
                selected = kb.get_task(conn, row["id"])
                _verify_claim_readback(
                    selected, run_id=run_id, claim_token=claim_token, candidate=row,
                )
                selected_route = route
                selected_run_id = run_id
                attempt_number = int(conn.execute(
                    "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (row["id"],),
                ).fetchone()[0])
                parents = kb.parent_ids(conn, row["id"])
                break
            if selected is None:
                return _error("no_candidate", "no eligible ready task")
    except _LeaseReadbackError as exc:
        return _error("lease_readback_failed", str(exc))

    assert selected_route is not None and selected_run_id is not None
    kb._fire_kanban_lifecycle_hook(
        "kanban_task_claimed", selected.id, board=resolved_board,
        assignee=selected.assignee, run_id=selected_run_id,
    )
    route = {
        "provider": selected_route.provider,
        "model": selected_route.model_id,
        "requested_effort": selected_route.requested_effort,
        "applied_effort": selected_route.applied_effort,
    }
    return {
        "ok": True,
        "task_id": selected.id,
        "worker_identity": worker,
        "claim_token": claim_token,
        "claim_lock": claim_token,
        "run_id": int(selected_run_id),
        "claim_expires": expires,
        "heartbeat_at": now,
        "last_heartbeat_at": selected.last_heartbeat_at,
        "heartbeat_ttl_seconds": ttl,
        "heartbeat_interval_seconds": max(1, ttl // 3),
        "attempt_number": attempt_number,
        "run_number": attempt_number,
        "board": resolved_board,
        "tenant": selected.tenant,
        "business_scope": selected.tenant,
        "scope": {"tenant": selected.tenant, "business": selected.tenant},
        "project_id": selected.project_id,
        "workspace_kind": selected.workspace_kind,
        "workspace_path": selected.workspace_path,
        "branch_name": selected.branch_name,
        "project": selected.project_id,
        "workspace": selected.workspace_path,
        "branch": selected.branch_name,
        "parent_links": parents,
        "bucket_key": selected.bucket_key,
        "route_policy_ref": selected.route_policy_ref,
        "route_policy_version": selected.route_policy_version,
        "route_policy": {"ref": selected.route_policy_ref,
                         "version": selected.route_policy_version},
        "provider": selected_route.provider,
        "model": selected_route.model_id,
        "requested_effort": selected_route.requested_effort,
        "applied_effort": selected_route.applied_effort,
        "route": route,
    }
