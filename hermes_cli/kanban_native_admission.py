"""Kanban-native work creation/admission helpers.

This module owns the first Linear-free admission frontier for the Hermes Work
Ledger. It writes native Kanban tasks with a separate public id (``PP-NNN``)
while preserving fail-closed admission rules and dry-run/no-dispatch safety.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb

DEFAULT_NATIVE_NAMESPACE = "HL"
NATIVE_IDEMPOTENCY_PREFIX = "kanban:"
_NAMESPACE_RE = re.compile(r"^[A-Z]{2}$")
_PUBLIC_ID_RE = re.compile(r"^[A-Z]{2}-\d{3,}$")


@dataclass(frozen=True)
class NativeAdmissionRequest:
    title: str
    tenant: str
    repo_full_name: str
    profile: str
    closeout_policy: str
    executor: str
    namespace: str = DEFAULT_NATIVE_NAMESPACE
    body: Optional[str] = None
    created_by: str = "kanban-native-admission"
    workspace_kind: str = "worktree"
    workspace_path: Optional[str] = None
    priority: int = 0
    parents: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    base_branch: Optional[str] = None
    worktree_branch: Optional[str] = None
    approval_boundary: str = "human_approval_required"
    public_id: Optional[str] = None
    idempotency_key: Optional[str] = None


def normalize_namespace(namespace: str) -> str:
    ns = (namespace or "").strip().upper()
    if not _NAMESPACE_RE.match(ns):
        raise ValueError("namespace must be exactly two uppercase letters")
    if ns == "CH":
        raise ValueError("CH is reserved for Linear legacy references and cannot be used for native work")
    return ns


def normalize_public_id(public_id: str) -> str:
    pid = (public_id or "").strip().upper()
    if not _PUBLIC_ID_RE.match(pid):
        raise ValueError("public_id must use PP-NNN format")
    normalize_namespace(pid.split("-", 1)[0])
    return pid


def _missing_required(req: NativeAdmissionRequest) -> list[str]:
    checks = {
        "title": req.title,
        "tenant": req.tenant,
        "repo_full_name": req.repo_full_name,
        "profile": req.profile,
        "closeout_policy": req.closeout_policy,
        "executor": req.executor,
    }
    return [name for name, value in checks.items() if not str(value or "").strip()]


def native_idempotency_key(public_id: str) -> str:
    return f"{NATIVE_IDEMPOTENCY_PREFIX}{normalize_public_id(public_id)}"


def next_public_id(conn: Any, namespace: str = DEFAULT_NATIVE_NAMESPACE) -> str:
    ns = normalize_namespace(namespace)
    rows = conn.execute(
        "SELECT public_id FROM tasks WHERE public_id LIKE ?",
        (f"{ns}-%",),
    ).fetchall()
    max_seen = 0
    for row in rows:
        value = str(row["public_id"] or "")
        if not value.startswith(f"{ns}-"):
            continue
        suffix = value.split("-", 1)[1]
        if suffix.isdigit():
            max_seen = max(max_seen, int(suffix))
    return f"{ns}-{max_seen + 1:03d}"


def existing_native_task_id(conn: Any, public_id: str) -> Optional[str]:
    pid = normalize_public_id(public_id)
    row = conn.execute(
        "SELECT id FROM tasks WHERE public_id = ? AND status != 'archived' "
        "ORDER BY created_at DESC LIMIT 1",
        (pid,),
    ).fetchone()
    return row["id"] if row else None


def build_native_body(req: NativeAdmissionRequest, public_id: str, idempotency_key: str) -> str:
    payload = {
        "source": "kanban_native",
        "public_id": public_id,
        "idempotency_key": idempotency_key,
        "tenant": req.tenant.strip(),
        "legacy_refs": [],
        "admission": {
            "mode": "native_dry_run_or_triage_admission",
            "linear_required": False,
            "executor_dispatch": "forbidden_during_admission",
            "approval_boundary": req.approval_boundary,
        },
        "repo_intent": {
            "repo_full_name": req.repo_full_name.strip(),
            "base_branch": (req.base_branch or "").strip() or None,
            "worktree_branch": (req.worktree_branch or "").strip() or None,
            "workspace_kind": req.workspace_kind,
            "workspace_path": req.workspace_path,
        },
        "execution_hints": {
            "executor": req.executor.strip(),
            "profile": req.profile.strip(),
            "skills": list(req.skills),
        },
        "routing": {
            "verdict": req.executor.strip(),
            "reason": "native admission captures requested executor; dispatch remains forbidden until live preflight",
            "approval_boundary": req.approval_boundary,
        },
        "closeout": {
            "policy": req.closeout_policy.strip(),
            "worker_done_review_ready_closed_are_distinct": True,
        },
        "parents": list(req.parents),
    }
    parts = [
        f"Kanban-native work item `{public_id}`.",
        "",
        "Linear source issue is not required. This admission record is safe-by-default: "
        "it creates a triage ledger row and does not dispatch an executor.",
        "",
        "```json source_payload",
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
    ]
    if req.body:
        parts.extend(["", "## Opening context", req.body.strip()])
    return "\n".join(parts).strip() + "\n"


def build_native_admission_payload(
    conn: Any,
    req: NativeAdmissionRequest,
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    missing = _missing_required(req)
    namespace = normalize_namespace(req.namespace)
    public_id = normalize_public_id(req.public_id) if req.public_id else next_public_id(conn, namespace)
    idempotency_key = req.idempotency_key or native_idempotency_key(public_id)
    existing_id = existing_native_task_id(conn, public_id)
    body = build_native_body(req, public_id, idempotency_key)
    status = "blocked" if missing else "would_create"
    return {
        "status": status,
        "reason": "native_admission_contract_ready" if not missing else "native_admission_missing_required_fields",
        "dry_run": dry_run,
        "missing": missing,
        "public_id": public_id,
        "task_id": existing_id,
        "created": False,
        "task": {
            "title": f"{public_id} — {req.title.strip()}",
            "body": body,
            "tenant": req.tenant.strip() or None,
            "public_id": public_id,
            "idempotency_key": idempotency_key,
            "workspace_kind": req.workspace_kind,
            "workspace_path": req.workspace_path,
            "status": "triage",
            "assignee": None,
            "priority": req.priority,
            "parents": list(req.parents),
            "skills": list(req.skills),
        },
        "repo_intent": {
            "repo_full_name": req.repo_full_name.strip() or None,
            "base_branch": (req.base_branch or "").strip() or None,
            "worktree_branch": (req.worktree_branch or "").strip() or None,
        },
        "execution_hints": {
            "executor": req.executor.strip() or None,
            "profile": req.profile.strip() or None,
            "skills": list(req.skills),
        },
        "authority": {
            "review_phase": None,
            "routing_verdict": {
                "verdict": "Hermes direct" if req.executor.strip() == "hermes-direct" else "blocked",
                "reason": "native admission records routing intent only; executor dispatch is forbidden during admission",
                "boundary": req.approval_boundary,
            },
            "admission_snapshot": {
                "source": "kanban_native",
                "linear_required": False,
                "public_id": public_id,
                "idempotency_key": idempotency_key,
                "tenant": req.tenant.strip() or None,
                "repo_full_name": req.repo_full_name.strip() or None,
                "profile": req.profile.strip() or None,
                "executor": req.executor.strip() or None,
                "closeout_policy": req.closeout_policy.strip() or None,
                "executor_dispatch": "forbidden_during_admission",
            },
            "closeout_evidence": {
                "policy": req.closeout_policy.strip() or None,
                "worker_done_review_ready_closed_are_distinct": True,
                "evidence_status": "not_started",
            },
        },
        "side_effects": {
            "kanban_task_written": False,
            "executor_spawned": False,
            "linear_required": False,
            "linear_mutated": False,
        },
    }


def create_native_work(
    conn: Any,
    req: NativeAdmissionRequest,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    payload = build_native_admission_payload(conn, req, dry_run=dry_run)
    if payload["missing"]:
        return payload
    if dry_run:
        return payload
    if payload["task_id"]:
        payload.update(
            {
                "status": "reused",
                "reason": "native_admission_existing_public_id",
                "dry_run": False,
                "created": False,
            }
        )
        return payload
    task = payload["task"]
    task_id = kb.create_task(
        conn,
        title=task["title"],
        body=task["body"],
        assignee=None,
        created_by=req.created_by,
        workspace_kind=task["workspace_kind"],
        workspace_path=task["workspace_path"],
        tenant=task["tenant"],
        priority=task["priority"],
        parents=tuple(task["parents"]),
        triage=True,
        idempotency_key=task["idempotency_key"],
        skills=task["skills"] or None,
        public_id=task["public_id"],
        review_phase=payload["authority"]["review_phase"],
        routing_verdict=payload["authority"]["routing_verdict"],
        admission_snapshot=payload["authority"]["admission_snapshot"],
        closeout_evidence=payload["authority"]["closeout_evidence"],
    )
    payload.update(
        {
            "status": "created",
            "reason": "native_admission_created_triage_task",
            "dry_run": False,
            "created": True,
            "task_id": task_id,
        }
    )
    payload["side_effects"]["kanban_task_written"] = True
    return payload
