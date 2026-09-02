"""Kanban-backed cross-profile delegation orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Optional

from agent.redact import redact_sensitive_text
from hermes_cli.capability_registry import ProfileCapability, find_capability
from hermes_cli.delegation_policy import (
    DelegationAction,
    DelegationRisk,
    PolicyDecision,
    ToolActionRequest,
    enforce_delegation_policy,
)
from hermes_cli.profiles import normalize_profile_name


@dataclass
class ProfileDelegationRequest:
    profile: Optional[str]
    task: str
    required_capability: str
    risk: str = "READ"
    requester_profile: str = "default"
    requester_session_key: Optional[str] = None
    requester_session_id: Optional[str] = None
    requester_platform: Optional[str] = None
    requester_chat_id: Optional[str] = None
    requester_thread_id: Optional[str] = None
    return_to: str = "current_session"
    timeout_seconds: int = 300
    max_runtime_seconds: int = 300
    board: Optional[str] = None
    idempotency_key: Optional[str] = None
    approval_id: Optional[str] = None
    tool_action: Optional[ToolActionRequest] = None
    max_concurrency: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.tool_action:
            data["tool_action"] = asdict(self.tool_action)
        return data


@dataclass
class ProfileDelegationResult:
    status: str
    delegation_id: str
    task_id: Optional[str]
    executor_profile: Optional[str]
    requester_profile: str
    capability: str
    risk: str
    result: Optional[dict[str, Any]] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    audit_ref: Optional[str] = None
    ranking: Optional[list[dict[str, Any]]] = None
    policy: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _idempotency_key(req: ProfileDelegationRequest, executor: str) -> str:
    if req.idempotency_key:
        return req.idempotency_key
    h = hashlib.sha256()
    h.update("\0".join([
        req.requester_session_key or "",
        req.requester_profile,
        executor,
        req.required_capability,
        " ".join(req.task.split()),
    ]).encode("utf-8"))
    return "profile-delegation:" + h.hexdigest()[:32]


def _candidate_rows(caps: list[ProfileCapability]) -> list[dict[str, Any]]:
    return [
        {
            "profile": c.profile,
            "capability": c.capability,
            "executable": c.executable,
            "enabled": c.enabled,
            "credential_present": c.credential_present,
            "credential_check": c.credential_check,
            "worker_available": c.worker_available,
            "gateway_running": c.gateway_running,
            "rank_score": c.rank_score,
            "rank_reasons": c.rank_reasons,
            "workload": c.workload.to_dict(),
            "notes": c.notes,
        }
        for c in caps
    ]


def select_executor(
    *,
    required_capability: str,
    requester_profile: str = "default",
    profile: Optional[str] = None,
    include_disabled: bool = True,
    max_concurrency: Optional[int] = None,
) -> tuple[Optional[ProfileCapability], list[dict[str, Any]], str]:
    query = find_capability(
        required_capability,
        requester_profile=requester_profile,
        include_disabled=include_disabled,
        max_concurrency=max_concurrency,
    )
    candidates = query.profiles
    if profile:
        target = normalize_profile_name(profile)
        matches = [c for c in candidates if c.profile == target]
        chosen = matches[0] if matches else None
        reason = "explicit_executor" if chosen else f"explicit executor {target!r} does not advertise {required_capability}"
        return chosen, _candidate_rows(candidates), reason
    executable = [c for c in candidates if c.executable and not c.workload.saturated]
    if executable:
        return executable[0], _candidate_rows(candidates), "auto_selected_workload_aware_executor"
    if candidates:
        return candidates[0], _candidate_rows(candidates), "best_candidate_not_currently_executable"
    return None, [], "no_candidate"


def build_delegation_worker_body(
    req: ProfileDelegationRequest,
    delegation_id: str,
    executor: str,
    capability_route: Optional[ProfileCapability] = None,
) -> str:
    route_source = capability_route.source if capability_route else "unknown"
    route_kind = capability_route.kind if capability_route else "unknown"
    composio_hint = ""
    if capability_route and capability_route.kind == "composio":
        composio_hint = (
            "\n## Capability route\n"
            f"Use Composio for this capability: {capability_route.source}. "
            "Do not try to enable or use the native Vercel MCP server for this delegated request.\n"
        )
    return f"""# INTERNAL PROFILE DELEGATION REQUEST

You are executing a delegated subtask for another Hermes executive profile.

## Delegation
- delegation_id: {delegation_id}
- requester_profile: {req.requester_profile}
- executor_profile: {executor}
- required_capability: {req.required_capability}
- risk: {req.risk}
- credential_export_allowed: false
- capability_route_kind: {route_kind}
- capability_route_source: {route_source}
{composio_hint}
## Task
{req.task}

## Hard rules
1. Run under your own profile only. Do not ask requester for credentials.
2. Do not reveal OAuth tokens, API keys, Authorization headers, cookies, or credential file contents.
3. For READ risk, inspect only. Do not deploy, delete, mutate config, change permissions, or spend money.
4. If the required capability is unavailable, block/complete with structured failure.
5. Finish using kanban_complete with metadata.profile_delegation exactly in this shape:

{{
  "delegation_id": "{delegation_id}",
  "capability": "{req.required_capability}",
  "risk": "{req.risk}",
  "status": "completed|failed|blocked_approval",
  "structured_result": {{ }},
  "redaction": {{"secrets_returned": false}}
}}
"""


def _safe_json_result(data: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(data, ensure_ascii=False, default=str)
    redacted = redact_sensitive_text(text, force=True)
    try:
        return json.loads(redacted)
    except Exception:
        return {"redacted_text": redacted}


def _extract_structured_result(run) -> tuple[dict[str, Any], str]:
    metadata = run.metadata or {}
    pd = metadata.get("profile_delegation") if isinstance(metadata, dict) else None
    if not isinstance(pd, dict):
        pd = {}
    if pd.get("redaction", {}).get("secrets_returned") is True:
        raise ValueError("Executor reported credential material in result; result withheld.")
    result = pd.get("structured_result") if isinstance(pd.get("structured_result"), dict) else {}
    if not result and run.summary:
        result = {"summary": run.summary}
    if run.error:
        result.setdefault("error", run.error)
    return _safe_json_result(result), run.summary or ""


def _emit_completion_event(result: ProfileDelegationResult, session_key: Optional[str]) -> None:
    if not session_key:
        return
    try:
        from tools.process_registry import process_registry
        process_registry.completion_queue.put({
            "type": "profile_delegation",
            "delegation_id": result.delegation_id,
            "task_id": result.task_id,
            "session_key": session_key,
            "requester_profile": result.requester_profile,
            "executor_profile": result.executor_profile,
            "capability": result.capability,
            "risk": result.risk,
            "status": result.status,
            "summary": result.summary,
            "result": result.result,
            "error": result.error,
            "completed_at": time.time(),
        })
    except Exception:
        pass


def delegate_to_profile(req: ProfileDelegationRequest, *, spawn_fn=None) -> ProfileDelegationResult:
    from hermes_cli import kanban_db as kb

    req.requester_profile = normalize_profile_name(req.requester_profile or "default")
    chosen, ranking, select_reason = select_executor(
        required_capability=req.required_capability,
        requester_profile=req.requester_profile,
        profile=req.profile,
        include_disabled=True,
        max_concurrency=req.max_concurrency,
    )
    delegation_id = "pd_" + uuid.uuid4().hex[:12]

    if not chosen:
        return ProfileDelegationResult(
            status="failed", delegation_id=delegation_id, task_id=None,
            executor_profile=None, requester_profile=req.requester_profile,
            capability=req.required_capability, risk=req.risk,
            error=f"No executor candidate found for {req.required_capability} ({select_reason}).",
            ranking=ranking,
        )

    executor = chosen.profile
    action = DelegationAction(
        requester_profile=req.requester_profile,
        executor_profile=executor,
        task=req.task,
        required_capability=req.required_capability,
        requested_risk=DelegationRisk(req.risk) if req.risk in DelegationRisk._value2member_map_ else DelegationRisk.READ,
        tool_action=req.tool_action,
        approval_id=req.approval_id,
    )
    decision: PolicyDecision = enforce_delegation_policy(action)
    req.risk = decision.risk.value
    if not decision.allowed:
        return ProfileDelegationResult(
            status="blocked_approval" if decision.approval_required else "denied",
            delegation_id=delegation_id, task_id=None, executor_profile=executor,
            requester_profile=req.requester_profile, capability=req.required_capability,
            risk=decision.risk.value, error=decision.reason, ranking=ranking,
            policy=decision.to_dict(),
        )

    if not chosen.executable:
        reasons = []
        if not chosen.enabled:
            reasons.append("capability is disabled")
        if not chosen.worker_available:
            reasons.append("executor profile is not worker-available")
        if chosen.workload.saturated:
            reasons.append("executor is at max concurrency")
        return ProfileDelegationResult(
            status="failed", delegation_id=delegation_id, task_id=None,
            executor_profile=executor, requester_profile=req.requester_profile,
            capability=req.required_capability, risk=decision.risk.value,
            error=f"Executor {executor} cannot currently run {req.required_capability}: {', '.join(reasons) or select_reason}.",
            ranking=ranking, policy=decision.to_dict(),
        )

    with kb.connect_closing(board=req.board) as conn:
        idem = _idempotency_key(req, executor)
        task_id = kb.create_task(
            conn,
            title=f"[internal delegation] {req.requester_profile} → {executor}: {req.required_capability}",
            body=build_delegation_worker_body(req, delegation_id, executor, chosen),
            assignee=executor,
            created_by=req.requester_profile,
            tenant=f"profile-delegation:{req.requester_profile}",
            priority=100,
            idempotency_key=idem,
            max_runtime_seconds=req.max_runtime_seconds,
            max_retries=2,
            initial_status="blocked",
            session_id=req.requester_session_id,
            board=req.board,
        )
        # create_task only allows running/blocked as initial states; promote
        # after the audit row exists so the dispatcher sees a ready card.
        kb.promote_task(conn, task_id, actor=req.requester_profile, reason="profile delegation dispatch", force=True)
        kb.create_profile_delegation(
            conn,
            delegation_id=delegation_id,
            task_id=task_id,
            requester_profile=req.requester_profile,
            executor_profile=executor,
            requester_session_key=req.requester_session_key,
            requester_session_id=req.requester_session_id,
            requester_platform=req.requester_platform,
            requester_chat_id=req.requester_chat_id,
            requester_thread_id=req.requester_thread_id,
            capability=req.required_capability,
            risk=decision.risk.value,
            request=req.to_dict(),
            approval_id=req.approval_id,
        )
        conn.commit()
        kb.dispatch_once(conn, spawn_fn=spawn_fn, max_spawn=1, max_in_progress_per_profile=req.max_concurrency, board=req.board)
        deadline = time.time() + max(0, min(int(req.timeout_seconds), 900))
        while time.time() <= deadline:
            task = kb.get_task(conn, task_id)
            if task and task.status == "running":
                kb.mark_profile_delegation_running(conn, delegation_id)
                conn.commit()
            if task and task.status in {"done", "blocked"}:
                runs = kb.list_runs(conn, task_id, include_active=False)
                latest = runs[-1] if runs else None
                if task.status == "done" and latest:
                    structured, summary = _extract_structured_result(latest)
                    kb.complete_profile_delegation(conn, delegation_id, result=structured)
                    conn.commit()
                    result = ProfileDelegationResult(
                        status="completed", delegation_id=delegation_id, task_id=task_id,
                        executor_profile=executor, requester_profile=req.requester_profile,
                        capability=req.required_capability, risk=decision.risk.value,
                        result=structured, summary=summary, audit_ref=f"profile_delegations:{delegation_id}",
                        ranking=ranking, policy=decision.to_dict(),
                    )
                    return result
                error = (latest.error or latest.summary) if latest else (task.result or "delegated task blocked")
                kb.fail_profile_delegation(conn, delegation_id, status="failed", error=error or "delegated task blocked")
                conn.commit()
                return ProfileDelegationResult(
                    status="failed", delegation_id=delegation_id, task_id=task_id,
                    executor_profile=executor, requester_profile=req.requester_profile,
                    capability=req.required_capability, risk=decision.risk.value,
                    error=error, audit_ref=f"profile_delegations:{delegation_id}",
                    ranking=ranking, policy=decision.to_dict(),
                )
            time.sleep(0.2)
        result = ProfileDelegationResult(
            status="queued", delegation_id=delegation_id, task_id=task_id,
            executor_profile=executor, requester_profile=req.requester_profile,
            capability=req.required_capability, risk=decision.risk.value,
            summary="Delegation is queued/running; result will be returned via session continuation when available.",
            audit_ref=f"profile_delegations:{delegation_id}", ranking=ranking,
            policy=decision.to_dict(),
        )
        return result
