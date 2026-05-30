"""Dev-first launch profiles and execution plans backed by Hermes state.db."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from gateway.dev_worker_runtimes import (
    DEFAULT_RUNTIME,
    WorkerRuntimeError,
    WorkerRuntimeRouter,
    list_worker_runtimes,
    normalize_runtime,
)
from gateway.dev_control.runtime_policy_evidence import latest_runtime_policy_evidence
from gateway.dev_control.runtime_selection import select_worker_runtime
from gateway.dev_control.worker_output_contract import (
    append_worker_output_contract,
    output_contract_fields_from_event,
    parse_worker_output_contract,
    worker_output_contract_score,
)
from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


DEFAULT_AGENT = "codex"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_OPENHANDS_MODEL = "openrouter/openai/gpt-4o-mini"
DEFAULT_REASONING_EFFORT = "medium"
TASK_RUNNING_STATUSES = {"queued", "spawning", "working", "running", "active", "started"}
TASK_COMPLETED_STATUSES = {"done", "merged", "completed", "complete", "success", "succeeded"}
TASK_FAILED_STATUSES = {"killed", "errored", "error", "terminated", "failed", "cancelled", "canceled"}
PLAN_STATUSES = {
    "planned",
    "launched",
    "running",
    "partially_completed",
    "completed",
    "failed",
    "needs_review",
}
SUPERVISOR_APPROVAL_TTL_SECONDS = 24 * 60 * 60
APPROVABLE_SUPERVISOR_ACTIONS = {"retry", "repair_retry", "reassign"}
DEV_TEST_STATES = {"completed_ok", "completed_weak", "failed_repairable", "failed_unrepairable", "running"}
DEFAULT_POLICY_PROFILE = "standard"
DEV_PLAN_DRAFT_STATUSES = {"draft", "revision_requested", "approved_for_launch", "cancelled"}
DEFAULT_SUPERVISOR_INTERVAL_SECONDS = 60
DEFAULT_SUPERVISOR_LIMIT = 10
MIN_SUPERVISOR_INTERVAL_SECONDS = 15
MAX_SUPERVISOR_INTERVAL_SECONDS = 3600
BUILTIN_POLICY_PROFILES: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "policy_profile": "conservative",
        "auto_accept": True,
        "auto_follow_up": False,
        "max_follow_ups_per_task": 0,
        "max_retries_per_task": 0,
        "auto_retry": False,
        "auto_repair_retry": False,
        "auto_reassign": False,
        "approval_required_for": ["retry", "repair_retry", "reassign"],
    },
    "standard": {
        "policy_profile": "standard",
        "auto_accept": True,
        "auto_follow_up": True,
        "max_follow_ups_per_task": 1,
        "max_retries_per_task": 0,
        "auto_retry": False,
        "auto_repair_retry": False,
        "auto_reassign": False,
        "approval_required_for": ["retry", "repair_retry", "reassign"],
    },
    "aggressive": {
        "policy_profile": "aggressive",
        "auto_accept": True,
        "auto_follow_up": True,
        "max_follow_ups_per_task": 2,
        "max_retries_per_task": 0,
        "auto_retry": False,
        "auto_repair_retry": False,
        "auto_reassign": False,
        "approval_required_for": ["retry", "repair_retry", "reassign"],
    },
}
WEAK_SUMMARY_PATTERNS = (
    "verification gap",
    "cannot confirm",
    "can't confirm",
    "could not confirm",
    "did not produce",
    "partial exploration",
    "partial inspection",
    "still searching",
    "unclear",
    "no definitive answer",
)
PROMPT_ECHO_SUMMARY_PATTERNS = (
    "hermes ao delegation contract",
    "canonical contract:",
    "coding-worker-contract.md",
    "## task brief",
    "## required actions",
    "## final summary requirement",
    "files to inspect:",
    "your task:",
    "output format:",
    "dev launch profile",
    "permissions: read_only",
    "## additional instructions",
    "before reporting completion, run:",
    "in your final summary",
    "return exactly this marker",
    "include exactly this marker",
    "your final summary must:",
    "your final summary must contain",
)


DEFAULT_LAUNCH_PROFILES: list[Dict[str, Any]] = [
    {
        "id": "workspace.inspect",
        "name": "Workspace Inspect",
        "runtime": DEFAULT_RUNTIME,
        "project_id": "OrynWorkspace",
        "agent": DEFAULT_AGENT,
        "model": DEFAULT_MODEL,
        "reasoning_effort": "medium",
        "permissions": "read_only",
        "contract": "Read-only inspection. Do not edit files, create branches, run builds, or open PRs unless explicitly asked.",
    },
    {
        "id": "workspace.openhands.inspect",
        "name": "Workspace OpenHands Inspect",
        "runtime": "openhands",
        "project_id": "OrynWorkspace",
        "agent": "openhands",
        "model": DEFAULT_OPENHANDS_MODEL,
        "reasoning_effort": "medium",
        "permissions": "read_only",
        "contract": "Experimental OpenHands read-only inspection. Do not edit files, create branches, run builds, or open PRs unless explicitly asked.",
    },
    {
        "id": "workspace.implement",
        "name": "Workspace Implement",
        "runtime": DEFAULT_RUNTIME,
        "project_id": "OrynWorkspace",
        "agent": DEFAULT_AGENT,
        "model": DEFAULT_MODEL,
        "reasoning_effort": "high",
        "permissions": "edit",
        "contract": "Implement scoped Oryn Workspace changes. Keep edits focused and verify with the relevant build or tests.",
    },
    {
        "id": "workspace.test",
        "name": "Workspace Test",
        "runtime": DEFAULT_RUNTIME,
        "project_id": "OrynWorkspace",
        "agent": DEFAULT_AGENT,
        "model": DEFAULT_MODEL,
        "reasoning_effort": "medium",
        "permissions": "verify",
        "contract": "Run verification, inspect failures, and report concrete evidence. Edit only when the task explicitly asks for a fix.",
    },
    {
        "id": "platform.inspect",
        "name": "Platform Inspect",
        "runtime": DEFAULT_RUNTIME,
        "project_id": "OrynPlatform",
        "agent": DEFAULT_AGENT,
        "model": DEFAULT_MODEL,
        "reasoning_effort": "medium",
        "permissions": "read_only",
        "contract": "Read-only inspection for Hermes/Oryn platform code. Return file-backed findings and avoid implementation changes.",
    },
    {
        "id": "platform.implement",
        "name": "Platform Implement",
        "runtime": DEFAULT_RUNTIME,
        "project_id": "OrynPlatform",
        "agent": DEFAULT_AGENT,
        "model": DEFAULT_MODEL,
        "reasoning_effort": "high",
        "permissions": "edit",
        "contract": "Implement scoped Hermes/Oryn platform changes. Keep compatibility with existing gateway and AO behavior.",
    },
    {
        "id": "review",
        "name": "Review",
        "runtime": DEFAULT_RUNTIME,
        "project_id": None,
        "agent": DEFAULT_AGENT,
        "model": DEFAULT_MODEL,
        "reasoning_effort": "high",
        "permissions": "review_only",
        "contract": "Review-only worker. Prioritize bugs, regressions, missing tests, and concrete file/line evidence.",
    },
]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_execution_plans (
    plan_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    vision_brief TEXT,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS dev_execution_plan_tasks (
    task_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    goal TEXT NOT NULL,
    prompt TEXT NOT NULL,
    profile_id TEXT,
    project_id TEXT,
    dependencies TEXT,
    acceptance_criteria TEXT,
    ao_session_id TEXT,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    payload TEXT,
    FOREIGN KEY(plan_id) REFERENCES dev_execution_plans(plan_id)
);

CREATE INDEX IF NOT EXISTS idx_dev_execution_plan_tasks_plan
    ON dev_execution_plan_tasks(plan_id, task_id);
CREATE INDEX IF NOT EXISTS idx_dev_execution_plan_tasks_ao_session
    ON dev_execution_plan_tasks(ao_session_id);

CREATE TABLE IF NOT EXISTS dev_execution_supervisor_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    plan_id TEXT,
    status TEXT NOT NULL,
    action TEXT,
    message TEXT,
    created_at REAL NOT NULL,
    completed_at REAL,
    payload TEXT
);

CREATE INDEX IF NOT EXISTS idx_dev_execution_supervisor_runs_plan
    ON dev_execution_supervisor_runs(plan_id, created_at DESC);

CREATE TABLE IF NOT EXISTS dev_execution_supervisor_approvals (
    approval_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    task_ids TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    suggested_instruction TEXT,
    action_overrides TEXT,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    resolved_at REAL,
    resolved_by TEXT,
    resolution_message TEXT,
    payload TEXT
);

CREATE INDEX IF NOT EXISTS idx_dev_execution_supervisor_approvals_plan
    ON dev_execution_supervisor_approvals(plan_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_execution_supervisor_approvals_status
    ON dev_execution_supervisor_approvals(status, expires_at);

CREATE TABLE IF NOT EXISTS dev_execution_runbooks (
    runbook_id TEXT PRIMARY KEY,
    project_id TEXT UNIQUE,
    policy_profile TEXT NOT NULL,
    max_follow_ups_per_task INTEGER NOT NULL,
    max_retries_per_task INTEGER NOT NULL,
    supervisor_enabled INTEGER NOT NULL DEFAULT 0,
    supervisor_interval_seconds INTEGER NOT NULL DEFAULT 60,
    supervisor_limit INTEGER NOT NULL DEFAULT 10,
    supervisor_include_synthesis INTEGER NOT NULL DEFAULT 0,
    supervisor_apply_guarded_actions INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    payload TEXT
);

CREATE INDEX IF NOT EXISTS idx_dev_execution_runbooks_project
    ON dev_execution_runbooks(project_id);

CREATE TABLE IF NOT EXISTS dev_execution_plan_runbooks (
    plan_id TEXT PRIMARY KEY,
    runbook_id TEXT,
    policy_profile TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    payload TEXT,
    FOREIGN KEY(plan_id) REFERENCES dev_execution_plans(plan_id)
);

CREATE TABLE IF NOT EXISTS dev_execution_supervisor_loop_state (
    project_id TEXT PRIMARY KEY,
    runbook_id TEXT,
    status TEXT NOT NULL,
    last_run_id TEXT,
    last_tick_at REAL,
    next_tick_at REAL,
    last_message TEXT,
    consecutive_error_count INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS dev_execution_plan_draft_reviews (
    plan_id TEXT PRIMARY KEY,
    plan_artifact_id TEXT NOT NULL,
    build_id TEXT,
    draft_status TEXT NOT NULL,
    version INTEGER NOT NULL,
    revision_history TEXT NOT NULL,
    source TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    approved_at REAL,
    cancelled_at REAL,
    payload TEXT,
    FOREIGN KEY(plan_id) REFERENCES dev_execution_plans(plan_id)
);

CREATE INDEX IF NOT EXISTS idx_dev_execution_plan_draft_reviews_artifact
    ON dev_execution_plan_draft_reviews(plan_artifact_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS dev_execution_plan_launch_records (
    launch_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    plan_artifact_id TEXT,
    build_id TEXT,
    draft_version INTEGER,
    launch_scope TEXT NOT NULL,
    requested_task_ids TEXT NOT NULL,
    launched_task_ids TEXT NOT NULL,
    failed_task_ids TEXT NOT NULL,
    launched_count INTEGER NOT NULL,
    failure_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    payload TEXT,
    FOREIGN KEY(plan_id) REFERENCES dev_execution_plans(plan_id)
);

CREATE INDEX IF NOT EXISTS idx_dev_execution_plan_launch_records_plan
    ON dev_execution_plan_launch_records(plan_id, created_at DESC);
"""


def list_launch_profiles() -> list[Dict[str, Any]]:
    return [dict(profile) for profile in DEFAULT_LAUNCH_PROFILES]


def get_launch_profile(profile_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not profile_id:
        return None
    wanted = str(profile_id).strip()
    for profile in DEFAULT_LAUNCH_PROFILES:
        if profile["id"] == wanted:
            return dict(profile)
    return None


def list_policy_profiles() -> list[Dict[str, Any]]:
    return [dict(profile) for profile in BUILTIN_POLICY_PROFILES.values()]


def _normalize_policy_profile(policy_profile: Optional[str]) -> str:
    value = str(policy_profile or DEFAULT_POLICY_PROFILE).strip().lower().replace("-", "_")
    return value if value in BUILTIN_POLICY_PROFILES else DEFAULT_POLICY_PROFILE


def _policy_profile(policy_profile: Optional[str]) -> Dict[str, Any]:
    return dict(BUILTIN_POLICY_PROFILES[_normalize_policy_profile(policy_profile)])


def _builtin_runbook(*, policy_profile: str, source: str) -> Dict[str, Any]:
    policy = _policy_profile(policy_profile)
    return {
        "runbook_id": f"builtin:{policy['policy_profile']}",
        "project_id": None,
        "policy_profile": policy["policy_profile"],
        "policy_source": source,
        "max_follow_ups_per_task": int(policy.get("max_follow_ups_per_task") or 0),
        "max_retries_per_task": int(policy.get("max_retries_per_task") or 0),
        "supervisor_enabled": False,
        "supervisor_interval_seconds": DEFAULT_SUPERVISOR_INTERVAL_SECONDS,
        "supervisor_limit": DEFAULT_SUPERVISOR_LIMIT,
        "supervisor_include_synthesis": False,
        "supervisor_apply_guarded_actions": True,
        "policy": policy,
    }


def _resolved_runbook(runbook: Dict[str, Any]) -> Dict[str, Any]:
    policy = {
        **_policy_profile(runbook.get("policy_profile")),
        **(runbook.get("policy") or {}),
    }
    policy["max_follow_ups_per_task"] = int(
        runbook.get("max_follow_ups_per_task")
        if runbook.get("max_follow_ups_per_task") is not None
        else policy.get("max_follow_ups_per_task") or 0
    )
    policy["max_retries_per_task"] = int(
        runbook.get("max_retries_per_task")
        if runbook.get("max_retries_per_task") is not None
        else policy.get("max_retries_per_task") or 0
    )
    return {
        **runbook,
        "policy_profile": policy["policy_profile"],
        "policy": policy,
        "max_follow_ups_per_task": policy["max_follow_ups_per_task"],
        "max_retries_per_task": policy["max_retries_per_task"],
        "supervisor_enabled": _as_bool(runbook.get("supervisor_enabled"), default=False),
        "supervisor_interval_seconds": _bounded_positive_int(
            runbook.get("supervisor_interval_seconds"),
            default=DEFAULT_SUPERVISOR_INTERVAL_SECONDS,
            minimum=MIN_SUPERVISOR_INTERVAL_SECONDS,
            maximum=MAX_SUPERVISOR_INTERVAL_SECONDS,
        ),
        "supervisor_limit": _bounded_positive_int(
            runbook.get("supervisor_limit"),
            default=DEFAULT_SUPERVISOR_LIMIT,
            minimum=1,
            maximum=100,
        ),
        "supervisor_include_synthesis": _as_bool(runbook.get("supervisor_include_synthesis"), default=False),
        "supervisor_apply_guarded_actions": _as_bool(runbook.get("supervisor_apply_guarded_actions"), default=True),
        "policy_source": runbook.get("policy_source") or "project",
    }


def _runbook_fields(runbook: Dict[str, Any]) -> Dict[str, Any]:
    policy = runbook.get("policy") or _policy_profile(runbook.get("policy_profile"))
    return {
        "runbook_id": runbook.get("runbook_id"),
        "policy_profile": policy.get("policy_profile") or runbook.get("policy_profile"),
        "policy_source": runbook.get("policy_source"),
        "max_follow_ups_per_task": int(policy.get("max_follow_ups_per_task") or 0),
        "max_retries_per_task": int(policy.get("max_retries_per_task") or 0),
        "supervisor_enabled": _as_bool(runbook.get("supervisor_enabled"), default=False),
        "supervisor_interval_seconds": _bounded_positive_int(
            runbook.get("supervisor_interval_seconds"),
            default=DEFAULT_SUPERVISOR_INTERVAL_SECONDS,
            minimum=MIN_SUPERVISOR_INTERVAL_SECONDS,
            maximum=MAX_SUPERVISOR_INTERVAL_SECONDS,
        ),
        "supervisor_limit": _bounded_positive_int(
            runbook.get("supervisor_limit"),
            default=DEFAULT_SUPERVISOR_LIMIT,
            minimum=1,
            maximum=100,
        ),
        "supervisor_include_synthesis": _as_bool(runbook.get("supervisor_include_synthesis"), default=False),
        "supervisor_apply_guarded_actions": _as_bool(runbook.get("supervisor_apply_guarded_actions"), default=True),
        "runbook": {
            "runbook_id": runbook.get("runbook_id"),
            "project_id": runbook.get("project_id"),
            "policy_profile": policy.get("policy_profile") or runbook.get("policy_profile"),
            "policy_source": runbook.get("policy_source"),
            "policy": policy,
            "supervisor_enabled": _as_bool(runbook.get("supervisor_enabled"), default=False),
            "supervisor_interval_seconds": _bounded_positive_int(
                runbook.get("supervisor_interval_seconds"),
                default=DEFAULT_SUPERVISOR_INTERVAL_SECONDS,
                minimum=MIN_SUPERVISOR_INTERVAL_SECONDS,
                maximum=MAX_SUPERVISOR_INTERVAL_SECONDS,
            ),
            "supervisor_limit": _bounded_positive_int(
                runbook.get("supervisor_limit"),
                default=DEFAULT_SUPERVISOR_LIMIT,
                minimum=1,
                maximum=100,
            ),
            "supervisor_include_synthesis": _as_bool(runbook.get("supervisor_include_synthesis"), default=False),
            "supervisor_apply_guarded_actions": _as_bool(runbook.get("supervisor_apply_guarded_actions"), default=True),
        },
    }


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _bounded_nonnegative_int(value: Optional[int], *, default: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else int(default)
    except Exception:
        parsed = int(default)
    return max(0, min(parsed, maximum))


def _bounded_positive_int(value: Optional[int], *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else int(default)
    except Exception:
        parsed = int(default)
    return max(minimum, min(parsed, maximum))


def resolve_launch_defaults(
    *,
    profile_id: Optional[str] = None,
    runtime: Optional[str] = None,
    project_id: Optional[str] = None,
    agent: Optional[str] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    permissions: Optional[str] = None,
    runtime_policy_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = get_launch_profile(profile_id) or {}
    runtime_selection = select_worker_runtime(
        goal=None,
        prompt=None,
        profile=profile,
        requested_runtime=runtime,
        permissions=permissions or profile.get("permissions"),
        project_id=project_id or profile.get("project_id") or "OrynWorkspace",
        runtimes=list_worker_runtimes(),
        runtime_policy_evidence=runtime_policy_evidence,
    )
    resolved_runtime = runtime_selection["selected_runtime"]
    resolved_model = model or profile.get("model") or DEFAULT_MODEL
    if resolved_runtime == "openhands" and resolved_model in {None, "", DEFAULT_MODEL, "gpt-5.5"}:
        resolved_model = os.getenv("OPENHANDS_MODEL") or DEFAULT_OPENHANDS_MODEL
    resolved = {
        "launch_profile_id": profile.get("id") or profile_id,
        "runtime": resolved_runtime,
        "project_id": project_id or profile.get("project_id") or "OrynWorkspace",
        "agent": agent or profile.get("agent") or DEFAULT_AGENT,
        "model": resolved_model,
        "reasoning_effort": reasoning_effort or profile.get("reasoning_effort") or DEFAULT_REASONING_EFFORT,
        "permissions": permissions or profile.get("permissions"),
        "contract": profile.get("contract"),
        "runtime_selection": runtime_selection,
        "selected_runtime": runtime_selection["selected_runtime"],
        "runtime_selection_reason": runtime_selection["reason"],
        "runtime_fallback_reason": runtime_selection.get("runtime_fallback_reason"),
        "runtime_policy_evidence": runtime_selection.get("runtime_policy_evidence") or {},
        "runtime_policy_status": runtime_selection.get("runtime_policy_status"),
        "runtime_policy_reason": runtime_selection.get("runtime_policy_reason"),
    }
    if resolved["launch_profile_id"] and not profile:
        resolved["profile_warning"] = f"Unknown launch profile: {resolved['launch_profile_id']}"
    return resolved


def select_execution_runtime(
    *,
    goal: Optional[str] = None,
    prompt: Optional[str] = None,
    profile_id: Optional[str] = None,
    runtime: Optional[str] = None,
    project_id: Optional[str] = None,
    permissions: Optional[str] = None,
    runtimes: Optional[Iterable[Dict[str, Any]]] = None,
    db_path: Optional[Path] = None,
    runtime_policy_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = get_launch_profile(profile_id) or {}
    evidence = runtime_policy_evidence
    if evidence is None:
        evidence = latest_runtime_policy_evidence(db_path)
    return select_worker_runtime(
        goal=goal,
        prompt=prompt,
        profile=profile,
        requested_runtime=runtime,
        permissions=permissions or profile.get("permissions"),
        project_id=project_id or profile.get("project_id") or "OrynWorkspace",
        runtimes=runtimes or list_worker_runtimes(),
        runtime_policy_evidence=evidence,
    )


def build_profiled_prompt(
    prompt: str,
    *,
    goal: Optional[str],
    profile: Dict[str, Any],
    acceptance_criteria: Optional[Iterable[str]] = None,
) -> str:
    parts = []
    if profile.get("launch_profile_id") or profile.get("permissions") or profile.get("contract"):
        parts.extend([
            "## Dev Launch Profile",
            f"Profile: {profile.get('launch_profile_id') or 'custom'}",
            f"Permissions: {profile.get('permissions') or 'unspecified'}",
        ])
        if profile.get("contract"):
            parts.append(f"Contract: {profile['contract']}")
        if profile.get("agent") or profile.get("model") or profile.get("reasoning_effort"):
            parts.append(
                "Runtime: "
                f"{profile.get('agent') or 'agent unspecified'} / "
                f"{profile.get('model') or 'model unspecified'} / "
                f"{profile.get('reasoning_effort') or 'reasoning unspecified'}"
            )
        if goal:
            parts.append(f"Goal: {goal}")
        criteria = [str(item).strip() for item in (acceptance_criteria or []) if str(item).strip()]
        if criteria:
            parts.extend(["", "Acceptance criteria:"])
            parts.extend(f"- {item}" for item in criteria)
        parts.append("")
    parts.append((prompt or "").strip())
    return append_worker_output_contract("\n".join(parts).strip())


def _runtime_label(runtime: Optional[str]) -> str:
    value = normalize_runtime(runtime)
    if value == DEFAULT_RUNTIME:
        return "AO"
    if value == "openhands":
        return "OpenHands"
    return value.replace("_", " ").title()


def _persist_runtime_start_event(
    *,
    session: Any,
    runtime: str,
    goal: str,
    prompt: str,
    project_id: str,
    issue_id: Optional[str],
    branch: Optional[str],
    preview: str,
    tool_name: str,
    event_store: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if event_store is None:
        return None
    metadata = metadata or {}
    session_id = str(getattr(session, "id", "") or "")
    if not session_id:
        return None
    try:
        event_store.upsert_ao_prompt(
            ao_session_id=session_id,
            project_id=project_id,
            prompt=prompt,
            goal=goal,
            issue_id=issue_id,
            branch=branch or getattr(session, "branch", None),
            agent=getattr(session, "agent", None) or metadata.get("agent"),
            model=getattr(session, "model", None) or metadata.get("model"),
            reasoning_effort=getattr(session, "reasoning_effort", None) or metadata.get("reasoning_effort"),
            launch_profile_id=metadata.get("launch_profile_id"),
            launch_plan_id=metadata.get("launch_plan_id"),
            launch_task_id=metadata.get("launch_task_id"),
            permissions=metadata.get("permissions"),
            acceptance_criteria=metadata.get("acceptance_criteria") or [],
            runtime_selection=metadata.get("runtime_selection"),
            selected_runtime=metadata.get("selected_runtime"),
            runtime_selection_reason=metadata.get("runtime_selection_reason"),
            runtime_fallback_reason=metadata.get("runtime_fallback_reason"),
        )
        try:
            session_fields = session.event_fields()
        except Exception:
            session_fields = {}
        payload: Dict[str, Any] = {
            "event": "subagent.start",
            "subagent_id": f"{runtime}:{session_id}",
            "depth": 0,
            "goal": goal,
            "tool": tool_name,
            "tool_name": tool_name,
            "preview": preview,
            "message": preview,
            "timestamp": time.time(),
            "runtime": runtime,
            "runtime_session_id": session_id,
            "runtime_project_id": project_id,
            "workspace_path": getattr(session, "workspace_path", None),
            "branch": getattr(session, "branch", None),
            "open_command": getattr(session, "open_command", None),
            "agent": getattr(session, "agent", None) or metadata.get("agent"),
            "model": getattr(session, "model", None) or metadata.get("model"),
            "reasoning_effort": getattr(session, "reasoning_effort", None) or metadata.get("reasoning_effort"),
            "status": getattr(session, "display_status", None) or "running",
            "launch_profile_id": metadata.get("launch_profile_id"),
            "launch_plan_id": metadata.get("launch_plan_id"),
            "launch_task_id": metadata.get("launch_task_id"),
            "permissions": metadata.get("permissions"),
            "acceptance_criteria": metadata.get("acceptance_criteria") or [],
        }
        for key in (
            "runtime_selection",
            "selected_runtime",
            "runtime_selection_reason",
            "runtime_fallback_reason",
            "runtime_policy_evidence",
            "runtime_policy_status",
            "runtime_policy_reason",
        ):
            if metadata.get(key) is not None:
                payload[key] = metadata.get(key)
        payload.update({key: value for key, value in session_fields.items() if value is not None})
        payload["runtime"] = runtime
        payload["runtime_session_id"] = session_id
        payload.setdefault("runtime_project_id", project_id)
        if runtime == DEFAULT_RUNTIME:
            payload["ao_session_id"] = session_id
            payload["ao_project_id"] = project_id
        else:
            payload.pop("ao_session_id", None)
            payload.pop("ao_project_id", None)
        return event_store.append_event(payload)
    except Exception:
        return None


@dataclass
class DevExecutionStore:
    db_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.db_path = self.db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(self._conn, db_label="state.db")
        self._lock = threading.Lock()
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)
            self._ensure_columns(
                "dev_execution_runbooks",
                {
                    "supervisor_enabled": "INTEGER NOT NULL DEFAULT 0",
                    "supervisor_interval_seconds": "INTEGER NOT NULL DEFAULT 60",
                    "supervisor_limit": "INTEGER NOT NULL DEFAULT 10",
                    "supervisor_include_synthesis": "INTEGER NOT NULL DEFAULT 0",
                    "supervisor_apply_guarded_actions": "INTEGER NOT NULL DEFAULT 1",
                },
            )

    def close(self) -> None:
        self._conn.close()

    def _ensure_columns(self, table: str, columns: Dict[str, str]) -> None:
        existing = {
            str(row["name"])
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for column, definition in columns.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def create_plan(
        self,
        *,
        title: str,
        vision_brief: Optional[str],
        tasks: list[Dict[str, Any]],
        runbook_id: Optional[str] = None,
        policy_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        title = (title or "").strip() or "Dev execution plan"
        now = time.time()
        plan_id = f"devplan-{uuid.uuid4().hex[:10]}"
        normalized_tasks = self._normalize_tasks(plan_id, tasks, now)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_execution_plans (plan_id, title, vision_brief, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (plan_id, title, vision_brief, "planned", now, now),
            )
            for task in normalized_tasks:
                self._insert_task(task)
            if runbook_id or policy_profile:
                self._conn.execute(
                    """
                    INSERT INTO dev_execution_plan_runbooks (
                        plan_id, runbook_id, policy_profile, created_at, updated_at, payload
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(plan_id) DO UPDATE SET
                        runbook_id = excluded.runbook_id,
                        policy_profile = excluded.policy_profile,
                        updated_at = excluded.updated_at,
                        payload = excluded.payload
                    """,
                    (
                        plan_id,
                        str(runbook_id).strip() if runbook_id else None,
                        _normalize_policy_profile(policy_profile) if policy_profile else None,
                        now,
                        now,
                        json.dumps({}, ensure_ascii=False),
                    ),
                )
        return self.get_plan(plan_id) or {"plan_id": plan_id, "title": title, "tasks": normalized_tasks}

    def list_plans(self, *, limit: int = 50, project_id: Optional[str] = None) -> list[Dict[str, Any]]:
        project_id = str(project_id or "").strip()
        if project_id:
            rows = self._conn.execute(
                """
                SELECT DISTINCT p.plan_id, p.updated_at
                FROM dev_execution_plans p
                JOIN dev_execution_plan_tasks t ON t.plan_id = p.plan_id
                WHERE t.project_id = ?
                ORDER BY p.updated_at DESC
                LIMIT ?
                """,
                (project_id, max(1, min(int(limit or 50), 200))),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT plan_id
                FROM dev_execution_plans
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, min(int(limit or 50), 200)),),
            ).fetchall()
        return [plan for row in rows if (plan := self.get_plan(row["plan_id"]))]

    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT plan_id, title, vision_brief, status, created_at, updated_at
            FROM dev_execution_plans
            WHERE plan_id = ?
            """,
            (plan_id,),
        ).fetchone()
        if not row:
            return None
        task_rows = self._conn.execute(
            """
            SELECT *
            FROM dev_execution_plan_tasks
            WHERE plan_id = ?
            ORDER BY created_at ASC, task_id ASC
            """,
            (plan_id,),
        ).fetchall()
        plan = dict(row)
        plan["tasks"] = [self._task_from_row(task_row) for task_row in task_rows]
        if draft_review := self.get_draft_review(plan_id):
            plan.update({
                "draft_review": draft_review,
                "draft_status": draft_review.get("draft_status"),
                "draft_version": draft_review.get("version"),
                "draft_plan_artifact_id": draft_review.get("plan_artifact_id"),
                "draft_build_id": draft_review.get("build_id"),
                "draft_approved_at": draft_review.get("approved_at"),
                "draft_cancelled_at": draft_review.get("cancelled_at"),
            })
        if latest_supervisor := self.latest_supervisor_record(plan_id):
            plan.update({
                "supervisor_status": latest_supervisor.get("status"),
                "supervisor_last_action": latest_supervisor.get("action"),
                "supervisor_last_message": latest_supervisor.get("message"),
                "supervisor_last_run_id": latest_supervisor.get("run_id"),
                "supervisor_last_run_at": latest_supervisor.get("completed_at") or latest_supervisor.get("created_at"),
            })
        if latest_approval := self.latest_supervisor_approval(plan_id):
            plan.update({
                "supervisor_approval_id": latest_approval.get("approval_id"),
                "supervisor_approval_status": latest_approval.get("status"),
                "supervisor_approval_expires_at": latest_approval.get("expires_at"),
            })
        runbook = self.resolve_runbook_for_plan(plan)
        plan.update(_runbook_fields(runbook))
        for project_id in _plan_project_ids(plan):
            if state := self.get_supervisor_loop_state(project_id):
                plan.update({
                    "supervisor_loop_status": state.get("status"),
                    "supervisor_loop_last_run_id": state.get("last_run_id"),
                    "supervisor_loop_last_tick_at": state.get("last_tick_at"),
                    "supervisor_loop_next_tick_at": state.get("next_tick_at"),
                    "supervisor_loop_last_message": state.get("last_message"),
                    "supervisor_loop_consecutive_error_count": state.get("consecutive_error_count"),
                })
                break
        return plan

    def list_runbooks(self, *, project_id: Optional[str] = None, limit: int = 100) -> list[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        sql = "SELECT * FROM dev_execution_runbooks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit or 100), 200)))
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [self._runbook_from_row(row) for row in rows]

    def get_runbook(self, runbook_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_execution_runbooks WHERE runbook_id = ?",
            (runbook_id,),
        ).fetchone()
        return self._runbook_from_row(row)

    def get_project_runbook(self, project_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_execution_runbooks WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        return self._runbook_from_row(row)

    def upsert_project_runbook(
        self,
        *,
        project_id: str,
        policy_profile: str = DEFAULT_POLICY_PROFILE,
        max_follow_ups_per_task: Optional[int] = None,
        max_retries_per_task: Optional[int] = None,
        supervisor_enabled: Optional[bool] = None,
        supervisor_interval_seconds: Optional[int] = None,
        supervisor_limit: Optional[int] = None,
        supervisor_include_synthesis: Optional[bool] = None,
        supervisor_apply_guarded_actions: Optional[bool] = None,
    ) -> Dict[str, Any]:
        project_id = str(project_id or "").strip()
        if not project_id:
            raise ValueError("project_id is required")
        policy = _policy_profile(policy_profile)
        max_followups = _bounded_nonnegative_int(
            max_follow_ups_per_task,
            default=int(policy.get("max_follow_ups_per_task") or 0),
            maximum=10,
        )
        max_retries = _bounded_nonnegative_int(
            max_retries_per_task,
            default=int(policy.get("max_retries_per_task") or 0),
            maximum=10,
        )
        existing = self.get_project_runbook(project_id)
        loop_enabled = _as_bool(
            supervisor_enabled,
            default=_as_bool((existing or {}).get("supervisor_enabled"), default=False),
        )
        loop_interval = _bounded_positive_int(
            supervisor_interval_seconds,
            default=int((existing or {}).get("supervisor_interval_seconds") or DEFAULT_SUPERVISOR_INTERVAL_SECONDS),
            minimum=MIN_SUPERVISOR_INTERVAL_SECONDS,
            maximum=MAX_SUPERVISOR_INTERVAL_SECONDS,
        )
        loop_limit = _bounded_positive_int(
            supervisor_limit,
            default=int((existing or {}).get("supervisor_limit") or DEFAULT_SUPERVISOR_LIMIT),
            minimum=1,
            maximum=100,
        )
        loop_include_synthesis = _as_bool(
            supervisor_include_synthesis,
            default=_as_bool((existing or {}).get("supervisor_include_synthesis"), default=False),
        )
        loop_apply_guarded = _as_bool(
            supervisor_apply_guarded_actions,
            default=_as_bool((existing or {}).get("supervisor_apply_guarded_actions"), default=True),
        )
        now = time.time()
        runbook_id = (existing or {}).get("runbook_id") or f"devrunbook-{uuid.uuid4().hex[:10]}"
        payload = {
            **policy,
            "max_follow_ups_per_task": max_followups,
            "max_retries_per_task": max_retries,
            "supervisor_enabled": loop_enabled,
            "supervisor_interval_seconds": loop_interval,
            "supervisor_limit": loop_limit,
            "supervisor_include_synthesis": loop_include_synthesis,
            "supervisor_apply_guarded_actions": loop_apply_guarded,
        }
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_execution_runbooks (
                    runbook_id, project_id, policy_profile, max_follow_ups_per_task,
                    max_retries_per_task, supervisor_enabled, supervisor_interval_seconds,
                    supervisor_limit, supervisor_include_synthesis, supervisor_apply_guarded_actions,
                    created_at, updated_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    policy_profile = excluded.policy_profile,
                    max_follow_ups_per_task = excluded.max_follow_ups_per_task,
                    max_retries_per_task = excluded.max_retries_per_task,
                    supervisor_enabled = excluded.supervisor_enabled,
                    supervisor_interval_seconds = excluded.supervisor_interval_seconds,
                    supervisor_limit = excluded.supervisor_limit,
                    supervisor_include_synthesis = excluded.supervisor_include_synthesis,
                    supervisor_apply_guarded_actions = excluded.supervisor_apply_guarded_actions,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    runbook_id,
                    project_id,
                    policy["policy_profile"],
                    max_followups,
                    max_retries,
                    1 if loop_enabled else 0,
                    loop_interval,
                    loop_limit,
                    1 if loop_include_synthesis else 0,
                    1 if loop_apply_guarded else 0,
                    now if not existing else existing.get("created_at") or now,
                    now,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
        return self.get_project_runbook(project_id) or {
            "runbook_id": runbook_id,
            "project_id": project_id,
            "policy_profile": policy["policy_profile"],
            "max_follow_ups_per_task": max_followups,
            "max_retries_per_task": max_retries,
            "supervisor_enabled": loop_enabled,
            "supervisor_interval_seconds": loop_interval,
            "supervisor_limit": loop_limit,
            "supervisor_include_synthesis": loop_include_synthesis,
            "supervisor_apply_guarded_actions": loop_apply_guarded,
            "policy_source": "project",
        }

    def resolve_runbook_for_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        plan_id = str(plan.get("plan_id") or "").strip()
        if plan_id:
            override = self._conn.execute(
                "SELECT * FROM dev_execution_plan_runbooks WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
            if override:
                override_record = dict(override)
                runbook_id = str(override_record.get("runbook_id") or "").strip()
                if runbook_id and (runbook := self.get_runbook(runbook_id)):
                    runbook["policy_source"] = "plan"
                    return _resolved_runbook(runbook)
                if override_record.get("policy_profile"):
                    return _builtin_runbook(
                        policy_profile=str(override_record.get("policy_profile")),
                        source="plan",
                    )
        for project_id in _plan_project_ids(plan):
            if runbook := self.get_project_runbook(project_id):
                runbook["policy_source"] = "project"
                return _resolved_runbook(runbook)
        return _builtin_runbook(policy_profile=DEFAULT_POLICY_PROFILE, source="global")

    def create_draft_review(
        self,
        *,
        plan_id: str,
        plan_artifact_id: str,
        build_id: Optional[str],
        source: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        plan_id = str(plan_id or "").strip()
        plan_artifact_id = str(plan_artifact_id or "").strip()
        if not plan_id or not plan_artifact_id:
            raise ValueError("plan_id and plan_artifact_id are required")
        if not self.get_plan(plan_id):
            raise KeyError(f"Dev execution plan not found: {plan_id}")
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_execution_plan_draft_reviews (
                    plan_id, plan_artifact_id, build_id, draft_status, version,
                    revision_history, source, created_at, updated_at,
                    approved_at, cancelled_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    plan_artifact_id = excluded.plan_artifact_id,
                    build_id = excluded.build_id,
                    draft_status = excluded.draft_status,
                    version = excluded.version,
                    revision_history = excluded.revision_history,
                    source = excluded.source,
                    updated_at = excluded.updated_at,
                    approved_at = excluded.approved_at,
                    cancelled_at = excluded.cancelled_at,
                    payload = excluded.payload
                """,
                (
                    plan_id,
                    plan_artifact_id,
                    str(build_id).strip() if build_id else None,
                    "draft",
                    1,
                    json.dumps([], ensure_ascii=False),
                    source,
                    now,
                    now,
                    None,
                    None,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
        review = self.get_draft_review(plan_id)
        if not review:
            raise RuntimeError("Draft review record was not persisted")
        return review

    def get_draft_review(self, plan_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_execution_plan_draft_reviews WHERE plan_id = ?",
            (str(plan_id or "").strip(),),
        ).fetchone()
        review = self._draft_review_from_row(row)
        if review:
            review["launch_records"] = self.list_launch_records(plan_id)
        return review

    def update_draft_review(self, plan_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_draft_review(plan_id)
        if not current:
            raise KeyError(f"Draft review not found for Dev execution plan: {plan_id}")
        payload = {**current, **updates, "updated_at": time.time()}
        status = str(payload.get("draft_status") or "").strip()
        if status not in DEV_PLAN_DRAFT_STATUSES:
            raise ValueError(f"Unsupported draft status: {status}")
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE dev_execution_plan_draft_reviews
                SET plan_artifact_id = ?, build_id = ?, draft_status = ?,
                    version = ?, revision_history = ?, source = ?,
                    updated_at = ?, approved_at = ?, cancelled_at = ?, payload = ?
                WHERE plan_id = ?
                """,
                (
                    payload["plan_artifact_id"],
                    payload.get("build_id"),
                    status,
                    int(payload.get("version") or 1),
                    json.dumps(payload.get("revision_history") or [], ensure_ascii=False),
                    payload.get("source"),
                    float(payload["updated_at"]),
                    payload.get("approved_at"),
                    payload.get("cancelled_at"),
                    json.dumps(payload.get("payload") or {}, ensure_ascii=False),
                    plan_id,
                ),
            )
        review = self.get_draft_review(plan_id)
        if not review:
            raise RuntimeError("Draft review record was not updated")
        return review

    def plan_has_launched_tasks(self, plan_id: str) -> bool:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM dev_execution_plan_tasks
            WHERE plan_id = ?
              AND (
                ao_session_id IS NOT NULL
                OR LOWER(status) NOT IN ('planned')
              )
            """,
            (str(plan_id or "").strip(),),
        ).fetchone()
        return bool(row and int(row["count"] or 0) > 0)

    def replace_plan_tasks(self, *, plan_id: str, tasks: list[Dict[str, Any]]) -> Dict[str, Any]:
        plan = self.get_plan(plan_id)
        if not plan:
            raise KeyError(f"Dev execution plan not found: {plan_id}")
        if self.plan_has_launched_tasks(plan_id):
            raise ValueError("Draft plan tasks cannot be revised after any task has launched")
        now = time.time()
        normalized_tasks = self._normalize_tasks(plan_id, tasks, now)
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM dev_execution_plan_tasks WHERE plan_id = ?", (plan_id,))
            for task in normalized_tasks:
                self._insert_task(task)
            self._conn.execute(
                """
                UPDATE dev_execution_plans
                SET status = ?, updated_at = ?
                WHERE plan_id = ?
                """,
                ("planned", now, plan_id),
            )
        return self.get_plan(plan_id) or {**plan, "tasks": normalized_tasks}

    def update_task_launch(self, *, plan_id: str, task_id: str, ao_session_id: str, status: str = "launched") -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE dev_execution_plan_tasks
                SET ao_session_id = ?, status = ?, updated_at = ?
                WHERE plan_id = ? AND task_id = ?
                """,
                (ao_session_id, status, now, plan_id, task_id),
            )
            self._conn.execute(
                """
                UPDATE dev_execution_plans
                SET status = ?, updated_at = ?
                WHERE plan_id = ?
                """,
                ("launched", now, plan_id),
            )

    def append_launch_record(
        self,
        *,
        plan_id: str,
        draft_review: Dict[str, Any],
        requested_task_ids: Optional[list[str]],
        launched: list[Dict[str, Any]],
        failures: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        plan_id = str(plan_id or "").strip()
        requested = [str(task_id).strip() for task_id in (requested_task_ids or []) if str(task_id).strip()]
        launched_task_ids = [
            str(item.get("task_id") or "").strip()
            for item in launched
            if str(item.get("task_id") or "").strip()
        ]
        failed_task_ids = [
            str(item.get("task_id") or "").strip()
            for item in failures
            if str(item.get("task_id") or "").strip()
        ]
        if not requested:
            launch_scope = "all"
        elif len(requested) == 1:
            launch_scope = "smoke"
        else:
            launch_scope = "subset"
        if launched_task_ids and failed_task_ids:
            status = "partial"
        elif launched_task_ids:
            status = "succeeded"
        elif failed_task_ids:
            status = "failed"
        else:
            status = "noop"
        launch_id = f"devlaunch-{uuid.uuid4().hex[:10]}"
        created_at = time.time()
        payload = {
            "launched": launched,
            "failures": failures,
        }
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_execution_plan_launch_records (
                    launch_id, plan_id, plan_artifact_id, build_id, draft_version,
                    launch_scope, requested_task_ids, launched_task_ids, failed_task_ids,
                    launched_count, failure_count, status, created_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    launch_id,
                    plan_id,
                    draft_review.get("plan_artifact_id"),
                    draft_review.get("build_id"),
                    int(draft_review.get("version") or 1),
                    launch_scope,
                    json.dumps(requested, ensure_ascii=False),
                    json.dumps(launched_task_ids, ensure_ascii=False),
                    json.dumps(failed_task_ids, ensure_ascii=False),
                    len(launched_task_ids),
                    len(failed_task_ids),
                    status,
                    created_at,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
        record = self.get_launch_record(launch_id)
        if not record:
            raise RuntimeError("Launch record was not persisted")
        return record

    def get_launch_record(self, launch_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_execution_plan_launch_records WHERE launch_id = ?",
            (str(launch_id or "").strip(),),
        ).fetchone()
        return self._launch_record_from_row(row)

    def list_launch_records(self, plan_id: str, *, limit: int = 20) -> list[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM dev_execution_plan_launch_records
            WHERE plan_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (str(plan_id or "").strip(), max(1, min(int(limit or 20), 100))),
        ).fetchall()
        return [record for row in rows if (record := self._launch_record_from_row(row))]

    def append_supervisor_record(
        self,
        *,
        run_id: str,
        plan_id: Optional[str],
        status: str,
        action: Optional[str],
        message: Optional[str],
        started_at: float,
        completed_at: Optional[float],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO dev_execution_supervisor_runs (
                    run_id, plan_id, status, action, message, created_at, completed_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    plan_id,
                    status,
                    action,
                    message,
                    started_at,
                    completed_at,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
        return {
            "id": cur.lastrowid,
            "run_id": run_id,
            "plan_id": plan_id,
            "status": status,
            "action": action,
            "message": message,
            "created_at": started_at,
            "completed_at": completed_at,
            "payload": payload or {},
        }

    def latest_supervisor_record(self, plan_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT *
            FROM dev_execution_supervisor_runs
            WHERE plan_id = ?
            ORDER BY COALESCE(completed_at, created_at) DESC, id DESC
            LIMIT 1
            """,
            (plan_id,),
        ).fetchone()
        if not row:
            return None
        record = dict(row)
        try:
            record["payload"] = json.loads(record.get("payload") or "{}")
        except Exception:
            record["payload"] = {}
        return record

    def list_enabled_supervisor_runbooks(self) -> list[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM dev_execution_runbooks
            WHERE supervisor_enabled = 1 AND project_id IS NOT NULL AND project_id != ''
            ORDER BY updated_at ASC
            """
        ).fetchall()
        return [runbook for row in rows if (runbook := self._runbook_from_row(row))]

    def get_supervisor_loop_state(self, project_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_execution_supervisor_loop_state WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        return self._loop_state_from_row(row)

    def list_supervisor_loop_state(self, *, project_id: Optional[str] = None) -> list[Dict[str, Any]]:
        params: list[Any] = []
        sql = "SELECT * FROM dev_execution_supervisor_loop_state"
        if project_id:
            sql += " WHERE project_id = ?"
            params.append(project_id)
        sql += " ORDER BY updated_at DESC"
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [state for row in rows if (state := self._loop_state_from_row(row))]

    def record_supervisor_loop_tick(
        self,
        *,
        project_id: str,
        runbook_id: Optional[str],
        status: str,
        last_run_id: Optional[str],
        last_tick_at: float,
        next_tick_at: float,
        last_message: Optional[str],
        consecutive_error_count: int,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        updated_at = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_execution_supervisor_loop_state (
                    project_id, runbook_id, status, last_run_id, last_tick_at,
                    next_tick_at, last_message, consecutive_error_count, updated_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    runbook_id = excluded.runbook_id,
                    status = excluded.status,
                    last_run_id = excluded.last_run_id,
                    last_tick_at = excluded.last_tick_at,
                    next_tick_at = excluded.next_tick_at,
                    last_message = excluded.last_message,
                    consecutive_error_count = excluded.consecutive_error_count,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    project_id,
                    runbook_id,
                    status,
                    last_run_id,
                    last_tick_at,
                    next_tick_at,
                    last_message,
                    max(0, int(consecutive_error_count or 0)),
                    updated_at,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
        return self.get_supervisor_loop_state(project_id) or {
            "project_id": project_id,
            "runbook_id": runbook_id,
            "status": status,
            "last_run_id": last_run_id,
            "last_tick_at": last_tick_at,
            "next_tick_at": next_tick_at,
            "last_message": last_message,
            "consecutive_error_count": max(0, int(consecutive_error_count or 0)),
            "updated_at": updated_at,
            "payload": payload or {},
        }

    def create_or_reuse_supervisor_approval(
        self,
        *,
        plan_id: str,
        task_ids: list[str],
        recommended_action: str,
        reason: Optional[str],
        suggested_instruction: Optional[str],
        action_overrides: Optional[Dict[str, Any]],
        payload: Dict[str, Any],
        ttl_seconds: int = SUPERVISOR_APPROVAL_TTL_SECONDS,
    ) -> Dict[str, Any]:
        now = time.time()
        normalized_task_ids = _normalize_task_ids(task_ids)
        action = str(recommended_action or "").strip()
        existing = self._find_pending_supervisor_approval(
            plan_id=plan_id,
            task_ids=normalized_task_ids,
            recommended_action=action,
            now=now,
        )
        if existing:
            return existing
        approval_id = f"devappr-{uuid.uuid4().hex[:10]}"
        expires_at = now + max(60, int(ttl_seconds or SUPERVISOR_APPROVAL_TTL_SECONDS))
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_execution_supervisor_approvals (
                    approval_id, plan_id, task_ids, recommended_action, status,
                    reason, suggested_instruction, action_overrides, created_at,
                    expires_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    plan_id,
                    json.dumps(normalized_task_ids, ensure_ascii=False),
                    action,
                    "pending",
                    reason,
                    suggested_instruction,
                    json.dumps(action_overrides or {}, ensure_ascii=False),
                    now,
                    expires_at,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
        return self.get_supervisor_approval(approval_id) or {
            "approval_id": approval_id,
            "plan_id": plan_id,
            "task_ids": normalized_task_ids,
            "recommended_action": action,
            "status": "pending",
            "reason": reason,
            "suggested_instruction": suggested_instruction,
            "action_overrides": action_overrides or {},
            "created_at": now,
            "expires_at": expires_at,
            "payload": payload or {},
        }

    def get_supervisor_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_execution_supervisor_approvals WHERE approval_id = ?",
            (approval_id,),
        ).fetchone()
        return self._approval_from_row(row)

    def list_supervisor_approvals(
        self,
        *,
        status: Optional[str] = None,
        plan_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if plan_id:
            clauses.append("plan_id = ?")
            params.append(plan_id)
        sql = "SELECT * FROM dev_execution_supervisor_approvals"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit or 50), 200)))
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [approval for row in rows if (approval := self._approval_from_row(row))]

    def latest_supervisor_approval(self, plan_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT *
            FROM dev_execution_supervisor_approvals
            WHERE plan_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (plan_id,),
        ).fetchone()
        return self._approval_from_row(row)

    def resolve_supervisor_approval(
        self,
        *,
        approval_id: str,
        status: str,
        resolved_by: Optional[str],
        resolution_message: Optional[str],
        action_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        approval = self.get_supervisor_approval(approval_id)
        if not approval:
            raise KeyError(f"Supervisor approval not found: {approval_id}")
        if approval["status"] == "expired":
            raise ValueError("Supervisor approval has expired.")
        if approval["status"] != "pending":
            raise ValueError(f"Supervisor approval is {approval['status']}, not pending.")
        now = time.time()
        overrides = dict(approval.get("action_overrides") or {})
        if action_overrides:
            overrides.update({key: value for key, value in action_overrides.items() if value is not None})
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE dev_execution_supervisor_approvals
                SET status = ?, resolved_at = ?, resolved_by = ?, resolution_message = ?,
                    action_overrides = ?
                WHERE approval_id = ?
                """,
                (
                    status,
                    now,
                    resolved_by,
                    resolution_message,
                    json.dumps(overrides, ensure_ascii=False),
                    approval_id,
                ),
            )
        return self.get_supervisor_approval(approval_id) or approval

    def consume_supervisor_approval(self, approval_id: str, *, message: Optional[str]) -> Dict[str, Any]:
        approval = self.get_supervisor_approval(approval_id)
        if not approval:
            raise KeyError(f"Supervisor approval not found: {approval_id}")
        if approval["status"] != "approved":
            raise ValueError(f"Supervisor approval is {approval['status']}, not approved.")
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE dev_execution_supervisor_approvals
                SET status = ?, resolved_at = ?, resolution_message = ?
                WHERE approval_id = ?
                """,
                ("consumed", now, message, approval_id),
            )
        return self.get_supervisor_approval(approval_id) or approval

    def _find_pending_supervisor_approval(
        self,
        *,
        plan_id: str,
        task_ids: list[str],
        recommended_action: str,
        now: float,
    ) -> Optional[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM dev_execution_supervisor_approvals
            WHERE plan_id = ? AND recommended_action = ? AND status = 'pending'
            ORDER BY created_at DESC
            """,
            (plan_id, recommended_action),
        ).fetchall()
        for row in rows:
            approval = self._approval_from_row(row, now=now)
            if approval and approval.get("status") == "pending" and approval.get("task_ids") == task_ids:
                return approval
        return None

    def _approval_from_row(self, row: Optional[sqlite3.Row], *, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        approval = dict(row)
        now = time.time() if now is None else now
        if approval.get("status") == "pending" and float(approval.get("expires_at") or 0) <= now:
            with self._lock, self._conn:
                self._conn.execute(
                    """
                    UPDATE dev_execution_supervisor_approvals
                    SET status = ?, resolved_at = ?, resolution_message = ?
                    WHERE approval_id = ?
                    """,
                    ("expired", now, "Approval expired.", approval["approval_id"]),
                )
            approval["status"] = "expired"
            approval["resolved_at"] = now
            approval["resolution_message"] = "Approval expired."
        for key, default in (("task_ids", []), ("action_overrides", {}), ("payload", {})):
            try:
                approval[key] = json.loads(approval.get(key) or json.dumps(default))
            except Exception:
                approval[key] = default
        return approval

    def _runbook_from_row(self, row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        runbook = dict(row)
        try:
            payload = json.loads(runbook.get("payload") or "{}")
        except Exception:
            payload = {}
        runbook["payload"] = payload
        runbook.update({
            "policy": {
                **_policy_profile(runbook.get("policy_profile")),
                **payload,
                "max_follow_ups_per_task": int(runbook.get("max_follow_ups_per_task") or 0),
                "max_retries_per_task": int(runbook.get("max_retries_per_task") or 0),
            },
            "policy_source": runbook.get("policy_source") or "project",
            "supervisor_enabled": _as_bool(runbook.get("supervisor_enabled"), default=False),
            "supervisor_interval_seconds": _bounded_positive_int(
                runbook.get("supervisor_interval_seconds"),
                default=DEFAULT_SUPERVISOR_INTERVAL_SECONDS,
                minimum=MIN_SUPERVISOR_INTERVAL_SECONDS,
                maximum=MAX_SUPERVISOR_INTERVAL_SECONDS,
            ),
            "supervisor_limit": _bounded_positive_int(
                runbook.get("supervisor_limit"),
                default=DEFAULT_SUPERVISOR_LIMIT,
                minimum=1,
                maximum=100,
            ),
            "supervisor_include_synthesis": _as_bool(runbook.get("supervisor_include_synthesis"), default=False),
            "supervisor_apply_guarded_actions": _as_bool(runbook.get("supervisor_apply_guarded_actions"), default=True),
        })
        return runbook

    def _loop_state_from_row(self, row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        state = dict(row)
        try:
            state["payload"] = json.loads(state.get("payload") or "{}")
        except Exception:
            state["payload"] = {}
        return state

    def _normalize_tasks(self, plan_id: str, tasks: list[Dict[str, Any]], now: float) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        runtime_policy_evidence = latest_runtime_policy_evidence(self.db_path)
        for index, raw in enumerate(tasks or [], start=1):
            prompt = str((raw or {}).get("prompt") or "").strip()
            if not prompt:
                continue
            profile_id = (raw or {}).get("profile_id") or (raw or {}).get("launch_profile_id")
            profile = resolve_launch_defaults(
                profile_id=profile_id,
                runtime=(raw or {}).get("runtime"),
                project_id=(raw or {}).get("project_id"),
                agent=(raw or {}).get("agent"),
                model=(raw or {}).get("model"),
                reasoning_effort=(raw or {}).get("reasoning_effort"),
                permissions=(raw or {}).get("permissions"),
                runtime_policy_evidence=runtime_policy_evidence,
            )
            runtime_selection = select_execution_runtime(
                goal=(raw or {}).get("goal") or prompt.splitlines()[0],
                prompt=prompt,
                profile_id=profile_id,
                runtime=(raw or {}).get("runtime"),
                project_id=profile.get("project_id"),
                permissions=profile.get("permissions"),
                runtime_policy_evidence=runtime_policy_evidence,
            )
            profile.update({
                "runtime": runtime_selection["selected_runtime"],
                "runtime_selection": runtime_selection,
                "selected_runtime": runtime_selection["selected_runtime"],
                "runtime_selection_reason": runtime_selection["reason"],
                "runtime_fallback_reason": runtime_selection.get("runtime_fallback_reason"),
                "runtime_policy_evidence": runtime_selection.get("runtime_policy_evidence") or {},
                "runtime_policy_status": runtime_selection.get("runtime_policy_status"),
                "runtime_policy_reason": runtime_selection.get("runtime_policy_reason"),
            })
            if profile["runtime"] == "openhands" and profile.get("model") in {None, "", DEFAULT_MODEL, "gpt-5.5"}:
                profile["model"] = os.getenv("OPENHANDS_MODEL") or DEFAULT_OPENHANDS_MODEL
            criteria = self._list_field((raw or {}).get("acceptance_criteria"))
            task = {
                "task_id": str((raw or {}).get("task_id") or f"{plan_id}-task-{index}"),
                "plan_id": plan_id,
                "goal": str((raw or {}).get("goal") or prompt.splitlines()[0])[:180],
                "prompt": prompt,
                "profile_id": profile.get("launch_profile_id") or profile_id,
                "project_id": profile.get("project_id"),
                "dependencies": self._list_field((raw or {}).get("dependencies")),
                "acceptance_criteria": criteria,
                "ao_session_id": None,
                "status": "planned",
                "created_at": now,
                "updated_at": now,
                "payload": {**(raw or {}), "resolved_profile": profile},
            }
            normalized.append(task)
        if not normalized:
            raise ValueError("dev execution plan requires at least one task with a prompt")
        return normalized

    def _insert_task(self, task: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO dev_execution_plan_tasks (
                task_id, plan_id, goal, prompt, profile_id, project_id,
                dependencies, acceptance_criteria, ao_session_id, status,
                created_at, updated_at, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task["task_id"],
                task["plan_id"],
                task["goal"],
                task["prompt"],
                task.get("profile_id"),
                task.get("project_id"),
                json.dumps(task.get("dependencies") or [], ensure_ascii=False),
                json.dumps(task.get("acceptance_criteria") or [], ensure_ascii=False),
                task.get("ao_session_id"),
                task.get("status") or "planned",
                task.get("created_at"),
                task.get("updated_at"),
                json.dumps(task.get("payload") or {}, ensure_ascii=False),
            ),
        )

    @staticmethod
    def _list_field(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> Dict[str, Any]:
        task = dict(row)
        for key in ("dependencies", "acceptance_criteria"):
            try:
                task[key] = json.loads(task.get(key) or "[]")
            except Exception:
                task[key] = []
        try:
            task["payload"] = json.loads(task.get("payload") or "{}")
        except Exception:
            task["payload"] = {}
        return task

    @staticmethod
    def _draft_review_from_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        review = dict(row)
        try:
            review["revision_history"] = json.loads(review.get("revision_history") or "[]")
        except Exception:
            review["revision_history"] = []
        try:
            review["payload"] = json.loads(review.get("payload") or "{}")
        except Exception:
            review["payload"] = {}
        review["version"] = int(review.get("version") or 1)
        review["object"] = "hermes.dev_execution_plan_draft_review"
        return review

    @staticmethod
    def _launch_record_from_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        record = dict(row)
        for key in ("requested_task_ids", "launched_task_ids", "failed_task_ids"):
            try:
                record[key] = json.loads(record.get(key) or "[]")
            except Exception:
                record[key] = []
        try:
            record["payload"] = json.loads(record.get("payload") or "{}")
        except Exception:
            record["payload"] = {}
        record["draft_version"] = int(record.get("draft_version") or 0) or None
        record["launched_count"] = int(record.get("launched_count") or 0)
        record["failure_count"] = int(record.get("failure_count") or 0)
        record["object"] = "hermes.dev_execution_plan_launch_record"
        return record


def launch_execution_plan(
    *,
    store: DevExecutionStore,
    plan_id: str,
    task_ids: Optional[list[str]] = None,
    bridge: Any = None,
    event_store: Any = None,
) -> Dict[str, Any]:
    plan = store.get_plan(plan_id)
    if not plan:
        raise KeyError(f"Dev execution plan not found: {plan_id}")
    draft_review = store.get_draft_review(plan_id)
    if draft_review and draft_review.get("draft_status") != "approved_for_launch":
        raise ValueError(
            f"Artifact-created Dev execution plan draft is {draft_review.get('draft_status')}; approve draft before launch."
        )

    from tools.ao_delegate_tool import _emit, _persist_start_event_direct, build_ao_worker_prompt

    router = _ensure_runtime_router(bridge)
    wanted = {str(task_id) for task_id in (task_ids or []) if str(task_id).strip()}
    launched: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    tasks = [task for task in plan.get("tasks") or [] if not wanted or task.get("task_id") in wanted]
    runtime_policy_evidence = latest_runtime_policy_evidence(store.db_path)
    for task in tasks:
        profile_payload = (task.get("payload") or {}).get("resolved_profile") or {}
        profile = resolve_launch_defaults(
            profile_id=task.get("profile_id"),
            runtime=profile_payload.get("runtime"),
            project_id=task.get("project_id"),
            agent=profile_payload.get("agent"),
            model=profile_payload.get("model"),
            reasoning_effort=profile_payload.get("reasoning_effort"),
            permissions=profile_payload.get("permissions"),
            runtime_policy_evidence=runtime_policy_evidence,
        )
        runtime_selection = select_execution_runtime(
            goal=task.get("goal"),
            prompt=task.get("prompt"),
            profile_id=task.get("profile_id"),
            runtime=(task.get("payload") or {}).get("runtime"),
            project_id=profile.get("project_id"),
            permissions=profile.get("permissions"),
            runtime_policy_evidence=runtime_policy_evidence,
        )
        profile.update({
            "runtime": runtime_selection["selected_runtime"],
            "runtime_selection": runtime_selection,
            "selected_runtime": runtime_selection["selected_runtime"],
            "runtime_selection_reason": runtime_selection["reason"],
            "runtime_fallback_reason": runtime_selection.get("runtime_fallback_reason"),
            "runtime_policy_evidence": runtime_selection.get("runtime_policy_evidence") or {},
            "runtime_policy_status": runtime_selection.get("runtime_policy_status"),
            "runtime_policy_reason": runtime_selection.get("runtime_policy_reason"),
        })
        if profile["runtime"] == "openhands" and profile.get("model") in {None, "", DEFAULT_MODEL, "gpt-5.5"}:
            profile["model"] = os.getenv("OPENHANDS_MODEL") or DEFAULT_OPENHANDS_MODEL
        task_payload = task.get("payload") or {}
        minimal_worker_prompt = bool(task_payload.get("minimal_worker_prompt"))
        if minimal_worker_prompt:
            profiled_prompt = (task.get("prompt") or "").strip()
        else:
            profiled_prompt = build_profiled_prompt(
                task.get("prompt") or "",
                goal=task.get("goal"),
                profile=profile,
                acceptance_criteria=task.get("acceptance_criteria") or [],
            )
        runtime = profile.get("runtime") or DEFAULT_RUNTIME
        if runtime == DEFAULT_RUNTIME and not minimal_worker_prompt:
            launch_prompt = build_ao_worker_prompt(profiled_prompt, goal=task.get("goal"))
        else:
            launch_prompt = profiled_prompt
        try:
            session = router.spawn(
                runtime,
                project_id=profile["project_id"],
                prompt=launch_prompt,
                issue_id=(task.get("payload") or {}).get("issue_id"),
                branch=(task.get("payload") or {}).get("branch"),
                agent=profile.get("agent"),
                model=profile.get("model"),
                reasoning_effort=profile.get("reasoning_effort"),
                minimal_worker_prompt=minimal_worker_prompt,
            )
        except Exception as exc:
            if (
                runtime == "openhands"
                and runtime_selection.get("selection_mode") in {"auto", "fallback"}
                and runtime_selection.get("fallback_runtime") == DEFAULT_RUNTIME
            ):
                fallback_reason = f"OpenHands launch failed, falling back to AO: {exc}"
                runtime = DEFAULT_RUNTIME
                profile["runtime"] = DEFAULT_RUNTIME
                runtime_selection = {
                    **runtime_selection,
                    "selected_runtime": DEFAULT_RUNTIME,
                    "selection_mode": "fallback",
                    "runtime_fallback_reason": fallback_reason,
                    "warnings": [*(runtime_selection.get("warnings") or []), fallback_reason],
                    "runtime_policy_status": "fallback",
                    "runtime_policy_reason": "OpenHands launch failed after runtime policy selection.",
                }
                profile.update({
                    "runtime_selection": runtime_selection,
                    "selected_runtime": DEFAULT_RUNTIME,
                    "runtime_selection_reason": runtime_selection["reason"],
                    "runtime_fallback_reason": fallback_reason,
                    "runtime_policy_evidence": runtime_selection.get("runtime_policy_evidence") or {},
                    "runtime_policy_status": runtime_selection.get("runtime_policy_status"),
                    "runtime_policy_reason": runtime_selection.get("runtime_policy_reason"),
                })
                if minimal_worker_prompt:
                    launch_prompt = profiled_prompt
                else:
                    launch_prompt = build_ao_worker_prompt(profiled_prompt, goal=task.get("goal"))
                try:
                    session = router.spawn(
                        runtime,
                        project_id=profile["project_id"],
                        prompt=launch_prompt,
                        issue_id=(task.get("payload") or {}).get("issue_id"),
                        branch=(task.get("payload") or {}).get("branch"),
                        agent=profile.get("agent"),
                        model=profile.get("model"),
                        reasoning_effort=profile.get("reasoning_effort"),
                        minimal_worker_prompt=minimal_worker_prompt,
                    )
                except Exception as fallback_exc:
                    failures.append({"task_id": task.get("task_id"), "goal": task.get("goal"), "error": str(fallback_exc)})
                    continue
            else:
                failures.append({"task_id": task.get("task_id"), "goal": task.get("goal"), "error": str(exc)})
                continue

        metadata = {
            "runtime": runtime,
            "prompt": task.get("prompt") or "",
            "goal": task.get("goal"),
            "project_id": profile["project_id"],
            "issue_id": (task.get("payload") or {}).get("issue_id"),
            "branch": (task.get("payload") or {}).get("branch"),
            "agent": getattr(session, "agent", None) or profile.get("agent"),
            "model": getattr(session, "model", None) or profile.get("model"),
            "reasoning_effort": getattr(session, "reasoning_effort", None) or profile.get("reasoning_effort"),
            "launch_profile_id": profile.get("launch_profile_id"),
            "launch_plan_id": plan_id,
            "launch_task_id": task.get("task_id"),
            "permissions": profile.get("permissions"),
            "acceptance_criteria": task.get("acceptance_criteria") or [],
            "runtime_selection": runtime_selection,
            "selected_runtime": runtime_selection["selected_runtime"],
            "runtime_selection_reason": runtime_selection["reason"],
            "runtime_fallback_reason": runtime_selection.get("runtime_fallback_reason"),
            "runtime_policy_evidence": runtime_selection.get("runtime_policy_evidence") or {},
            "runtime_policy_status": runtime_selection.get("runtime_policy_status"),
            "runtime_policy_reason": runtime_selection.get("runtime_policy_reason"),
        }
        preview = f"{_runtime_label(runtime)} session {session.id} spawned from Dev plan {plan_id}"
        if runtime == DEFAULT_RUNTIME:
            _emit(
                None,
                "subagent.start",
                session,
                task.get("goal") or session.id,
                preview=preview,
                tool_name="dev_launch_execution_plan",
                _ao_prompt_metadata=metadata,
                launch_profile_id=metadata["launch_profile_id"],
                launch_plan_id=plan_id,
                launch_task_id=task.get("task_id"),
                permissions=metadata["permissions"],
                acceptance_criteria=metadata["acceptance_criteria"],
                runtime_selection=runtime_selection,
                selected_runtime=runtime_selection["selected_runtime"],
                runtime_selection_reason=runtime_selection["reason"],
                runtime_fallback_reason=runtime_selection.get("runtime_fallback_reason"),
                runtime_policy_evidence=runtime_selection.get("runtime_policy_evidence") or {},
                runtime_policy_status=runtime_selection.get("runtime_policy_status"),
                runtime_policy_reason=runtime_selection.get("runtime_policy_reason"),
            )
            _persist_start_event_direct(
                session=session,
                goal=task.get("goal") or session.id,
                prompt=task.get("prompt") or "",
                project_id=profile["project_id"],
                issue_id=metadata["issue_id"],
                branch=metadata["branch"],
                preview=preview,
                tool_name="dev_launch_execution_plan",
                event_store=event_store,
                launch_profile_id=metadata["launch_profile_id"],
                launch_plan_id=plan_id,
                launch_task_id=task.get("task_id"),
                permissions=metadata["permissions"],
                acceptance_criteria=metadata["acceptance_criteria"],
                runtime_selection=runtime_selection,
                selected_runtime=runtime_selection["selected_runtime"],
                runtime_selection_reason=runtime_selection["reason"],
                runtime_fallback_reason=runtime_selection.get("runtime_fallback_reason"),
                runtime_policy_evidence=runtime_selection.get("runtime_policy_evidence") or {},
                runtime_policy_status=runtime_selection.get("runtime_policy_status"),
                runtime_policy_reason=runtime_selection.get("runtime_policy_reason"),
            )
        else:
            _persist_runtime_start_event(
                session=session,
                runtime=runtime,
                goal=task.get("goal") or session.id,
                prompt=task.get("prompt") or "",
                project_id=profile["project_id"],
                issue_id=metadata["issue_id"],
                branch=metadata["branch"],
                preview=preview,
                tool_name="dev_launch_execution_plan",
                event_store=event_store,
                metadata=metadata,
            )
        store.update_task_launch(plan_id=plan_id, task_id=task["task_id"], ao_session_id=session.id)
        launched.append({
            "task_id": task["task_id"],
            "goal": task.get("goal"),
            "ao_session_id": session.id,
            "runtime": metadata["runtime"],
            "runtime_session_id": session.id,
            "runtime_project_id": profile["project_id"],
            "runtime_selection": runtime_selection,
            "selected_runtime": runtime_selection["selected_runtime"],
            "runtime_selection_reason": runtime_selection["reason"],
            "runtime_fallback_reason": runtime_selection.get("runtime_fallback_reason"),
            "runtime_policy_evidence": runtime_selection.get("runtime_policy_evidence") or {},
            "runtime_policy_status": runtime_selection.get("runtime_policy_status"),
            "runtime_policy_reason": runtime_selection.get("runtime_policy_reason"),
            "session": session.event_fields(),
            "launch_profile_id": metadata["launch_profile_id"],
        })

    launch_record = None
    if draft_review:
        launch_record = store.append_launch_record(
            plan_id=plan_id,
            draft_review=draft_review,
            requested_task_ids=task_ids,
            launched=launched,
            failures=failures,
        )

    return {
        "ok": bool(launched),
        "plan": store.get_plan(plan_id),
        "launched": launched,
        "failures": failures,
        "launch_record": launch_record,
    }


def derive_execution_plan_status(
    *,
    store: DevExecutionStore,
    plan_id: str,
    bridge: Any = None,
    event_store: Any = None,
    verification_store: Any = None,
) -> Dict[str, Any]:
    """Derive truthful Dev plan/task status from linked AO sessions and events."""
    plan = store.get_plan(plan_id)
    if not plan:
        raise KeyError(f"Dev execution plan not found: {plan_id}")

    bridge = _ensure_runtime_router(bridge)
    tasks = [
        _derive_task_status(task, bridge=bridge, event_store=event_store)
        for task in (plan.get("tasks") or [])
    ]
    status = _rollup_plan_status(tasks)
    review = _review_decision_from_status(status=status, tasks=tasks)
    runbook = store.resolve_runbook_for_plan({**plan, "tasks": tasks})
    counts = _plan_action_counts(event_store=event_store, plan_id=plan_id, tasks=tasks)
    next_step, next_step_reason = _next_step_from_review(
        review={"status": status, **review},
        runbook=runbook,
        counts=counts,
        approval=store.latest_supervisor_approval(plan_id),
    )
    plan = dict(plan)
    plan["status"] = status
    plan["tasks"] = tasks
    plan.update(review)
    plan.update(_runbook_fields(runbook))
    max_followups = int((runbook.get("policy") or {}).get("max_follow_ups_per_task") or 0)
    max_retries = int((runbook.get("policy") or {}).get("max_retries_per_task") or 0)
    plan.update({
        "follow_up_count": counts["follow_up_count"],
        "retry_count": counts["retry_count"],
        "repair_retry_count": counts["repair_retry_count"],
        "reassign_count": counts["reassign_count"],
        "max_follow_ups_reached": max_followups > 0 and counts["follow_up_count"] >= max_followups,
        "max_retries_reached": max_retries > 0 and counts["retry_count"] >= max_retries,
        "next_step": next_step,
        "next_step_reason": next_step_reason,
    })
    if verification_store is not None:
        try:
            from gateway.dev_control.acceptance_verification import annotate_plan_with_verification

            plan = annotate_plan_with_verification(plan, verification_store)
            tasks = plan.get("tasks") or tasks
        except Exception:
            pass
    return {
        "ok": True,
        "object": "hermes.dev_execution_plan_status",
        "plan_id": plan_id,
        "status": status,
        **review,
        **_runbook_fields(runbook),
        "counts": counts,
        "next_step": next_step,
        "next_step_reason": next_step_reason,
        "plan": plan,
        "tasks": tasks,
        "summary": _plan_status_summary(tasks),
    }


def synthesize_execution_plan(
    *,
    store: DevExecutionStore,
    plan_id: str,
    bridge: Any = None,
    event_store: Any = None,
) -> Dict[str, Any]:
    """Return a compact Dev-facing implementation report for one plan."""
    status_payload = derive_execution_plan_status(
        store=store,
        plan_id=plan_id,
        bridge=bridge,
        event_store=event_store,
    )
    plan = status_payload["plan"]
    tasks = status_payload["tasks"]
    unresolved_gaps: list[str] = []
    report_lines = [
        f"{plan.get('title') or plan_id}: {status_payload['status']}",
        "",
        "Tasks:",
    ]
    for task in tasks:
        task_status = task.get("status") or "unknown"
        summary = _clip_text(task.get("summary") or task.get("status_reason") or "No summary.", 4000)
        session_text = f" ({task.get('ao_session_id')})" if task.get("ao_session_id") else ""
        report_lines.append(f"- {task.get('goal') or task.get('task_id')}: {task_status}{session_text}. {summary}")
        if task.get("summary_warning"):
            warning = f"{task.get('goal') or task.get('task_id')}: {task['summary_warning']}"
            unresolved_gaps.append(warning)
            report_lines.append(f"  Warning: {task['summary_warning']}")
        if task.get("recent_action"):
            report_lines.append(
                "  Recent action: "
                f"{task.get('recent_action')} {task.get('recent_action_status') or ''} "
                f"{task.get('recent_action_message') or ''}".strip()
            )
        files = _list_unique((task.get("files_read") or []) + (task.get("files_written") or []))
        if files:
            report_lines.append(f"  Files: {', '.join(files[:8])}")
        findings = task.get("findings") or []
        if findings:
            report_lines.append(f"  Findings: {'; '.join(str(item) for item in findings[:4])}")
        changed = task.get("files_changed") or []
        if changed:
            report_lines.append(f"  Files changed: {', '.join(str(item) for item in changed[:8])}")
        commands = task.get("commands_run") or []
        if commands:
            report_lines.append(f"  Commands: {'; '.join(str(item) for item in commands[:4])}")
        evidence = task.get("verification_evidence") or []
        if evidence:
            verification_status = task.get("verification_status")
            label = f"Verification ({verification_status})" if verification_status else "Verification"
            report_lines.append(f"  {label}: {'; '.join(str(item) for item in evidence[:4])}")
        gaps = task.get("unresolved_gaps") or []
        if gaps:
            for gap in gaps:
                unresolved_gaps.append(f"{task.get('goal') or task.get('task_id')}: {gap}")
            report_lines.append(f"  Unresolved gaps: {'; '.join(str(item) for item in gaps[:4])}")
        if task_status in {"failed", "needs_review"} and not task.get("summary_warning"):
            unresolved_gaps.append(f"{task.get('goal') or task.get('task_id')}: {task.get('status_reason') or task_status}")

    if unresolved_gaps:
        report_lines.extend(["", "Unresolved gaps:"])
        report_lines.extend(f"- {gap}" for gap in unresolved_gaps)

    return {
        "ok": True,
        "object": "hermes.dev_execution_plan_synthesis",
        "plan_id": plan_id,
        "status": status_payload["status"],
        "plan": plan,
        "tasks": tasks,
        "summary": status_payload["summary"],
        "unresolved_gaps": unresolved_gaps,
        "report": "\n".join(report_lines).strip(),
    }


def review_execution_plan(
    *,
    store: DevExecutionStore,
    plan_id: str,
    bridge: Any = None,
    event_store: Any = None,
    include_synthesis: bool = True,
) -> Dict[str, Any]:
    """Return Dev's on-demand review recommendation for an execution plan."""
    status_payload = derive_execution_plan_status(
        store=store,
        plan_id=plan_id,
        bridge=bridge,
        event_store=event_store,
    )
    synthesis = (
        synthesize_execution_plan(
            store=store,
            plan_id=plan_id,
            bridge=bridge,
            event_store=event_store,
        )
        if include_synthesis
        else None
    )
    unresolved_gaps = (synthesis or {}).get("unresolved_gaps") or []
    review = _review_decision_from_status(
        status=status_payload["status"],
        tasks=status_payload["tasks"],
        unresolved_gaps=unresolved_gaps,
    )
    return {
        "ok": True,
        "object": "hermes.dev_execution_plan_review",
        "plan_id": plan_id,
        "status": status_payload["status"],
        **review,
        "status_payload": status_payload,
        "synthesis": synthesis,
    }


def apply_execution_plan_review(
    *,
    store: DevExecutionStore,
    plan_id: str,
    bridge: Any = None,
    event_store: Any = None,
    include_synthesis: bool = True,
    message: Optional[str] = None,
    instruction: Optional[str] = None,
    project_id: Optional[str] = None,
    agent: Optional[str] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    expected_action: Optional[str] = None,
    target_task_ids: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Apply the current Dev review recommendation for a plan on demand."""
    bridge = _ensure_runtime_router(bridge)
    event_store = event_store
    review = review_execution_plan(
        store=store,
        plan_id=plan_id,
        bridge=bridge,
        event_store=event_store,
        include_synthesis=include_synthesis,
    )
    action = str(review.get("recommended_action") or "none")
    if expected_action and action != expected_action:
        return _review_application_response(
            plan_id=plan_id,
            review=review,
            applied_action=expected_action,
            results=[],
            skipped=[{
                "reason": f"Current recommendation is {action}, not approved action {expected_action}.",
                "target_task_ids": target_task_ids or [],
            }],
            status="skipped",
        )
    tasks = (review.get("status_payload") or {}).get("tasks") or []
    target_tasks = _review_target_tasks(action=action, review=review, tasks=tasks)
    if target_task_ids is not None:
        allowed = set(_normalize_task_ids(target_task_ids))
        target_tasks = [
            task for task in target_tasks
            if str(task.get("task_id") or "").strip() in allowed
        ]
    results: list[Dict[str, Any]] = []
    skipped: list[Dict[str, Any]] = []

    if action in {"none", "human_review"}:
        skipped.append({
            "reason": review.get("reason") or "Review recommendation is not actionable.",
            "target_task_ids": review.get("target_task_ids") or [],
        })
        return _review_application_response(
            plan_id=plan_id,
            review=review,
            applied_action=action,
            results=results,
            skipped=skipped,
            status="no_op",
        )

    if not target_tasks:
        skipped.append({
            "reason": "Review recommendation did not identify any actionable worker tasks.",
            "target_task_ids": review.get("target_task_ids") or [],
        })
        return _review_application_response(
            plan_id=plan_id,
            review=review,
            applied_action=action,
            results=results,
            skipped=skipped,
            status="skipped",
        )

    for task in target_tasks:
        task_id = str(task.get("task_id") or "").strip()
        source_session_id = str(task.get("ao_session_id") or "").strip()
        if not source_session_id:
            skipped.append({
                "task_id": task_id,
                "action": action,
                "reason": "Task has no linked worker session.",
            })
            continue
        try:
            if action == "accept":
                results.append(_apply_accept_review_action(
                    task=task,
                    bridge=bridge,
                    event_store=event_store,
                ))
            elif action == "follow_up":
                results.append(_apply_follow_up_review_action(
                    task=task,
                    bridge=bridge,
                    event_store=event_store,
                    message=(message or review.get("suggested_message") or ""),
                ))
            elif action in {"retry", "repair_retry", "reassign"}:
                results.append(_apply_spawn_review_action(
                    store=store,
                    plan_id=plan_id,
                    task=task,
                    bridge=bridge,
                    event_store=event_store,
                    review_action=action,
                    instruction=instruction or review.get("suggested_instruction"),
                    project_id=project_id,
                    agent=agent,
                    model=model,
                    reasoning_effort=reasoning_effort,
                ))
            else:
                skipped.append({
                    "task_id": task_id,
                    "ao_session_id": source_session_id,
                    "action": action,
                    "reason": f"Review action {action!r} is not supported by apply-review.",
                })
        except Exception as exc:
            skipped.append({
                "task_id": task_id,
                "ao_session_id": source_session_id,
                "action": action,
                "reason": str(exc),
            })

    return _review_application_response(
        plan_id=plan_id,
        review=review,
        applied_action=action,
        results=results,
        skipped=skipped,
        status="applied" if results else "skipped",
    )


def supervise_execution_plans(
    *,
    store: DevExecutionStore,
    plan_ids: Optional[list[str]] = None,
    limit: int = 20,
    project_id: Optional[str] = None,
    reviewable_only: bool = False,
    bridge: Any = None,
    event_store: Any = None,
    apply_guarded_actions: bool = True,
    include_synthesis: bool = False,
) -> Dict[str, Any]:
    """Audit recent Dev plans and apply only guarded low-risk recommendations."""
    bridge = _ensure_runtime_router(bridge)
    event_store = event_store
    started_at = time.time()
    run_id = f"devsup-{uuid.uuid4().hex[:10]}"
    plans: list[Dict[str, Any]] = []
    applied: list[Dict[str, Any]] = []
    skipped: list[Dict[str, Any]] = []

    selected_plan_ids = _supervisor_plan_ids(
        store=store,
        plan_ids=plan_ids,
        limit=limit,
        project_id=project_id,
        reviewable_only=reviewable_only,
    )
    for plan_id in selected_plan_ids:
        try:
            review = review_execution_plan(
                store=store,
                plan_id=plan_id,
                bridge=bridge,
                event_store=event_store,
                include_synthesis=include_synthesis,
            )
        except Exception as exc:
            item = {
                "plan_id": plan_id,
                "status": "error",
                "supervisor_status": "error",
                "message": str(exc),
            }
            skipped.append(item)
            plans.append(item)
            store.append_supervisor_record(
                run_id=run_id,
                plan_id=plan_id,
                status="error",
                action=None,
                message=str(exc),
                started_at=started_at,
                completed_at=time.time(),
                payload=item,
            )
            continue

        action = str(review.get("recommended_action") or "none")
        status_payload = review.get("status_payload") or {}
        tasks = status_payload.get("tasks") or []
        runbook = store.resolve_runbook_for_plan(status_payload.get("plan") or {"plan_id": plan_id, "tasks": tasks})
        policy = runbook.get("policy") or {}
        counts = _plan_action_counts(event_store=event_store, plan_id=plan_id, tasks=tasks)
        plan_item: Dict[str, Any] = {
            "plan_id": plan_id,
            "status": review.get("status"),
            "review_status": review.get("review_status"),
            "recommended_action": action,
            "reason": review.get("reason"),
            "confidence": review.get("confidence"),
            "target_task_ids": review.get("target_task_ids") or [],
            **_runbook_fields(runbook),
            "follow_up_count": counts["follow_up_count"],
            "retry_count": counts["retry_count"],
            "repair_retry_count": counts["repair_retry_count"],
            "reassign_count": counts["reassign_count"],
            "max_follow_ups_reached": int(policy.get("max_follow_ups_per_task") or 0) > 0
            and counts["follow_up_count"] >= int(policy.get("max_follow_ups_per_task") or 0),
            "max_retries_reached": int(policy.get("max_retries_per_task") or 0) > 0
            and counts["retry_count"] >= int(policy.get("max_retries_per_task") or 0),
        }
        result_status = "skipped"
        result_message = _supervisor_skip_reason(action=action, review=review)
        result_action = action

        if action in {"accept", "follow_up"}:
            target_tasks = _review_target_tasks(action=action, review=review, tasks=tasks)
            target_task_ids = _task_ids(target_tasks)
            policy_blocked = False
            if action == "accept" and not policy.get("auto_accept"):
                result_status = "skipped"
                result_message = "Current runbook does not auto-accept plans."
                policy_blocked = True
                skipped.append({
                    "plan_id": plan_id,
                    "action": action,
                    "status": result_status,
                    "message": result_message,
                    "review_status": review.get("review_status"),
                })
            elif action == "follow_up" and not policy.get("auto_follow_up"):
                result_status = "skipped"
                result_message = "Current runbook does not auto-send follow-ups."
                policy_blocked = True
                skipped.append({
                    "plan_id": plan_id,
                    "action": action,
                    "status": result_status,
                    "message": result_message,
                    "review_status": review.get("review_status"),
                })
            if policy_blocked:
                pass
            elif action == "follow_up" and policy.get("auto_follow_up"):
                max_followups = int(policy.get("max_follow_ups_per_task") or 0)
                eligible_tasks = _target_tasks_with_capacity(
                    tasks=target_tasks,
                    counts=counts,
                    count_key="follow_up_count",
                    max_count=max_followups,
                )
                if not eligible_tasks:
                    result_status = "skipped"
                    result_message = "Follow-up limit has been reached."
                    skipped.append({
                        "plan_id": plan_id,
                        "action": action,
                        "status": result_status,
                        "message": result_message,
                        "review_status": review.get("review_status"),
                        "target_task_ids": target_task_ids,
                    })
                elif not apply_guarded_actions:
                    result_status = "observed"
                    result_message = "Guarded action available but apply_guarded_actions is false."
                    skipped.append({
                        "plan_id": plan_id,
                        "action": action,
                        "status": result_status,
                        "message": result_message,
                        "review_status": review.get("review_status"),
                        "target_task_ids": _task_ids(eligible_tasks),
                    })
                elif not str(review.get("suggested_message") or "").strip():
                    result_status = "skipped"
                    result_message = "Follow-up recommendation did not include a suggested message."
                    skipped.append({
                        "plan_id": plan_id,
                        "action": action,
                        "status": result_status,
                        "message": result_message,
                        "review_status": review.get("review_status"),
                    })
                else:
                    application = apply_execution_plan_review(
                        store=store,
                        plan_id=plan_id,
                        bridge=bridge,
                        event_store=event_store,
                        include_synthesis=include_synthesis,
                        target_task_ids=_task_ids(eligible_tasks),
                    )
                    result_status = application.get("status") or "applied"
                    result_message = _supervisor_application_message(application)
                    plan_item["application"] = application
                    applied.append({
                        "plan_id": plan_id,
                        "action": action,
                        "status": result_status,
                        "message": result_message,
                        "application": application,
                    })
            elif _supervisor_action_already_applied(action=action, review=review):
                result_status = "skipped"
                result_message = "Guarded action already applied."
                skipped.append({
                    "plan_id": plan_id,
                    "action": action,
                    "status": result_status,
                    "message": result_message,
                    "review_status": review.get("review_status"),
                })
            elif not apply_guarded_actions:
                result_status = "observed"
                result_message = "Guarded action available but apply_guarded_actions is false."
                skipped.append({
                    "plan_id": plan_id,
                    "action": action,
                    "status": result_status,
                    "message": result_message,
                    "review_status": review.get("review_status"),
                })
            else:
                application = apply_execution_plan_review(
                    store=store,
                    plan_id=plan_id,
                    bridge=bridge,
                    event_store=event_store,
                    include_synthesis=include_synthesis,
                )
                result_status = application.get("status") or "applied"
                result_message = _supervisor_application_message(application)
                plan_item["application"] = application
                applied.append({
                    "plan_id": plan_id,
                    "action": action,
                    "status": result_status,
                    "message": result_message,
                    "application": application,
                })
        else:
            skipped_item = {
                "plan_id": plan_id,
                "action": action,
                "status": result_status,
                "message": result_message,
                "review_status": review.get("review_status"),
            }
            if action in APPROVABLE_SUPERVISOR_ACTIONS:
                approval = store.create_or_reuse_supervisor_approval(
                    plan_id=plan_id,
                    task_ids=review.get("target_task_ids") or [],
                    recommended_action=action,
                    reason=review.get("reason"),
                    suggested_instruction=review.get("suggested_instruction"),
                    action_overrides={},
                    payload={"review": review},
                )
                approval_fields = _supervisor_approval_fields(approval)
                skipped_item.update(approval_fields)
                plan_item.update(approval_fields)
                result_message = f"{result_message} Approval required."
                skipped_item["message"] = result_message
            skipped.append(skipped_item)

        plan_item.update({
            "supervisor_status": result_status,
            "supervisor_last_action": result_action,
            "supervisor_last_message": result_message,
            "supervisor_last_run_id": run_id,
        })
        plans.append(plan_item)
        store.append_supervisor_record(
            run_id=run_id,
            plan_id=plan_id,
            status=result_status,
            action=result_action,
            message=result_message,
            started_at=started_at,
            completed_at=time.time(),
            payload=plan_item,
        )

    completed_at = time.time()
    status = "completed"
    if any(plan.get("supervisor_status") == "error" for plan in plans):
        status = "completed_with_errors"
    return {
        "ok": True,
        "object": "hermes.dev_execution_plan_supervision_run",
        "run_id": run_id,
        "status": status,
        "plans": plans,
        "applied": applied,
        "skipped": skipped,
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _supervisor_plan_ids(
    *,
    store: DevExecutionStore,
    plan_ids: Optional[list[str]],
    limit: int,
    project_id: Optional[str] = None,
    reviewable_only: bool = False,
) -> list[str]:
    explicit = [str(plan_id).strip() for plan_id in (plan_ids or []) if str(plan_id).strip()]
    if explicit:
        return list(dict.fromkeys(explicit))
    bounded = max(1, min(int(limit or 20), 100))
    selected: list[str] = []
    scan_limit = max(bounded, min(200, bounded * 5))
    for plan in store.list_plans(limit=scan_limit):
        if project_id and project_id not in _plan_project_ids(plan):
            continue
        if reviewable_only and not _plan_reviewable_for_supervisor(plan):
            continue
        if plan.get("plan_id"):
            selected.append(plan["plan_id"])
        if len(selected) >= bounded:
            break
    return selected


def list_runbooks(
    *,
    store: DevExecutionStore,
    project_id: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    return {
        "ok": True,
        "object": "list",
        "data": store.list_runbooks(project_id=project_id, limit=limit),
        "policy_profiles": list_policy_profiles(),
    }


def list_supervisor_loop_status(
    *,
    store: DevExecutionStore,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    runbooks = store.list_runbooks(project_id=project_id, limit=200)
    states = {
        state["project_id"]: state
        for state in store.list_supervisor_loop_state(project_id=project_id)
        if state.get("project_id")
    }
    data = []
    for runbook in runbooks:
        project = runbook.get("project_id")
        if not project:
            continue
        state = states.get(project) or {}
        enabled = _as_bool(runbook.get("supervisor_enabled"), default=False)
        data.append({
            "project_id": project,
            "runbook_id": runbook.get("runbook_id"),
            "supervisor_enabled": enabled,
            "supervisor_interval_seconds": runbook.get("supervisor_interval_seconds"),
            "supervisor_limit": runbook.get("supervisor_limit"),
            "supervisor_include_synthesis": _as_bool(runbook.get("supervisor_include_synthesis"), default=False),
            "supervisor_apply_guarded_actions": _as_bool(runbook.get("supervisor_apply_guarded_actions"), default=True),
            "loop_state": state or None,
            "status": state.get("status") or ("enabled" if enabled else "disabled"),
            "last_run_id": state.get("last_run_id"),
            "last_tick_at": state.get("last_tick_at"),
            "next_tick_at": state.get("next_tick_at"),
            "last_message": state.get("last_message"),
            "consecutive_error_count": state.get("consecutive_error_count") or 0,
        })
    return {
        "ok": True,
        "object": "list",
        "data": data,
        "total": len(data),
    }


def set_supervisor_loop(
    *,
    store: DevExecutionStore,
    project_id: str,
    supervisor_enabled: Optional[bool] = None,
    supervisor_interval_seconds: Optional[int] = None,
    supervisor_limit: Optional[int] = None,
    supervisor_include_synthesis: Optional[bool] = None,
    supervisor_apply_guarded_actions: Optional[bool] = None,
) -> Dict[str, Any]:
    existing = store.get_project_runbook(project_id)
    runbook = store.upsert_project_runbook(
        project_id=project_id,
        policy_profile=(existing or {}).get("policy_profile") or DEFAULT_POLICY_PROFILE,
        max_follow_ups_per_task=(existing or {}).get("max_follow_ups_per_task"),
        max_retries_per_task=(existing or {}).get("max_retries_per_task"),
        supervisor_enabled=supervisor_enabled,
        supervisor_interval_seconds=supervisor_interval_seconds,
        supervisor_limit=supervisor_limit,
        supervisor_include_synthesis=supervisor_include_synthesis,
        supervisor_apply_guarded_actions=supervisor_apply_guarded_actions,
    )
    return {
        "ok": True,
        "object": "hermes.dev_supervisor_loop",
        "project_id": project_id,
        "runbook": runbook,
        "loop": (list_supervisor_loop_status(store=store, project_id=project_id).get("data") or [None])[0],
    }


def set_execution_plan_test_state(
    *,
    store: DevExecutionStore,
    plan_id: str,
    task_id: str,
    state: str,
    event_store: Any,
    summary: Optional[str] = None,
    status_reason: Optional[str] = None,
    ao_session_id: Optional[str] = None,
    runtime: Optional[str] = None,
    project_id: Optional[str] = None,
    files_read: Optional[list[str]] = None,
    files_written: Optional[list[str]] = None,
    verification_evidence: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Inject deterministic normalized subagent lifecycle events for Dev supervisor tests."""
    if not event_store:
        raise ValueError("subagent event store is required")
    normalized_state = str(state or "").strip().lower()
    if normalized_state not in DEV_TEST_STATES:
        allowed = ", ".join(sorted(DEV_TEST_STATES))
        raise ValueError(f"Unsupported Dev test state: {state}. Expected one of: {allowed}")
    plan = store.get_plan(plan_id)
    if not plan:
        raise KeyError(f"Dev execution plan not found: {plan_id}")
    task = next((item for item in plan.get("tasks") or [] if item.get("task_id") == task_id), None)
    if not task:
        raise KeyError(f"Dev execution plan task not found: {task_id}")

    fixture_session_id = str(ao_session_id or f"fixture-{task_id}").strip()
    resolved_project_id = str(project_id or task.get("project_id") or _first_non_empty(*_plan_project_ids(plan)) or "OrynWorkspace")
    resolved_runtime = "fixture"
    goal = task.get("goal") or task_id
    state_status = {
        "completed_ok": "completed",
        "completed_weak": "completed",
        "failed_repairable": "failed",
        "failed_unrepairable": "failed",
        "running": "running",
    }[normalized_state]
    default_summary = {
        "completed_ok": "PHASE16_FIXTURE_OK_DONE Verified deterministic fixture completion with no unresolved gaps.",
        "completed_weak": "unclear",
        "failed_repairable": "Fixture worker failed and can be repair-retried.",
        "failed_unrepairable": "Fixture worker failed without reliable prompt metadata.",
        "running": "Fixture worker is running.",
    }[normalized_state]
    final_summary = str(summary or default_summary)
    has_prompt_metadata = normalized_state != "failed_unrepairable"
    now = time.time()

    store.update_task_launch(
        plan_id=plan_id,
        task_id=task_id,
        ao_session_id=fixture_session_id,
        status="running" if normalized_state == "running" else "launched",
    )
    if has_prompt_metadata:
        try:
            event_store.upsert_ao_prompt(
                ao_session_id=fixture_session_id,
                project_id=resolved_project_id,
                prompt=task.get("prompt") or final_summary,
                goal=goal,
                issue_id=(task.get("payload") or {}).get("issue_id") if isinstance(task.get("payload"), dict) else None,
                branch=(task.get("payload") or {}).get("branch") if isinstance(task.get("payload"), dict) else None,
                agent="fixture",
                model="fixture",
                reasoning_effort="test",
                launch_profile_id=task.get("profile_id"),
                launch_plan_id=plan_id,
                launch_task_id=task_id,
                permissions="fixture",
                acceptance_criteria=task.get("acceptance_criteria") or [],
            )
        except Exception:
            pass

    base_payload = {
        "subagent_id": f"fixture:{task_id}",
        "runtime": resolved_runtime,
        "runtime_session_id": fixture_session_id,
        "runtime_project_id": resolved_project_id,
        "fixture": True,
        "ao_session_id": fixture_session_id,
        "ao_project_id": resolved_project_id,
        "goal": goal,
        "agent": "fixture",
        "model": "fixture",
        "reasoning_effort": "test",
        "launch_plan_id": plan_id,
        "launch_task_id": task_id,
        "launch_profile_id": task.get("profile_id"),
        "permissions": "fixture",
        "acceptance_criteria": task.get("acceptance_criteria") or [],
        "has_prompt_metadata": has_prompt_metadata,
        "files_read": files_read or [],
        "files_written": files_written or [],
        "verification_evidence": verification_evidence or [],
        "timestamp": now,
    }
    start_event = event_store.append_event({
        **base_payload,
        "event": "subagent.start",
        "status": "running",
        "summary": None,
        "message": f"Fixture {normalized_state} state injected for {task_id}.",
        "preview": f"Fixture {normalized_state} state injected.",
    })
    if normalized_state == "running":
        final_event = event_store.append_event({
            **base_payload,
            "event": "subagent.progress",
            "status": "running",
            "summary": final_summary,
            "message": status_reason or final_summary,
            "preview": status_reason or final_summary,
        })
    else:
        final_event = event_store.append_event({
            **base_payload,
            "event": "subagent.complete",
            "status": state_status,
            "summary": final_summary,
            "message": status_reason or final_summary,
            "preview": status_reason or final_summary,
        })
    return {
        "ok": True,
        "object": "hermes.dev_execution_plan_test_state",
        "plan_id": plan_id,
        "task_id": task_id,
        "state": normalized_state,
        "ao_session_id": fixture_session_id,
        "runtime": resolved_runtime,
        "events": [start_event, final_event],
    }


def supervisor_loop_tick(
    *,
    store: DevExecutionStore,
    event_store: Any = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    now = time.time() if now is None else now
    ticks: list[Dict[str, Any]] = []
    for runbook in store.list_enabled_supervisor_runbooks():
        project_id = str(runbook.get("project_id") or "").strip()
        if not project_id:
            continue
        interval = _bounded_positive_int(
            runbook.get("supervisor_interval_seconds"),
            default=DEFAULT_SUPERVISOR_INTERVAL_SECONDS,
            minimum=MIN_SUPERVISOR_INTERVAL_SECONDS,
            maximum=MAX_SUPERVISOR_INTERVAL_SECONDS,
        )
        current_state = store.get_supervisor_loop_state(project_id) or {}
        if float(current_state.get("next_tick_at") or 0) > now:
            continue
        try:
            result = supervise_execution_plans(
                store=store,
                project_id=project_id,
                limit=int(runbook.get("supervisor_limit") or DEFAULT_SUPERVISOR_LIMIT),
                reviewable_only=True,
                event_store=event_store,
                apply_guarded_actions=_as_bool(runbook.get("supervisor_apply_guarded_actions"), default=True),
                include_synthesis=_as_bool(runbook.get("supervisor_include_synthesis"), default=False),
            )
            state = store.record_supervisor_loop_tick(
                project_id=project_id,
                runbook_id=runbook.get("runbook_id"),
                status=result.get("status") or "completed",
                last_run_id=result.get("run_id"),
                last_tick_at=now,
                next_tick_at=now + interval,
                last_message=_loop_result_message(result),
                consecutive_error_count=0,
                payload=result,
            )
            ticks.append({"project_id": project_id, "result": result, "loop_state": state})
        except Exception as exc:
            error_count = int(current_state.get("consecutive_error_count") or 0) + 1
            state = store.record_supervisor_loop_tick(
                project_id=project_id,
                runbook_id=runbook.get("runbook_id"),
                status="error",
                last_run_id=None,
                last_tick_at=now,
                next_tick_at=now + interval,
                last_message=str(exc),
                consecutive_error_count=error_count,
                payload={"error": str(exc)},
            )
            ticks.append({"project_id": project_id, "error": str(exc), "loop_state": state})
    return {
        "ok": True,
        "object": "hermes.dev_supervisor_loop_tick",
        "status": "completed",
        "tick_count": len(ticks),
        "ticks": ticks,
        "checked_at": now,
    }


def get_runbook(*, store: DevExecutionStore, runbook_id: str) -> Dict[str, Any]:
    runbook = store.get_runbook(runbook_id)
    if not runbook:
        raise KeyError(f"Dev runbook not found: {runbook_id}")
    return {
        "ok": True,
        "object": "hermes.dev_runbook",
        "runbook": runbook,
        "policy_profiles": list_policy_profiles(),
    }


def set_project_runbook(
    *,
    store: DevExecutionStore,
    project_id: str,
    policy_profile: str = DEFAULT_POLICY_PROFILE,
    max_follow_ups_per_task: Optional[int] = None,
    max_retries_per_task: Optional[int] = None,
    supervisor_enabled: Optional[bool] = None,
    supervisor_interval_seconds: Optional[int] = None,
    supervisor_limit: Optional[int] = None,
    supervisor_include_synthesis: Optional[bool] = None,
    supervisor_apply_guarded_actions: Optional[bool] = None,
) -> Dict[str, Any]:
    runbook = store.upsert_project_runbook(
        project_id=project_id,
        policy_profile=policy_profile,
        max_follow_ups_per_task=max_follow_ups_per_task,
        max_retries_per_task=max_retries_per_task,
        supervisor_enabled=supervisor_enabled,
        supervisor_interval_seconds=supervisor_interval_seconds,
        supervisor_limit=supervisor_limit,
        supervisor_include_synthesis=supervisor_include_synthesis,
        supervisor_apply_guarded_actions=supervisor_apply_guarded_actions,
    )
    return {
        "ok": True,
        "object": "hermes.dev_runbook",
        "runbook": runbook,
        "policy_profiles": list_policy_profiles(),
    }


def next_execution_step(
    *,
    store: DevExecutionStore,
    plan_id: str,
    bridge: Any = None,
    event_store: Any = None,
    include_synthesis: bool = False,
) -> Dict[str, Any]:
    review = review_execution_plan(
        store=store,
        plan_id=plan_id,
        bridge=bridge,
        event_store=event_store,
        include_synthesis=include_synthesis,
    )
    plan = (review.get("status_payload") or {}).get("plan") or store.get_plan(plan_id) or {}
    runbook = store.resolve_runbook_for_plan(plan)
    counts = _plan_action_counts(
        event_store=event_store,
        plan_id=plan_id,
        tasks=(review.get("status_payload") or {}).get("tasks") or plan.get("tasks") or [],
    )
    approval = store.latest_supervisor_approval(plan_id)
    next_step, reason = _next_step_from_review(
        review=review,
        runbook=runbook,
        counts=counts,
        approval=approval,
    )
    return {
        "ok": True,
        "object": "hermes.dev_execution_plan_next_step",
        "plan_id": plan_id,
        "status": review.get("status"),
        "review_status": review.get("review_status"),
        "recommended_action": review.get("recommended_action"),
        "next_step": next_step,
        "reason": reason,
        "target_task_ids": review.get("target_task_ids") or [],
        "approval_id": (approval or {}).get("approval_id"),
        "approval_status": (approval or {}).get("status"),
        "runbook": _runbook_fields(runbook)["runbook"],
        "counts": counts,
        "review": review,
    }


def list_supervisor_approvals(
    *,
    store: DevExecutionStore,
    status: Optional[str] = None,
    plan_id: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    approvals = store.list_supervisor_approvals(status=status, plan_id=plan_id, limit=limit)
    return {
        "ok": True,
        "object": "list",
        "data": approvals,
        "total": len(approvals),
    }


def get_supervisor_approval(*, store: DevExecutionStore, approval_id: str) -> Dict[str, Any]:
    approval = store.get_supervisor_approval(approval_id)
    if not approval:
        raise KeyError(f"Supervisor approval not found: {approval_id}")
    return {
        "ok": True,
        "object": "hermes.dev_supervisor_approval",
        "approval": approval,
    }


def approve_supervisor_approval(
    *,
    store: DevExecutionStore,
    approval_id: str,
    resolved_by: Optional[str] = None,
    message: Optional[str] = None,
    instruction: Optional[str] = None,
    project_id: Optional[str] = None,
    agent: Optional[str] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    approval = store.resolve_supervisor_approval(
        approval_id=approval_id,
        status="approved",
        resolved_by=resolved_by or "dev",
        resolution_message=message or "Approved.",
        action_overrides={
            "instruction": instruction,
            "project_id": project_id,
            "agent": agent,
            "model": model,
            "reasoning_effort": reasoning_effort,
        },
    )
    return {
        "ok": True,
        "object": "hermes.dev_supervisor_approval_resolution",
        "approval": approval,
    }


def deny_supervisor_approval(
    *,
    store: DevExecutionStore,
    approval_id: str,
    resolved_by: Optional[str] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    approval = store.resolve_supervisor_approval(
        approval_id=approval_id,
        status="denied",
        resolved_by=resolved_by or "dev",
        resolution_message=message or "Denied.",
    )
    return {
        "ok": True,
        "object": "hermes.dev_supervisor_approval_resolution",
        "approval": approval,
    }


def apply_supervisor_approval(
    *,
    store: DevExecutionStore,
    approval_id: str,
    bridge: Any = None,
    event_store: Any = None,
    include_synthesis: bool = True,
) -> Dict[str, Any]:
    bridge = _ensure_runtime_router(bridge)
    approval = store.get_supervisor_approval(approval_id)
    if not approval:
        raise KeyError(f"Supervisor approval not found: {approval_id}")
    if approval.get("status") != "approved":
        return _supervisor_approval_application_response(
            approval=approval,
            application=None,
            status="rejected",
            message=f"Supervisor approval is {approval.get('status')}, not approved.",
        )
    action = str(approval.get("recommended_action") or "").strip()
    if action not in APPROVABLE_SUPERVISOR_ACTIONS:
        return _supervisor_approval_application_response(
            approval=approval,
            application=None,
            status="rejected",
            message=f"Action {action} is not approvable.",
        )
    overrides = approval.get("action_overrides") or {}
    application = apply_execution_plan_review(
        store=store,
        plan_id=str(approval.get("plan_id") or ""),
        bridge=bridge,
        event_store=event_store,
        include_synthesis=include_synthesis,
        instruction=overrides.get("instruction") or approval.get("suggested_instruction"),
        project_id=overrides.get("project_id"),
        agent=overrides.get("agent"),
        model=overrides.get("model"),
        reasoning_effort=overrides.get("reasoning_effort"),
        expected_action=action,
        target_task_ids=approval.get("task_ids") or [],
    )
    if application.get("status") == "applied" and application.get("results"):
        consumed = store.consume_supervisor_approval(
            approval_id,
            message=f"Applied {action}.",
        )
        return _supervisor_approval_application_response(
            approval=consumed,
            application=application,
            status="applied",
            message=f"Applied {action}.",
        )
    return _supervisor_approval_application_response(
        approval=approval,
        application=application,
        status="rejected",
        message=_supervisor_application_message(application),
    )


def _supervisor_approval_application_response(
    *,
    approval: Dict[str, Any],
    application: Optional[Dict[str, Any]],
    status: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "ok": status == "applied",
        "object": "hermes.dev_supervisor_approval_application",
        "approval_id": approval.get("approval_id"),
        "plan_id": approval.get("plan_id"),
        "status": status,
        "message": message,
        "approval": approval,
        "application": application,
    }


def _supervisor_skip_reason(*, action: str, review: Dict[str, Any]) -> str:
    if action == "none":
        return review.get("reason") or "Plan is not ready for supervision."
    if action in {"retry", "repair_retry", "reassign", "human_review"}:
        return f"{action.replace('_', '-')} requires manual approval."
    return review.get("reason") or f"Action {action} is not guarded."


def _supervisor_approval_fields(approval: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "approval_id": approval.get("approval_id"),
        "approval_status": approval.get("status"),
        "approval_expires_at": approval.get("expires_at"),
        "supervisor_approval_id": approval.get("approval_id"),
        "supervisor_approval_status": approval.get("status"),
        "supervisor_approval_expires_at": approval.get("expires_at"),
    }


def _supervisor_action_already_applied(*, action: str, review: Dict[str, Any]) -> bool:
    tasks = (review.get("status_payload") or {}).get("tasks") or []
    target_tasks = _review_target_tasks(action=action, review=review, tasks=tasks)
    if not target_tasks:
        return False
    expected_action = "follow-up" if action == "follow_up" else action
    expected_statuses = {"accepted"} if action == "accept" else {"succeeded", "success", "sent"}
    for task in target_tasks:
        recent_action = str(task.get("recent_action") or "").lower()
        recent_status = str(task.get("recent_action_status") or "").lower()
        if recent_action != expected_action or recent_status not in expected_statuses:
            return False
    return True


def _normalize_task_ids(task_ids: Iterable[Any]) -> list[str]:
    return sorted({
        str(task_id).strip()
        for task_id in (task_ids or [])
        if str(task_id).strip()
    })


def _plan_project_ids(plan: Dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for task in plan.get("tasks") or []:
        project_id = str(task.get("project_id") or "").strip()
        if project_id:
            ids.append(project_id)
    return list(dict.fromkeys(ids))


def _plan_reviewable_for_supervisor(plan: Dict[str, Any]) -> bool:
    status = str(plan.get("status") or "").lower()
    if status in {"completed", "needs_review", "failed"}:
        return True
    if str(plan.get("supervisor_approval_status") or "").lower() in {"pending", "approved"}:
        return True
    if any(str(task.get("ao_session_id") or "").strip() for task in plan.get("tasks") or []):
        return True
    return False


def _loop_result_message(result: Dict[str, Any]) -> str:
    applied = result.get("applied") or []
    skipped = result.get("skipped") or []
    if applied:
        return f"Applied {len(applied)} guarded supervisor action(s)."
    if skipped:
        return f"Skipped {len(skipped)} supervisor recommendation(s)."
    return "No reviewable plans found."


def _plan_action_counts(
    *,
    event_store: Any,
    plan_id: str,
    tasks: list[Dict[str, Any]],
) -> Dict[str, Any]:
    counts = {
        "follow_up_count": 0,
        "retry_count": 0,
        "repair_retry_count": 0,
        "reassign_count": 0,
        "by_task": {},
    }
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        if task_id:
            counts["by_task"][task_id] = {
                "follow_up_count": 0,
                "retry_count": 0,
                "repair_retry_count": 0,
                "reassign_count": 0,
            }
    if not event_store:
        return counts
    try:
        events = event_store.list_events(limit=2000)
    except Exception:
        return counts
    for event in events:
        if event.get("event") != "subagent.action":
            continue
        if str(event.get("launch_plan_id") or "").strip() != str(plan_id or "").strip():
            continue
        task_id = str(event.get("launch_task_id") or "").strip()
        action = str(event.get("action") or "").strip().lower().replace("-", "_")
        key = {
            "follow_up": "follow_up_count",
            "retry": "retry_count",
            "repair_retry": "repair_retry_count",
            "reassign": "reassign_count",
        }.get(action)
        if not key:
            continue
        counts[key] += 1
        if task_id:
            counts["by_task"].setdefault(task_id, {
                "follow_up_count": 0,
                "retry_count": 0,
                "repair_retry_count": 0,
                "reassign_count": 0,
            })
            counts["by_task"][task_id][key] += 1
    return counts


def _task_count(counts: Dict[str, Any], task_id: str, key: str) -> int:
    return int(((counts.get("by_task") or {}).get(task_id) or {}).get(key) or 0)


def _target_tasks_with_capacity(
    *,
    tasks: list[Dict[str, Any]],
    counts: Dict[str, Any],
    count_key: str,
    max_count: int,
) -> list[Dict[str, Any]]:
    if max_count <= 0:
        return []
    return [
        task for task in tasks
        if _task_count(counts, str(task.get("task_id") or ""), count_key) < max_count
    ]


def _next_step_from_review(
    *,
    review: Dict[str, Any],
    runbook: Dict[str, Any],
    counts: Dict[str, Any],
    approval: Optional[Dict[str, Any]],
) -> tuple[str, str]:
    status = str(review.get("status") or "").lower()
    action = str(review.get("recommended_action") or "none").lower()
    policy = runbook.get("policy") or {}
    if status in {"planned", "launched", "running", "partially_completed"}:
        return "wait", "Execution plan still has tasks that are not terminal."
    if approval and approval.get("status") == "approved":
        return "apply_approval", "Approved supervisor action is ready to apply."
    if approval and approval.get("status") == "pending":
        return "approve", "Supervisor approval is waiting for a decision."
    if action == "human_review":
        return "ask_human", review.get("reason") or "Human review is required."
    if action == "follow_up":
        max_followups = int(policy.get("max_follow_ups_per_task") or 0)
        if not policy.get("auto_follow_up"):
            return "ask_human", "Current runbook does not auto-send follow-ups."
        if max_followups <= 0 or counts.get("follow_up_count", 0) >= max_followups:
            return "ask_human", "Follow-up limit has been reached."
        return "supervise", "Run supervisor to send the allowed follow-up."
    if action in APPROVABLE_SUPERVISOR_ACTIONS:
        return "supervise", "Run supervisor to create a durable approval request."
    if action == "accept":
        return "synthesize", "Plan is accepted; synthesize the final implementation report."
    if status == "completed":
        return "synthesize", "Plan is completed; synthesize the final implementation report."
    return "none", review.get("reason") or "No next Dev action is required."


def _supervisor_application_message(application: Dict[str, Any]) -> str:
    for result in application.get("results") or []:
        if result.get("message"):
            return str(result["message"])
    for skipped in application.get("skipped") or []:
        if skipped.get("reason"):
            return str(skipped["reason"])
    return str(application.get("status") or "applied")


def _review_application_response(
    *,
    plan_id: str,
    review: Dict[str, Any],
    applied_action: str,
    results: list[Dict[str, Any]],
    skipped: list[Dict[str, Any]],
    status: str,
) -> Dict[str, Any]:
    return {
        "ok": status != "skipped" or bool(results) or applied_action in {"none", "human_review"},
        "object": "hermes.dev_execution_plan_review_application",
        "plan_id": plan_id,
        "review": review,
        "applied_action": applied_action,
        "status": status,
        "results": results,
        "skipped": skipped,
    }


def _review_target_tasks(*, action: str, review: Dict[str, Any], tasks: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    target_ids = {
        str(task_id).strip()
        for task_id in (review.get("target_task_ids") or [])
        if str(task_id).strip()
    }
    if target_ids:
        return [task for task in tasks if str(task.get("task_id") or "").strip() in target_ids]
    if action == "accept":
        return [
            task for task in tasks
            if str(task.get("ao_session_id") or "").strip()
            and str(task.get("status") or "").lower() == "completed"
        ]
    return []


def _apply_accept_review_action(*, task: Dict[str, Any], bridge: Any, event_store: Any) -> Dict[str, Any]:
    source_session_id = str(task.get("ao_session_id") or "").strip()
    runtime = _task_runtime(task, _review_prompt_metadata(event_store, task), task.get("last_event"))
    session = _find_runtime_session(bridge, runtime, source_session_id, project_id=task.get("project_id"))
    action_event = _append_review_action_event(
        event_store=event_store,
        task=task,
        action="accept",
        source_session_id=source_session_id,
        status="accepted",
        message="Dev review accepted this AO task.",
        session=session,
    )
    return _review_action_result(
        task=task,
        source_session_id=source_session_id,
        action="accept",
        status="accepted",
        message="Dev review accepted this AO task.",
        action_event=action_event,
    )


def _apply_follow_up_review_action(
    *,
    task: Dict[str, Any],
    bridge: Any,
    event_store: Any,
    message: str,
) -> Dict[str, Any]:
    source_session_id = str(task.get("ao_session_id") or "").strip()
    message = str(message or "").strip()
    if not message:
        raise ValueError("Follow-up recommendation has no message to send.")
    if _is_fixture_task(task):
        latest = _latest_event(_events_for_ao_session(event_store, source_session_id), include_actions=False)
        action_event = _append_review_action_event(
            event_store=event_store,
            task=task,
            action="follow-up",
            source_session_id=source_session_id,
            status="succeeded",
            message="Fixture follow-up recorded",
            base_event=latest,
            extra={"sent_message": message, "fixture": True},
        )
        return _review_action_result(
            task=task,
            source_session_id=source_session_id,
            action="follow-up",
            status="succeeded",
            message="Fixture follow-up recorded",
            action_event=action_event,
        )
    runtime = _task_runtime(task, _review_prompt_metadata(event_store, task), task.get("last_event"))
    session = bridge.send(runtime, source_session_id, message) or _find_runtime_session(
        bridge,
        runtime,
        source_session_id,
        project_id=task.get("project_id"),
    )
    action_event = _append_review_action_event(
        event_store=event_store,
        task=task,
        action="follow-up",
        source_session_id=source_session_id,
        status="succeeded",
        message="Follow-up sent",
        session=session,
        extra={"sent_message": message},
    )
    return _review_action_result(
        task=task,
        source_session_id=source_session_id,
        action="follow-up",
        status="succeeded",
        message="Follow-up sent",
        action_event=action_event,
    )


def _is_fixture_task(task: Dict[str, Any]) -> bool:
    runtime = str(task.get("runtime") or "").strip().lower()
    session_id = str(task.get("ao_session_id") or "").strip()
    return runtime == "fixture" or session_id.startswith("fixture-") or bool(task.get("fixture"))


def _apply_spawn_review_action(
    *,
    store: DevExecutionStore,
    plan_id: str,
    task: Dict[str, Any],
    bridge: Any,
    event_store: Any,
    review_action: str,
    instruction: Optional[str],
    project_id: Optional[str],
    agent: Optional[str],
    model: Optional[str],
    reasoning_effort: Optional[str],
) -> Dict[str, Any]:
    source_session_id = str(task.get("ao_session_id") or "").strip()
    prompt_meta = _review_prompt_metadata(event_store, task)
    original_prompt = str(prompt_meta.get("prompt") or "").strip()
    if not original_prompt:
        raise ValueError("Original AO prompt metadata is unavailable for this task.")

    mode = _review_action_event_name(review_action)
    latest = _latest_event(_events_for_ao_session(event_store, source_session_id), include_actions=False)
    source_runtime = _task_runtime(task, prompt_meta, task.get("last_event"))
    source_session = _find_runtime_session(bridge, source_runtime, source_session_id, project_id=task.get("project_id"))
    action_runtime = DEFAULT_RUNTIME
    diagnostic_context = ""
    if review_action == "repair_retry":
        diagnostic_context = _repair_retry_diagnostic_context(
            bridge=bridge,
            runtime=source_runtime,
            session=source_session,
        )

    prior_summary = _first_non_empty(
        task.get("summary"),
        task.get("status_reason"),
        (latest or {}).get("summary"),
        (latest or {}).get("message"),
        (latest or {}).get("preview"),
    )
    requested_instruction = str(instruction or "").strip()
    if diagnostic_context:
        requested_instruction = "\n\n".join(part for part in (requested_instruction, diagnostic_context) if part)

    resolved_project_id = str(project_id or prompt_meta.get("project_id") or task.get("project_id") or "OrynWorkspace")
    resolved_agent = agent or prompt_meta.get("agent")
    resolved_model = model or prompt_meta.get("model")
    resolved_reasoning_effort = reasoning_effort or prompt_meta.get("reasoning_effort")
    goal = prompt_meta.get("goal") or task.get("goal") or f"{mode.title()} AO worker"
    prompt = _related_review_action_prompt(
        mode=mode,
        original_prompt=original_prompt,
        prior_summary=prior_summary,
        instruction=requested_instruction,
        model=resolved_model,
    )

    from tools.ao_delegate_tool import build_ao_worker_prompt

    runtime_prompt = build_ao_worker_prompt(prompt, goal=f"{mode.title()}: {goal}")
    session = bridge.spawn(
        action_runtime,
        project_id=resolved_project_id,
        prompt=runtime_prompt,
        issue_id=prompt_meta.get("issue_id"),
        agent=resolved_agent,
        model=resolved_model,
        reasoning_effort=resolved_reasoning_effort,
    )
    if event_store:
        event_store.upsert_ao_prompt(
            ao_session_id=session.id,
            project_id=resolved_project_id,
            prompt=prompt,
            goal=f"{mode.title()}: {goal}",
            issue_id=prompt_meta.get("issue_id"),
            branch=getattr(session, "branch", None),
            agent=getattr(session, "agent", None) or resolved_agent,
            model=getattr(session, "model", None) or resolved_model,
            reasoning_effort=getattr(session, "reasoning_effort", None) or resolved_reasoning_effort,
            launch_profile_id=prompt_meta.get("launch_profile_id") or task.get("profile_id"),
            launch_plan_id=plan_id,
            launch_task_id=task.get("task_id"),
            permissions=prompt_meta.get("permissions") or ((task.get("payload") or {}).get("resolved_profile") or {}).get("permissions"),
            acceptance_criteria=prompt_meta.get("acceptance_criteria") or task.get("acceptance_criteria") or [],
            runtime_selection={
                "selected_runtime": action_runtime,
                "selection_mode": "profile",
                "reason": "Retry, repair-retry, and reassign actions use AO in this phase.",
                "candidate_runtimes": [action_runtime],
                "fallback_runtime": None,
                "required_capabilities": ["can_spawn", "supports_worktree", "supports_terminal", "can_capture_output"],
                "warnings": [],
                "runtime_fallback_reason": None,
            },
            selected_runtime=action_runtime,
            runtime_selection_reason="Retry, repair-retry, and reassign actions use AO in this phase.",
        )
        _append_lifecycle_start_event(
            event_store=event_store,
            task=task,
            session=session,
            parent_event=latest,
            mode=mode,
            goal=goal,
            prompt_meta=prompt_meta,
        )

    store.update_task_launch(
        plan_id=plan_id,
        task_id=str(task.get("task_id") or ""),
        ao_session_id=session.id,
        status="launched",
    )
    action_event = _append_review_action_event(
        event_store=event_store,
        task=task,
        action=mode,
        source_session_id=source_session_id,
        target_session_id=session.id,
        status="succeeded",
        message=f"{_runtime_label(action_runtime)} {mode} session spawned",
        session=session,
        base_event=latest,
    )
    return _review_action_result(
        task=task,
        source_session_id=source_session_id,
        action=mode,
        status="succeeded",
        message=f"{_runtime_label(action_runtime)} {mode} session spawned",
        target_session_id=session.id,
        action_event=action_event,
    )


def _review_action_result(
    *,
    task: Dict[str, Any],
    source_session_id: str,
    action: str,
    status: str,
    message: str,
    target_session_id: Optional[str] = None,
    action_event: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "ao_session_id": source_session_id,
        "source_ao_session_id": source_session_id,
        "target_ao_session_id": target_session_id,
        "action": action,
        "status": status,
        "message": message,
        "action_event": action_event,
    }


def _append_review_action_event(
    *,
    event_store: Any,
    task: Dict[str, Any],
    action: str,
    source_session_id: str,
    status: str,
    message: str,
    session: Any = None,
    target_session_id: Optional[str] = None,
    base_event: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not event_store:
        return None
    event_payload: Dict[str, Any] = dict(base_event or {})
    if session is not None:
        try:
            event_payload.update(session.event_fields())
        except Exception:
            pass
    event_payload.update({
        "event": "subagent.action",
        "subagent_id": event_payload.get("subagent_id") or f"{event_payload.get('runtime') or task.get('runtime') or DEFAULT_RUNTIME}:{source_session_id}",
        "runtime": event_payload.get("runtime") or task.get("runtime") or DEFAULT_RUNTIME,
        "runtime_session_id": event_payload.get("runtime_session_id") or source_session_id,
        "runtime_project_id": event_payload.get("runtime_project_id") or event_payload.get("ao_project_id") or task.get("project_id"),
        "action": action,
        "action_status": status,
        "status": event_payload.get("status") or status,
        "source_ao_session_id": source_session_id,
        "target_ao_session_id": target_session_id,
        "message": message,
        "preview": message,
        "launch_plan_id": task.get("plan_id"),
        "launch_task_id": task.get("task_id"),
        "launch_profile_id": task.get("profile_id") or event_payload.get("launch_profile_id"),
        "goal": event_payload.get("goal") or task.get("goal"),
        "timestamp": time.time(),
    })
    resolved_profile = (task.get("payload") or {}).get("resolved_profile") or {}
    for key in (
        "runtime_selection",
        "selected_runtime",
        "runtime_selection_reason",
        "runtime_fallback_reason",
        "runtime_policy_evidence",
        "runtime_policy_status",
        "runtime_policy_reason",
    ):
        value = event_payload.get(key) or task.get(key) or resolved_profile.get(key)
        if value is not None:
            event_payload[key] = value
    if event_payload.get("runtime") in {DEFAULT_RUNTIME, "fixture"}:
        event_payload["ao_session_id"] = source_session_id
    else:
        event_payload.pop("ao_session_id", None)
        event_payload.pop("ao_project_id", None)
    if extra:
        event_payload.update(extra)
    event_payload.pop("event_id", None)
    event_payload.pop("created_at", None)
    return event_store.append_event(event_payload)


def _append_lifecycle_start_event(
    *,
    event_store: Any,
    task: Dict[str, Any],
    session: Any,
    parent_event: Optional[Dict[str, Any]],
    mode: str,
    goal: str,
    prompt_meta: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not event_store:
        return None
    try:
        session_fields = session.event_fields()
    except Exception:
        session_fields = {}
    payload = {
        "event": "subagent.start",
        "subagent_id": f"{session_fields.get('runtime') or DEFAULT_RUNTIME}:{getattr(session, 'id', '')}",
        "parent_id": (parent_event or {}).get("subagent_id"),
        "depth": int((parent_event or {}).get("depth") or 0),
        "goal": f"{mode.title()}: {goal}",
        "runtime": session_fields.get("runtime") or DEFAULT_RUNTIME,
        "runtime_session_id": getattr(session, "id", None),
        "runtime_project_id": session_fields.get("runtime_project_id") or prompt_meta.get("project_id") or task.get("project_id"),
        "status": session_fields.get("status") or "running",
        "message": f"{_runtime_label(session_fields.get('runtime') or DEFAULT_RUNTIME)} {mode} session spawned from {task.get('ao_session_id')}",
        "preview": f"{_runtime_label(session_fields.get('runtime') or DEFAULT_RUNTIME)} {mode} session spawned from {task.get('ao_session_id')}",
        "launch_profile_id": prompt_meta.get("launch_profile_id") or task.get("profile_id"),
        "launch_plan_id": task.get("plan_id"),
        "launch_task_id": task.get("task_id"),
        "permissions": prompt_meta.get("permissions") or ((task.get("payload") or {}).get("resolved_profile") or {}).get("permissions"),
        "acceptance_criteria": prompt_meta.get("acceptance_criteria") or task.get("acceptance_criteria") or [],
        "timestamp": time.time(),
    }
    resolved_profile = (task.get("payload") or {}).get("resolved_profile") or {}
    for key in (
        "runtime_selection",
        "selected_runtime",
        "runtime_selection_reason",
        "runtime_fallback_reason",
        "runtime_policy_evidence",
        "runtime_policy_status",
        "runtime_policy_reason",
    ):
        value = prompt_meta.get(key) or task.get(key) or resolved_profile.get(key)
        if value is not None:
            payload[key] = value
    payload.update(session_fields)
    if payload.get("runtime") in {DEFAULT_RUNTIME, "fixture"}:
        payload["ao_session_id"] = getattr(session, "id", None)
    else:
        payload.pop("ao_session_id", None)
        payload.pop("ao_project_id", None)
    return event_store.append_event(payload)


def _review_prompt_metadata(event_store: Any, task: Dict[str, Any]) -> Dict[str, Any]:
    ao_session_id = str(task.get("ao_session_id") or "").strip()
    metadata: Dict[str, Any] = {}
    if event_store and ao_session_id:
        try:
            metadata = event_store.get_ao_prompt(ao_session_id) or {}
        except Exception:
            metadata = {}
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    profile = payload.get("resolved_profile") if isinstance(payload.get("resolved_profile"), dict) else {}
    if ao_session_id.startswith("oryn-workspace-"):
        runtime = DEFAULT_RUNTIME
    elif ao_session_id.startswith("fixture-"):
        runtime = "fixture"
    else:
        runtime = metadata.get("runtime") or task.get("runtime") or profile.get("runtime") or DEFAULT_RUNTIME
    return {
        "prompt": metadata.get("prompt") or task.get("prompt"),
        "goal": metadata.get("goal") or task.get("goal"),
        "project_id": metadata.get("project_id") or task.get("project_id") or profile.get("project_id"),
        "runtime": runtime,
        "issue_id": metadata.get("issue_id") or payload.get("issue_id"),
        "branch": metadata.get("branch") or payload.get("branch"),
        "agent": metadata.get("agent") or profile.get("agent"),
        "model": metadata.get("model") or profile.get("model"),
        "reasoning_effort": metadata.get("reasoning_effort") or profile.get("reasoning_effort"),
        "launch_profile_id": metadata.get("launch_profile_id") or task.get("profile_id") or profile.get("launch_profile_id"),
        "launch_plan_id": metadata.get("launch_plan_id") or task.get("plan_id"),
        "launch_task_id": metadata.get("launch_task_id") or task.get("task_id"),
        "permissions": metadata.get("permissions") or profile.get("permissions"),
        "acceptance_criteria": metadata.get("acceptance_criteria") or task.get("acceptance_criteria") or [],
    }


def _review_action_event_name(action: str) -> str:
    return {
        "follow_up": "follow-up",
        "repair_retry": "repair-retry",
    }.get(action, action)


def _repair_retry_diagnostic_context(*, bridge: Any, runtime: Optional[str], session: Any) -> str:
    if session is None:
        return "Recovery diagnostics:\nRuntime health: missing\nRuntime warning: Worker session metadata is unavailable."
    parts = ["Recovery diagnostics:"]
    try:
        health = bridge.runtime_health(runtime, session) or {}
        parts.append(f"Runtime health: {health.get('runtime_health') or 'unknown'}")
        if health.get("runtime_warning"):
            parts.append(f"Runtime warning: {health.get('runtime_warning')}")
    except Exception as exc:
        parts.append(f"Runtime warning: diagnostics failed: {exc}")
    try:
        tail = bridge.capture_output(runtime, session, lines=120) or ""
    except Exception:
        tail = ""
    if tail:
        parts.extend(["Recent transcript tail:", tail[-8000:]])
    return "\n".join(parts)


def _related_review_action_prompt(
    *,
    mode: str,
    original_prompt: str,
    prior_summary: Optional[str],
    instruction: str,
    model: Optional[str],
) -> str:
    parts = [
        f"You are a {mode} worker for a Dev execution plan review action.",
        "",
        "Original task:",
        original_prompt.strip(),
    ]
    if prior_summary:
        parts.extend(["", "Previous session summary/status:", str(prior_summary).strip()])
    if model:
        parts.extend(["", f"Requested model/agent preference: {model}"])
    if instruction:
        parts.extend(["", "Additional instruction:", instruction])
    return append_worker_output_contract("\n".join(part for part in parts if part is not None))


def _review_decision_from_status(
    *,
    status: str,
    tasks: list[Dict[str, Any]],
    unresolved_gaps: Optional[list[str]] = None,
) -> Dict[str, Any]:
    unresolved_gaps = unresolved_gaps or []
    normalized_status = str(status or "planned").lower()
    if normalized_status in {"planned", "launched", "running", "partially_completed"}:
        active = [
            str(task.get("task_id") or "")
            for task in tasks
            if str(task.get("status") or "planned").lower() in {"planned", "launched", "running"}
        ]
        return {
            "review_status": "not_ready",
            "recommended_action": "none",
            "reason": "Execution plan still has tasks that are not terminal.",
            "confidence": 0.9,
            "target_task_ids": [task_id for task_id in active if task_id],
            "suggested_message": None,
            "suggested_instruction": None,
        }

    failed_tasks = [task for task in tasks if str(task.get("status") or "").lower() == "failed"]
    weak_tasks = [
        task
        for task in tasks
        if str(task.get("status") or "").lower() == "needs_review"
        or bool(task.get("summary_warning"))
        or str(task.get("summary_quality") or "").lower() == "warning"
    ]

    if failed_tasks:
        without_prompt = [task for task in failed_tasks if not _task_has_prompt_metadata(task)]
        target_tasks = without_prompt or failed_tasks
        if without_prompt:
            return {
                "review_status": "human_review_required",
                "recommended_action": "human_review",
                "reason": "One or more failed tasks do not have stored prompt metadata for a reliable repair retry.",
                "confidence": 0.86,
                "target_task_ids": _task_ids(target_tasks),
                "suggested_message": None,
                "suggested_instruction": "Inspect the failed worker manually, then create a new scoped Dev plan task.",
            }
        return {
            "review_status": "retry_recommended",
            "recommended_action": "repair_retry",
            "reason": "One or more tasks failed, and prompt metadata is available for repair retry.",
            "confidence": 0.84,
            "target_task_ids": _task_ids(failed_tasks),
            "suggested_message": None,
            "suggested_instruction": "Repair retry with the original prompt, prior status, summary, and diagnostics context.",
        }

    if weak_tasks or unresolved_gaps:
        target_tasks = weak_tasks or tasks
        if any(_task_worker_available(task) for task in target_tasks):
            return {
                "review_status": "needs_follow_up",
                "recommended_action": "follow_up",
                "reason": "The plan reached a terminal state, but one or more task summaries are weak or incomplete.",
                "confidence": 0.78,
                "target_task_ids": _task_ids(target_tasks),
                "suggested_message": "Please provide a concise final implementation summary with verification evidence and unresolved gaps.",
                "suggested_instruction": None,
            }
        return {
            "review_status": "retry_recommended",
            "recommended_action": "retry",
            "reason": "The plan reached a terminal state with weak or missing summaries, and the original worker is unavailable.",
            "confidence": 0.72,
            "target_task_ids": _task_ids(target_tasks),
            "suggested_message": None,
            "suggested_instruction": "Retry the affected task and require a concrete final summary with verification evidence.",
        }

    if normalized_status == "completed":
        return {
            "review_status": "accepted",
            "recommended_action": "accept",
            "reason": "All tasks completed with acceptable summaries and no unresolved gaps.",
            "confidence": 0.88,
            "target_task_ids": [],
            "suggested_message": None,
            "suggested_instruction": None,
        }

    return {
        "review_status": "human_review_required",
        "recommended_action": "human_review",
        "reason": f"Plan status {normalized_status} does not map to a deterministic review recommendation.",
        "confidence": 0.55,
        "target_task_ids": _task_ids(tasks),
        "suggested_message": None,
        "suggested_instruction": "Review the plan status and task evidence manually.",
    }


def _task_ids(tasks: list[Dict[str, Any]]) -> list[str]:
    return [str(task.get("task_id") or "").strip() for task in tasks if str(task.get("task_id") or "").strip()]


def _task_worker_available(task: Dict[str, Any]) -> bool:
    if not str(task.get("ao_session_id") or "").strip():
        return False
    if str(task.get("runtime_health") or "").lower() not in {"", "ok", "unknown"}:
        return False
    if str(task.get("status") or "").lower() == "failed":
        return False
    return True


def _task_has_prompt_metadata(task: Dict[str, Any]) -> bool:
    if task.get("has_prompt_metadata") is False:
        return False
    if str(task.get("prompt") or "").strip():
        return True
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    if str((payload or {}).get("original_prompt") or "").strip():
        return True
    if isinstance((payload or {}).get("resolved_profile"), dict):
        return True
    return False


def _ensure_runtime_router(bridge: Any = None) -> WorkerRuntimeRouter:
    if isinstance(bridge, WorkerRuntimeRouter):
        return bridge
    return WorkerRuntimeRouter(ao_bridge=bridge)


def _derive_task_status(task: Dict[str, Any], *, bridge: Any, event_store: Any = None) -> Dict[str, Any]:
    derived = dict(task)
    ao_session_id = str(task.get("ao_session_id") or "").strip()
    if not ao_session_id:
        runtime = _task_runtime(task, None, None)
        profile_payload = (task.get("payload") or {}).get("resolved_profile") or {}
        derived.update({
            "status": "planned",
            "status_reason": "Task has not been launched.",
            "runtime": runtime,
            "runtime_session_id": None,
            "runtime_project_id": task.get("project_id"),
            "runtime_selection": profile_payload.get("runtime_selection"),
            "selected_runtime": profile_payload.get("selected_runtime") or runtime,
            "runtime_selection_reason": profile_payload.get("runtime_selection_reason"),
            "runtime_fallback_reason": profile_payload.get("runtime_fallback_reason"),
            "summary_quality": "pending",
            "summary_warning": None,
        })
        return derived

    events = _events_for_ao_session(event_store, ao_session_id)
    task_events = _events_for_task(events, task)
    latest_event = _latest_event(task_events, include_actions=False)
    latest_action = _latest_event(task_events, include_actions=True, only_actions=True)
    prompt_meta = _ao_prompt_metadata(event_store, ao_session_id)
    runtime = _task_runtime(task, prompt_meta, latest_event)
    session_reused = _ao_session_reused_by_newer_lifecycle(events, task_events)
    session = None if session_reused else _find_runtime_session(
        bridge,
        runtime,
        ao_session_id,
        project_id=task.get("project_id"),
    )
    if runtime == "fixture":
        runtime_health = {"runtime_health": "ok", "runtime_warning": None}
    else:
        runtime_health = _runtime_health(bridge, runtime, session)
    goal_text = " ".join(str(part or "") for part in (
        task.get("goal"),
        task.get("prompt"),
        (prompt_meta or {}).get("goal"),
        (prompt_meta or {}).get("prompt"),
    ))
    synced_event = _sync_completion_from_transcript(
        task=task,
        session=session,
        bridge=bridge,
        runtime=runtime,
        event_store=event_store,
        task_events=task_events,
        latest_event=latest_event,
        goal_text=goal_text,
    )
    if synced_event:
        events.append(synced_event)
        task_events.append(synced_event)
        latest_event = _latest_event(task_events, include_actions=False)
        latest_action = _latest_event(task_events, include_actions=True, only_actions=True)

    if session is None and latest_event is None:
        task_status = "failed"
        reason = f"{_runtime_label(runtime)} session metadata is unavailable."
        session_summary = None
    else:
        task_status, reason = _status_from_session_or_event(session, latest_event, runtime_health)
        if session_reused and latest_event is not None and task_status == "failed":
            reason = "Worker session id was reused by a newer lifecycle; using this task's persisted terminal event."
        session_summary = _session_attr(session, "summary")

    event_summary = _first_non_empty(
        (latest_event or {}).get("summary"),
        (latest_event or {}).get("message"),
        (latest_event or {}).get("preview"),
    )
    usable_session_summary = None if _is_status_like_summary(session_summary) else session_summary
    usable_event_summary = None if _is_status_like_summary(event_summary) else event_summary
    summary = _first_non_empty(usable_session_summary, usable_event_summary, session_summary, event_summary)
    if (
        task_status == "completed"
        and event_summary
        and _summary_warning(goal_text, summary, "completed")
        and not _summary_warning(goal_text, event_summary, "completed")
    ):
        summary = event_summary
    contract_fields = output_contract_fields_from_event(latest_event)
    if not contract_fields.get("output_contract_status"):
        contract_text = "\n".join(
            str(part or "")
            for part in (
                summary,
                event_summary,
                (latest_event or {}).get("output_tail"),
            )
        )
        contract_fields = parse_worker_output_contract(contract_text)
        marker = contract_fields.get("final_marker")
        contract_fields["output_contract_score"] = worker_output_contract_score(contract_fields, required_marker=marker)
    if contract_fields.get("structured_summary"):
        summary = contract_fields["structured_summary"]
    summary_for_warning = summary
    if contract_fields.get("final_marker"):
        summary_for_warning = f"{summary or ''}\n{contract_fields['final_marker']}"
    summary_warning = _summary_warning(goal_text, summary_for_warning, task_status)
    if (
        task_status == "completed"
        and contract_fields.get("unresolved_gaps")
        and not summary_warning
    ):
        summary_warning = "Worker reported unresolved gaps in structured evidence."
    if task_status == "completed" and summary_warning:
        task_status = "needs_review"
        reason = summary_warning

    files_read = _list_unique(
        (latest_event or {}).get("files_read") or contract_fields.get("files_read") or []
    )
    files_written = _list_unique(
        (latest_event or {}).get("files_written") or contract_fields.get("files_changed") or []
    )
    evidence = _list_unique((contract_fields.get("verification_evidence") or []) + _verification_evidence(summary, task_events))
    profile_payload = (task.get("payload") or {}).get("resolved_profile") or {}
    runtime_selection = _first_non_empty(
        (latest_event or {}).get("runtime_selection"),
        (prompt_meta or {}).get("runtime_selection"),
        profile_payload.get("runtime_selection"),
    )
    runtime_policy_evidence = _first_non_empty(
        (latest_event or {}).get("runtime_policy_evidence"),
        (runtime_selection or {}).get("runtime_policy_evidence") if isinstance(runtime_selection, dict) else None,
        profile_payload.get("runtime_policy_evidence"),
    )
    runtime_policy_status = _first_non_empty(
        (latest_event or {}).get("runtime_policy_status"),
        (runtime_selection or {}).get("runtime_policy_status") if isinstance(runtime_selection, dict) else None,
        profile_payload.get("runtime_policy_status"),
    )
    runtime_policy_reason = _first_non_empty(
        (latest_event or {}).get("runtime_policy_reason"),
        (runtime_selection or {}).get("runtime_policy_reason") if isinstance(runtime_selection, dict) else None,
        profile_payload.get("runtime_policy_reason"),
    )

    derived.update({
        "status": task_status,
        "derived_status": task_status,
        "status_reason": reason,
        "summary": summary,
        "summary_quality": "warning" if summary_warning else ("pending" if task_status in {"planned", "running", "launched"} else "ok"),
        "summary_warning": summary_warning,
        **contract_fields,
        "runtime": runtime,
        "runtime_session_id": ao_session_id,
        "runtime_project_id": _first_non_empty(_session_attr(session, "project_id"), task.get("project_id"), (prompt_meta or {}).get("project_id")),
        "runtime_selection": runtime_selection,
        "runtime_policy_evidence": runtime_policy_evidence,
        "runtime_policy_status": runtime_policy_status,
        "runtime_policy_reason": runtime_policy_reason,
        "selected_runtime": _first_non_empty(
            (latest_event or {}).get("selected_runtime"),
            (prompt_meta or {}).get("selected_runtime"),
            runtime if runtime != profile_payload.get("selected_runtime") else None,
            profile_payload.get("selected_runtime"),
            runtime,
        ),
        "runtime_selection_reason": _first_non_empty((latest_event or {}).get("runtime_selection_reason"), (prompt_meta or {}).get("runtime_selection_reason"), profile_payload.get("runtime_selection_reason")),
        "runtime_fallback_reason": _first_non_empty((latest_event or {}).get("runtime_fallback_reason"), (prompt_meta or {}).get("runtime_fallback_reason"), profile_payload.get("runtime_fallback_reason")),
        "ao_project_id": _first_non_empty(_session_attr(session, "project_id"), task.get("project_id"), (prompt_meta or {}).get("project_id")),
        "agent": _first_non_empty(_session_attr(session, "agent"), (prompt_meta or {}).get("agent")),
        "model": _first_non_empty(_session_attr(session, "model"), (prompt_meta or {}).get("model")),
        "reasoning_effort": _first_non_empty(_session_attr(session, "reasoning_effort"), (prompt_meta or {}).get("reasoning_effort")),
        "workspace_path": _session_attr(session, "workspace_path"),
        "branch": _first_non_empty(_session_attr(session, "branch"), (prompt_meta or {}).get("branch")),
        "tmux_name": _session_attr(session, "tmux_name"),
        "runtime_health": runtime_health.get("runtime_health"),
        "runtime_warning": runtime_health.get("runtime_warning"),
        "recent_action": (latest_action or {}).get("action"),
        "recent_action_status": (latest_action or {}).get("action_status") or (latest_action or {}).get("status"),
        "recent_action_message": (latest_action or {}).get("message"),
        "recent_action_at": (latest_action or {}).get("created_at"),
        "files_read": files_read,
        "files_written": files_written,
        "verification_evidence": evidence,
        "has_prompt_metadata": (
            bool((latest_event or {}).get("has_prompt_metadata"))
            if "has_prompt_metadata" in (latest_event or {})
            else None
        ),
        "last_event": latest_event,
    })
    return derived


def _events_for_task(events: list[Dict[str, Any]], task: Dict[str, Any]) -> list[Dict[str, Any]]:
    plan_id = str(task.get("plan_id") or "").strip()
    task_id = str(task.get("task_id") or "").strip()
    matched = [
        event for event in events
        if (not plan_id or event.get("launch_plan_id") == plan_id)
        and (not task_id or event.get("launch_task_id") == task_id)
    ]
    if matched:
        return matched

    updated_at = _float_or_none(task.get("updated_at"))
    if updated_at is None:
        return events
    return [
        event for event in events
        if _float_or_none(event.get("created_at") or event.get("timestamp")) is not None
        and (_float_or_none(event.get("created_at") or event.get("timestamp")) or 0) >= updated_at - 2
    ] or events


def _ao_session_reused_by_newer_lifecycle(
    all_events: list[Dict[str, Any]],
    task_events: list[Dict[str, Any]],
) -> bool:
    latest_task_lifecycle = _latest_event(task_events, include_actions=False)
    latest_global_lifecycle = _latest_event(all_events, include_actions=False)
    if not latest_task_lifecycle or not latest_global_lifecycle:
        return False
    return _event_identity(latest_task_lifecycle) != _event_identity(latest_global_lifecycle)


def _sync_completion_from_transcript(
    *,
    task: Dict[str, Any],
    session: Any,
    bridge: Any,
    runtime: Optional[str],
    event_store: Any,
    task_events: list[Dict[str, Any]],
    latest_event: Optional[Dict[str, Any]],
    goal_text: str,
) -> Optional[Dict[str, Any]]:
    if not event_store or session is None:
        return None
    markers = _required_completion_markers(goal_text)
    if _task_has_terminal_lifecycle(task_events):
        existing_summary = _first_non_empty(
            (latest_event or {}).get("summary"),
            (latest_event or {}).get("message"),
            (latest_event or {}).get("preview"),
        )
        existing_warning = _summary_warning(goal_text, existing_summary, "completed")
        if not existing_warning:
            return None
    try:
        transcript = bridge.capture_output(runtime, session, lines=160) or ""
    except Exception:
        return None
    marker_complete = bool(markers) and all(marker in transcript for marker in markers)
    transcript_complete = marker_complete or _transcript_has_terminal_answer(transcript)
    if not transcript_complete:
        return None

    summary = (
        _extract_completion_summary(transcript, markers)
        if marker_complete
        else _extract_terminal_transcript_summary(transcript)
    )
    if not summary:
        return None
    existing_summary = _first_non_empty(
        (latest_event or {}).get("summary"),
        (latest_event or {}).get("message"),
        (latest_event or {}).get("preview"),
    )
    if (
        ((latest_event or {}).get("transcript_corrected") or (latest_event or {}).get("transcript_inferred_completion"))
        and summary.strip() == (existing_summary or "").strip()
    ):
        return None
    payload: Dict[str, Any] = dict(latest_event or {})
    try:
        payload.update(session.event_fields())
    except Exception:
        pass
    payload.update({
        "event": "subagent.complete",
        "subagent_id": payload.get("subagent_id") or f"{runtime or DEFAULT_RUNTIME}:{task.get('ao_session_id')}",
        "runtime": runtime or DEFAULT_RUNTIME,
        "runtime_session_id": task.get("ao_session_id"),
        "runtime_project_id": payload.get("runtime_project_id") or task.get("project_id"),
        "goal": task.get("goal") or payload.get("goal"),
        "status": "completed",
        "summary": summary,
        "message": summary,
        "preview": summary,
        "transcript_corrected": marker_complete,
        "transcript_inferred_completion": not marker_complete,
        "tool": payload.get("tool") or "dev_execution_plan_status",
        "tool_name": payload.get("tool_name") or "dev_execution_plan_status",
        "launch_profile_id": payload.get("launch_profile_id") or task.get("profile_id"),
        "launch_plan_id": task.get("plan_id"),
        "launch_task_id": task.get("task_id"),
        "permissions": payload.get("permissions") or ((task.get("payload") or {}).get("resolved_profile") or {}).get("permissions"),
        "acceptance_criteria": payload.get("acceptance_criteria") or task.get("acceptance_criteria") or [],
        "timestamp": time.time(),
    })
    contract_fields = parse_worker_output_contract(transcript)
    marker = contract_fields.get("final_marker") or (markers[0] if len(markers) == 1 else None)
    contract_fields["output_contract_score"] = worker_output_contract_score(contract_fields, required_marker=marker)
    payload.update(contract_fields)
    if (runtime or DEFAULT_RUNTIME) in {DEFAULT_RUNTIME, "fixture"}:
        payload["ao_session_id"] = task.get("ao_session_id")
        payload["ao_project_id"] = payload.get("ao_project_id") or task.get("project_id")
    else:
        payload.pop("ao_session_id", None)
        payload.pop("ao_project_id", None)
    payload.pop("event_id", None)
    payload.pop("created_at", None)
    try:
        return event_store.append_event(payload)
    except Exception:
        return None


def _task_has_terminal_lifecycle(events: list[Dict[str, Any]]) -> bool:
    for event in reversed(events):
        if event.get("event") == "subagent.action":
            continue
        status = str(event.get("status") or "").lower()
        event_type = str(event.get("event") or "").lower()
        if event_type == "subagent.complete" or status in TASK_COMPLETED_STATUSES or status in TASK_FAILED_STATUSES:
            return True
    return False


def _events_for_ao_session(event_store: Any, ao_session_id: str) -> list[Dict[str, Any]]:
    if not event_store:
        return []
    try:
        events = event_store.list_events(ao_session_id=ao_session_id, limit=1000)
        if events:
            return events
        # The task table still stores the linked runtime session id in the
        # legacy ao_session_id column. Non-AO runtimes keep AO-only metadata out
        # of event payloads, so resolve their events by normalized subagent id.
        for runtime in ("openhands", "fixture", DEFAULT_RUNTIME):
            events = event_store.list_events(subagent_id=f"{runtime}:{ao_session_id}", limit=1000)
            if events:
                return events
        return []
    except Exception:
        return []


def _ao_prompt_metadata(event_store: Any, ao_session_id: str) -> Optional[Dict[str, Any]]:
    if not event_store:
        return None
    try:
        return event_store.get_ao_prompt(ao_session_id)
    except Exception:
        return None


def _task_runtime(
    task: Dict[str, Any],
    prompt_meta: Optional[Dict[str, Any]],
    latest_event: Optional[Dict[str, Any]],
) -> str:
    session_id = str(task.get("ao_session_id") or "").strip()
    if not latest_event:
        if session_id.startswith("oryn-workspace-"):
            return DEFAULT_RUNTIME
        if session_id.startswith("fixture-"):
            return "fixture"
    profile = (task.get("payload") or {}).get("resolved_profile") or {}
    return normalize_runtime(
        (latest_event or {}).get("runtime")
        or (prompt_meta or {}).get("runtime")
        or profile.get("runtime")
    )


def _find_runtime_session(
    bridge: Any,
    runtime: Optional[str],
    session_id: str,
    *,
    project_id: Optional[str],
) -> Any:
    try:
        session = bridge.status(runtime, session_id)
        if session:
            return session
    except Exception:
        pass
    try:
        for session in bridge.list(runtime, project_id=project_id):
            if getattr(session, "id", None) == session_id:
                return session
    except Exception:
        pass
    return None


def _runtime_health(bridge: Any, runtime: Optional[str], session: Any) -> Dict[str, Any]:
    if session is None:
        return {"runtime_health": "missing", "runtime_warning": "Worker session metadata is unavailable."}
    try:
        health = bridge.runtime_health(runtime, session) or {}
        return {
            "runtime_health": health.get("runtime_health") or "ok",
            "runtime_warning": health.get("runtime_warning"),
        }
    except Exception as exc:
        return {"runtime_health": "unknown", "runtime_warning": f"Worker runtime health check failed: {exc}"}


def _status_from_session_or_event(session: Any, latest_event: Optional[Dict[str, Any]], runtime_health: Dict[str, Any]) -> tuple[str, str]:
    health = str(runtime_health.get("runtime_health") or "").lower()
    if health == "stale":
        return "failed", runtime_health.get("runtime_warning") or "Worker runtime is stale."

    event_status = str((latest_event or {}).get("status") or "").lower()
    if event_status in TASK_COMPLETED_STATUSES:
        return "completed", "Worker session completed."
    if event_status in TASK_FAILED_STATUSES:
        return "failed", f"Worker session ended with status {event_status}."
    if event_status in TASK_RUNNING_STATUSES or event_status == "running":
        return "running", "Worker session is running."

    raw_status = _first_non_empty(
        _session_attr(session, "display_status"),
        _session_attr(session, "status"),
    )
    status = str(raw_status or "").lower()
    if status in TASK_COMPLETED_STATUSES:
        return "completed", "Worker session completed."
    if status in TASK_FAILED_STATUSES:
        return "failed", f"Worker session ended with status {status}."
    if status in TASK_RUNNING_STATUSES or status == "running":
        return "running", "Worker session is running."
    if session is None:
        return "failed", "Worker session metadata is unavailable."
    return "running", "Worker session has been launched."


def _rollup_plan_status(tasks: list[Dict[str, Any]]) -> str:
    if not tasks:
        return "planned"
    statuses = [str(task.get("status") or "planned").lower() for task in tasks]
    if any(status == "failed" for status in statuses):
        return "failed"
    if all(status == "planned" for status in statuses):
        return "planned"
    if all(status in {"completed", "needs_review"} for status in statuses):
        return "needs_review" if any(status == "needs_review" for status in statuses) else "completed"
    if any(status in {"completed", "needs_review"} for status in statuses):
        return "partially_completed"
    if any(status == "running" for status in statuses):
        return "running"
    return "launched"


def _plan_status_summary(tasks: list[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for task in tasks:
        status = str(task.get("status") or "planned")
        counts[status] = counts.get(status, 0) + 1
    return {
        "task_count": len(tasks),
        "counts": counts,
        "ao_session_ids": [task["ao_session_id"] for task in tasks if task.get("ao_session_id")],
        "needs_review_count": counts.get("needs_review", 0),
        "failed_count": counts.get("failed", 0),
    }


def _summary_warning(goal_text: str, summary: Optional[str], status: str) -> Optional[str]:
    if status != "completed":
        return None
    text = (summary or "").strip()
    if not text:
        return "Completed worker did not provide a summary."
    if _is_status_like_summary(text):
        return "Completed worker summary appears to be terminal status/progress text instead of final results."
    if len(text) < 24:
        return "Completed worker summary is too short to verify."
    lowered = text.lower()
    if any(pattern in lowered for pattern in PROMPT_ECHO_SUMMARY_PATTERNS):
        return "Completed worker summary appears to echo the prompt or worker contract instead of reporting results."
    if any(pattern in lowered for pattern in WEAK_SUMMARY_PATTERNS):
        return "Completed worker summary looks weak or inconclusive."
    required_markers = set(_required_completion_markers(goal_text))
    if required_markers and not any(marker in text for marker in required_markers):
        markers = ", ".join(sorted(required_markers))
        return f"Completed worker summary is missing required completion marker: {markers}."
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3:
        toolish = sum(1 for line in lines if any(word in line for word in ("searched", "read", "grep", "rg ", "inspected", "activated")))
        if toolish / max(1, len(lines)) >= 0.75:
            return "Completed worker summary mostly describes tool activity instead of conclusions."
    return None


def _required_completion_markers(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b[A-Z][A-Z0-9_]{3,}_DONE\b", text or "")))


def _extract_completion_summary(transcript: str, markers: list[str]) -> str:
    marker_positions = [transcript.rfind(marker) for marker in markers]
    marker_index = max(marker_positions) if marker_positions else -1
    if marker_index < 0:
        return _clip_text(transcript, 1200)
    marker_line_start = transcript.rfind("\n", 0, marker_index) + 1
    marker_line_end = transcript.find("\n", marker_index)
    if marker_line_end < 0:
        marker_line_end = len(transcript)
    marker_line = transcript[marker_line_start:marker_line_end].strip()
    if len(markers) == 1 and marker_line == markers[0]:
        preceding_lines = transcript[:marker_line_start].splitlines()[-12:]
        meaningful_lines = [
            line.strip()
            for line in preceding_lines
            if line.strip()
            and markers[0] not in line
            and not line.strip().startswith((
                "## Dev Launch Profile",
                "Profile:",
                "Permissions:",
                "Contract:",
                "Runtime:",
                "Goal:",
                "last_user_message_id:",
                "execution_status:",
                "stats:",
            ))
            and not _is_status_like_summary(line.strip())
        ]
        if not meaningful_lines:
            return markers[0]
    bullet_prefix = transcript[:marker_index]
    bullet_matches = [
        match.strip()
        for match in re.findall(r"(?:^|\n)\s*•\s+([^\n]+)", bullet_prefix + transcript[marker_index:marker_index + 500])
        if (
            match.strip()
            and not match.strip().lower().startswith("called ")
            and not _is_status_like_summary(match.strip())
        )
    ]
    marker_bullets = [
        bullet for bullet in bullet_matches
        if any(marker in bullet for marker in markers)
    ]
    if marker_bullets:
        return _clip_text(marker_bullets[-1], 4000)
    prefix = transcript[:marker_index]
    answer_start = max(
        prefix.rfind("\n• ## "),
        prefix.rfind("\n## "),
        prefix.rfind("\n• FINDING"),
        prefix.rfind("\nFINDING"),
        prefix.rfind("\n• Read both requested"),
        prefix.rfind("\nRead both requested"),
    )
    if answer_start < 0:
        separator_matches = list(re.finditer(r"\n[─━-]{20,}\n", prefix))
        answer_start = separator_matches[-1].end() if separator_matches else -1
    start = answer_start if answer_start >= 0 else max(0, marker_index - 3200)
    end = min(len(transcript), marker_index + 400)
    snippet = transcript[start:end].strip()
    lines = [line.rstrip() for line in snippet.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned and cleaned[-1]:
                cleaned.append("")
            continue
        if stripped.startswith(("• Called", "└", "─", "› ", "gpt-")) or _is_status_like_summary(stripped):
            continue
        cleaned.append(line)
    summary = "\n".join(cleaned).strip()
    return _clip_text(summary or transcript[marker_index:end], 4000)


def _transcript_has_terminal_answer(transcript: str) -> bool:
    text = transcript or ""
    return bool(
        re.search(r"\n[─━-]?\s*Worked for\s+\d", text, re.IGNORECASE)
        or _codex_idle_prompt_index(text) >= 0
    )


def _extract_terminal_transcript_summary(transcript: str) -> str:
    text = transcript or ""
    worked_match = list(re.finditer(r"\n[─━-]?\s*Worked for\s+\d[^\n]*", text, re.IGNORECASE))
    idle_index = _codex_idle_prompt_index(text)
    if worked_match:
        before_terminal = text[:worked_match[-1].start()]
    elif idle_index >= 0:
        before_terminal = text[:idle_index]
    else:
        return ""
    bullet_matches = [
        match.strip()
        for match in re.findall(r"(?:^|\n)\s*•\s+([^\n]+)", before_terminal)
        if (
            match.strip()
            and not match.strip().lower().startswith("called ")
            and not _is_status_like_summary(match.strip())
        )
    ]
    if bullet_matches:
        return _clip_text(bullet_matches[-1], 4000)
    separator_matches = list(re.finditer(r"\n[─━-]{20,}[^\n]*\n", before_terminal))
    start = separator_matches[-1].end() if separator_matches else max(0, len(before_terminal) - 3200)
    snippet = before_terminal[start:].strip()
    lines = [line.rstrip() for line in snippet.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned and cleaned[-1]:
                cleaned.append("")
            continue
        if stripped.startswith(("• Called", "└", "─", "› ", "gpt-", "tokens used")) or _is_status_like_summary(stripped):
            continue
        if stripped.startswith("• "):
            line = stripped[2:].strip()
        cleaned.append(line)
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return _clip_text("\n".join(cleaned).strip(), 4000)


def _codex_idle_prompt_index(transcript: str) -> int:
    text = transcript or ""
    prompt_match = re.search(r"\n›\s+[^\n]*\n\s*\n\s*(?:gpt-[^\n]+|[a-z0-9_.-]+/[a-z0-9_.-]+[^\n]*)\s*$", text, re.IGNORECASE)
    return prompt_match.start() if prompt_match else -1


def _is_status_like_summary(summary: Optional[str]) -> bool:
    text = (summary or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if "esc to interrupt" in lowered:
        return True
    compact = re.sub(r"\s+", " ", text).strip()
    status_patterns = (
        r"^working\s*\(\s*\d+\s*(?:s|sec|secs|seconds|m|min|mins|minutes)\b.*\)$",
        r"^thinking\s*\(\s*\d+\s*(?:s|sec|secs|seconds|m|min|mins|minutes)\b.*\)$",
        r"^running\s*\(\s*\d+\s*(?:s|sec|secs|seconds|m|min|mins|minutes)\b.*\)$",
        r"^working\s+\d+\s*(?:s|sec|secs|seconds|m|min|mins|minutes)\b",
        r"^thinking\s+\d+\s*(?:s|sec|secs|seconds|m|min|mins|minutes)\b",
    )
    return any(re.search(pattern, compact, re.IGNORECASE) for pattern in status_patterns)


def _verification_evidence(summary: Optional[str], events: list[Dict[str, Any]]) -> list[str]:
    evidence: list[str] = []
    for line in (summary or "").splitlines():
        stripped = line.strip()
        if any(word in stripped.lower() for word in ("test", "build", "verified", "passed", "failed", "swift test", "pytest")):
            evidence.append(_clip_text(stripped, 180))
    for event in events:
        text = _first_non_empty(event.get("message"), event.get("summary"), event.get("preview"))
        if text and any(word in text.lower() for word in ("test", "build", "verified", "passed", "failed")):
            evidence.append(_clip_text(text, 180))
    return _list_unique(evidence)


def _latest_event(events: list[Dict[str, Any]], *, include_actions: bool, only_actions: bool = False) -> Optional[Dict[str, Any]]:
    for event in reversed(events):
        is_action = event.get("event") == "subagent.action"
        if only_actions and is_action:
            return event
        if not only_actions and (include_actions or not is_action):
            return event
    return None


def _event_identity(event: Dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        event.get("event_id"),
        event.get("created_at") or event.get("timestamp"),
        event.get("launch_plan_id"),
    )


def _session_attr(session: Any, name: str) -> Any:
    if session is None:
        return None
    try:
        return getattr(session, name)
    except Exception:
        return None


def _first_non_empty(*values: Any) -> Optional[Any]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
            continue
        return value
    return None


def _list_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_text(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"
