"""Agent Office control loop for Hermes Kanban.

This module keeps the durable Kanban kernel generic while adding the
Agent Office product layer: automatic triage/specification, strict
role-to-profile routing, supervisor diagnostics, and human-friendly event
aliases for the dashboard.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli import kanban_db as kb

OFFICE_ROLES: tuple[str, ...] = (
    "triage", "supervisor", "chief", "pm", "architect", "research",
    "coder", "reviewer", "qa", "security", "memory", "tooling",
    "approval", "observability", "devops", "docs", "demo",
)

DEFAULT_ROLE_SEATS: dict[str, list[str]] = {
    "triage": ["triage"],
    "supervisor": ["supervisor", "supervisor-1"],
    "chief": ["chief"],
    "pm": ["pm", "pm-1", "pm-2"],
    "architect": ["architect", "architect-1", "architect-2"],
    "research": ["research", "research-1", "research-2"],
    "coder": ["coder", "coder-1", "coder-2", "coder-3"],
    "reviewer": ["reviewer", "reviewer-1", "reviewer-2"],
    "qa": ["qa", "qa-1", "qa-2"],
    "security": ["security", "security-1", "security-2"],
    "memory": ["memory", "memory-1"],
    "tooling": ["tooling", "tooling-1", "tooling-2"],
    "approval": ["approval", "approval-1"],
    "observability": ["observability", "observability-1"],
    "devops": ["devops", "devops-1", "devops-2"],
    "docs": ["docs", "docs-1", "docs-2"],
    "demo": ["demo", "demo-1"],
}

ROLE_MANAGERS: dict[str, str] = {
    "triage": "chief",
    "supervisor": "chief",
    "chief": "Akhil",
    "pm": "chief",
    "architect": "chief",
    "research": "pm",
    "coder": "chief",
    "reviewer": "chief",
    "qa": "chief",
    "security": "chief",
    "memory": "chief",
    "tooling": "architect",
    "approval": "chief",
    "observability": "devops",
    "devops": "chief",
    "docs": "pm",
    "demo": "pm",
}

_APPROVAL_REQUIRED_MARKERS: tuple[str, ...] = (
    "human_review_required: true",
    "approval_required: true",
    "human review required",
    "requires human review",
    "need human review",
    "needs human review",
    "human approval required",
    "requires human approval",
    "explicit approval",
    "approval required",
    "requires approval",
    "need approval",
    "needs approval",
    "get approval",
    "obtain approval",
    "must approve",
    "must be approved",
    "approved by",
    "ask akhil",
    "ask the user",
    "user approval",
    "manual approval",
    "sign-off",
    "sign off",
    "signoff",
    "user sign-off",
    "user sign off",
    "user signoff",
    "security approval",
    "legal approval",
    "compliance approval",
    "before merge",
    "before deploy",
    "production deploy",
    "prod deploy",
)


def _requires_explicit_approval(reason: str) -> bool:
    reason_cf = reason.casefold()
    return any(marker in reason_cf for marker in _APPROVAL_REQUIRED_MARKERS)


def office_config() -> dict[str, Any]:
    """Return merged Agent Office config with safe defaults.

    ``load_config`` already merges ``DEFAULT_CONFIG`` with the user's
    config.yaml, so this helper is intentionally tiny and cheap to monkeypatch
    in tests. Runtime policy lives in config so Akhil can tune the office
    without changing code.
    """
    defaults = {
        "enabled": True,
        "board": "inbox",
        "auto_specify": True,
        "auto_route": True,
        "auto_supervise": True,
        "config_dir": ".hermes/agent-office-config",
        "references_dir": ".hermes/skills/agent-office/references",
        "workspace_root": "/Users/akhilkinnera/Documents/My Workspace",
        "default_mode": "yolo",
        "approval_mode_keywords": [
            "keep me in the loop", "ask me", "take my permission",
            "permission first", "approval required", "do not yolo",
            "not yolo", "manual approval",
        ],
        "role_seats": DEFAULT_ROLE_SEATS,
        "quality_gates": {
            "enabled": True,
            "apply_to_every_task": True,
            "default_final_verifier": "qa",
            "default_reviewer": "reviewer",
            "do_not_trust_worker_done_text": True,
            "require_gate_scorecard": True,
            "require_real_benchmark_artifacts": True,
            "forbid_silent_scope_reduction": True,
            "scope_change_block": "SCOPE_CHANGE_REQUEST",
            "minimum_evidence": [
                "commands_run_with_exit_codes",
                "tests_or_reason_not_applicable",
                "artifact_paths_or_reason_not_applicable",
                "gate_verdicts_pass_fail_partial_blocked",
            ],
        },
    }
    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("agent_office") or {}
        if isinstance(cfg, dict):
            merged = dict(defaults)
            merged.update(cfg)
            if isinstance(cfg.get("role_seats"), dict):
                seats = {k: list(v) for k, v in DEFAULT_ROLE_SEATS.items()}
                for role, names in cfg["role_seats"].items():
                    if isinstance(names, (list, tuple)):
                        seats[str(role)] = [str(n) for n in names if str(n).strip()]
                merged["role_seats"] = seats
            defaults = merged
    except Exception:
        pass
    return defaults


def configured_board() -> str:
    board = office_config().get("board") or "inbox"
    return str(board)


def role_seats(role: str, *, cfg: Optional[dict[str, Any]] = None) -> list[str]:
    cfg = cfg or office_config()
    seats = (cfg.get("role_seats") or {}).get(role)
    if isinstance(seats, (list, tuple)):
        out = [str(s).strip() for s in seats if str(s).strip()]
        if out:
            return out
    return [role]


def office_profiles(cfg: Optional[dict[str, Any]] = None) -> tuple[str, ...]:
    cfg = cfg or office_config()
    seen: set[str] = set()
    out: list[str] = []
    for role in OFFICE_ROLES:
        for seat in role_seats(role, cfg=cfg):
            if seat not in seen:
                seen.add(seat)
                out.append(seat)
    return tuple(out)


OFFICE_PROFILES = tuple(dict.fromkeys(p for seats in DEFAULT_ROLE_SEATS.values() for p in seats))

_EVENT_ALIASES = {
    "created": "card.created",
    "specified": "triaged",
    "promoted": "ready_for_assignment",
    "assigned": "assigned",
    "claimed": "started",
    "spawned": "started",
    "completed": "completed",
    "blocked": "blocked",
    "reclaimed": "reclaimed",
    "gave_up": "needs_supervisor",
    "timed_out": "needs_supervisor",
    "crashed": "needs_supervisor",
}

_KEYWORD_ROUTES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("security", ("security", "threat", "auth", "token", "permission", "vulnerability", "secret")),
    ("devops", ("deploy", "deployment", "docker", "infra", "migration", "server", "ci", "cd", "kubernetes", "install", "brew", "download app")),
    ("observability", ("observability", "logging", "metrics", "trace", "tracing", "dashboard", "alert")),
    ("docs", ("doc", "docs", "documentation", "readme", "runbook", "guide")),
    ("qa", ("test", "tests", "testing", "eval", "evaluation", "quality", "regression")),
    ("reviewer", ("review", "code review", "pr review", "pull request")),
    ("demo", ("demo", "showcase", "presentation", "founder", "story")),
    ("architect", ("architecture", "design", "system", "schema", "boundary", "data flow")),
    ("pm", ("prd", "requirements", "roadmap", "milestone", "user story")),
    ("research", ("research", "survey", "benchmark", "compare", "market", "paper")),
    ("memory", ("memory", "rag", "retrieval", "vector", "embedding")),
    ("tooling", ("mcp", "tool", "tools", "integration", "connector", "api")),
    ("approval", ("approval", "human", "permission", "risky", "policy")),
    ("coder", ("implement", "build", "fix", "code", "bug", "feature", "refactor")),
)


OFFICE_WORKFLOW_STATES: tuple[str, ...] = (
    "intake",
    "specified",
    "routed",
    "running",
    "implementation_done",
    "verification_pending",
    "verification_passed",
    "verification_failed",
    "review_pending",
    "approved",
    "final_done",
)


def default_verification_gates(task_id: str, title: str | None = None) -> list[dict[str, Any]]:
    """Machine-readable fallback gates attached to every Office-created task.

    Explicit spec gates can override this later, but Office final-close should
    never depend only on prose. The default gate forces a real scorecard artifact
    that reviewers/verifiers can inspect.
    """
    return [
        {
            "id": "office-gate-scorecard",
            "title": "Evidence-backed gate scorecard",
            "requirement_ref": task_id,
            "type": "quality_gate_scorecard",
            "required": True,
            "commands": [],
            "expected_artifacts": [
                {
                    "path": ".hermes/office/gate-scorecard.json",
                    "must_exist": True,
                    "min_bytes": 20,
                    "content_regex": "gate|verdict|evidence|pass|fail|blocked",
                }
            ],
            "thresholds": [],
            "evidence_policy": "inspect_existing",
            "allow_blocked": False,
        }
    ]


def attach_default_gates_to_body(body: str | None, task_id: str, title: str | None = None) -> str:
    """Append default Office verification gates unless the body already has gates."""
    text = body or ""
    if "verification_gates" in text:
        return text
    gates = json.dumps(default_verification_gates(task_id, title), indent=2)
    guidance = (
        "\n\nOffice verification contract:\n"
        "Before requesting final close, produce .hermes/office/gate-scorecard.json "
        "with gates, commands/checks, artifact paths, verdicts, and rationale. "
        "Do not silently reduce scope; use SCOPE_CHANGE_REQUEST ... END_SCOPE_CHANGE_REQUEST.\n"
        "```json verification_gates\n" + gates + "\n```"
    )
    return text.rstrip() + guidance


@dataclass(frozen=True)
class RouteDecision:
    profile: str
    reason: str
    skills: tuple[str, ...] = ()
    role: str = ""
    manager: str = "chief"


@dataclass
class OfficeTickResult:
    specified: list[str] = field(default_factory=list)
    specify_failed: dict[str, str] = field(default_factory=dict)
    routed: list[tuple[str, str]] = field(default_factory=list)
    supervised: list[str] = field(default_factory=list)
    dispatched: kb.DispatchResult = field(default_factory=kb.DispatchResult)


def office_event_alias(kind: Optional[str]) -> Optional[str]:
    if kind is None:
        return None
    return _EVENT_ALIASES.get(kind, kind)


def _task_text(task: kb.Task) -> str:
    return f"{task.title or ''}\n{task.body or ''}".casefold()


def _role_for_task(task: kb.Task) -> tuple[str, str]:
    text = _task_text(task)
    for role, needles in _KEYWORD_ROUTES:
        if any(n in text for n in needles):
            return role, f"keyword route -> {role}"
    return "chief", "fallback route -> chief"


def _profile_exists_fn():
    try:
        from hermes_cli.profiles import profile_exists
        return profile_exists
    except Exception:
        return None


def active_counts_by_assignee(conn) -> dict[str, int]:
    rows = conn.execute(
        "SELECT assignee, COUNT(*) AS n FROM tasks "
        "WHERE assignee IS NOT NULL AND status IN ('ready', 'running') "
        "GROUP BY assignee"
    ).fetchall()
    return {str(r["assignee"]): int(r["n"]) for r in rows}


def resolve_profile_for_role(conn, role: str) -> str:
    """Resolve an office role to a real spawnable profile seat.

    Strict assignment rule: Kanban assignees must be concrete Hermes profile
    names, not abstract skills or imaginary role labels. The base role profile
    is preferred when load is tied; additional seats absorb parallel work.
    """
    cfg = office_config()
    candidates = role_seats(role, cfg=cfg)
    profile_exists = _profile_exists_fn()
    spawnable = [p for p in candidates if profile_exists is None or profile_exists(p)]
    if not spawnable:
        return role
    counts = active_counts_by_assignee(conn)
    return min(enumerate(spawnable), key=lambda pair: (counts.get(pair[1], 0), pair[0]))[1]


def route_task(task: kb.Task) -> RouteDecision:
    """Return the default Agent Office route for ``task``.

    This cheap, deterministic classifier yields a role and its preferred base
    profile. ``route_ready_unassigned`` upgrades that base profile to the
    least-loaded real seat at assignment time.
    """
    role, reason = _role_for_task(task)
    profile = role_seats(role)[0]
    return RouteDecision(
        profile=profile,
        role=role,
        reason=reason,
        manager=ROLE_MANAGERS.get(role, "chief"),
    )


def validate_office_profiles() -> dict[str, Any]:
    missing: list[str] = []
    present: list[str] = []
    profile_exists = _profile_exists_fn()
    expected = office_profiles()
    for profile in expected:
        ok = True if profile_exists is None else bool(profile_exists(profile))
        (present if ok else missing).append(profile)
    return {"expected": len(expected), "present": present, "missing": missing}


def status(board: Optional[str] = None) -> dict[str, Any]:
    """Status/preflight payload for the dashboard Office cockpit."""
    cfg = office_config()
    preferred = str(cfg.get("board") or "inbox")
    board_slug = board or preferred
    try:
        board_exists = kb.board_exists(board_slug)
    except Exception:
        board_exists = False
    try:
        from gateway.status import get_running_pid  # type: ignore
        gateway_pid = get_running_pid()
    except Exception:
        gateway_pid = None
    profiles = validate_office_profiles()
    workspace_root = Path(str(cfg.get("workspace_root") or "/Users/akhilkinnera/Documents/My Workspace")).expanduser()
    health_checks = [
        {"id": "office_enabled", "label": "Office enabled", "status": "pass" if cfg.get("enabled", True) else "fail", "detail": "agent_office.enabled"},
        {"id": "board_exists", "label": "Office board exists", "status": "pass" if board_exists else "fail", "detail": board_slug},
        {"id": "gateway", "label": "Gateway/dispatcher", "status": "pass" if gateway_pid else "warn", "detail": f"pid {gateway_pid}" if gateway_pid else "not detected"},
        {"id": "profiles", "label": "Office profiles", "status": "pass" if not profiles.get("missing") else "fail", "detail": f"{len(profiles.get('present', []))}/{profiles.get('expected', 0)} present"},
        {"id": "workspace_root", "label": "Workspace root", "status": "pass" if workspace_root.exists() and os.access(workspace_root, os.W_OK) else "fail", "detail": str(workspace_root)},
        {"id": "quality_gates", "label": "Quality gates", "status": "pass" if (cfg.get("quality_gates") or {}).get("apply_to_every_task") else "warn", "detail": "default gates attached to Office-created work"},
    ]
    health = {
        "checks": health_checks,
        "summary": {
            "pass": sum(1 for c in health_checks if c["status"] == "pass"),
            "warn": sum(1 for c in health_checks if c["status"] == "warn"),
            "fail": sum(1 for c in health_checks if c["status"] == "fail"),
        },
    }
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "preferred_board": preferred,
        "board": board_slug,
        "board_exists": bool(board_exists),
        "gateway_running": bool(gateway_pid),
        "gateway_pid": gateway_pid,
        "profiles": profiles,
        "roles": {role: role_seats(role, cfg=cfg) for role in OFFICE_ROLES},
        "workspace_root": str(workspace_root),
        "workflow_states": list(OFFICE_WORKFLOW_STATES),
        "health": health,
        "default_mode": cfg.get("default_mode", "yolo"),
        "event_aliases": dict(_EVENT_ALIASES),
        "quality_gates": cfg.get("quality_gates") or {},
    }


def _append_office_event(conn, task_id: str, kind: str, payload: Optional[dict[str, Any]] = None) -> None:
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
        (task_id, kind, json.dumps(payload) if payload is not None else None, int(time.time())),
    )


def specify_triage(
    conn,
    *,
    specify_fn: Optional[Callable[..., Any]] = None,
    author: str = "agent-office",
    limit: Optional[int] = None,
) -> tuple[list[str], dict[str, str]]:
    from hermes_cli import kanban_specify

    fn = specify_fn or kanban_specify.specify_task
    tasks = kb.list_tasks(conn, status="triage", include_archived=False, limit=limit)
    ok: list[str] = []
    failed: dict[str, str] = {}
    for task in tasks:
        outcome = fn(task.id, author=author)
        if getattr(outcome, "ok", False):
            ok.append(task.id)
        else:
            reason = str(getattr(outcome, "reason", "failed"))
            if "unavailable" in reason.casefold():
                fallback_ok = kb.specify_triage_task(
                    conn,
                    task.id,
                    title=task.title,
                    body=task.body or task.title,
                    author=author,
                )
                if fallback_ok:
                    kb.add_comment(
                        conn,
                        task.id,
                        author="agent-office",
                        body=f"LLM specifier unavailable; used original intake text as fallback specification ({reason}).",
                    )
                    ok.append(task.id)
                    continue
            failed[task.id] = reason
    return ok, failed


def route_ready_unassigned(conn, *, limit: Optional[int] = None) -> list[tuple[str, str]]:
    rows = conn.execute(
        "SELECT * FROM tasks WHERE status = 'ready' AND assignee IS NULL "
        "ORDER BY priority DESC, created_at ASC" + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    routed: list[tuple[str, str]] = []
    cfg = office_config()
    default_mode = str(cfg.get("default_mode") or "yolo")
    workspace_root = str(cfg.get("workspace_root") or "/Users/akhilkinnera/Documents/My Workspace")
    for row in rows:
        task = kb.Task.from_row(row)
        decision = route_task(task)
        assignee = resolve_profile_for_role(conn, decision.role or decision.profile)
        manager = ROLE_MANAGERS.get(decision.role or assignee, "chief")
        if kb.assign_task(conn, task.id, assignee):
            kb.add_comment(
                conn,
                task.id,
                author="agent-office-router",
                body=(
                    f"Strict route: role={decision.role or assignee}, assignee=@{assignee}, "
                    f"reports_to={manager}, mode={default_mode}, workspace_root={workspace_root}. "
                    "Quality gates apply to every Office task: provide a token-efficient "
                    "evidence-backed gate scorecard, no silent scope reduction, real "
                    "benchmark artifacts when benchmark/performance is claimed, and "
                    "reviewer/QA evidence review for substantive work. Prefer artifact "
                    "paths and concise excerpts over pasted logs. "
                    f"Reason: {decision.reason}."
                ),
            )
            try:
                with kb.write_txn(conn):
                    _append_office_event(
                        conn,
                        task.id,
                        "office.routed",
                        {
                            "role": decision.role or assignee,
                            "assignee": assignee,
                            "reports_to": manager,
                            "mode": default_mode,
                            "workspace_root": workspace_root,
                            "reason": decision.reason,
                        },
                    )
            except Exception:
                pass
            routed.append((task.id, assignee))
    return routed


def _latest_block_reason(conn, task_id: str) -> str:
    row = conn.execute(
        "SELECT summary, error FROM task_runs WHERE task_id = ? "
        "AND outcome = 'blocked' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if not row:
        return ""
    return str(row["summary"] or row["error"] or "")


def _auto_route_review_required_block(conn, task: kb.Task) -> bool:
    """In yolo mode, turn human-review blocks into reviewer handoffs.

    Workers must be honest and evidence-backed, but they should not stop the
    whole Office on generic human review. Route the card to the reviewer seat;
    real external blockers and explicit human approval still remain blocked.
    """
    cfg = office_config()
    if str(cfg.get("default_mode") or "yolo").casefold() != "yolo":
        return False
    reason = _latest_block_reason(conn, task.id)
    reason_cf = reason.casefold()
    if "review-required" not in reason_cf:
        return False
    if _requires_explicit_approval(reason):
        return False
    reviewer_role = str((cfg.get("quality_gates") or {}).get("default_reviewer") or "reviewer")
    reviewer = resolve_profile_for_role(conn, reviewer_role)
    if not kb.assign_task(conn, task.id, reviewer):
        return False
    if not kb.unblock_task(conn, task.id):
        return False
    kb.add_comment(
        conn,
        task.id,
        author="agent-office-supervisor",
        body=(
            "Auto-routed review-required block to reviewer in yolo mode. "
            "Worker honesty is preserved by requiring evidence and routing to "
            f"@{reviewer}; this is not treated as final user-visible success. "
            f"Original block reason: {reason}"
        ),
    )
    try:
        with kb.write_txn(conn):
            _append_office_event(
                conn,
                task.id,
                "office.review_routed",
                {"assignee": reviewer, "source_reason": reason, "mode": "yolo"},
            )
    except Exception:
        pass
    return True


def supervise(conn, *, stale_seconds: int = 30 * 60) -> list[str]:
    """Lightweight supervisor pass that annotates stale/blocked work."""
    now = int(time.time())
    rows = conn.execute(
        "SELECT * FROM tasks WHERE status IN ('running', 'blocked', 'ready') AND status != 'archived'"
    ).fetchall()
    touched: list[str] = []
    for row in rows:
        task = kb.Task.from_row(row)
        reasons: list[str] = []
        if task.status == "blocked" and _auto_route_review_required_block(conn, task):
            touched.append(task.id)
            continue
        if task.status == "running" and task.started_at and now - int(task.started_at) >= stale_seconds:
            reasons.append("running task is stale")
        if task.status == "blocked":
            reasons.append("task is blocked")
        if task.status == "ready" and not task.assignee:
            reasons.append("ready task is unassigned")
        if not reasons:
            continue
        body = "Supervisor notice: " + "; ".join(reasons)
        existing = conn.execute(
            "SELECT 1 FROM task_comments WHERE task_id = ? AND author = 'agent-office-supervisor' AND body = ?",
            (task.id, body),
        ).fetchone()
        if existing:
            continue
        kb.add_comment(conn, task.id, author="agent-office-supervisor", body=body)
        try:
            with kb.write_txn(conn):
                _append_office_event(conn, task.id, "office.supervised", {"reasons": reasons})
        except Exception:
            pass
        touched.append(task.id)
    return touched


def tick(
    conn,
    *,
    board: Optional[str] = None,
    specify_fn: Optional[Callable[..., Any]] = None,
    spawn_fn: Optional[Callable[..., Any]] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    failure_limit: int = kb.DEFAULT_SPAWN_FAILURE_LIMIT,
    auto_specify: Optional[bool] = None,
    auto_route: Optional[bool] = None,
    auto_supervise: Optional[bool] = None,
) -> OfficeTickResult:
    cfg = office_config()
    enabled = bool(cfg.get("enabled", True))
    if auto_specify is None:
        auto_specify = enabled and bool(cfg.get("auto_specify", True))
    if auto_route is None:
        auto_route = enabled and bool(cfg.get("auto_route", True))
    if auto_supervise is None:
        auto_supervise = enabled and bool(cfg.get("auto_supervise", True))
    if board is None:
        board = str(cfg.get("board") or "inbox")

    result = OfficeTickResult()
    if auto_specify:
        result.specified, result.specify_failed = specify_triage(conn, specify_fn=specify_fn)
    if auto_route:
        result.routed = route_ready_unassigned(conn)
    if auto_supervise:
        result.supervised = supervise(conn)
    result.dispatched = kb.dispatch_once(
        conn,
        spawn_fn=spawn_fn,
        dry_run=dry_run,
        max_spawn=max_spawn,
        failure_limit=failure_limit,
        board=board,
    )
    return result
