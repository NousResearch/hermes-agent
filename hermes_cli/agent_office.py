"""Agent Office control loop for Hermes Kanban.

This module keeps the durable Kanban kernel generic while adding the
Agent Office product layer: automatic triage/specification, lightweight
pick-rule routing to office profiles, supervisor diagnostics, and
human-friendly event aliases for the dashboard.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli import kanban_db as kb


def office_config() -> dict[str, Any]:
    """Return merged Agent Office config with safe defaults.

    ``load_config`` already merges ``DEFAULT_CONFIG`` with the user's
    config.yaml, so this helper is intentionally tiny and cheap to monkeypatch
    in tests.
    """
    defaults = {
        "enabled": True,
        "board": "inbox",
        "auto_specify": True,
        "auto_route": True,
        "auto_supervise": True,
        "config_dir": ".hermes/agent-office-config",
        "references_dir": ".hermes/skills/agent-office/references",
    }
    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("agent_office") or {}
        if isinstance(cfg, dict):
            defaults.update(cfg)
    except Exception:
        pass
    return defaults


def configured_board() -> str:
    board = office_config().get("board") or "inbox"
    return str(board)


OFFICE_PROFILES: tuple[str, ...] = (
    "triage", "supervisor", "chief", "pm", "architect", "research",
    "coder", "reviewer", "qa", "security", "memory", "tooling",
    "approval", "observability", "devops", "docs", "demo",
)

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
    ("devops", ("deploy", "deployment", "docker", "infra", "migration", "server", "ci", "cd", "kubernetes")),
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


@dataclass(frozen=True)
class RouteDecision:
    profile: str
    reason: str
    skills: tuple[str, ...] = ()


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


def route_task(task: kb.Task) -> RouteDecision:
    """Return the Agent Office profile that should own ``task``.

    This is intentionally deterministic and cheap for the first version. It
    turns the pack's pick-rule intent into runtime behavior without doing an
    LLM call on every dispatcher tick. The rules can later be hydrated from
    ``pick_rules.yaml`` without changing callers.
    """
    text = _task_text(task)
    for profile, needles in _KEYWORD_ROUTES:
        if any(n in text for n in needles):
            return RouteDecision(profile=profile, reason=f"keyword route -> {profile}")
    return RouteDecision(profile="chief", reason="fallback route -> chief")


def validate_office_profiles() -> dict[str, Any]:
    missing: list[str] = []
    present: list[str] = []
    try:
        from hermes_cli.profiles import profile_exists
    except Exception:
        profile_exists = None  # type: ignore[assignment]
    for profile in OFFICE_PROFILES:
        ok = True if profile_exists is None else bool(profile_exists(profile))
        (present if ok else missing).append(profile)
    return {"expected": len(OFFICE_PROFILES), "present": present, "missing": missing}


def status(board: Optional[str] = None) -> dict[str, Any]:
    """Small status payload for the dashboard Office pill."""
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
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "preferred_board": preferred,
        "board": board_slug,
        "board_exists": bool(board_exists),
        "gateway_running": bool(gateway_pid),
        "gateway_pid": gateway_pid,
        "profiles": profiles,
        "event_aliases": dict(_EVENT_ALIASES),
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
            # Product-quality fallback: if the LLM specifier is unavailable,
            # don't leave the whole office stalled in triage. Preserve the
            # user's title/body as the specification and promote to ready so
            # deterministic routing can continue. This is intentionally used
            # only for provider/client availability failures, not malformed
            # specs or policy errors.
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
    for row in rows:
        task = kb.Task.from_row(row)
        decision = route_task(task)
        if kb.assign_task(conn, task.id, decision.profile):
            kb.add_comment(
                conn,
                task.id,
                author="agent-office-router",
                body=f"Routed to @{decision.profile}: {decision.reason}",
            )
            try:
                with kb.write_txn(conn):
                    _append_office_event(
                        conn,
                        task.id,
                        "office.routed",
                        {"assignee": decision.profile, "reason": decision.reason},
                    )
            except Exception:
                pass
            routed.append((task.id, decision.profile))
    return routed


def supervise(conn, *, stale_seconds: int = 30 * 60) -> list[str]:
    """Lightweight supervisor pass that annotates stale/blocked work.

    This first version avoids disruptive reassignment; it creates durable
    comments/events so the UI and human operator can see what needs attention.
    """
    now = int(time.time())
    rows = conn.execute(
        "SELECT * FROM tasks WHERE status IN ('running', 'blocked', 'ready') AND status != 'archived'"
    ).fetchall()
    touched: list[str] = []
    for row in rows:
        task = kb.Task.from_row(row)
        reasons: list[str] = []
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
