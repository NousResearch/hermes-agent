"""Feature-gated canonical ``/fix-review`` correction adapter.

This is deliberately a thin implementation-lane adapter.  Changes-requested
history is authoritative; claiming, routing persistence, CAS, and spawning are
all delegated to the existing implementation dispatcher.
"""

from __future__ import annotations

import shlex
from typing import Any, Optional

from agent.routing_decision import build_routing_decision
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_routing as kbr
from hermes_cli.kanban_implement import (
    _DEFAULT_IMPLEMENTATION_PROFILE,
    _kanban_config,
    _profile_runtime_identity,
    resolve_human_gate,
    select_implementation_route,
)
from hermes_cli.kanban_status import resolve_status_reference
from utils import is_truthy_value

_GATE_KEY = "fix_review_command"
_COMPLETED_IMPLEMENTATION_OUTCOMES = frozenset({"completed", "review_requested"})


def fix_review_command_enabled() -> bool:
    return is_truthy_value(_kanban_config().get(_GATE_KEY), default=False)


def _result(**fields: Any) -> dict[str, Any]:
    result = {
        "command": "fix-review",
        "task_id": None,
        "board": None,
        "task_status": None,
        "review_status": None,
        "changes_requested": False,
        "changes_run_id": None,
        "correction_required": False,
        "run_id": None,
        "decision_id": None,
        "implementation_profile": None,
        "implementation_provider": None,
        "implementation_model": None,
        "reviewer_profile": None,
        "reviewer_provider": None,
        "reviewer_model": None,
        "dispatch_status": "failed",
        "human_gate_required": None,
        "message": None,
    }
    result.update(fields)
    return result


def _parse_args(text: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    raw = str(text or "").strip().lstrip("/")
    if raw == "fix-review" or raw.startswith("fix-review "):
        raw = raw[len("fix-review"):].strip()
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return None, None, None, f"Invalid arguments: {exc}"
    refs: list[str] = []
    board = profile = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--board", "--profile"}:
            index += 1
            if index >= len(tokens) or tokens[index].startswith("--"):
                return None, None, None, f"{token} requires a value"
            if token == "--board":
                if board is not None:
                    return None, None, None, "--board may be specified only once"
                board = tokens[index]
            else:
                if profile is not None:
                    return None, None, None, "--profile may be specified only once"
                profile = tokens[index]
        elif token.startswith("--board="):
            if board is not None:
                return None, None, None, "--board may be specified only once"
            board = token.partition("=")[2]
        elif token.startswith("--profile="):
            if profile is not None:
                return None, None, None, "--profile may be specified only once"
            profile = token.partition("=")[2]
        elif token.startswith("--"):
            return None, None, None, f"Unknown option: {token}"
        else:
            refs.append(token)
        index += 1
    if not refs:
        return None, None, None, "Usage: /fix-review <task|reference> [--board <board>] [--profile <profile>]"
    return " ".join(refs), board, profile, None


def _latest_changes_run(conn, task_id: str):
    row = conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? AND outcome = 'changes_requested' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if row is None:
        return None
    event = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND run_id = ? AND kind = 'changes_requested' LIMIT 1",
        (task_id, int(row["id"])),
    ).fetchone()
    return int(row["id"]) if event is not None else None


def _implementation_run_after(conn, task_id: str, changes_run_id: int, *, active: Optional[bool] = None):
    rows = conn.execute(
        "SELECT id, outcome, ended_at FROM task_runs WHERE task_id = ? AND id > ? ORDER BY id DESC",
        (task_id, int(changes_run_id)),
    ).fetchall()
    for row in rows:
        if active is True and row["ended_at"] is not None:
            continue
        if active is False and row["ended_at"] is None:
            continue
        decision = kbr.load_run_routing_decision(conn, task_id=task_id, run_id=int(row["id"]))
        if not isinstance(decision, dict) or decision.get("task_type") != "implementation":
            continue
        if active is True or row["outcome"] in _COMPLETED_IMPLEMENTATION_OUTCOMES:
            return int(row["id"]), decision
    return None


def _route_result(route, *, reviewer_model: Optional[str] = None) -> dict[str, Any]:
    return {
        "implementation_profile": route.selected_profile,
        "implementation_provider": route.implementation_provider,
        "implementation_model": route.selected_model,
        "reviewer_profile": route.reviewer_profile,
        "reviewer_provider": route.reviewer_provider,
        "reviewer_model": reviewer_model,
    }


def run_fix_review_slash(text: str) -> dict[str, Any]:
    reference, explicit_board, requested_profile, error = _parse_args(text)
    if error:
        return _result(dispatch_status="invalid", message=error)
    if not fix_review_command_enabled():
        return _result(dispatch_status="disabled", message="/fix-review is disabled (kanban.fix_review_command)")
    assert reference is not None
    resolution = resolve_status_reference(reference, board=explicit_board)
    if not resolution.ok or resolution.scope != "task" or not resolution.task_id or not resolution.board:
        return _result(board=resolution.board, dispatch_status="not_eligible", message=resolution.error or "task is not eligible")

    board, task_id = resolution.board, resolution.task_id
    with kbc.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            return _result(task_id=task_id, board=board, dispatch_status="not_eligible", message="task no longer exists")
        human_gate_required, human_gate_satisfied = resolve_human_gate(conn, task)
        changes_run_id = _latest_changes_run(conn, task_id)
        base = dict(
            task_id=task_id, board=board, task_status=task.status,
            review_status="changes_requested" if changes_run_id is not None else None,
            changes_requested=changes_run_id is not None, changes_run_id=changes_run_id,
            correction_required=changes_run_id is not None,
            human_gate_required=human_gate_required,
        )
        if changes_run_id is None:
            return _result(**{**base, "correction_required": False}, dispatch_status="not_eligible",
                           message="no authoritative changes_requested review requires correction")
        if task.status in {"done", "archived"}:
            return _result(**base, dispatch_status="not_eligible", message=f"task is {task.status}; it will not be restarted")
        if task.status == "running":
            active = _implementation_run_after(conn, task_id, changes_run_id, active=True)
            if active is not None:
                run_id, decision = active
                return _result(
                    **{**base, "task_status": task.status}, run_id=run_id,
                    decision_id=decision.get("decision_id"), dispatch_status="already_active",
                    **{key: decision.get(key) for key in (
                        "implementation_profile", "implementation_provider", "implementation_model",
                        "reviewer_profile", "reviewer_provider", "reviewer_model",
                    )}, message="correction implementation is already active",
                )
            return _result(**base, dispatch_status="not_eligible", message="task is currently running and is not an active correction")
        if task.status != "ready" or task.claim_lock is not None:
            return _result(**base, dispatch_status="not_eligible", message="task is not ready for correction")
        completed = _implementation_run_after(conn, task_id, changes_run_id, active=False)
        if completed is not None:
            run_id, decision = completed
            return _result(
                **{**base, "correction_required": False}, run_id=run_id,
                decision_id=decision.get("decision_id"), dispatch_status="already_completed",
                **{key: decision.get(key) for key in (
                    "implementation_profile", "implementation_provider", "implementation_model",
                    "reviewer_profile", "reviewer_provider", "reviewer_model",
                )}, message="correction implementation is already completed",
            )
        graph = kb.task_graph_context(conn, task_id)
        if not all(parent.get("status") in {"done", "archived"} for parent in graph.get("parents", [])):
            return _result(**base, dispatch_status="not_eligible", message="dependencies do not permit correction")
        if human_gate_required and not human_gate_satisfied:
            return _result(**base, dispatch_status="not_eligible", message="required human gate is not satisfied")

        policy_profile = _kanban_config().get("implement_profile")
        selected = requested_profile or (str(policy_profile).strip() if policy_profile else _DEFAULT_IMPLEMENTATION_PROFILE)
        try:
            route = select_implementation_route(selected)
            _, reviewer_model = _profile_runtime_identity(route.reviewer_profile)
        except (ValueError, TypeError) as exc:
            return _result(**base, dispatch_status="routing_failed", message=str(exc))
        route_fields = _route_result(route, reviewer_model=reviewer_model)
        if not route.implementation_provider or not route.reviewer_provider:
            return _result(**base, **route_fields, dispatch_status="routing_failed",
                           message="authoritative implementation/reviewer provider identity is required")

        decision_box: dict[str, Any] = {}

        def persist_before_spawn(run_conn, claimed, *, board, lane, workspace):
            decision = build_routing_decision(
                task_id=claimed.id, board=board, task_type="implementation", capability="implement",
                risk="normal", code_change=True, independent_review=True,
                preferred_profile=route.selected_profile, reviewer_profile=route.reviewer_profile,
                selected_profile=route.selected_profile,
                selected_provider=claimed.provider_override or route.implementation_provider,
                selected_model=claimed.model_override or route.selected_model,
                human_gate_required=human_gate_required, independence_valid=route.independence_valid,
                policy_digest=None, selected_by="fix-review", run_id=claimed.current_run_id,
                session_id=claimed.session_id,
            )
            if not kbr.persist_run_routing_decision(
                run_conn, task_id=claimed.id, run_id=int(claimed.current_run_id),
                decision=decision, event_kind="routing_selected",
            ):
                raise RuntimeError("authoritative correction routing decision could not be persisted")
            decision_box.update(decision)

        try:
            dispatch = kbd.dispatch_once(
                conn, task_id=task_id, board=board, max_spawn=1,
                before_spawn_fn=persist_before_spawn, spawn_profile=route.selected_profile,
            )
        except Exception as exc:
            current = kb.get_task(conn, task_id)
            return _result(
                **{**base, "task_status": current.status if current else None}, **route_fields,
                dispatch_status="routing_failed", message=f"correction dispatch failed: {exc}",
            )
        after = kb.get_task(conn, task_id)
        status = "started" if dispatch.spawned else ("already_active" if dispatch.skipped_locked else "not_dispatched")
        return _result(
            **{**base, "task_status": after.status if after else None},
            run_id=after.current_run_id if after else None, decision_id=decision_box.get("decision_id"),
            **route_fields, dispatch_status=status, message="correction implementation started" if status == "started" else "correction was not dispatched",
        )


def render_fix_review_result(result: dict[str, Any]) -> str:
    if result.get("dispatch_status") == "started":
        return "\n".join(("Fix-review correction started", f"Task: {result.get('task_id') or '-'}",
                             f"Run: {result.get('run_id') or '-'}",
                             f"Implementation lane: {result.get('implementation_profile') or '-'}",
                             f"Expected reviewer: {result.get('reviewer_profile') or '-'}"))
    return str(result.get("message") or "Fix-review was not started")


def run_fix_review_slash_rendered(text: str) -> str:
    return render_fix_review_result(run_fix_review_slash(text))
