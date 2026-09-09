"""Thin, feature-gated adapter for the canonical ``/implement`` command."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Optional

from agent.routing_decision import build_routing_decision, evaluate_reviewer_independence
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_routing as kbr
from hermes_cli.kanban_status import resolve_status_reference
from utils import is_truthy_value


_IMPLEMENT_GATE_KEY = "implement_command"


@dataclass(frozen=True)
class ImplementationRoute:
    selected_profile: Optional[str]
    reviewer_profile: Optional[str]
    independence_valid: bool


def implement_command_enabled() -> bool:
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        return is_truthy_value(cfg_get(read_raw_config(), "kanban", _IMPLEMENT_GATE_KEY), default=False)
    except Exception:
        return False


def select_implementation_route(profile: Optional[str]) -> ImplementationRoute:
    selected = str(profile or "").strip() or None
    if selected is None:
        raise ValueError("implementation profile is required; profile=null is non-authoritative")
    reviewer = {
        "rozmilo-codex": "rozmilo-claude",
        "rozmilo-claude": "rozmilo-codex",
    }.get(selected)
    independent = evaluate_reviewer_independence(
        implementation_profile=selected,
        reviewer_profile=reviewer,
        independent_review=True,
    )
    if not independent:
        raise ValueError("required independent review is not valid")
    return ImplementationRoute(selected, reviewer, independent)


def _result(**fields: Any) -> dict[str, Any]:
    defaults = {
        "command": "implement", "task_id": None, "board": None,
        "task_status": None, "run_id": None, "decision_id": None,
        "selected_profile": None, "selected_provider": None, "selected_model": None,
        "reviewer_profile": None, "dispatch_status": "failed",
        "human_gate_required": None, "routing_contract_version": None,
        "independence_valid": None, "fallback_used": None, "message": None,
    }
    defaults.update(fields)
    return defaults


def _parse_args(text: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    raw = str(text or "").strip().lstrip("/")
    if raw == "implement" or raw.startswith("implement "):
        raw = raw[len("implement"):].strip()
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return None, None, None, f"Invalid arguments: {exc}"
    reference: list[str] = []
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
            board = token.partition("=")[2] or None
        elif token.startswith("--profile="):
            profile = token.partition("=")[2] or None
        elif token.startswith("--"):
            return None, None, None, f"Unknown option: {token}"
        else:
            reference.append(token)
        index += 1
    if not reference:
        return None, None, None, "Usage: /implement <task|reference> [--board <board>] [--profile <profile>]"
    return " ".join(reference), board, profile, None


def run_implement_slash(text: str) -> dict[str, Any]:
    reference, explicit_board, requested_profile, error = _parse_args(text)
    if error:
        return _result(dispatch_status="invalid", message=error)
    if not implement_command_enabled():
        return _result(dispatch_status="disabled", message="/implement is disabled (kanban.implement_command)")
    assert reference is not None
    resolution = resolve_status_reference(reference, board=explicit_board)
    if not resolution.ok or resolution.scope != "task" or not resolution.task_id or not resolution.board:
        return _result(board=resolution.board, dispatch_status="not_eligible", message=resolution.error or "task is not eligible")

    board, task_id = resolution.board, resolution.task_id
    with kbc.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            return _result(task_id=task_id, board=board, dispatch_status="not_eligible", message="task no longer exists")
        if task.status == "running":
            existing = (
                kbr.load_run_routing_decision(
                    conn, task_id=task_id, run_id=int(task.current_run_id)
                )
                if task.current_run_id is not None else None
            ) or {}
            return _result(
                task_id=task_id, board=board, task_status=task.status,
                run_id=task.current_run_id, decision_id=existing.get("decision_id"),
                selected_profile=existing.get("selected_profile"),
                selected_provider=existing.get("selected_provider"),
                selected_model=existing.get("selected_model"),
                reviewer_profile=existing.get("reviewer_profile"),
                routing_contract_version=existing.get("routing_contract_version"),
                independence_valid=existing.get("independence_valid"),
                fallback_used=existing.get("fallback_used"),
                human_gate_required=existing.get("human_gate_required"),
                dispatch_status="already_running",
                message="implementation is already running",
            )
        if task.status in {"done", "archived"}:
            return _result(task_id=task_id, board=board, task_status=task.status,
                           dispatch_status="not_eligible", message=f"task is {task.status}; it will not be restarted")
        if task.status != "ready" or task.claim_lock is not None:
            return _result(task_id=task_id, board=board, task_status=task.status,
                           dispatch_status="not_eligible", message="task is not ready for implementation")
        graph = kb.task_graph_context(conn, task_id)
        if not all(parent.get("status") in {"done", "archived"} for parent in graph.get("parents", [])):
            return _result(task_id=task_id, board=board, task_status=task.status,
                           dispatch_status="not_eligible", message="dependencies do not permit execution")
        profile = requested_profile or task.assignee
        try:
            route = select_implementation_route(profile)
        except ValueError as exc:
            return _result(task_id=task_id, board=board, task_status=task.status,
                           dispatch_status="routing_failed", message=str(exc))
        if task.assignee != route.selected_profile:
            return _result(task_id=task_id, board=board, task_status=task.status,
                           selected_profile=route.selected_profile, reviewer_profile=route.reviewer_profile,
                           independence_valid=route.independence_valid, dispatch_status="not_eligible",
                           message="task assignee does not match the requested implementation profile")

        decision_box: dict[str, Any] = {}

        def persist_before_spawn(run_conn, claimed, *, board, lane, workspace):
            decision = build_routing_decision(
                task_id=claimed.id, board=board, task_type="implementation", capability="implement",
                risk="normal", code_change=True, independent_review=True,
                preferred_profile=route.selected_profile, reviewer_profile=route.reviewer_profile,
                selected_profile=route.selected_profile,
                selected_provider=claimed.provider_override, selected_model=claimed.model_override,
                human_gate_required=False, independence_valid=route.independence_valid,
                policy_digest=None, selected_by="implement", run_id=claimed.current_run_id,
                session_id=claimed.session_id,
            )
            if not kbr.persist_run_routing_decision(
                run_conn, task_id=claimed.id, run_id=int(claimed.current_run_id),
                decision=decision, event_kind="routing_selected",
            ):
                raise RuntimeError("authoritative routing decision could not be persisted")
            decision_box.update(decision)

        try:
            dispatch = kbd.dispatch_once(
                conn, task_id=task_id, board=board, max_spawn=1,
                before_spawn_fn=persist_before_spawn,
            )
        except Exception as exc:
            return _result(
                task_id=task_id, board=board, task_status=kb.get_task(conn, task_id).status,
                selected_profile=route.selected_profile, reviewer_profile=route.reviewer_profile,
                independence_valid=route.independence_valid, dispatch_status="routing_failed",
                message=str(exc),
            )
        task_after = kb.get_task(conn, task_id)
        decision = decision_box
        status = "started" if dispatch.spawned else "not_dispatched"
        if dispatch.skipped_locked:
            status = "already_running"
        return _result(
            task_id=task_id, board=board, task_status=task_after.status if task_after else None,
            run_id=task_after.current_run_id if task_after else None,
            decision_id=decision.get("decision_id"), selected_profile=route.selected_profile,
            selected_provider=decision.get("selected_provider"), selected_model=decision.get("selected_model"),
            reviewer_profile=route.reviewer_profile, dispatch_status=status,
            human_gate_required=False, routing_contract_version=decision.get("routing_contract_version"),
            independence_valid=route.independence_valid, fallback_used=decision.get("fallback_used"),
            message="implementation started" if status == "started" else "task was not dispatched",
        )


def render_implement_result(result: dict[str, Any]) -> str:
    if result.get("dispatch_status") == "started":
        return "\n".join((
            "Implement started",
            f"Task: {result.get('task_id') or '-'}",
            f"Run: {result.get('run_id') or '-'}",
            f"Implementation lane: {result.get('selected_profile') or '-'}",
            f"Expected reviewer: {result.get('reviewer_profile') or '-'}",
        ))
    return str(result.get("message") or "Implement was not started")


def run_implement_slash_rendered(text: str) -> str:
    return render_implement_result(run_implement_slash(text))
