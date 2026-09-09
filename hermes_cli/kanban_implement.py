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
_DEFAULT_IMPLEMENTATION_PROFILE = "rozmilo-codex"
_REVIEWER_BY_PROFILE = {
    "rozmilo-codex": "rozmilo-claude",
    "rozmilo-claude": "rozmilo-codex",
}
_HUMAN_GATE_SATISFIED_EVENTS = frozenset({"human_gate_satisfied", "human_approved", "approved"})


@dataclass(frozen=True)
class ImplementationRoute:
    selected_profile: Optional[str]
    reviewer_profile: Optional[str]
    implementation_provider: Optional[str]
    reviewer_provider: Optional[str]
    selected_model: Optional[str]
    independence_valid: bool


def _kanban_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        raw = read_raw_config()
        return cfg_get(raw, "kanban", default={}) if isinstance(cfg_get(raw, "kanban", default={}), dict) else {}
    except Exception:
        return {}


def implement_command_enabled() -> bool:
    return is_truthy_value(_kanban_config().get(_IMPLEMENT_GATE_KEY), default=False)


def _profile_runtime_identity(profile: str) -> tuple[Optional[str], Optional[str]]:
    """Read only authoritative profile config; unavailable fields remain ``None``."""
    from hermes_cli.profiles import get_profile_dir, profile_exists, _read_config_model

    if not profile_exists(profile):
        return None, None
    model, provider = _read_config_model(get_profile_dir(profile))
    return (
        str(provider).strip() if provider else None,
        str(model).strip() if model else None,
    )


def select_implementation_route(profile: Optional[str]) -> ImplementationRoute:
    selected = str(profile or "").strip() or None
    if selected is None:
        raise ValueError("implementation profile is required; profile=null is non-authoritative")
    selected = selected.casefold()
    reviewer = _REVIEWER_BY_PROFILE.get(selected)
    if reviewer is None:
        raise ValueError(f"implementation profile is not authoritative: {selected}")
    implementation_provider, selected_model = _profile_runtime_identity(selected)
    reviewer_provider, _ = _profile_runtime_identity(reviewer)
    independent = evaluate_reviewer_independence(
        implementation_profile=selected,
        reviewer_profile=reviewer,
        independent_review=True,
        implementation_provider=implementation_provider,
        reviewer_provider=reviewer_provider,
    )
    if not independent:
        raise ValueError("required independent review is not valid")
    return ImplementationRoute(
        selected, reviewer, implementation_provider, reviewer_provider, selected_model, independent,
    )


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
            reference.append(token)
        index += 1
    if not reference:
        return None, None, None, "Usage: /implement <task|reference> [--board <board>] [--profile <profile>]"
    return " ".join(reference), board, profile, None


def resolve_human_gate(conn, task: kb.Task) -> tuple[bool, bool]:
    """Resolve the existing Kanban/policy gate without creating another state store."""
    policy = _kanban_config()
    required = policy.get("implement_human_gate_required", policy.get("human_gate_required"))
    if not isinstance(required, bool):
        required = next(
            (
                event.payload.get("required")
                for event in reversed(kb.list_events(conn, task.id))
                if event.kind == "human_gate_required"
                and isinstance(event.payload, dict)
                and isinstance(event.payload.get("required"), bool)
            ),
            False,
        )
    if not required:
        return False, True
    satisfied = any(event.kind in _HUMAN_GATE_SATISFIED_EVENTS for event in kb.list_events(conn, task.id))
    return True, satisfied


def _preflight(
    conn, task: kb.Task, requested_profile: Optional[str],
) -> tuple[Optional[dict[str, Any]], Optional[ImplementationRoute]]:
    human_gate_required, human_gate_satisfied = resolve_human_gate(conn, task)
    if task.status in {"done", "archived"}:
        return _result(task_id=task.id, task_status=task.status, human_gate_required=human_gate_required,
                       dispatch_status="not_eligible", message=f"task is {task.status}; it will not be restarted"), None
    if task.status != "ready" or task.claim_lock is not None:
        return _result(task_id=task.id, task_status=task.status, human_gate_required=human_gate_required,
                       dispatch_status="not_eligible", message="task is not ready for implementation"), None
    graph = kb.task_graph_context(conn, task.id)
    if not all(parent.get("status") in {"done", "archived"} for parent in graph.get("parents", [])):
        return _result(task_id=task.id, task_status=task.status, dispatch_status="not_eligible",
                       human_gate_required=human_gate_required,
                       message="dependencies do not permit execution"), None
    if human_gate_required and not human_gate_satisfied:
        return _result(task_id=task.id, task_status=task.status, human_gate_required=True,
                       dispatch_status="not_eligible", message="required human gate is not satisfied"), None
    policy = _kanban_config()
    policy_profile = policy.get("implement_profile")
    requested = requested_profile if requested_profile is not None else (
        str(policy_profile).strip() if policy_profile else _DEFAULT_IMPLEMENTATION_PROFILE
    )
    try:
        route = select_implementation_route(requested)
    except ValueError as exc:
        return _result(task_id=task.id, task_status=task.status,
                       human_gate_required=human_gate_required, dispatch_status="routing_failed",
                       message=str(exc)), None
    return None, route


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
            existing = kbr.load_run_routing_decision(conn, task_id=task_id, run_id=int(task.current_run_id)) if task.current_run_id else None
            existing = existing or {}
            return _result(
                task_id=task_id, board=board, task_status=task.status, run_id=task.current_run_id,
                decision_id=existing.get("decision_id"), selected_profile=existing.get("selected_profile"),
                selected_provider=existing.get("selected_provider"), selected_model=existing.get("selected_model"),
                reviewer_profile=existing.get("reviewer_profile"), routing_contract_version=existing.get("routing_contract_version"),
                independence_valid=existing.get("independence_valid"), fallback_used=existing.get("fallback_used"),
                human_gate_required=existing.get("human_gate_required"), dispatch_status="already_running",
                message="implementation is already running",
            )
        preflight_result, route = _preflight(conn, task, requested_profile)
        if preflight_result is not None:
            preflight_result["board"] = board
            return preflight_result
        assert route is not None
        decision_box: dict[str, Any] = {}
        human_gate_required, _ = resolve_human_gate(conn, task)

        def persist_before_spawn(run_conn, claimed, *, board, lane, workspace):
            selected_provider = claimed.provider_override or route.implementation_provider
            selected_model = claimed.model_override or route.selected_model
            decision = build_routing_decision(
                task_id=claimed.id, board=board, task_type="implementation", capability="implement",
                risk="normal", code_change=True, independent_review=True,
                preferred_profile=route.selected_profile, reviewer_profile=route.reviewer_profile,
                selected_profile=route.selected_profile, selected_provider=selected_provider,
                selected_model=selected_model, human_gate_required=human_gate_required,
                independence_valid=route.independence_valid, policy_digest=None,
                selected_by="implement", run_id=claimed.current_run_id, session_id=claimed.session_id,
            )
            if not kbr.persist_run_routing_decision(run_conn, task_id=claimed.id, run_id=int(claimed.current_run_id),
                                                   decision=decision, event_kind="routing_selected"):
                raise RuntimeError("authoritative routing decision could not be persisted")
            decision_box.update(decision)

        try:
            dispatch = kbd.dispatch_once(
                conn, task_id=task_id, board=board, max_spawn=1,
                before_spawn_fn=persist_before_spawn, spawn_profile=route.selected_profile,
            )
        except Exception as exc:
            current = kb.get_task(conn, task_id)
            return _result(
                task_id=task_id, board=board, task_status=current.status if current else None,
                selected_profile=route.selected_profile, selected_provider=route.implementation_provider,
                selected_model=route.selected_model, reviewer_profile=route.reviewer_profile,
                human_gate_required=human_gate_required, independence_valid=route.independence_valid,
                dispatch_status="routing_failed", message=f"implementation dispatch failed: {exc}",
            )
        task_after = kb.get_task(conn, task_id)
        status = "started" if dispatch.spawned else "not_dispatched"
        if dispatch.skipped_locked:
            status = "already_running"
        return _result(
            task_id=task_id, board=board, task_status=task_after.status if task_after else None,
            run_id=task_after.current_run_id if task_after else None,
            decision_id=decision_box.get("decision_id"), selected_profile=route.selected_profile,
            selected_provider=decision_box.get("selected_provider"), selected_model=decision_box.get("selected_model"),
            reviewer_profile=route.reviewer_profile, dispatch_status=status,
            human_gate_required=human_gate_required, routing_contract_version=decision_box.get("routing_contract_version"),
            independence_valid=route.independence_valid, fallback_used=decision_box.get("fallback_used"),
            message="implementation started" if status == "started" else "task was not dispatched",
        )


def render_implement_result(result: dict[str, Any]) -> str:
    if result.get("dispatch_status") == "started":
        return "\n".join(("Implement started", f"Task: {result.get('task_id') or '-'}",
                             f"Run: {result.get('run_id') or '-'}",
                             f"Implementation lane: {result.get('selected_profile') or '-'}",
                             f"Expected reviewer: {result.get('reviewer_profile') or '-'}"))
    return str(result.get("message") or "Implement was not started")


def run_implement_slash_rendered(text: str) -> str:
    return render_implement_result(run_implement_slash(text))
