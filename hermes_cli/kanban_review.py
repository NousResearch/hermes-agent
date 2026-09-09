"""Feature-gated canonical ``/review`` adapter for the existing Kanban review lane."""

from __future__ import annotations

import shlex
from typing import Any, Optional

from agent.routing_decision import build_routing_decision, evaluate_reviewer_independence
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_routing as kbr
from hermes_cli.kanban_implement import resolve_human_gate
from hermes_cli.kanban_status import resolve_status_reference
from utils import is_truthy_value

_REVIEW_GATE_KEY = "review_command"
_REVIEWER_BY_PROFILE = {
    "rozmilo-codex": "rozmilo-claude",
    "rozmilo-claude": "rozmilo-codex",
}


def _kanban_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        raw = read_raw_config()
        value = cfg_get(raw, "kanban", default={})
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def review_command_enabled() -> bool:
    return is_truthy_value(_kanban_config().get(_REVIEW_GATE_KEY), default=False)


def _profile_runtime_identity(profile: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not profile:
        return None, None
    try:
        from hermes_cli.profiles import _read_config_model, get_profile_dir, profile_exists
        if not profile_exists(profile):
            return None, None
        model, provider = _read_config_model(get_profile_dir(profile))
        return (str(provider).strip() if provider else None, str(model).strip() if model else None)
    except Exception:
        return None, None


def _result(**fields: Any) -> dict[str, Any]:
    result = {
        "command": "review",
        "task_id": None,
        "board": None,
        "task_status": None,
        "review_status": None,
        "run_id": None,
        "decision_id": None,
        "implementation_profile": None,
        "implementation_provider": None,
        "implementation_model": None,
        "reviewer_profile": None,
        "reviewer_provider": None,
        "reviewer_model": None,
        "independence_valid": None,
        "dispatch_status": "failed",
        "human_gate_required": None,
        "message": None,
    }
    result.update(fields)
    return result


def _parse_args(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    raw = str(text or "").strip().lstrip("/")
    if raw == "review" or raw.startswith("review "):
        raw = raw[len("review"):].strip()
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return None, None, f"Invalid arguments: {exc}"
    refs: list[str] = []
    board = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--board":
            index += 1
            if index >= len(tokens) or tokens[index].startswith("--"):
                return None, None, "--board requires a value"
            if board is not None:
                return None, None, "--board may be specified only once"
            board = tokens[index]
        elif token.startswith("--board="):
            if board is not None:
                return None, None, "--board may be specified only once"
            board = token.partition("=")[2]
        elif token.startswith("--"):
            return None, None, f"Unknown option: {token}"
        else:
            refs.append(token)
        index += 1
    if not refs:
        return None, None, "Usage: /review <task|reference> [--board <board>]"
    return " ".join(refs), board, None


def _latest_implementation_route(conn, task_id: str) -> Optional[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC", (task_id,)
    ).fetchall()
    for row in rows:
        decision = kbr.load_run_routing_decision(conn, task_id=task_id, run_id=int(row["id"]))
        if isinstance(decision, dict) and decision.get("task_type") == "implementation":
            return decision
    return None


def _route_from_implementation(conn, task_id: str) -> Optional[dict[str, Any]]:
    decision = _latest_implementation_route(conn, task_id)
    if not decision:
        return None
    implementation_profile = decision.get("selected_profile") or decision.get("preferred_profile")
    if not isinstance(implementation_profile, str) or not implementation_profile.strip():
        return None
    implementation_profile = implementation_profile.strip().casefold()
    reviewer_profile = _REVIEWER_BY_PROFILE.get(implementation_profile)
    if reviewer_profile is None:
        return None
    implementation_provider = decision.get("selected_provider") or decision.get("initial_provider")
    implementation_model = decision.get("selected_model") or decision.get("initial_model")
    if not implementation_provider:
        return None
    reviewer_provider, reviewer_model = _profile_runtime_identity(reviewer_profile)
    if not reviewer_provider:
        return None
    independent = evaluate_reviewer_independence(
        implementation_profile=implementation_profile,
        reviewer_profile=reviewer_profile,
        independent_review=True,
        implementation_provider=str(implementation_provider),
        reviewer_provider=reviewer_provider,
    )
    if not independent:
        return None
    return {
        "implementation_profile": implementation_profile,
        "implementation_provider": str(implementation_provider),
        "implementation_model": implementation_model,
        "reviewer_profile": reviewer_profile,
        "reviewer_provider": reviewer_provider,
        "reviewer_model": reviewer_model,
        "independence_valid": True,
    }


def _active_review_result(conn, task: kb.Task, board: str) -> Optional[dict[str, Any]]:
    if task.status != "running" or not task.current_run_id:
        return None
    claimed = kb._latest_event(conn, task.id, "claimed", int(task.current_run_id))
    payload = kb._json_dict(kb._row_get(claimed, "payload"))
    if payload.get("source_status") != "review":
        return None
    decision = kbr.load_run_routing_decision(conn, task_id=task.id, run_id=int(task.current_run_id)) or {}
    return _result(
        task_id=task.id, board=board, task_status=task.status, review_status="running",
        run_id=task.current_run_id, decision_id=decision.get("decision_id"),
        implementation_profile=decision.get("implementation_profile"),
        implementation_provider=decision.get("implementation_provider"),
        implementation_model=decision.get("implementation_model"),
        reviewer_profile=decision.get("selected_profile"),
        reviewer_provider=decision.get("selected_provider"),
        reviewer_model=decision.get("selected_model"),
        independence_valid=decision.get("independence_valid"), dispatch_status="already_active",
        human_gate_required=decision.get("human_gate_required"), message="review is already active",
    )


def run_review_slash(text: str) -> dict[str, Any]:
    reference, explicit_board, error = _parse_args(text)
    if error:
        return _result(dispatch_status="invalid", message=error)
    if not review_command_enabled():
        return _result(dispatch_status="disabled", message="/review is disabled (kanban.review_command)")
    assert reference is not None
    resolution = resolve_status_reference(reference, board=explicit_board)
    if not resolution.ok or resolution.scope != "task" or not resolution.task_id or not resolution.board:
        return _result(board=resolution.board, dispatch_status="not_eligible", message=resolution.error or "task is not eligible")
    board, task_id = resolution.board, resolution.task_id
    with kbc.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            return _result(task_id=task_id, board=board, dispatch_status="not_eligible", message="task no longer exists")
        active = _active_review_result(conn, task, board)
        if active is not None:
            return active
        human_gate_required, human_gate_satisfied = resolve_human_gate(conn, task)
        base = dict(task_id=task.id, board=board, task_status=task.status,
                    review_status=task.status if task.status in {"review", "running"} else None,
                    human_gate_required=human_gate_required)
        if task.status in {"done", "archived"}:
            return _result(**base, dispatch_status="not_eligible", message=f"task is {task.status}; it will not be restarted")
        if task.status not in {"ready", "review"} or task.claim_lock is not None:
            return _result(**base, dispatch_status="not_eligible", message="task is not eligible for review")
        graph = kb.task_graph_context(conn, task.id)
        if not all(parent.get("status") in {"done", "archived"} for parent in graph.get("parents", [])):
            return _result(**base, dispatch_status="not_eligible", message="dependencies do not permit review")
        if human_gate_required and not human_gate_satisfied:
            return _result(**base, dispatch_status="not_eligible", message="required human gate is not satisfied")
        route = _route_from_implementation(conn, task.id)
        if route is None:
            return _result(**base, dispatch_status="routing_failed", message="authoritative implementation provenance is required for independent review")
        if task.status == "ready":
            ok, reason = kb.request_review(
                conn, task.id, summary="canonical /review handoff", reviewer=route["reviewer_profile"], with_reason=True,
            )
            if not ok:
                return _result(**base, **route, dispatch_status="not_eligible", message=reason or "review handoff was refused")
        decision_box: dict[str, Any] = {}

        def persist_before_spawn(run_conn, claimed, *, board, lane, workspace):
            decision = build_routing_decision(
                task_id=claimed.id, board=board, task_type="review", capability="review",
                risk="normal", code_change=True, independent_review=True,
                preferred_profile=route["reviewer_profile"], reviewer_profile=route["reviewer_profile"],
                selected_profile=route["reviewer_profile"], selected_provider=route["reviewer_provider"],
                selected_model=route["reviewer_model"], human_gate_required=human_gate_required,
                independence_valid=True, policy_digest=None, selected_by="review",
                run_id=claimed.current_run_id, session_id=claimed.session_id,
            )
            decision.update({
                "implementation_profile": route["implementation_profile"],
                "implementation_provider": route["implementation_provider"],
                "implementation_model": route["implementation_model"],
            })
            if not kbr.persist_run_routing_decision(run_conn, task_id=claimed.id, run_id=int(claimed.current_run_id), decision=decision, event_kind="routing_selected"):
                raise RuntimeError("authoritative review routing decision could not be persisted")
            decision_box.update(decision)

        try:
            dispatch = kbd.dispatch_once(
                conn, task_id=task_id, board=board, max_spawn=1,
                before_spawn_fn=persist_before_spawn, spawn_profile=route["reviewer_profile"],
            )
        except Exception as exc:
            current = kb.get_task(conn, task_id)
            return _result(task_id=task_id, board=board, task_status=current.status if current else None,
                           review_status=current.status if current and current.status == "review" else None,
                           **route, human_gate_required=human_gate_required, dispatch_status="routing_failed",
                           message=f"review dispatch failed: {exc}")
        after = kb.get_task(conn, task_id)
        status = "started" if dispatch.spawned else ("already_active" if dispatch.skipped_locked else "not_dispatched")
        return _result(task_id=task_id, board=board, task_status=after.status if after else None,
                       review_status="running" if after and after.status == "running" else (after.status if after else None),
                       run_id=after.current_run_id if after else None, decision_id=decision_box.get("decision_id"),
                       **route, human_gate_required=human_gate_required, dispatch_status=status,
                       message="review started" if status == "started" else "review was not dispatched")


def render_review_result(result: dict[str, Any]) -> str:
    status = result.get("dispatch_status")
    if status in {"started", "already_active"}:
        return "\n".join(("Review started" if status == "started" else "Review already active",
                             f"Task: {result.get('task_id') or '-'}", f"Run: {result.get('run_id') or '-'}",
                             f"Reviewer: {result.get('reviewer_profile') or '-'}"))
    return str(result.get("message") or "Review was not started")


def run_review_slash_rendered(text: str) -> str:
    return render_review_result(run_review_slash(text))
