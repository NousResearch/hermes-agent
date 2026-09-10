"""Thin, state-aware adapter for the canonical ``/continue`` command."""

from __future__ import annotations

import shlex
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_routing as kbr
from hermes_cli import kanban_fix_review as fix_review
from hermes_cli import kanban_implement as implement
from hermes_cli import kanban_review as review
from hermes_cli.kanban_status import resolve_status_reference
from utils import is_truthy_value


_GATE_KEY = "continue_command"


def _kanban_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        value = cfg_get(read_raw_config(), "kanban", default={})
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def continue_command_enabled() -> bool:
    return is_truthy_value(_kanban_config().get(_GATE_KEY), default=False)


def _result(**fields: Any) -> dict[str, Any]:
    result = {
        "command": "continue", "task_id": None, "board": None, "task_status": None,
        "continuation_state": "unknown", "selected_action": None, "delegated_command": None,
        "dispatch_status": "failed", "run_id": None, "decision_id": None,
        "implementation_profile": None, "implementation_provider": None, "implementation_model": None,
        "reviewer_profile": None, "reviewer_provider": None, "reviewer_model": None,
        "human_gate_required": None, "recovery_required": False, "terminal": False, "message": None,
    }
    result.update(fields)
    return result


def _parse_args(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    raw = str(text or "").strip().lstrip("/")
    if raw == "continue" or raw.startswith("continue "):
        raw = raw[len("continue"):].strip()
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
        return None, None, "Usage: /continue <task|reference> [--board <board>]"
    return " ".join(refs), board, None


def _copy_delegated(result: dict[str, Any], *, action: str, state: str) -> dict[str, Any]:
    fields = {key: result.get(key) for key in (
        "task_id", "board", "task_status", "dispatch_status", "run_id", "decision_id",
        "implementation_profile", "implementation_provider", "implementation_model",
        "reviewer_profile", "reviewer_provider", "reviewer_model", "human_gate_required",
        "message",
    )}
    return _result(
        **fields, continuation_state=state, selected_action=action,
        delegated_command=action, terminal=False,
    )


def _active_result(conn, task: kb.Task, board: str) -> Optional[dict[str, Any]]:
    if task.status != "running" or not task.current_run_id:
        return None
    decision = kbr.load_run_routing_decision(conn, task_id=task.id, run_id=int(task.current_run_id)) or {}
    kind = decision.get("task_type")
    if kind not in {"implementation", "review"}:
        return None
    fields = {key: decision.get(key) for key in (
        "implementation_profile", "implementation_provider", "implementation_model",
        "reviewer_profile", "reviewer_provider", "reviewer_model", "human_gate_required",
    )}
    return _result(
        task_id=task.id, board=board, task_status=task.status, continuation_state="active-implementation" if kind == "implementation" else "active-review",
        selected_action="already-active", delegated_command=None, dispatch_status="already_active",
        run_id=task.current_run_id, decision_id=decision.get("decision_id"), **fields,
        message="implementation is already active" if kind == "implementation" else "review is already active",
    )


def _recovery_result(conn, task: kb.Task, board: str) -> Optional[dict[str, Any]]:
    reason = kbd.recovery_requirement_for_task(conn, task.id)
    if reason is not None:
        return _result(
            task_id=task.id, board=board, task_status=task.status,
            continuation_state="recovery-required", selected_action="recover-required",
            dispatch_status="recovery_required", recovery_required=True,
            message=f"authoritative recovery state ({reason}) requires /recover; no retry was performed",
        )
    return None


def run_continue_slash(text: str) -> dict[str, Any]:
    reference, explicit_board, error = _parse_args(text)
    if error:
        return _result(dispatch_status="invalid", message=error)
    if not continue_command_enabled():
        return _result(dispatch_status="disabled", message="/continue is disabled (kanban.continue_command)")
    assert reference is not None
    resolution = resolve_status_reference(
        reference, board=explicit_board, include_archived=True,
    )
    if not resolution.ok or resolution.scope != "task" or not resolution.task_id or not resolution.board:
        return _result(board=resolution.board, dispatch_status="not_eligible", continuation_state="unknown", message=resolution.error or "task is not eligible")
    board, task_id = resolution.board, resolution.task_id
    with kbc.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            return _result(task_id=task_id, board=board, dispatch_status="not_eligible", message="task no longer exists")
        if task.status in {"done", "archived"}:
            return _result(task_id=task.id, board=board, task_status=task.status,
                           continuation_state="terminal", selected_action="terminal",
                           dispatch_status="not_eligible", terminal=True,
                           message=f"task is {task.status}; it will not be restarted")
        active = _active_result(conn, task, board)
        if active is not None:
            return active
        recovery = _recovery_result(conn, task, board)
        if recovery is not None:
            return recovery
        if task.status in {"blocked", "todo", "scheduled"}:
            return _result(task_id=task.id, board=board, task_status=task.status,
                           continuation_state="waiting", selected_action="wait",
                           dispatch_status="not_eligible", message="task is blocked or waiting; no action was taken")
        if task.status == "ready":
            changes_id = fix_review._latest_changes_run(conn, task.id)
            if changes_id is not None:
                if fix_review._implementation_run_after(conn, task.id, changes_id, active=False) is None:
                    action, state, delegate = "fix-review", "ready-changes-requested", fix_review.run_fix_review_slash
                else:
                    action, state, delegate = "review", "review-eligible", review.run_review_slash
            else:
                latest = kb.latest_run(conn, task.id)
                if latest and latest.outcome in {"completed", "review_requested"}:
                    action, state, delegate = "review", "review-eligible", review.run_review_slash
                else:
                    action, state, delegate = "implement", "ready-implementation", implement.run_implement_slash
        elif task.status == "review":
            action, state, delegate = "review", "review-eligible", review.run_review_slash
        else:
            return _result(task_id=task.id, board=board, task_status=task.status,
                           continuation_state="unknown", selected_action="none",
                           dispatch_status="not_eligible", message="authoritative task state is inconsistent; no action was taken")
    try:
        return _copy_delegated(delegate(f"{task_id} --board {board}"), action=action, state=state)
    except Exception as exc:
        return _result(task_id=task_id, board=board, selected_action=action,
                       delegated_command=action, continuation_state=state,
                       dispatch_status="failed", message=f"delegated /{action} failed: {exc}")


def render_continue_result(result: dict[str, Any]) -> str:
    return str(result.get("message") or f"Continue: {result.get('continuation_state') or 'unknown'}")


def run_continue_slash_rendered(text: str) -> str:
    return render_continue_result(run_continue_slash(text))