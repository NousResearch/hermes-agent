"""Thin, feature-gated adapter for the canonical ``/close-task`` command.

This is deliberately a thin wrapper over the existing Kanban lifecycle engine
(``kanban_db.complete_task``). It introduces no new state authority: task
status, ownership, and CAS protections all remain owned by ``kanban_db``.

Closing a task means the authoritative Kanban lifecycle state becomes
``done`` — nothing more. It does NOT mean merged, deployed, released, or
live; this adapter never touches git, CI, or any deployment surface.
"""

from __future__ import annotations

import shlex
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_status import resolve_status_reference
from utils import is_truthy_value

_GATE_KEY = "close_task_command"
_ELIGIBLE_STATUSES = frozenset({"running", "ready", "blocked", "review"})


def _kanban_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        value = cfg_get(read_raw_config(), "kanban", default={})
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def close_task_command_enabled() -> bool:
    return is_truthy_value(_kanban_config().get(_GATE_KEY), default=False)


def _result(**fields: Any) -> dict[str, Any]:
    result = {
        "command": "close-task", "task_id": None, "board": None, "task_status": None,
        "closure_state": "unknown", "action": "none", "mutation_performed": False,
        "run_id": None, "dispatch_status": "failed", "message": None,
    }
    result.update(fields)
    return result


def _parse_args(text: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    raw = str(text or "").strip().lstrip("/")
    if raw == "close-task" or raw.startswith("close-task "):
        raw = raw[len("close-task"):].strip()
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return None, None, None, None, f"Invalid arguments: {exc}"
    refs: list[str] = []
    board = result_text = summary = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--board", "--result", "--summary"}:
            index += 1
            if index >= len(tokens) or tokens[index].startswith("--"):
                return None, None, None, None, f"{token} requires a value"
            if token == "--board":
                if board is not None:
                    return None, None, None, None, "--board may be specified only once"
                board = tokens[index]
            elif token == "--result":
                if result_text is not None:
                    return None, None, None, None, "--result may be specified only once"
                result_text = tokens[index]
            else:
                if summary is not None:
                    return None, None, None, None, "--summary may be specified only once"
                summary = tokens[index]
        elif token.startswith("--board="):
            if board is not None:
                return None, None, None, None, "--board may be specified only once"
            board = token.partition("=")[2]
        elif token.startswith("--result="):
            if result_text is not None:
                return None, None, None, None, "--result may be specified only once"
            result_text = token.partition("=")[2]
        elif token.startswith("--summary="):
            if summary is not None:
                return None, None, None, None, "--summary may be specified only once"
            summary = token.partition("=")[2]
        elif token.startswith("--"):
            return None, None, None, None, f"Unknown option: {token}"
        else:
            refs.append(token)
        index += 1
    if not refs:
        return None, None, None, None, "Usage: /close-task <task|reference> [--board <board>] [--result <text>] [--summary <text>]"
    return " ".join(refs), board, result_text, summary, None


def run_close_task_slash(text: str) -> dict[str, Any]:
    reference, explicit_board, result_text, summary, error = _parse_args(text)
    if error:
        return _result(dispatch_status="invalid", message=error)
    if not close_task_command_enabled():
        return _result(dispatch_status="disabled", message="/close-task is disabled (kanban.close_task_command)")
    assert reference is not None
    resolution = resolve_status_reference(reference, board=explicit_board, include_archived=True)
    if not resolution.ok or resolution.scope != "task" or not resolution.task_id or not resolution.board:
        return _result(board=resolution.board, dispatch_status="not_eligible",
                       message=resolution.error or "task is not eligible")

    board, task_id = resolution.board, resolution.task_id
    with kbc.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            return _result(task_id=task_id, board=board, dispatch_status="not_eligible",
                           message="task no longer exists")
        base = {"task_id": task.id, "board": board, "task_status": task.status, "run_id": task.current_run_id}
        if task.status in {"done", "archived"}:
            # Idempotent per the canonical lifecycle engine: closing an already-terminal
            # task is a no-op, never a second behavior (never merge/deploy/release).
            return _result(**base, closure_state="already-closed", action="none",
                           mutation_performed=False, dispatch_status="already_closed",
                           message=f"task is already {task.status}; no action was taken")
        if task.status not in _ELIGIBLE_STATUSES:
            return _result(**base, closure_state="not-eligible", action="none",
                           dispatch_status="not_eligible",
                           message=f"task status ({task.status}) is not eligible for closure")

        changed = kb.complete_task(conn, task_id, result=result_text, summary=summary)
        after = kb.get_task(conn, task_id)
        if not changed:
            # complete_task fails closed on: parent-not-satisfied, CAS mismatch, and any
            # concurrent status transition out of the eligible set observed above. A
            # concurrent caller may have already driven the task to a terminal state
            # between our initial read and this CAS attempt; classify against the fresh
            # re-read (`after`, via the canonical kb.get_task accessor) rather than the
            # stale pre-mutation snapshot, so the losing caller reports idempotent
            # already-closed instead of a false rejection.
            if after is not None and after.status in {"done", "archived"}:
                return _result(task_id=task_id, board=board, task_status=after.status,
                               closure_state="already-closed", action="none", mutation_performed=False,
                               dispatch_status="already_closed",
                               message=f"task is already {after.status}; no action was taken")
            return _result(task_id=task_id, board=board, task_status=after.status if after else task.status,
                           closure_state="rejected", action="none", mutation_performed=False,
                           dispatch_status="not_eligible",
                           message="closure was rejected by the lifecycle engine (dependency, ownership, or state check failed)")
        return _result(task_id=task_id, board=board, task_status=after.status if after else "done",
                       closure_state="closed", action="closed", mutation_performed=True,
                       run_id=after.current_run_id if after else None, dispatch_status="closed",
                       message="task closed")


def render_close_task_result(result: dict[str, Any]) -> str:
    return str(result.get("message") or "Close-task completed")


def run_close_task_slash_rendered(text: str) -> str:
    return render_close_task_result(run_close_task_slash(text))
