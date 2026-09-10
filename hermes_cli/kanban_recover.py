"""Thin, feature-gated recovery adapter for the canonical ``/recover`` command."""

from __future__ import annotations

import shlex
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_status import resolve_status_reference
from utils import is_truthy_value

_GATE_KEY = "recover_command"
_BREAKER_OUTCOMES = frozenset({"gave_up"})
_RECOVERABLE_OUTCOMES = frozenset({"crashed", "timed_out", "spawn_failed", "reclaimed", "stale"})


def _kanban_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        value = cfg_get(read_raw_config(), "kanban", default={})
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def recover_command_enabled() -> bool:
    return is_truthy_value(_kanban_config().get(_GATE_KEY), default=False)


def _result(**fields: Any) -> dict[str, Any]:
    result = {
        "command": "recover", "task_id": None, "board": None, "task_status": None,
        "recovery_state": "unknown", "eligible": False, "action": "none",
        "mutation_performed": False, "run_id": None, "retry_info": None,
        "cooldown": None, "dispatch_status": "failed", "next_action": None,
        "message": None,
    }
    result.update(fields)
    return result


def _parse_args(text: str) -> tuple[Optional[str], Optional[str], bool, Optional[str]]:
    raw = str(text or "").strip().lstrip("/")
    if raw == "recover" or raw.startswith("recover "):
        raw = raw[len("recover"):].strip()
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return None, None, False, f"Invalid arguments: {exc}"
    refs: list[str] = []
    board = None
    requeue = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--requeue":
            if requeue:
                return None, None, False, "--requeue may be specified only once"
            requeue = True
        elif token == "--board":
            index += 1
            if index >= len(tokens) or tokens[index].startswith("--"):
                return None, None, False, "--board requires a value"
            if board is not None:
                return None, None, False, "--board may be specified only once"
            board = tokens[index]
        elif token.startswith("--board="):
            if board is not None:
                return None, None, False, "--board may be specified only once"
            board = token.partition("=")[2]
        elif token.startswith("--"):
            return None, None, False, f"Unknown option: {token}"
        else:
            refs.append(token)
        index += 1
    if not refs:
        return None, None, False, "Usage: /recover <task|reference> [--requeue] [--board <board>]"
    return " ".join(refs), board, requeue, None


def _latest_run(conn, task_id: str):
    return conn.execute(
        "SELECT id, outcome, status, ended_at FROM task_runs "
        "WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()


def _cooldown(conn, task_id: str) -> Optional[int]:
    latest = _latest_run(conn, task_id)
    if latest is None or latest["outcome"] != "rate_limited" or latest["ended_at"] is None:
        return None
    try:
        seconds = int(kb._resolve_rate_limit_cooldown_seconds())
    except Exception:
        seconds = 0
    remaining = seconds - (int(__import__("time").time()) - int(latest["ended_at"]))
    return max(remaining, 0) if remaining > 0 else None


def _sticky_block(conn, task_id: str) -> bool:
    return bool(getattr(kb, "_has_sticky_block")(conn, task_id))


def _mark_recovered(conn, task_id: str, *, outcome: str) -> bool:
    """Record the idempotent lifecycle acknowledgement consumed by /continue."""
    with kb.write_txn(conn):
        exists = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'recovered' "
            "ORDER BY id DESC LIMIT 1", (task_id,),
        ).fetchone()
        if exists:
            return False
        kb._append_event(conn, task_id, "recovered", {"outcome": outcome})
        return True


def _dispatch_fields(dispatch: kbd.DispatchResult) -> dict[str, Any]:
    return {
        "reclaimed": dispatch.reclaimed,
        "reconciled_orphans": list(dispatch.reconciled_orphans),
        "crashed": list(dispatch.crashed),
        "stale": list(dispatch.stale),
        "timed_out": list(dispatch.timed_out),
        "spawned": list(dispatch.spawned),
        "rate_limited": list(dispatch.rate_limited),
        "auto_blocked": list(dispatch.auto_blocked),
    }


def run_recover_slash(text: str) -> dict[str, Any]:
    reference, explicit_board, requeue, error = _parse_args(text)
    if error:
        return _result(dispatch_status="invalid", message=error)
    if not recover_command_enabled():
        return _result(dispatch_status="disabled", message="/recover is disabled (kanban.recover_command)")
    assert reference is not None
    resolution = resolve_status_reference(reference, board=explicit_board, include_archived=True)
    if not resolution.ok or resolution.scope != "task" or not resolution.task_id or not resolution.board:
        return _result(board=resolution.board, dispatch_status="not_eligible", message=resolution.error or "task is not eligible")

    board, task_id = resolution.board, resolution.task_id
    with kbc.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            return _result(task_id=task_id, board=board, dispatch_status="not_eligible", message="task no longer exists")
        base = {"task_id": task.id, "board": board, "task_status": task.status,
                "run_id": task.current_run_id}
        if task.status in {"done", "archived"}:
            return _result(**base, recovery_state="terminal", action="terminal", dispatch_status="not_eligible",
                           message=f"task is {task.status}; it will not be changed")
        if task.status == "triage":
            return _result(**base, recovery_state="waiting", action="none", dispatch_status="not_eligible",
                           message="task is in triage; /recover does not act on triage tasks")
        if kbd.dispatch_paused():
            return _result(**base, recovery_state="paused", action="wait", dispatch_status="paused",
                           message="recovery paused by ESTOP; no mutation was performed")

        latest = _latest_run(conn, task_id)
        latest_outcome = latest["outcome"] if latest else None
        sticky = _sticky_block(conn, task_id)
        if requeue and sticky:
            return _result(**base, recovery_state="sticky-blocked", action="operator-unblock",
                           dispatch_status="not_eligible", retry_info={"outcome": latest_outcome},
                           message="explicit operator block remains; use the existing unblock/control command")

        if task.status in {"todo", "scheduled"}:
            return _result(**base, recovery_state="waiting", action="wait", dispatch_status="not_eligible",
                           retry_info={"outcome": latest_outcome} if latest_outcome else None,
                           message="task is waiting; no action was taken")
        if task.status == "blocked" and not (requeue and latest_outcome in _BREAKER_OUTCOMES):
            if latest_outcome in _BREAKER_OUTCOMES:
                return _result(**base, recovery_state="gave-up", action="requeue-required",
                               dispatch_status="not_eligible", retry_info={"outcome": latest_outcome, "reset": False},
                               message="retry budget is exhausted; use /recover <task> --requeue")
            return _result(**base, recovery_state="blocked", action="wait", dispatch_status="not_eligible",
                           retry_info={"outcome": latest_outcome} if latest_outcome else None,
                           message="task is blocked; no action was taken")

        if requeue:
            if latest_outcome not in _BREAKER_OUTCOMES:
                return _result(**base, recovery_state="not-requeueable", action="none", dispatch_status="not_eligible",
                               retry_info={"outcome": latest_outcome},
                               message="--requeue is allowed only for breaker/gave_up retry exhaustion")
            changed = kb.unblock_task(conn, task_id)
            after = kb.get_task(conn, task_id)
            return _result(task_id=task_id, board=board, task_status=after.status if after else None,
                           recovery_state="requeued" if changed else "already-requeued",
                           eligible=bool(changed), action="requeue", mutation_performed=bool(changed),
                           retry_info={"outcome": latest_outcome, "reset": bool(changed), "consecutive_failures": 0},
                           dispatch_status="requeued" if changed else "already_recovered",
                           next_action=f"/continue {task_id}" if after and after.status in {"ready", "review"} else None,
                           message="breaker reset and task requeued" if changed else "breaker was already requeued")

        dispatch = kbd.dispatch_once(conn, task_id=task_id, max_spawn=0)
        after = kb.get_task(conn, task_id)
        if dispatch.paused:
            return _result(**base, recovery_state="paused", action="wait", dispatch_status="paused",
                           message="recovery paused by ESTOP; no mutation was performed")
        task_status = after.status if after else task.status
        latest = _latest_run(conn, task_id)
        latest_outcome = latest["outcome"] if latest else None
        cooldown = _cooldown(conn, task_id)
        if cooldown is not None:
            return _result(task_id=task_id, board=board, task_status=task_status,
                           recovery_state="rate-limited", action="wait", dispatch_status="not_eligible",
                           cooldown={"remaining_seconds": cooldown}, retry_info={"outcome": latest_outcome},
                           message=f"rate-limit cooldown is active; wait {cooldown}s")
        if kbd.check_respawn_guard(conn, task_id) == "blocker_auth":
            return _result(task_id=task_id, board=board, task_status=task_status,
                           recovery_state="blocker-auth", action="operator-remediation", dispatch_status="not_eligible",
                           retry_info={"outcome": latest_outcome},
                           message="provider/auth blocker requires human remediation; failure details were preserved")
        if latest_outcome == "gave_up" or (task_status == "blocked" and int(after.consecutive_failures or 0) > 0):
            return _result(task_id=task_id, board=board, task_status=task_status,
                           recovery_state="gave-up", action="requeue-required", dispatch_status="not_eligible",
                           retry_info={"outcome": latest_outcome, "reset": False},
                           message="retry budget is exhausted; use /recover <task> --requeue")
        if latest_outcome in _RECOVERABLE_OUTCOMES or any(_dispatch_fields(dispatch)[key] for key in ("reclaimed", "reconciled_orphans", "crashed", "stale", "timed_out")):
            mutation = _mark_recovered(conn, task_id, outcome=str(latest_outcome or "reclaimed"))
            after = kb.get_task(conn, task_id)
            return _result(task_id=task_id, board=board, task_status=after.status if after else task_status,
                           recovery_state="recovered", eligible=True, action="recover", mutation_performed=mutation,
                           retry_info={"outcome": latest_outcome, "reset": False}, dispatch_status="recovered",
                           next_action=f"/continue {task_id}", message="task recovered and is ready for /continue")
        if latest_outcome == "rate_limited":
            return _result(task_id=task_id, board=board, task_status=task_status,
                           recovery_state="healthy", action="none", dispatch_status="already_recovered",
                           retry_info={"outcome": latest_outcome, "reset": False}, message="rate-limit cooldown expired; task is ready")
        return _result(task_id=task_id, board=board, task_status=task_status,
                       recovery_state="healthy", eligible=task_status in {"ready", "review"}, action="none",
                       dispatch_status="already_recovered", retry_info={"outcome": latest_outcome} if latest_outcome else None,
                       message="task is already healthy; no recovery was needed")


def render_recover_result(result: dict[str, Any]) -> str:
    return str(result.get("message") or "Recovery completed")


def run_recover_slash_rendered(text: str) -> str:
    return render_recover_result(run_recover_slash(text))
