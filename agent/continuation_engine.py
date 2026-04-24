from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from agent.gap_check import analyze_gap, should_skip_next_iteration
from agent.runtime_modes import resolve_runtime_mode


DEFAULT_MAX_RUNTIME_RESUMES = 2
DEFAULT_ITERATION_CAP = 100
PROMISE_DONE_TAG = "<promise>DONE</promise>"
_ACTIVE_TODO_STATUSES = {"pending", "in_progress"}
_STOP_OUTCOME_STATUSES = {"completed", "stopped", "cancelled", "canceled"}

logger = logging.getLogger(__name__)


def _normalize_runtime_mode(runtime_mode: Optional[str]) -> str:
    mode = str(runtime_mode or "").strip().lower().replace("-", "_")
    if mode == "ralph_loop":
        return "ralph"
    if mode in {"ulw", "ulw_loop"}:
        return "ultrawork"
    return mode


def _normalize_todos(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "id": str(item.get("id") or "?").strip() or "?",
                "content": str(item.get("content") or item.get("id") or "pending work").strip() or "pending work",
                "status": str(item.get("status") or "pending").strip().lower() or "pending",
            }
        )
    return normalized


def response_contains_promise_done(response_text: Any) -> bool:
    if response_text is None:
        return False
    return PROMISE_DONE_TAG in str(response_text)


def resolve_iteration_cap(runtime_mode: Optional[str], iteration_cap: Optional[int] = None) -> int:
    if iteration_cap is not None:
        return max(int(iteration_cap), 0)
    resolved_mode = resolve_runtime_mode(runtime_mode)
    return max(int(getattr(resolved_mode, "iteration_cap", DEFAULT_ITERATION_CAP) or DEFAULT_ITERATION_CAP), 0)


def resolve_max_resumes(
    *, runtime_mode: Optional[str], max_resumes: Optional[int] = None, iteration_cap: Optional[int] = None
) -> int:
    if max_resumes is not None:
        return max(int(max_resumes), 0)
    if _normalize_runtime_mode(runtime_mode) == "ralph":
        return resolve_iteration_cap(runtime_mode, iteration_cap)
    return DEFAULT_MAX_RUNTIME_RESUMES


def build_continuation_snapshot(agent: Any, result: Optional[dict[str, Any]]) -> dict[str, Any]:
    payload = result if isinstance(result, dict) else {}
    getter = getattr(agent, "get_orchestration_continuation_snapshot", None)
    if callable(getter):
        try:
            snapshot = getter(payload)
            if isinstance(snapshot, dict):
                todos = _normalize_todos(snapshot.get("todoItems") or [])
                active = _normalize_todos(snapshot.get("activeTodos") or [])
                return {
                    "sessionId": str(snapshot.get("sessionId") or getattr(agent, "session_id", "") or ""),
                    "outcomeStatus": str(snapshot.get("outcomeStatus") or "incomplete").strip().lower() or "incomplete",
                    "todoItems": todos,
                    "activeTodos": active,
                    "responsePreview": str(snapshot.get("responsePreview") or payload.get("final_response") or payload.get("error") or "").strip(),
                    "apiCalls": int(snapshot.get("apiCalls") or payload.get("api_calls") or 0),
                    "stopRequested": bool(snapshot.get("stopRequested") or payload.get("stopRequested") or payload.get("stop_requested")),
                    "retryRequested": bool(snapshot.get("retryRequested") or payload.get("retryRequested") or payload.get("retry_requested")),
                }
        except Exception:
            pass

    todos = _normalize_todos(payload.get("todoItems") or payload.get("todos") or [])
    active = [item for item in todos if item["status"] in _ACTIVE_TODO_STATUSES]
    if payload.get("interrupted"):
        outcome = "interrupted"
    elif payload.get("failed"):
        outcome = "failed"
    elif payload.get("completed") and not active:
        outcome = "completed"
    elif active:
        outcome = "incomplete"
    else:
        outcome = "incomplete"
    return {
        "sessionId": str(getattr(agent, "session_id", "") or ""),
        "outcomeStatus": outcome,
        "todoItems": todos,
        "activeTodos": active,
        "responsePreview": str(payload.get("final_response") or payload.get("error") or "").strip(),
        "apiCalls": int(payload.get("api_calls") or 0),
        "stopRequested": bool(payload.get("stopRequested") or payload.get("stop_requested")),
        "retryRequested": bool(payload.get("retryRequested") or payload.get("retry_requested")),
    }


def should_use_continuation_engine(runtime_mode: Optional[str], snapshot: dict[str, Any]) -> bool:
    mode = _normalize_runtime_mode(runtime_mode)
    if mode not in {"ultrawork", "ralph"}:
        return False
    if bool(snapshot.get("stopRequested")):
        return False
    if response_contains_promise_done(snapshot.get("responsePreview")):
        return False

    outcome = str(snapshot.get("outcomeStatus") or "").strip().lower()
    active_todos = bool(snapshot.get("activeTodos"))
    retry_requested = bool(snapshot.get("retryRequested"))

    if outcome in _STOP_OUTCOME_STATUSES:
        return False
    if mode == "ultrawork":
        return active_todos
    return retry_requested or outcome in {"failed", "interrupted", "incomplete"} or active_todos


def build_runtime_resume_message(
    snapshot: dict[str, Any], *, runtime_mode: Optional[str], attempt: int, max_attempts: int
) -> str:
    todos = snapshot.get("activeTodos") or []
    todo_lines = "\n".join(
        f"- [{item.get('status') or 'pending'}] {item.get('content') or item.get('id') or 'pending work'}"
        for item in todos
    ) or "- Continue the unfinished task"
    reason = str(snapshot.get("outcomeStatus") or "interrupted")
    mode = _normalize_runtime_mode(runtime_mode)
    if mode == "ralph":
        return (
            f"[System note: Ralph-loop continuation engine retry. "
            f"Attempt {attempt}/{max_attempts}. Previous outcome: {reason}. "
            "Retry the unfinished or failed work now. Stop cleanly if the loop should end, otherwise keep going until the objective is complete.]\n\n"
            "Open work or retry target:\n"
            f"{todo_lines}"
        )
    return (
        f"[System note: Ultrawork continuation engine resume. "
        f"Attempt {attempt}/{max_attempts}. Previous outcome: {reason}. "
        "Finish the remaining open work before declaring completion. Do not stop at a status update.]\n\n"
        "Open work:\n"
        f"{todo_lines}"
    )


def _resolve_gap_check(
    *,
    plan: Any,
    result: dict[str, Any],
    evidence: Any,
    caller_role: str | None = None,
) -> dict[str, Any] | None:
    if not plan:
        return None
    gap_result = analyze_gap(
        plan=plan,
        result=result,
        evidence=evidence if evidence is not None else result,
        caller_role=caller_role,
    )
    return gap_result.to_dict()


def _append_gap_to_snapshot(snapshot: dict[str, Any], gap_result: dict[str, Any] | None) -> None:
    if isinstance(gap_result, dict):
        snapshot["gapCheck"] = gap_result


def apply_bounded_continuation_engine(
    child: Any,
    initial_result: dict[str, Any],
    *,
    runtime_mode: Optional[str],
    max_resumes: Optional[int] = None,
    iteration_cap: Optional[int] = None,
    on_snapshot: Optional[Callable[[dict[str, Any], int], None]] = None,
    gap_check_plan: Any = None,
    gap_check_evidence: Any = None,
    gap_check_role: str | None = None,
) -> dict[str, Any]:
    result = dict(initial_result or {})
    snapshot = build_continuation_snapshot(child, result)
    snapshots = [snapshot]
    normalized_mode = _normalize_runtime_mode(runtime_mode) or "default"
    effective_iteration_cap = resolve_iteration_cap(runtime_mode, iteration_cap)
    effective_max_resumes = resolve_max_resumes(
        runtime_mode=runtime_mode,
        max_resumes=max_resumes,
        iteration_cap=effective_iteration_cap,
    )
    gap_result = _resolve_gap_check(
        plan=gap_check_plan,
        result=result,
        evidence=gap_check_evidence,
        caller_role=gap_check_role,
    )
    _append_gap_to_snapshot(snapshot, gap_result)
    if callable(on_snapshot):
        on_snapshot(snapshot, 0)

    if response_contains_promise_done(snapshot.get("responsePreview")):
        logger.info(
            "Continuation engine exiting on promise DONE tag (%s mode, initial response)",
            normalized_mode,
        )

    resume_count = 0
    while (
        resume_count < effective_max_resumes
        and resume_count < effective_iteration_cap
        and not should_skip_next_iteration(gap_result or {})
        and should_use_continuation_engine(runtime_mode, snapshot)
    ):
        resume_count += 1
        if isinstance(gap_result, dict) and gap_result.get("next_prompt"):
            continuation_message = str(gap_result.get("next_prompt"))
        else:
            continuation_message = build_runtime_resume_message(
                snapshot,
                runtime_mode=runtime_mode,
                attempt=resume_count,
                max_attempts=effective_max_resumes,
            )
        next_result = child.run_conversation(user_message=continuation_message)
        if isinstance(next_result, dict):
            result = next_result
        else:
            result = {"final_response": str(next_result)}
        snapshot = build_continuation_snapshot(child, result)
        gap_result = _resolve_gap_check(
            plan=gap_check_plan,
            result=result,
            evidence=gap_check_evidence if gap_check_evidence is not None else result,
            caller_role=gap_check_role,
        )
        _append_gap_to_snapshot(snapshot, gap_result)
        snapshots.append(snapshot)
        if callable(on_snapshot):
            on_snapshot(snapshot, resume_count)
        if response_contains_promise_done(snapshot.get("responsePreview")):
            logger.info(
                "Continuation engine exiting on promise DONE tag (%s mode, attempt %s)",
                normalized_mode,
                resume_count,
            )

    if resume_count >= effective_iteration_cap and should_use_continuation_engine(runtime_mode, snapshot):
        logger.info(
            "Continuation engine hit-max iteration cap (%s mode, cap=%s)",
            normalized_mode,
            effective_iteration_cap,
        )

    exhausted = (not should_skip_next_iteration(gap_result or {})) and should_use_continuation_engine(runtime_mode, snapshot)
    continuation_state = {
        "mode": normalized_mode,
        "resume_count": resume_count,
        "attempt_count": len(snapshots),
        "snapshot_count": len(snapshots),
        "iteration_cap": effective_iteration_cap,
        "max_resumes": effective_max_resumes,
        "done": response_contains_promise_done(snapshot.get("responsePreview")),
        "gap_check": gap_result,
        "gap_complete": should_skip_next_iteration(gap_result or {}),
        "exhausted": exhausted,
        "stop_requested": bool(snapshot.get("stopRequested")),
        "session_id": str(snapshot.get("sessionId") or getattr(child, "session_id", "") or ""),
    }
    result["continuation_state"] = continuation_state
    for item in snapshots:
        item["continuation_state"] = continuation_state

    return {
        "result": result,
        "snapshot": snapshot,
        "resume_count": resume_count,
        "attempt_count": len(snapshots),
        "snapshots": snapshots,
        "exhausted": exhausted,
        "iteration_cap": effective_iteration_cap,
        "max_resumes": effective_max_resumes,
        "mode": normalized_mode,
        "gap_check": gap_result,
        "continuation_state": continuation_state,
    }
