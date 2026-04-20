from __future__ import annotations

from typing import Any, Callable, Dict, Optional


DEFAULT_MAX_RUNTIME_RESUMES = 2
_ACTIVE_TODO_STATUSES = {"pending", "in_progress"}
_STOP_OUTCOME_STATUSES = {"completed", "stopped", "cancelled", "canceled"}


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
        outcome = "completed"
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


def apply_bounded_continuation_engine(
    child: Any,
    initial_result: dict[str, Any],
    *,
    runtime_mode: Optional[str],
    max_resumes: int = DEFAULT_MAX_RUNTIME_RESUMES,
    on_snapshot: Optional[Callable[[dict[str, Any], int], None]] = None,
) -> dict[str, Any]:
    result = dict(initial_result or {})
    snapshot = build_continuation_snapshot(child, result)
    snapshots = [snapshot]
    if callable(on_snapshot):
        on_snapshot(snapshot, 0)

    resume_count = 0
    while resume_count < max(int(max_resumes or 0), 0) and should_use_continuation_engine(runtime_mode, snapshot):
        resume_count += 1
        continuation_message = build_runtime_resume_message(
            snapshot,
            runtime_mode=runtime_mode,
            attempt=resume_count,
            max_attempts=max(int(max_resumes or 0), 0),
        )
        next_result = child.run_conversation(user_message=continuation_message)
        if isinstance(next_result, dict):
            result = next_result
        else:
            result = {"final_response": str(next_result)}
        snapshot = build_continuation_snapshot(child, result)
        snapshots.append(snapshot)
        if callable(on_snapshot):
            on_snapshot(snapshot, resume_count)

    return {
        "result": result,
        "snapshot": snapshot,
        "resume_count": resume_count,
        "attempt_count": len(snapshots),
        "snapshots": snapshots,
        "exhausted": should_use_continuation_engine(runtime_mode, snapshot),
        "mode": str(runtime_mode or "").strip().lower() or None,
    }
