"""Formatter for periodic long-running gateway heartbeat bubbles."""

from __future__ import annotations

from typing import Any, Optional

_MAX_LINE = 180
_MAX_LINES = 9
_MAX_TOTAL = 900
_STATUS_LABELS = {
    "in_progress": "now",
    "pending": "next",
    "completed": "done",
    "cancelled": "cancelled",
}


def _oneline(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(value: Any, max_len: int = _MAX_LINE) -> str:
    text = _oneline(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _format_elapsed(seconds: Any) -> str:
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        total = 0
    if total < 0:
        total = 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _code_value(value: Any, max_len: Optional[int] = None) -> str:
    text = _oneline(value).replace("`", "ˋ")
    if max_len is not None:
        text = _truncate(text, max_len)
    return f"`{text}`"


def _code_line(prefix: str, value: Any, suffix: str = "", max_len: int = _MAX_LINE) -> str:
    inner_max = max(0, max_len - len(_oneline(prefix)) - len(_oneline(suffix)) - 2)
    return f"{prefix}{_code_value(value, inner_max)}{suffix}"


def _duration_suffix(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return ""
    try:
        if float(seconds) < 0:
            return ""
    except Exception:
        return ""
    return f" · {_format_elapsed(float(seconds))}"


def _action_state(item: dict[str, Any]) -> str:
    return str(item.get("state") or ("failed" if item.get("is_error") else "done"))


def _action_duration_suffix(item: dict[str, Any]) -> str:
    state = _action_state(item)
    duration = _format_elapsed(item.get("duration") or 0)
    if state == "running":
        return f" · {duration}"
    return f" · took {duration}"



def format_long_running_heartbeat(
    elapsed_seconds: float,
    activity: Optional[dict[str, Any]] = None,
    *,
    want_iteration_detail: bool = False,
    now: Optional[float] = None,
) -> str:
    """Build the edited-in-place "Working" heartbeat text.

    Keeps the first line backward-compatible and appends bounded detail lines
    when the agent exposes them.
    """
    activity = activity if isinstance(activity, dict) else {}

    lines = [f"⏳ Working — {_format_elapsed(elapsed_seconds)}"]

    if want_iteration_detail:
        api = activity.get("api_call_count")
        max_iter = activity.get("max_iterations")
        if api is not None and max_iter is not None:
            lines.append(_code_line("• iteration: ", f"{api}/{max_iter}"))

    todo = activity.get("current_todo")
    if isinstance(todo, dict) and todo.get("content"):
        status = _STATUS_LABELS.get(str(todo.get("status") or ""), str(todo.get("status") or "todo"))
        todo_label = "• todo:" if status == "now" else f"• todo {status}:"
        lines.append(_code_line(
            f"{todo_label} ",
            todo.get("content"),
            _duration_suffix(todo.get("elapsed_seconds")),
        ))

    current_tool = _oneline(activity.get("current_tool"))
    current_elapsed = activity.get("current_tool_elapsed")
    if current_tool:
        lines.append(_code_line("• tool: ", current_tool, _duration_suffix(current_elapsed)))
    else:
        desc = _oneline(activity.get("last_activity_desc"))
        if desc:
            lines.append(_code_line("• status: ", desc))

    history = activity.get("recent_tool_activity")
    if isinstance(history, list) and history:
        item = next((item for item in reversed(history) if isinstance(item, dict)), None)
        if item is not None:
            lines.append(_code_line(
                f"• doing: {_action_state(item)} · ",
                item.get("label") or item.get("name") or "tool",
                _action_duration_suffix(item),
            ))

    # Bound vertical and total size; this bubble edits every minute.
    lines = lines[:_MAX_LINES]
    text = "\n".join(lines)
    if len(text) > _MAX_TOTAL:
        text = text[: _MAX_TOTAL - 1].rstrip() + "…"
    return text
