"""Human-readable todo rendering for gateway progress bubbles."""

from __future__ import annotations

import json
from typing import Any, Optional


_STATUS_ICON = {
    "completed": "✅",
    "in_progress": "🔄",
    "pending": "⏳",
    "cancelled": "✗",
}

_STATUS_LABEL = {
    "completed": "completed",
    "in_progress": "in progress",
    "pending": "pending",
    "cancelled": "cancelled",
}


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _format_elapsed(seconds: Any) -> Optional[str]:
    """Compact human duration for a per-item span. None when unmeasured.

    Mirrors the gateway's tool-progress duration buckets (ms / s / m / h) so a
    todo item's time reads the same as any other tool's completion time.
    """
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        return None
    value = float(seconds)
    if value < 0:
        value = 0.0
    if value < 0.1:
        return f"{int(round(value * 1000))}ms"
    if value < 10:
        return f"{value:.1f}s"
    total = int(round(value))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _coerce_todo_items(payload: Any) -> Optional[list]:
    """Extract the todo item list from a tool result (JSON string or dict)."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return None
    if isinstance(payload, dict):
        items = payload.get("todos")
        return items if isinstance(items, list) else None
    return None


def extract_todo_items(payload: Any) -> Optional[list]:
    """Public alias for the tool-result item coercion (JSON string or dict).

    Shared by the renderer and the persistent-card finished-check so both read
    the todo list the same way.
    """
    return _coerce_todo_items(payload)


def _plan_wall_clock_seconds(items: Any, *, now: Optional[float] = None) -> Optional[float]:
    """Whole-plan wall-clock span: earliest start to latest end (or now).

    Returns the real elapsed time across the plan, NOT the sum of per-item
    spans (which double-counts parallel/overlapping work). Reads the raw
    ``started_at`` / ``ended_at`` epoch stamps the todo result payload carries
    (tools/todo_tool.py read_with_timing). An item that is in_progress has a
    ``started_at`` but no ``ended_at`` -> counted up to ``now``. An item that
    never started has no ``started_at`` -> excluded from the earliest-start.
    Returns None when no item carries a ``started_at`` (e.g. the model's start
    args, which have no stamps) so the caller appends no header suffix.
    """
    if not isinstance(items, list):
        return None
    if now is None:
        import time as _time

        now = _time.time()

    starts: list = []
    ends: list = []
    for item in items:
        if not isinstance(item, dict):
            continue
        started = item.get("started_at")
        if not isinstance(started, (int, float)) or isinstance(started, bool):
            continue
        starts.append(float(started))
        ended = item.get("ended_at")
        if isinstance(ended, (int, float)) and not isinstance(ended, bool):
            ends.append(float(ended))
        else:
            # Still running: count its span up to now.
            ends.append(float(now))

    # Empty-sequence guard BEFORE min()/max() (no started_at anywhere).
    if not starts:
        return None
    span = max(ends) - min(starts)
    return span if span >= 0 else 0.0


def format_todo_progress(
    args: Optional[dict],
    *,
    result: Any = None,
    max_items: int = 0,
    max_chars: int = 0,
    content_limit: int = 100,
) -> Optional[str]:
    """Render ``todo`` tool args as a compact plan card.

    Gateway tool-progress events are emitted at tool start, so by default this
    renders the input args, not the completed result.  Initial planning calls
    pass the full list. Merge calls often pass only changed items, so label them
    as updates.

    When ``result`` is provided (the tool's completion payload), per-item
    wall-clock durations are read from it and shown as ``(2m 14s)`` suffixes.
    The model's args never carry timing, so durations only appear on the
    completion re-render. If ``result`` is provided but carries no usable item
    list, returns None (the caller should keep the existing start card) rather
    than falling back to args or the "Reading task list" sentinel.

    Row caps. Two independent caps decide how many task rows render; both are
    ``0`` (off) by default, so the default renders EVERY task with no cap and no
    "... N more" footer:
      * ``max_items`` — a hard row count cap (used by tests / explicit callers).
      * ``max_chars`` — a length budget so the rendered card fits ONE adapter
        message. The gateway passes the platform's message limit here. When the
        full card would exceed it, only the overflow TAIL collapses into a
        "... N more" footer; everything that fits still renders. This keeps the
        persistent plan card a single editable message even on adapters whose
        edit path truncates (Discord) or sends unsplit (Slack/Mattermost/Feishu)
        instead of splitting like Telegram.
    When both caps are set the stricter one wins.
    """
    if not isinstance(args, dict):
        return None

    if result is not None:
        # Completion re-render path: the result is authoritative. No usable
        # items → no card (caller leaves the start card untouched).
        result_items = _coerce_todo_items(result)
        if result_items is None:
            return None
        todos = result_items
    else:
        # Tool-start path: render from args (no timing available yet).
        todos = args.get("todos")
        if todos is None:
            return "📋 Todo\nReading task list"
    if not isinstance(todos, list):
        return None

    merge = bool(args.get("merge", False))
    title = "📋 Plan update" if merge else "📋 Plan"
    count = len(todos)
    noun = "task" if count == 1 else "tasks"
    # Whole-plan wall-clock (DECISION B): earliest start -> latest end/now.
    # Only renders when the items carry started_at/ended_at stamps (the result
    # payload), so the model's start args (no stamps) and existing test
    # fixtures (elapsed_seconds only) produce no suffix -> back-compat safe.
    wall = _format_elapsed(_plan_wall_clock_seconds(todos))
    header = f"{title} ({count} {noun})"
    if wall:
        header = f"{header} · {wall}"
    lines = [header]

    if not todos:
        lines.append("No tasks")
        return "\n".join(lines)

    # Build every renderable row first (skipping non-dict items, numbered by
    # the running shown counter), then decide how many fit under the caps.
    rows: list = []
    shown = 0
    for item in todos:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "pending")
        label = _STATUS_LABEL.get(status, status.replace("_", " "))
        icon = _STATUS_ICON.get(status, "•")
        content = _one_line(item.get("content") or item.get("id") or "<untitled task>")
        content = _truncate(content, content_limit)
        elapsed = _format_elapsed(item.get("elapsed_seconds"))
        duration = f" ({elapsed})" if elapsed else ""
        shown += 1
        rows.append(f"{shown}. {icon} {label}{duration} - {content}")

    # Hard row-count cap (explicit callers / tests).
    if max_items > 0 and len(rows) > max_items:
        rows = rows[:max_items]

    def _assemble(keep: int) -> list:
        body = rows[:keep]
        dropped = count - keep  # count = len(todos); non-dict skips count too
        out = [header] + body
        if dropped > 0:
            out.append(f"... {dropped} more")
        return out

    if max_chars > 0:
        # Length budget: keep the longest prefix of rows whose card (header +
        # kept rows + "... N more" footer) fits one adapter message. Only the
        # overflow tail collapses; everything that fits still renders. Reserve a
        # small margin: platforms measure UTF-16 code units and MarkdownV2
        # escaping inflates the payload past a raw char count, so leave room
        # (mirrors the gateway's _PROGRESS_TEXT_LIMIT margin).
        budget = max_chars - 64 if max_chars > 128 else max_chars
        keep = len(rows)
        lines = _assemble(keep)
        while keep > 0 and len("\n".join(lines)) > budget:
            keep -= 1
            lines = _assemble(keep)
    else:
        lines = _assemble(len(rows))

    text = "\n".join(lines)
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text)
    except Exception:
        pass
    return text
