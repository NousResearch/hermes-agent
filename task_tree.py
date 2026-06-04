from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


@dataclass(frozen=True)
class TaskButton:
    label: str
    callback_data: str


@dataclass(frozen=True)
class TaskView:
    text: str
    buttons: list[list[TaskButton]]


@dataclass(frozen=True)
class ParentTask:
    index: int
    bucket: str
    data: dict[str, Any]


@dataclass(frozen=True)
class ParentMatch:
    index: int
    bucket: str
    task: dict[str, Any]


@dataclass(frozen=True)
class TaskTreeState:
    source_path: Path
    updated_at: str | None
    parents: list[ParentTask]


_STATUS_ICONS = {
    "active": "🟢",
    "processing": "🟣",
    "pending": "⏳",
    "resolved": "✅",
    "cancelled": "⚪️",
}

_AUTH_BEARER_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(\bauthorization\b\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
)

_KEY_VALUE_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)([\"']?\b(?:[A-Za-z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|COOKIE)|api[_-]?key|secret|password|passwd|token|cookie|chat_id)\b[\"']?\s*[:=]\s*)([\"']?)[^\s,'\"}]{6,}([\"']?)"
    ),
)

_BARE_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{12,}|xox[baprs]-[A-Za-z0-9-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|AIza[A-Za-z0-9_-]{20,})\b"),
)


def get_task_tree_path() -> Path:
    return get_hermes_home() / "hermes-daily-state" / "todo-state.json"


def load_task_tree_state(path: str | Path | None = None) -> TaskTreeState:
    source = Path(path) if path is not None else get_task_tree_path()
    if not source.exists():
        return TaskTreeState(source_path=source, updated_at=None, parents=[])

    raw = json.loads(source.read_text(encoding="utf-8"))
    parents: list[ParentTask] = []
    for bucket in ("active", "pending", "resolved_recent"):
        for item in raw.get(bucket) or []:
            if isinstance(item, dict):
                parents.append(ParentTask(index=len(parents), bucket=bucket, data=item))
    return TaskTreeState(source_path=source, updated_at=raw.get("updated_at"), parents=parents)


def redact_sensitive(text: Any) -> str:
    value = str(text or "")
    for pattern in _AUTH_BEARER_SECRET_PATTERNS:
        value = pattern.sub(lambda m: f"{m.group(1)}[REDACTED]" if m.lastindex else "Bearer [REDACTED]", value)
    for pattern in _KEY_VALUE_SECRET_PATTERNS:
        value = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]{m.group(3)}", value)
    for pattern in _BARE_SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value


def _short(text: Any, limit: int = 86) -> str:
    value = " ".join(redact_sensitive(text).split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _join_bounded(lines: list[str], *, max_chars: int = 3600) -> str:
    rendered: list[str] = []
    used = 0
    truncated = False
    reserve = 96
    for line in lines:
        clean = redact_sensitive(line)
        extra = len(clean) + (1 if rendered else 0)
        if used + extra > max_chars - reserve:
            truncated = True
            break
        rendered.append(clean)
        used += extra
    if truncated:
        rendered.append(f"… truncated to keep Telegram message under {max_chars} chars; use buttons for details.")
    return "\n".join(rendered)


def _status_icon(status: Any) -> str:
    return _STATUS_ICONS.get(str(status or "").lower(), "•")


def _counts_from_items(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _counts_text(counts: dict[str, Any] | None) -> str:
    if not counts:
        return "none"
    preferred = ["processing", "active", "pending", "resolved", "cancelled", "unknown"]
    parts: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        val = counts.get(key)
        if val:
            parts.append(f"{key} {val}")
            seen.add(key)
    for key in sorted(k for k in counts if k not in seen):
        val = counts.get(key)
        if val:
            parts.append(f"{key} {val}")
    return " / ".join(parts) if parts else "none"


def _subtask_counts(task: dict[str, Any]) -> dict[str, int]:
    explicit = task.get("subtask_counts")
    if isinstance(explicit, dict) and explicit:
        return {str(k): int(v) for k, v in explicit.items() if isinstance(v, int)}
    return _counts_from_items([s for s in task.get("subtasks") or [] if isinstance(s, dict)])


def _step_counts_for_subtask(subtask: dict[str, Any]) -> dict[str, int]:
    explicit = subtask.get("step_counts")
    if isinstance(explicit, dict) and explicit:
        return {str(k): int(v) for k, v in explicit.items() if isinstance(v, int)}
    return _counts_from_items([s for s in subtask.get("acceptance_steps") or [] if isinstance(s, dict)])


def _aggregate_step_counts(task: dict[str, Any]) -> dict[str, int]:
    total: dict[str, int] = {}
    for subtask in task.get("subtasks") or []:
        if not isinstance(subtask, dict):
            continue
        for key, val in _step_counts_for_subtask(subtask).items():
            total[key] = total.get(key, 0) + val
    return total


def _field_lines(item: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for field in fields:
        value = item.get(field)
        if value in (None, "", [], {}):
            continue
        lines.append(f"{field}: {_short(value, 160)}")
    return lines


def build_task_index_view(*, path: str | Path | None = None, page: int = 0, per_page: int = 8) -> TaskView:
    state = load_task_tree_state(path)
    page = max(0, int(page or 0))
    total = len(state.parents)
    start = page * per_page
    end = start + per_page
    shown = state.parents[start:end]

    lines = ["📋 Tasks"]
    if state.updated_at:
        lines.append(f"updated: {state.updated_at}")
    lines.append(f"parents: {total}")
    lines.append("")

    if not shown:
        lines.append("(no tasks)")
    for parent in shown:
        task = parent.data
        subtasks = _subtask_counts(task)
        steps = _aggregate_step_counts(task)
        title = _short(task.get("title"), 72)
        status = task.get("status") or parent.bucket
        lines.append(f"{parent.index + 1}. {_status_icon(status)} [{parent.bucket}] {title}")
        meta: list[str] = []
        execution_type = task.get("execution_type")
        if execution_type:
            meta.append(f"type: {execution_type}")
        if subtasks:
            meta.append(f"subtasks: {_counts_text(subtasks)}")
        if steps:
            meta.append(f"steps: {_counts_text(steps)}")
        if meta:
            lines.append("   " + " | ".join(meta))

    buttons: list[list[TaskButton]] = []
    for parent in shown:
        task = parent.data
        status = task.get("status") or parent.bucket
        buttons.append([
            TaskButton(
                f"{parent.index + 1}. {_status_icon(status)} {_short(task.get('title'), 30)}",
                f"task:p:{parent.index}",
            )
        ])
    nav: list[TaskButton] = []
    if page > 0:
        nav.append(TaskButton("‹ Prev", f"task:list:{page - 1}"))
    if end < total:
        nav.append(TaskButton("Next ›", f"task:list:{page + 1}"))
    if nav:
        buttons.append(nav)
    buttons.append([TaskButton("Refresh", f"task:list:{page}")])
    return TaskView(_join_bounded(lines), buttons)


def find_parent_task(query: str, *, path: str | Path | None = None) -> ParentMatch | None:
    state = load_task_tree_state(path)
    q = str(query or "").strip()
    if not q:
        return None

    m = re.fullmatch(r"(?i)task\s*(\d+)", q) or re.fullmatch(r"#?(\d+)", q)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(state.parents):
            parent = state.parents[idx]
            return ParentMatch(parent.index, parent.bucket, parent.data)

    q_norm = q.casefold()
    exact: list[ParentTask] = []
    partial: list[ParentTask] = []
    for parent in state.parents:
        task = parent.data
        hay = [
            str(task.get("id") or ""),
            str(task.get("fingerprint") or ""),
            str(task.get("title") or ""),
            str(task.get("section") or ""),
        ]
        folded = [h.casefold() for h in hay]
        if q_norm in folded:
            exact.append(parent)
        elif any(q_norm in h for h in folded):
            partial.append(parent)
    chosen = exact[0] if exact else (partial[0] if partial else None)
    if not chosen:
        return None
    return ParentMatch(chosen.index, chosen.bucket, chosen.data)


def _parent_by_index(index: int, state: TaskTreeState) -> ParentTask | None:
    if 0 <= index < len(state.parents):
        return state.parents[index]
    return None


def build_task_tree_view(query: str, *, path: str | Path | None = None) -> TaskView:
    match = find_parent_task(query, path=path)
    if not match:
        lines = [f"No matching parent task: {_short(query, 240)}", "Use /task to list parent tasks."]
        return TaskView(_join_bounded(lines), [[TaskButton("Task list", "task:list:0")]])
    return build_parent_view(match.index, path=path)


def build_parent_view(parent_index: int, *, path: str | Path | None = None) -> TaskView:
    state = load_task_tree_state(path)
    parent = _parent_by_index(parent_index, state)
    if not parent:
        return TaskView("Task not found. It may have moved after the list refreshed.", [[TaskButton("Task list", "task:list:0")]])

    task = parent.data
    title = _short(task.get("title"), 120)
    status = task.get("status") or parent.bucket
    lines = [f"📌 Task {parent.index + 1}: {title}", f"status: {status}", f"bucket: {parent.bucket}"]
    lines.extend(_field_lines(task, ("id", "section", "execution_type", "cron_job_id", "cron_policy", "reasoning_level")))
    subtasks = [s for s in task.get("subtasks") or [] if isinstance(s, dict)]
    sub_counts = _subtask_counts(task)
    step_counts = _aggregate_step_counts(task)
    if sub_counts:
        lines.append(f"subtasks: {_counts_text(sub_counts)}")
    if step_counts:
        lines.append(f"steps: {_counts_text(step_counts)}")
    acceptance = task.get("acceptance")
    if acceptance:
        lines.extend(["", "acceptance:", _short(acceptance, 260)])
    lines.append("")
    lines.append("Task tree:")
    if not subtasks:
        lines.append("└─ (no subtasks)")
    for si, subtask in enumerate(subtasks):
        st = subtask.get("status") or "pending"
        lines.append(f"├─ {si + 1}. {_status_icon(st)} {_short(subtask.get('title'), 92)}")
        if subtask.get("reasoning_level"):
            lines.append(f"│  reasoning: {redact_sensitive(subtask.get('reasoning_level'))}")
        steps = [s for s in subtask.get("acceptance_steps") or [] if isinstance(s, dict)]
        for ki, step in enumerate(steps[:6]):
            ss = step.get("status") or "pending"
            lines.append(f"│  ├─ {ki + 1}. {_status_icon(ss)} {_short(step.get('title'), 82)}")
        if len(steps) > 6:
            lines.append(f"│  └─ … +{len(steps) - 6} more steps")

    buttons: list[list[TaskButton]] = []
    for si, subtask in enumerate(subtasks[:12]):
        st = subtask.get("status") or "pending"
        buttons.append([TaskButton(f"{si + 1}. {_status_icon(st)} {_short(subtask.get('title'), 34)}", f"task:s:{parent.index}:{si}")])
    buttons.append([TaskButton("‹ List", "task:list:0"), TaskButton("Refresh", f"task:p:{parent.index}")])
    return TaskView(_join_bounded(lines), buttons)


def build_subtask_view(parent_index: int, subtask_index: int, *, path: str | Path | None = None) -> TaskView:
    state = load_task_tree_state(path)
    parent = _parent_by_index(parent_index, state)
    if not parent:
        return TaskView("Parent task not found.", [[TaskButton("Task list", "task:list:0")]])
    subtasks = [s for s in parent.data.get("subtasks") or [] if isinstance(s, dict)]
    if not (0 <= subtask_index < len(subtasks)):
        return TaskView("Subtask not found.", [[TaskButton("‹ Parent", f"task:p:{parent.index}")]])
    subtask = subtasks[subtask_index]
    status = subtask.get("status") or "pending"
    lines = [
        f"🔎 Subtask {parent.index + 1}.{subtask_index + 1}: {_short(subtask.get('title'), 120)}",
        f"status: {status}",
    ]
    lines.extend(_field_lines(subtask, ("id", "fingerprint", "reasoning_level", "reasoning_reason", "cron_job_id", "artifact", "output_file", "workdir", "repo_path", "resolution_evidence", "evidence", "blocker", "acceptance")))
    steps = [s for s in subtask.get("acceptance_steps") or [] if isinstance(s, dict)]
    if steps:
        lines.append("")
        lines.append(f"steps: {_counts_text(_step_counts_for_subtask(subtask))}")
        for ki, step in enumerate(steps):
            ss = step.get("status") or "pending"
            lines.append(f"{ki + 1}. {_status_icon(ss)} {_short(step.get('title'), 100)}")

    buttons: list[list[TaskButton]] = []
    for ki, step in enumerate(steps[:12]):
        ss = step.get("status") or "pending"
        buttons.append([TaskButton(f"{ki + 1}. {_status_icon(ss)} {_short(step.get('title'), 34)}", f"task:step:{parent.index}:{subtask_index}:{ki}")])
    buttons.append([TaskButton("‹ Parent", f"task:p:{parent.index}"), TaskButton("Refresh", f"task:s:{parent.index}:{subtask_index}")])
    return TaskView(_join_bounded(lines), buttons)


def build_step_view(parent_index: int, subtask_index: int, step_index: int, *, path: str | Path | None = None) -> TaskView:
    state = load_task_tree_state(path)
    parent = _parent_by_index(parent_index, state)
    if not parent:
        return TaskView("Parent task not found.", [[TaskButton("Task list", "task:list:0")]])
    subtasks = [s for s in parent.data.get("subtasks") or [] if isinstance(s, dict)]
    if not (0 <= subtask_index < len(subtasks)):
        return TaskView("Subtask not found.", [[TaskButton("‹ Parent", f"task:p:{parent.index}")]])
    steps = [s for s in subtasks[subtask_index].get("acceptance_steps") or [] if isinstance(s, dict)]
    if not (0 <= step_index < len(steps)):
        return TaskView("Step not found.", [[TaskButton("‹ Subtask", f"task:s:{parent.index}:{subtask_index}")]])
    step = steps[step_index]
    status = step.get("status") or "pending"
    lines = [
        f"🧩 Step {parent.index + 1}.{subtask_index + 1}.{step_index + 1}: {_short(step.get('title'), 120)}",
        f"status: {status}",
    ]
    lines.extend(_field_lines(step, ("id", "evidence", "artifact", "output_file", "commit", "validation_command", "pass_condition", "blocker", "notes")))
    return TaskView(_join_bounded(lines), [[TaskButton("‹ Subtask", f"task:s:{parent.index}:{subtask_index}"), TaskButton("Parent", f"task:p:{parent.index}")]])


def build_task_callback_view(callback_data: str, *, path: str | Path | None = None) -> TaskView:
    parts = str(callback_data or "").split(":")
    if len(parts) < 2 or parts[0] != "task":
        return build_task_index_view(path=path)
    kind = parts[1]
    try:
        if kind == "list":
            page = int(parts[2]) if len(parts) > 2 else 0
            return build_task_index_view(path=path, page=page)
        if kind == "p" and len(parts) >= 3:
            return build_parent_view(int(parts[2]), path=path)
        if kind == "s" and len(parts) >= 4:
            return build_subtask_view(int(parts[2]), int(parts[3]), path=path)
        if kind == "step" and len(parts) >= 5:
            return build_step_view(int(parts[2]), int(parts[3]), int(parts[4]), path=path)
    except ValueError:
        pass
    return TaskView("Invalid task browser action.", [[TaskButton("Task list", "task:list:0")]])


def build_task_command_view(args: str, *, path: str | Path | None = None) -> TaskView:
    query = str(args or "").strip()
    if not query:
        return build_task_index_view(path=path)
    return build_task_tree_view(query, path=path)
