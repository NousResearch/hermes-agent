"""Subagent control panel — live tracking overlay for delegate_task children."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SubagentRecord:
    index: int                    # 0-based task_index
    goal: str                     # full goal string
    start_time: float             # time.monotonic() at spawn
    session_id: str = ""

    # Live-updated by progress callback
    status: str = "running"       # running|completed|failed|interrupted|error
    last_tool: str = ""
    last_tool_preview: str = ""
    tool_count: int = 0

    # Filled on completion
    duration_seconds: float = 0.0
    api_calls: int = 0
    exit_reason: str = ""
    error: Optional[str] = None

    child_ref: Any = field(default=None, repr=False)  # AIAgent; None after done

    @property
    def elapsed(self) -> float:
        if self.status == "running":
            return time.monotonic() - self.start_time
        return self.duration_seconds


STATUS_ICONS = {
    "running":     "●",
    "completed":   "✓",
    "failed":      "✗",
    "error":       "✗",
    "interrupted": "⚡",
}

STATUS_STYLES = {
    "running":     "class:subagent-running",
    "completed":   "class:subagent-done",
    "failed":      "class:subagent-error",
    "error":       "class:subagent-error",
    "interrupted": "class:subagent-warn",
}


def _fmt_elapsed(r: SubagentRecord) -> str:
    secs = int(r.elapsed)
    m, s = divmod(secs, 60)
    suffix = "" if r.status == "running" else " done"
    return f"{m}:{s:02d}{suffix}"


def _tool_emoji(tool_name: str) -> str:
    t = tool_name.lower()
    if "web" in t or "search" in t or "browser" in t:
        return "🌐"
    if "file" in t or "read" in t or "write" in t:
        return "📁"
    if "terminal" in t or "bash" in t or "shell" in t:
        return "💻"
    if "memory" in t:
        return "🧠"
    if "skill" in t:
        return "📚"
    return "🔧"


def render_panel(
    records: list,
    cursor: int,
    width: int,
) -> list:
    """Return a prompt_toolkit formatted_text fragment list for the full panel box."""
    W = min(width - 4, 80)
    n_running = sum(1 for r in records if r.status == "running")
    title = f" Subagents ({n_running} running) "
    fill = W - len(title) - 14  # '╭─' + ' Ctrl+X ─╮'
    frags = []

    def line(text: str, style: str = "") -> None:
        frags.append((style, text + "\n"))

    # Header
    line(f"╭─{title}{'─' * max(fill, 0)} Ctrl+X ─╮", "class:subagent-border")

    if not records:
        line(f"│  (no subagents){'':>{W - 17}}│", "class:subagent-border")
    else:
        for i, r in enumerate(records):
            icon  = STATUS_ICONS.get(r.status, "?")
            istyle = STATUS_STYLES.get(r.status, "")
            elapsed = _fmt_elapsed(r)
            goal_w = W - 12  # icon+index+elapsed+padding
            goal  = r.goal[:goal_w - 1] if len(r.goal) > goal_w else r.goal
            row   = f"│ {icon} [{r.index+1}] {goal:<{goal_w}} {elapsed:>7} │"
            if i == cursor:
                frags.append(("class:subagent-selected", row + "\n"))
            else:
                frags.append(("", "│ "))
                frags.append((istyle, f"{icon} [{r.index+1}]"))
                frags.append(("", f" {goal:<{goal_w}} {elapsed:>7} │\n"))
            # Tool sub-row (running only)
            if r.status == "running" and r.last_tool:
                emoji = _tool_emoji(r.last_tool)
                preview = r.last_tool_preview[:W - 18]
                sub = f"│   └─ {emoji} {r.last_tool:<14} {preview:<{W-18}} │"
                line(sub, "class:subagent-sub")

    # Footer
    line(f"╰{'─' * (W - 2)} ↑↓ K=interrupt ─╯", "class:subagent-border")
    return frags
