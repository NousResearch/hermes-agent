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


# All icons exactly 1 display column wide (avoid emoji that render as 2-wide)
STATUS_ICONS = {
    "running":     "●",
    "completed":   "✓",
    "failed":      "✗",
    "error":       "✗",
    "interrupted": "~",
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
    if r.status == "running":
        return f"{m}:{s:02d}"
    return f"{m}:{s:02d} done"


def _tool_emoji(tool_name: str) -> str:
    t = tool_name.lower()
    if "web" in t or "search" in t or "browser" in t:
        return ">"   # avoid 2-wide emoji in fixed-width box
    if "file" in t or "read" in t or "write" in t:
        return "f"
    if "terminal" in t or "bash" in t or "shell" in t:
        return "$"
    if "memory" in t:
        return "m"
    if "skill" in t:
        return "s"
    return "*"


def render_panel(
    records: list,
    cursor: int,
    width: int,
) -> list:
    """Return prompt_toolkit formatted_text fragments for the panel box.

    All measurements use display-column counts, assuming every character is
    exactly 1 column wide (no emoji, no CJK).  W is the total box width
    including the │ border characters on both sides.
    """
    W = min(width - 4, 80)

    # Fixed border strings — measure by len() since all chars are 1-wide
    HDR_PREFIX = "╭─"          # 2
    HDR_SUFFIX = " Ctrl+X ─╮"  # 10
    FTR_PREFIX = "╰"           # 1
    FTR_SUFFIX = " ↑↓ K=interrupt ─╯"  # 18

    n_running = sum(1 for r in records if r.status == "running")
    title = f" Subagents ({n_running} running) "

    hdr_dashes = max(0, W - len(HDR_PREFIX) - len(title) - len(HDR_SUFFIX))
    ftr_dashes = max(0, W - len(FTR_PREFIX) - len(FTR_SUFFIX))

    # Row layout (all 1-wide):
    # │ I [N] <goal_w> <ELAPSED_W> │
    # 1+1+1+1+1+1+1+1 + goal_w + 1+ELAPSED_W+1+1 = goal_w + ELAPSED_W + 11 = W
    ELAPSED_W = 9   # "0:00 done" = 9 chars; running "0:00" left-padded to 9
    goal_w = max(10, W - ELAPSED_W - 11)

    frags: list = []

    def line(text: str, style: str = "") -> None:
        frags.append((style, text + "\n"))

    line(f"{HDR_PREFIX}{title}{'─' * hdr_dashes}{HDR_SUFFIX}",
         "class:subagent-border")

    if not records:
        content = "(no subagents)"
        pad = max(0, W - 4 - len(content))
        line(f"│  {content}{' ' * pad} │", "class:subagent-border")
    else:
        for i, r in enumerate(records):
            icon   = STATUS_ICONS.get(r.status, "?")
            istyle = STATUS_STYLES.get(r.status, "")
            elapsed = _fmt_elapsed(r).ljust(ELAPSED_W)
            goal    = r.goal[:goal_w].ljust(goal_w)
            idx     = str(r.index + 1)
            row = f"│ {icon} [{idx}] {goal} {elapsed} │"
            if i == cursor:
                frags.append(("class:subagent-selected", row + "\n"))
            else:
                frags.append(("", "│ "))
                frags.append((istyle, f"{icon} [{idx}]"))
                frags.append(("", f" {goal} {elapsed} │\n"))

            # Tool sub-row (running agents only)
            if r.status == "running" and r.last_tool:
                sym     = _tool_emoji(r.last_tool)
                tool_s  = r.last_tool[:12].ljust(12)
                prev_w  = max(0, W - 23)
                preview = r.last_tool_preview[:prev_w].ljust(prev_w)
                line(f"│  └─ {sym} {tool_s} {preview} │", "class:subagent-sub")

    line(f"{FTR_PREFIX}{'─' * ftr_dashes}{FTR_SUFFIX}", "class:subagent-border")
    return frags
