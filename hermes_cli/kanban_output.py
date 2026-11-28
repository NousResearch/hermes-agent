"""Text / ``--json`` output helpers shared by the ``hermes kanban`` CLI modules."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Callable, Iterable, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_workflow

_STATUS_ICONS = {c.key: c.icon for c in kanban_workflow.DEFAULT_WORKFLOW} | {kanban_workflow.ARCHIVED: "—"}

_TASK_DICT_FIELDS = (
    "id", "title", "body", "assignee", "status", "priority", "tenant",
    "workspace_kind", "workspace_path", "branch_name", "project_id",
    "created_by", "created_at", "started_at", "completed_at", "result",
    "skills", "max_runtime_seconds", "max_retries", "model_override", "provider_override",
    "session_id", "workflow_template_id", "current_step_key", "completion_contract", "last_failure_error",
)
_SHOW_RUN_FIELDS = (
    "id", "profile", "step_key", "status", "outcome", "summary", "error",
    "metadata", "worker_pid", "started_at", "ended_at",
)
_RUNS_RUN_FIELDS = (
    "id", "profile", "status", "outcome", "started_at", "ended_at",
    "summary", "error", "metadata", "worker_pid", "step_key",
)
_ATTACHMENT_FIELDS = ("id", "filename", "content_type", "size", "uploaded_by", "stored_path", "created_at")


def _fmt_ts(ts: Optional[int]) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else ""


# Readable ``hermes kanban watch`` rendering: kind -> (icon, label, ANSI codes).
# Icons are single-cell so the columns stay aligned.
# Unknown kinds fall back to a dim "·" with the upper-cased kind.
_WATCH_STYLES: dict[str, tuple[str, str, str]] = {
    "spawned": ("▶", "START", "\033[36m"),
    "completed": ("✓", "DONE", "\033[1;32m"),
    "blocked": ("⏸", "BLOCKED", "\033[1;33m"),
    "unblocked": ("↺", "UNBLOCK", "\033[33m"),
    "review_requested": ("◎", "REVIEW", "\033[1;35m"),
    "review_approved": ("✔", "APPROVED", "\033[1;32m"),
    "changes_requested": ("✎", "CHANGES", "\033[1;35m"),
    "gave_up": ("✗", "GAVE UP", "\033[1;31m"),
    "crashed": ("✗", "CRASHED", "\033[1;31m"),
    "timed_out": ("⧗", "TIMEOUT", "\033[1;31m"),
    "spawn_failed": ("✗", "NOSPAWN", "\033[1;31m"),
    "protocol_violation": ("!", "PROTOCOL", "\033[31m"),
    "created": ("+", "NEW", "\033[2m"),
}
# Kinds hidden from the readable feed unless requested explicitly via --kinds.
WATCH_QUIET_KINDS = frozenset({"heartbeat"})
_WATCH_DETAIL_KEYS = ("summary", "reason", "error", "worker_output")
_WATCH_WIDE_MIN = 60  # below this many cells, drop the assignee column and indent less
_DIM, _BOLD, _RESET = "\033[2m", "\033[1m", "\033[0m"


def _graphemes(text: str):
    """Lazily yield user-perceived characters (emoji ZWJ/VS16 sequences stay whole)."""
    from wcwidth import iter_graphemes

    return iter_graphemes(text)


def _grapheme_cells(g: str) -> int:
    from wcwidth import wcswidth

    return max(wcswidth(g), 0)


def _cells(text: str) -> int:
    """Terminal cells ``text`` occupies (wide CJK/emoji count as two)."""
    return sum(_grapheme_cells(g) for g in _graphemes(text))


def _clip_cells(text: str, budget: int, ellipsis: str = "…") -> str:
    """Clip ``text`` to ``budget`` cells, ending in ``ellipsis`` when cut. Reads at most ~budget graphemes."""
    if budget <= 0:
        return ""
    kept, used, cut_at = [], 0, None
    for g in _graphemes(text):
        w = _grapheme_cells(g)
        if used + w > budget:
            break
        if cut_at is None and used + w > budget - len(ellipsis):
            cut_at = len(kept)  # where the ellipsis would go if the text turns out too long
        kept.append(g)
        used += w
    else:
        return text
    return "".join(kept[:cut_at]).rstrip() + ellipsis


def _wrap_cells(text: str, width: int, max_rows: int) -> tuple[list[str], bool]:
    """Greedy word wrap by cells, stopping after ``max_rows`` rows. Returns ``(rows, truncated)``.

    Words wider than a row are hard-split between graphemes. Work is bounded by the rows
    produced, not by the input length, so a huge unbroken token stays cheap.
    """
    import re

    rows: list[str] = []
    row, used = "", 0

    def push() -> bool:
        nonlocal row, used
        rows.append(row)
        row, used = "", 0
        return len(rows) >= max_rows

    for match in re.finditer(r"\S+", text):
        word = match.group()
        fits = _clip_cells(word, width + 1, ellipsis="") == word and _cells(word) <= width
        if fits:
            w = _cells(word)
            if row and used + 1 + w > width and push():
                return rows, True
            row, used = (f"{row} {word}", used + 1 + w) if row else (word, w)
            continue
        if row and push():
            return rows, True
        for g in _graphemes(word):
            gw = _grapheme_cells(g)
            if row and used + gw > width and push():
                return rows, True
            row += g
            used += gw
    if row:
        rows.append(row)
    return rows[:max_rows], len(rows) > max_rows


def _watch_detail(payload: Any) -> str:
    """Pick the one human-meaningful string out of an event payload, if any."""
    if not isinstance(payload, dict):
        return ""
    for key in _WATCH_DETAIL_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    if "elapsed_seconds" in payload and "limit_seconds" in payload:
        return f"ran {payload['elapsed_seconds']}s, limit {payload['limit_seconds']}s"
    return ""


def format_watch_event(*, kind: str, task_id: str, assignee: Optional[str], title: Optional[str],
                       payload: Any, ts: Optional[int], width: int = 100,
                       use_color: bool = False, detail_lines: int = 2) -> list[str]:
    """Render one task event as a headline plus up to ``detail_lines`` wrapped detail lines.

    ``HH:MM ✓ DONE     assignee      t_abcd1234  Task title``
    ``                 └ summary / block reason / error, wrapped to the terminal``

    Every returned line fits in ``width`` terminal cells (wide CJK/emoji
    characters count as two). Below 60 cells the assignee column is dropped.
    """
    width = max(20, width)
    wide = width >= _WATCH_WIDE_MIN
    icon, label, codes = _WATCH_STYLES.get(kind, ("·", kind.upper()[:8], _DIM))
    clock = time.strftime("%H:%M", time.localtime(ts)) if ts else "--:--"

    # (text, ansi) segments, clipped left to right against the cell budget.
    segments = [(clock, _DIM), (" ", ""), (f"{icon} {label:<8}", codes), (" ", "")]
    if wide:
        who = _clip_cells(assignee or "-", 13)
        segments += [(who + " " * (13 - _cells(who)), ""), (" ", "")]
    segments += [(task_id, _DIM), ("  ", ""), (" ".join((title or "").split()), _BOLD)]
    budget, parts = width, []
    for text, seg_codes in segments:
        if budget <= 0:
            break
        text = _clip_cells(text, budget)
        budget -= _cells(text)
        parts.append(f"{seg_codes}{text}{_RESET}" if use_color and seg_codes and text else text)
    lines = ["".join(parts).rstrip()]

    detail = _watch_detail(payload)
    if detail and detail_lines > 0:
        lead = " " * (17 if wide else 2)
        first, rest = lead + "└ ", lead + "  "
        room = width - _cells(first)
        wrapped, truncated = _wrap_cells(detail, room, detail_lines)
        if truncated:
            wrapped[-1] = _clip_cells(wrapped[-1], room - 1, ellipsis="").rstrip() + "…"
        for i, text in enumerate(wrapped):
            line = (first if i == 0 else rest) + text
            lines.append(f"{_DIM}{line}{_RESET}" if use_color else line)
    return lines


def _print_json(obj: Any, *, ascii: bool = False) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=ascii))


def _json_out(args: argparse.Namespace, obj: Any, *, ascii: bool = False) -> bool:
    """Print ``obj`` as JSON and return True when ``--json`` was passed."""
    if not getattr(args, "json", False):
        return False
    _print_json(obj, ascii=ascii)
    return True


def _fmt_counts(counts: dict, empty: str = "") -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or empty


def _err(msg: str, rc: int = 1) -> int:
    print(msg, file=sys.stderr)
    return rc


def _bulk_apply(ids: Iterable[str], op: Callable[[str], Any],
                ok_msg: Callable[[str], str], fail_msg: Callable[[str], str]) -> int:
    """Run ``op(tid) -> bool`` per id, print ok/fail lines, exit 1 if any failed."""
    failed = False
    for tid in ids:
        if op(tid):
            print(ok_msg(tid))
        else:
            failed = True
            print(fail_msg(tid), file=sys.stderr)
    return 1 if failed else 0


def _fmt_task_line(t: kb.Task) -> str:
    icon = _STATUS_ICONS.get(t.status, "?")
    assignee = t.assignee or "(unassigned)"
    tenant = f" [{t.tenant}]" if t.tenant else ""
    return f"{icon} {t.id}  {t.status:8s}  {assignee:20s}{tenant}  {t.title}"


def _obj_dict(obj: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {k: getattr(obj, k) for k in fields}


def _task_to_dict(t: kb.Task) -> dict[str, Any]:
    d = _obj_dict(t, _TASK_DICT_FIELDS)
    d["skills"] = list(t.skills) if t.skills else []
    return d
