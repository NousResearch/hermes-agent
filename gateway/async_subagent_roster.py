"""Rows for watcher-owned background delegate_task roster bubbles."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from gateway.subagent_roster import STATUS_GLYPH, _inline_code, roster_label

_PENDING_STATUSES = {"pending", "queued", "dispatched", "running"}


def _normalise_status(raw: Any) -> str:
    status = str(raw or "").strip().lower()
    if status == "success":
        return "completed"
    if status in STATUS_GLYPH:
        return status
    if status in _PENDING_STATUSES:
        return "pending"
    return "error"


def _children_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    children = record.get("children")
    if isinstance(children, list) and children:
        out = [dict(c) for c in children if isinstance(c, dict)]
    else:
        goals = record.get("goals") or []
        results = {
            int(r.get("task_index", -1)): r
            for r in (record.get("results") or [])
            if isinstance(r, dict)
        }
        out = []
        for i, goal in enumerate(goals):
            result = results.get(i, {})
            out.append(
                {
                    "task_index": i,
                    "subagent_id": str(result.get("subagent_id") or ""),
                    "goal": goal,
                    "model": record.get("model"),
                    "status": result.get("status") or "pending",
                    "duration_seconds": result.get("duration_seconds"),
                    "tool_count": result.get("tool_count"),
                    "api_calls": result.get("api_calls"),
                    "completed_at": record.get("completed_at"),
                }
            )

    out.sort(key=lambda c: int(c.get("task_index", 0) or 0))
    return out


def build_async_subagent_roster_rows(
    record: Dict[str, Any],
    active_subagents: List[Dict[str, Any]],
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build formatter rows for one async delegation record.

    Async record children provide stable membership and terminal status.
    The active-subagents registry provides live running elapsed and tool counts.
    """
    now = time.time() if now is None else float(now)
    active_by_id = {
        str(item.get("subagent_id") or ""): item
        for item in active_subagents or []
        if item.get("subagent_id")
    }

    rows: List[Dict[str, Any]] = []
    for child in _children_from_record(record):
        sid = str(child.get("subagent_id") or "")
        active = active_by_id.get(sid)
        status = _normalise_status(child.get("status"))
        label = roster_label(child.get("goal"))
        model = child.get("model") or record.get("model") or ""
        profile = child.get("profile") or ""
        reasoning = child.get("reasoning")
        if reasoning is None:
            reasoning = record.get("reasoning")

        if active is not None and status == "pending":
            started = (
                active.get("started_at")
                or child.get("started_at")
                or record.get("dispatched_at")
                or now
            )
            try:
                elapsed = max(0.0, now - float(started))
            except Exception:
                elapsed = 0.0
            rows.append(
                {
                    "glyph": "▶",
                    "label": label,
                    "elapsed": elapsed,
                    "running": True,
                    "tools": int(active.get("tool_count") or 0),
                    "bucket": "running",
                    "model": model,
                    "profile": profile,
                    "reasoning": reasoning,
                }
            )
            continue

        if status == "pending":
            rows.append(
                {
                    "glyph": "◦",
                    "label": label,
                    "elapsed": 0.0,
                    "running": False,
                    "tools": 0,
                    "bucket": "pending",
                    "model": model,
                    "profile": profile,
                    "reasoning": reasoning,
                }
            )
            continue

        glyph, bucket = STATUS_GLYPH.get(status, ("?", "errored"))
        duration = child.get("duration_seconds")
        if duration is None:
            started = child.get("started_at") or record.get("dispatched_at")
            completed = child.get("completed_at") or record.get("completed_at")
            try:
                duration = max(0.0, float(completed) - float(started))
            except Exception:
                duration = 0.0

        # Final tool count: the child record carries tool_count (falling back to
        # api_calls). The live registry entry is gone once the child completes,
        # so the finished row keeps the count from the record.
        try:
            tools = int(child.get("tool_count") or child.get("api_calls") or 0)
        except (TypeError, ValueError):
            tools = 0

        # Final cost: threaded as a PUBLIC cost_usd onto the child record (see
        # delegate_tool._run_single_child + async_delegation._update_child_result_locked).
        # Completion-only; missing/non-numeric -> None so the row shows no cost cell.
        _row_cost = child.get("cost_usd")
        rows.append(
            {
                "glyph": glyph,
                "label": label,
                "elapsed": float(duration or 0.0),
                "running": False,
                "tools": tools,
                "bucket": bucket,
                "model": model,
                "profile": profile,
                "reasoning": reasoning,
                "cost_usd": float(_row_cost) if isinstance(_row_cost, (int, float)) else None,
            }
        )

    return rows


def build_async_dispatched_header(record: Dict[str, Any]) -> str:
    """Pinned 'what I dispatched' header line for a BACKGROUND delegation.

    Rendered as the FIRST line of the bubble and KEPT for the whole lifecycle
    (it does NOT morph away into the roster the way the old seed-card frame did).
    The live/collapsed roster rows are appended BELOW it.

    Shape example: 🔀 Delegate task — N agents · profile: `x` · toolsets=`a,b`
      * ``N agents`` — the post-expansion child count (a ``dual-review`` fans
        into one agent per lane, so this is the real number of agents running,
        not the number of model-issued tasks). Singular ``— 1 agent``.
      * ``<profile>`` — the top-level delegation profile when one was set
        (``dual-review`` / ``coder`` / …); rendered as profile: `none` when no
        profile was passed, so a plain delegate is EXPLICITLY marked rather than
        leaving the reader to wonder whether the cell is missing.
      * ``toolsets=…`` — ONLY when toolsets were EXPLICITLY passed on the
        dispatch (an audit signal for "did I under-provision a child"); hidden
        when the children just inherited the parent toolset (the common case),
        to avoid noise on every row.
    Returns "" when the record has no children/goals.
    """
    children = _children_from_record(record)
    n = len(children)
    if n <= 0:
        return ""

    head = f"🔀 Delegate task — {n} agent" + ("" if n == 1 else "s")

    profile = record.get("header_profile") or record.get("profile")
    if profile not in (None, "", [], {}):
        head += " · profile: " + _inline_code(profile)
    else:
        # No delegation profile -> mark it explicitly so a plain delegate is
        # unambiguous (not "did the profile cell go missing?").
        head += " · profile: " + _inline_code("none")

    toolsets = record.get("header_toolsets")
    if toolsets in (None, "", [], {}):
        toolsets = record.get("toolsets")
    if isinstance(toolsets, (list, tuple)) and toolsets:
        cleaned = [str(t).replace("`", "").strip() for t in toolsets if str(t).strip()]
        if cleaned:
            head += " · toolsets=" + _inline_code(",".join(cleaned))
    return head
