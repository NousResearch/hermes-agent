"""
Read-only snapshot builder for /control-status.

Produces a Discord-friendly text snapshot of all active session-orchestration
tasks: state, age since created_at / last_output_ts, unresolved attention
items, and the oldest actionable item so the user never misses an
outstanding query.

Architecture note
-----------------
This module is **read-only** — snapshot helpers only call registry read APIs
(``list()``, ``list_unresolved_attention_items()``, and ``get()``).  The
single-writer discipline is enforced by the cron watcher (see registry.py);
this module is a pure observer.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Age helpers
# ---------------------------------------------------------------------------

def _age_seconds(row: Dict[str, Any], now: float) -> float:
    """Return the effective age in seconds for a row.

    Prefers ``last_output_ts`` (float epoch) when available and non-zero;
    falls back to ``created_at`` (ISO-8601 UTC string from SQLite).
    Returns 0 when neither is parseable.
    """
    last_ts = row.get("last_output_ts")
    if last_ts:
        try:
            ts = float(last_ts)
            if ts > 0:
                return max(0.0, now - ts)
        except (ValueError, TypeError):
            pass

    created = row.get("created_at")
    if created:
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                # Naive UTC string from SQLite — attach UTC explicitly.
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, now - dt.timestamp())
        except (ValueError, TypeError):
            pass

    return 0.0


def _format_age(seconds: float) -> str:
    """Return a human-readable age string e.g. '3m 12s', '1h 5m', '45s'."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m"


# ---------------------------------------------------------------------------
# Attention-item helpers
# ---------------------------------------------------------------------------

def _attention_now(now: float) -> Any:
    """Return a timezone-aware datetime for attention age rendering."""
    from datetime import datetime, timezone

    from session_orchestration.feed import _coerce_datetime

    return _coerce_datetime(now) or datetime.now(timezone.utc)


def _attention_opened_age(item: Mapping[str, Any], now: Any) -> str:
    """Return the digest-compatible opened_at age label for an attention item."""
    from session_orchestration.feed import _format_age as _format_digest_age

    opened_age = _format_digest_age(item.get("opened_at"), now)
    return f"opened {opened_age} ago" if opened_age != "unknown" else "opened unknown"


def _attention_priority_label(item: Mapping[str, Any]) -> str:
    """Return the digest-compatible priority / staleness label."""
    from session_orchestration.feed import _priority_label

    return _priority_label(item)


def _attention_sort_key(item: Mapping[str, Any]) -> tuple:
    """Return the digest-compatible actionable ordering key."""
    from session_orchestration.feed import _sort_attention_item

    return _sort_attention_item(item)


def _sessions_by_task_id(
    rows: Sequence[Mapping[str, Any]],
    explicit_sessions: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Mapping[str, Any]]:
    """Build task_id -> session mapping without mutating row dictionaries."""
    sessions: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if task_id:
            sessions[str(task_id)] = row
    if explicit_sessions:
        for task_id, session in explicit_sessions.items():
            sessions[str(task_id)] = session
    return sessions


def _render_attention_item(
    item: Mapping[str, Any],
    sessions: Mapping[str, Mapping[str, Any]],
    now: Any,
) -> str:
    """Render one unresolved attention item in status-snapshot form."""
    task_id = str(item.get("task_id") or "unknown-task")
    session = sessions.get(task_id, {})
    reason = str(item.get("reason") or "attention")
    agent = session.get("agent") or item.get("agent")
    project = (
        session.get("project")
        or session.get("repo")
        or item.get("project")
        or item.get("repo")
    )
    owner_parts: List[str] = []
    if agent:
        owner_parts.append(str(agent))
    if project:
        owner_parts.append(str(project))
    owner = f" · {'/'.join(owner_parts)}" if owner_parts else ""
    detail = item.get("detail")
    detail_part = f" · {detail}" if detail else ""
    thread_id = session.get("discord_thread_id") or item.get("discord_thread_id")
    if thread_id:
        return (
            f"- <#{thread_id}> · reason `{reason}`{detail_part} · "
            f"{_attention_priority_label(item)} · {_attention_opened_age(item, now)}"
        )
    return (
        f"- `{task_id}`{owner} · reason `{reason}`{detail_part} · "
        f"{_attention_priority_label(item)} · {_attention_opened_age(item, now)}"
    )


def _render_oldest_actionable(item: Mapping[str, Any], now: Any) -> str:
    """Render the actionable highlight chosen from priority + opened_at."""
    task_id = str(item.get("task_id") or "unknown-task")
    reason = str(item.get("reason") or "attention")
    return (
        f"**Oldest actionable item:** `{task_id}` "
        f"reason `{reason}` · {_attention_priority_label(item)} · "
        f"{_attention_opened_age(item, now)}"
    )


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def build_snapshot(
    rows: List[Dict[str, Any]],
    *,
    now: Optional[float] = None,
    attention_items: Optional[Sequence[Mapping[str, Any]]] = None,
    sessions_by_task_id: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    """Build a Discord-friendly snapshot string from registry rows.

    Parameters
    ----------
    rows:
        List of row dicts as returned by ``SessionOrchestrationRegistry.list()``.
        May be empty (produces a "no active tasks" message unless unresolved
        attention items are supplied).
    now:
        Epoch timestamp to use as "current time" for age calculations.
        Defaults to ``time.time()``.  Overridable in tests for determinism.
    attention_items:
        Optional unresolved attention rows as returned by
        ``SessionOrchestrationRegistry.list_unresolved_attention_items()``.
        These rows are rendered read-only and never updated here.
    sessions_by_task_id:
        Optional task_id -> session rows, typically from ``registry.get()``.
        Values supplement ``rows`` for attention items whose session row is not
        in the active listing.

    Returns
    -------
    A plain-text, Discord-safe string.  No markdown that Discord doesn't render
    (e.g. no tables); uses bullet points and inline code spans for task IDs.
    """
    if now is None:
        now = time.time()

    ordered_attention = sorted(attention_items or (), key=_attention_sort_key)
    attention_now = _attention_now(now) if ordered_attention else None
    sessions = _sessions_by_task_id(rows, sessions_by_task_id)

    if not rows and not ordered_attention:
        return "**Session Orchestration Status**\n\nNo active tasks."

    # Compute ages and find oldest WAITING_USER for legacy rows that predate
    # attention items.
    enriched: List[Dict[str, Any]] = []
    oldest_waiting: Optional[Dict[str, Any]] = None
    oldest_waiting_age: float = -1.0

    for row in rows:
        age = _age_seconds(row, now)
        entry = {**row, "_age_s": age}
        enriched.append(entry)

        state = str(row.get("state", "")).upper()
        if state == "WAITING_USER":
            if age > oldest_waiting_age:
                oldest_waiting_age = age
                oldest_waiting = entry

    # Sort: WAITING_USER first, then by age descending (longest-running at top)
    def _sort_key(e: Dict[str, Any]) -> tuple:
        state = str(e.get("state", "")).upper()
        waiting_first = 0 if state == "WAITING_USER" else 1
        return (waiting_first, -e["_age_s"])

    enriched.sort(key=_sort_key)

    lines: List[str] = [
        f"**Session Orchestration Status** — {len(enriched)} active task(s)",
        "",
    ]

    if enriched:
        state_counts: Dict[str, int] = {}
        for entry in enriched:
            state = str(entry.get("state", "UNKNOWN")).upper()
            state_counts[state] = state_counts.get(state, 0) + 1

        # Summary counts line
        summary_parts = [f"{v} {k}" for k, v in sorted(state_counts.items())]
        lines.append("  " + " · ".join(summary_parts))
        lines.append("")

        # Per-task lines
        for entry in enriched:
            task_id = entry.get("task_id", "?")
            agent = entry.get("agent", "?")
            state = str(entry.get("state", "UNKNOWN")).upper()
            age_str = _format_age(entry["_age_s"])
            project = entry.get("project") or entry.get("workdir") or ""
            project_label = f" `{project}`" if project else ""

            # State emoji
            state_icon = {
                "RUNNING": "🟢",
                "WAITING_USER": "🔔",
                "PAUSED_HANDOFF": "⏸️",
                "STALLED": "⚠️",
                "DONE": "✅",
                "ERROR": "❌",
            }.get(state, "❓")

            line = (
                f"{state_icon} `{task_id}` [{agent}]{project_label} — "
                f"**{state}** · age {age_str}"
            )
            lines.append(line)
    else:
        lines.append("No active tasks.")

    if ordered_attention:
        lines.append("")
        lines.append(
            f"**Unresolved attention items** — {len(ordered_attention)} item(s)"
        )
        for item in ordered_attention:
            lines.append(_render_attention_item(item, sessions, attention_now))

        lines.append("")
        lines.append(_render_oldest_actionable(ordered_attention[0], attention_now))
    elif oldest_waiting is not None:
        # Legacy fallback for old registries that have WAITING_USER rows but no
        # watcher-owned attention table rows yet.
        ow_id = oldest_waiting.get("task_id", "?")
        ow_age = _format_age(oldest_waiting["_age_s"])
        lines.append("")
        lines.append(f"**Oldest WAITING_USER:** `{ow_id}` (waiting {ow_age})")

    return "\n".join(lines)


def build_registry_snapshot(
    registry: Any,
    *,
    now: Optional[float] = None,
) -> str:
    """Build a read-only snapshot directly from a registry instance."""
    rows = registry.list()
    attention_items = registry.list_unresolved_attention_items()
    task_ids = sorted(
        {str(item.get("task_id")) for item in attention_items if item.get("task_id")}
    )
    sessions = {task_id: (registry.get(task_id) or {}) for task_id in task_ids}
    return build_snapshot(
        rows,
        now=now,
        attention_items=attention_items,
        sessions_by_task_id=sessions,
    )
