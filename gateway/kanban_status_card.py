"""Pure Kanban status-card projection for gateway notifications.

The gateway watcher owns event claiming, receipts, and delivery. This module
turns existing lifecycle state into one concise English status card.
"""

from __future__ import annotations

import re
import time
from typing import Any, cast

from hermes_cli.kanban_status_timing import format_elapsed_age, format_relative_age


def _attr(value: Any, name: str, default: Any = "") -> Any:
    return getattr(value, name, default) if value is not None else default


def _clean_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].rstrip()
    return (clipped or text[:limit]).rstrip() + "…"


def _payload(event: Any) -> dict[str, Any]:
    payload = _attr(event, "payload", None)
    return payload if isinstance(payload, dict) else {}


def _last_event(timeline: list[Any], kinds: set[str]) -> Any | None:
    for event in reversed(timeline):
        if _attr(event, "kind") in kinds:
            return event
    return None


def _is_human_note(text: str) -> bool:
    """Reject implementation plumbing from a compact end-user status card."""
    lowered = text.casefold()
    forbidden = (
        "receipt", "lease", "sqlite", "database", "schema", "gateway",
        "notifier", "adapter", "architecture", "py_compile", "pytest",
        "run_tests", "git diff", "checksum", "sha256", "traceback",
    )
    if any(word in lowered for word in forbidden):
        return False
    if re.search(r"(?:^|\s)(?:~?/|[a-z]:[\\/])|\.(?:py|json|yaml|yml|toml|log)\b", lowered):
        return False
    if re.search(r"\b[0-9a-f]{12,}\b", lowered):
        return False
    return bool(text)


def _human_note(value: Any) -> str:
    text = _clean_text(value, 220)
    text = re.sub(r"^ready\s+for\s+review\s*:\s*", "", text, flags=re.IGNORECASE)
    return text if _is_human_note(text) else ""


def _is_superseded_event(event: Any) -> bool:
    """Recognize typed replacement outcomes and the legacy block wording."""
    kind = str(_attr(event, "kind", "") or "").casefold()
    payload = _payload(event)
    outcome = str(payload.get("outcome") or payload.get("status") or "").casefold()
    reason = str(payload.get("reason") or payload.get("message") or "").casefold()
    return (
        kind in {"superseded", "replaced"}
        or outcome in {"superseded", "replaced"}
        or bool(re.search(r"\b(?:superseded|replaced)\b", reason))
    )


def _detail(timeline: list[Any], latest_comment: Any, current_run: Any = None) -> str:
    """Return only an explicit, safe worker update; never expose comments."""
    del latest_comment
    source = _attr(current_run, "events", timeline) if current_run is not None else timeline
    for event in reversed(source):
        payload = _payload(event)
        if _attr(event, "kind") == "progress":
            text = _human_note(payload.get("checkpoint"))
        elif _attr(event, "kind") == "heartbeat":
            text = _human_note(payload.get("note") or payload.get("message"))
        else:
            text = ""
        if text:
            return text
    return ""


def _age(timestamp: Any, now: int) -> int | None:
    try:
        return max(0, now - int(timestamp))
    except (TypeError, ValueError):
        return None


def _age_line(prefix: str, timestamp: Any, now: int, *, fresh: int = 120, stale: int = 900) -> str:
    age = _age(timestamp, now)
    if age is None:
        return f"🟡 {prefix}: unavailable"
    marker = "🟢" if age < fresh else "🟡" if age < stale else "🔴"
    return f"{marker} {prefix}: {format_relative_age(age)}"


def _elapsed_line(prefix: str, timestamp: Any, now: int, *, icon: str = "⏱") -> str:
    age = _age(timestamp, now)
    return f"{icon} {prefix}: {'unavailable' if age is None else format_elapsed_age(age)}"


def _compact_elapsed(age: int | None) -> str:
    """Render a 15-second-clock duration without prose labels for live cards."""
    if age is None:
        return "unavailable"
    bucketed = max(0, int(age)) // 15 * 15
    if bucketed < 15:
        return "under 15s"
    minutes, seconds = divmod(bucketed, 60)
    if minutes:
        return f"{minutes}m" + (f" {seconds}s" if seconds else "")
    return f"{seconds}s"


def _first_timestamp(timeline: list[Any], kinds: set[str], fallback: Any = None) -> Any:
    for event in timeline:
        if _attr(event, "kind") in kinds:
            return _attr(event, "created_at", fallback)
    return fallback


def _run_clock(current_run: Any, timeline: list[Any], kind: str) -> Any:
    """Read the canonical live-run clock, with a lightweight test fallback."""
    attr = "substantive_progress_at" if kind == "progress" else f"{kind}_at"
    timestamp = _attr(current_run, attr, None) if current_run is not None else None
    if timestamp is not None:
        return timestamp
    events = _attr(current_run, "events", timeline) if current_run is not None else timeline
    return _attr(_last_event(events, {kind}), "created_at", None)


def _header(title: str, task_id: str) -> str:
    """Keep the task id on the first mobile-safe header line."""
    prefix = "🗂 "
    suffix = f" · {task_id}"
    title_limit = max(12, 72 - len(prefix) - len(suffix))
    return f"{prefix}{_clean_text(title, title_limit)}{suffix}"


def _dependency_wait_copy(parents: list[Any]) -> tuple[str, str]:
    """Describe a parent gate in user terms without exposing board internals."""
    if len(parents) == 1:
        parent = parents[0]
        title = _clean_text(_attr(parent, "title", ""), 80)
        if title:
            if str(_attr(parent, "status", "") or "").lower() == "review":
                return f"Starts after review of '{title}'", "A related task is still in progress."
            return f"Starts after '{title}'", "A related task is still in progress."
        return "Starts after a related task", "A related task is still in progress."
    if len(parents) > 1:
        completed = sum(
            str(_attr(parent, "status", "") or "").lower() == "done"
            for parent in parents
        )
        count = len(parents)
        return (
            f"Starts after {count} related tasks",
            f"{completed} of {count} related tasks complete.",
        )
    return "Queued", "Waiting for an available worker."


def _state(
    task: Any, timeline: list[Any], parents: list[Any], now: int, current_run: Any = None,
) -> tuple[str, str, str]:
    status = str(_attr(task, "status", "") or "").lower()
    block_kind = str(_attr(task, "block_kind", "") or "").lower()
    assignee = _clean_text(_attr(task, "assignee", ""), 80)
    worker = f"@{assignee}" if assignee else "Worker"
    latest = _last_event(timeline, {
        "review_requested", "review_rejected", "review_accepted", "crashed",
        "timed_out", "gave_up", "reclaimed", "archived", "review_retry_scheduled",
        "review_recovered", "review_job_reconciled", "auditor_review_claimed",
        "auditor_review_spawned", "needs_auditor",
    })
    if status == "running" and current_run is not None:
        latest = _last_event(_attr(current_run, "events", ()), {
            "review_requested", "review_rejected", "review_accepted", "crashed",
            "timed_out", "gave_up", "reclaimed", "archived", "review_retry_scheduled",
            "review_recovered", "review_job_reconciled", "auditor_review_claimed",
            "auditor_review_spawned", "needs_auditor",
        })
    latest_kind = _attr(latest, "kind", "")
    if status == "archived" or latest_kind == "archived":
        return "📦", "Archived", "This task was closed in the archive."
    if latest_kind == "needs_auditor":
        return "🔐", "Manual auditor review required", "Automatic review is unavailable."
    if latest_kind == "review_retry_scheduled":
        return "⚠️", "Auditor review is restarting", "Review will be retried automatically."
    if latest_kind == "review_recovered":
        return "🔎", "Auditor review restored", "The auditor will receive this task again."
    if latest_kind == "review_job_reconciled":
        return "🔁", "Auditor review restored", "The review queue is synchronized."
    if status == "review":
        requests = sum(1 for event in timeline if _attr(event, "kind") == "review_requested")
        if latest_kind in {"auditor_review_claimed", "auditor_review_spawned"}:
            return "🔎", "Auditor started review", "The auditor is reviewing the result."
        if requests > 1:
            return "🤔", "Auditor is reviewing again", "The auditor is reviewing the result again."
        return "🔎", "Auditor is reviewing", "The auditor is reviewing the result."
    if latest_kind == "review_rejected":
        if status == "running":
            return "🤝", f"{worker} is addressing auditor feedback", "The worker is addressing auditor feedback."
        return "😡", "Auditor returned the task for changes", "The task is waiting for the worker to restart."
    if status == "done" or latest_kind == "review_accepted":
        return "✅", "Accepted by auditor", "Review complete."
    if latest_kind == "gave_up":
        return "❌", "Could not complete automatically", "A decision is needed to continue."
    if latest_kind in {"timed_out", "crashed", "reclaimed"}:
        return "⚠️", "Worker is restarting", "The dispatcher will retry automatically."
    if status == "blocked":
        replacement = _last_event(timeline, {"blocked", "dependency_wait", "superseded", "replaced"})
        if replacement is not None and _is_superseded_event(replacement):
            return "🔁", "Work continues in a new task", "Further work is tracked in a new task."
        if block_kind == "dependency":
            heading, detail = _dependency_wait_copy(parents)
            return "⏳", heading, detail
        if block_kind == "transient":
            return "⚠️", "Restarting after a temporary failure", "This is temporary; no reply is needed."
        if block_kind == "capability":
            return "🔐", "Manual help required", "A user action or access is required."
        return "📞", "Your reply is needed", "A reply is needed to continue."
    if status == "running":
        heartbeat_age = _age(_run_clock(current_run, timeline, "heartbeat"), now)
        progress_at = _run_clock(current_run, timeline, "progress")
        progress_age = _age(progress_at, now)
        if heartbeat_age is not None and heartbeat_age < 900 and (
            progress_at is None or (progress_age is not None and progress_age >= 900)
        ):
            return "🤷‍♂️", f"{worker} is online, but no confirmed progress", "The worker is online; no new confirmed progress yet."
        return "🔫", f"{worker} is working", "The worker is continuing the task."
    if status == "todo":
        if not parents:
            return "⏳", f"{worker} is waiting to start", "Waiting for an available worker."
        heading, detail = _dependency_wait_copy(parents)
        return "⏳", heading, detail
    if status in {"ready", "triage", "scheduled"}:
        return "⏳", f"{worker} is waiting to start", "Waiting for an available worker."
    return "⏳", f"{worker} is waiting to start", "Status updated."


def _metrics(
    task: Any, timeline: list[Any], status: str, now: int, current_run: Any = None,
) -> list[str]:
    """Independent clocks: raw liveness, durable progress, and task lifetime."""
    created_at = _attr(task, "created_at", None)
    started_at = _attr(current_run, "started_at", None) if current_run is not None else None
    started_at = started_at or _attr(task, "started_at", None) or _first_timestamp(
        timeline, {"claimed", "spawned"}, created_at,
    )
    if status == "running":
        heartbeat_at = _run_clock(current_run, timeline, "heartbeat")
        progress_at = _run_clock(current_run, timeline, "progress")
        heartbeat_age = _age(heartbeat_at, now)
        marker = "🟢" if (heartbeat_age or 0) < 120 else "🟡" if (heartbeat_age or 0) < 900 else "🔴"
        if progress_at is not None:
            progress_text = _compact_elapsed(_age(progress_at, now))
        else:
            progress_text = "unavailable"
        return [
            f"{marker} {_compact_elapsed(heartbeat_age)} · 🛠 {progress_text} · "
            f"⏱️ {_compact_elapsed(_age(started_at or created_at, now))}"
        ]
    if status in {"ready", "triage", "scheduled"}:
        return [f"⏱️ {_compact_elapsed(_age(created_at, now))}"]
    if status == "review":
        event = _last_event(timeline, {"review_requested"})
        return [_elapsed_line("In review", _attr(event, "created_at", created_at), now)]
    if status == "todo" or (
        status == "blocked" and str(_attr(task, "block_kind", "") or "").lower() == "dependency"
    ):
        event = _last_event(timeline, {"blocked", "dependency_wait"})
        return [f"⏱️ {_compact_elapsed(_age(_attr(event, 'created_at', created_at), now))}"]
    if status == "blocked":
        event = _last_event(timeline, {"blocked", "dependency_wait"})
        return [_elapsed_line("Waiting", _attr(event, "created_at", created_at), now)]
    if status in {"done", "archived"}:
        ended_at = _attr(task, "completed_at", None) or _attr(
            _last_event(timeline, {"review_accepted", "completed", "archived"}), "created_at", None,
        )
        age = _age(started_at or created_at, int(ended_at)) if ended_at is not None else None
        return [f"⏱ Completed in: {age // 60}m"] if age is not None else []
    return []


def user_facing_title(task: Any, timeline: list[Any], fallback: str = "") -> str:
    """Prefer an explicitly supplied end-user title over an internal task name."""
    task_metadata = _attr(task, "metadata", {})
    if not isinstance(task_metadata, dict):
        task_metadata = {}
    title = _clean_text(task_metadata.get("user_facing_title") or task_metadata.get("display_title"), 140)
    if title:
        return title
    for event in reversed(timeline):
        payload = _payload(event)
        raw_metadata = payload.get("metadata")
        nested = cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
        title = _clean_text(
            payload.get("user_facing_title") or payload.get("display_title")
            or nested.get("user_facing_title") or nested.get("display_title"),
            140,
        )
        if title:
            return title
    return _clean_text(_attr(task, "title", "") or fallback, 140)


def render_kanban_status_card(
    *, sub: dict[str, Any], task: Any, timeline: list[Any], latest_comment: Any = None,
    parents: list[Any] | None = None, now: int | None = None, current_run: Any = None,
) -> str:
    """Render a compact status snapshot from existing Kanban lifecycle data."""
    now = int(time.time()) if now is None else int(now)
    emoji, heading, fallback = _state(task, timeline, parents or [], now, current_run)
    task_id = str(sub.get("task_id") or _attr(task, "id", ""))
    title = user_facing_title(task, timeline, task_id)
    detail = _detail(timeline, latest_comment, current_run)
    status = str(_attr(task, "status", "") or "").lower()
    lines = [_header(title, task_id), "", f"{emoji} {heading}", "", "🧭 Now:"]
    lines.append(detail or fallback)
    metrics = _metrics(task, timeline, status, now, current_run)
    if metrics:
        lines.extend(("", *metrics))
    return "\n".join(lines)


def _compact_age(timestamp: Any, now: int) -> str:
    age = _age(timestamp, now)
    if age is None:
        return "unavailable"
    return format_elapsed_age(age)


def _active_index_title(icon: str, title: str, status_card_url: str) -> str:
    del status_card_url
    return f"{icon} {title}"


def _active_index_item(
    task: Any, timeline: list[Any], parents: list[Any], now: int, current_run: Any = None,
    status_card_url: str = "",
) -> tuple[int, int, str, str]:
    """Return ordering data and one compact end-user task item."""
    status = str(_attr(task, "status", "") or "").lower()
    task_id = str(_attr(task, "id", ""))
    title = _clean_text(user_facing_title(task, timeline, task_id), 72)
    created_at = _attr(task, "created_at", None)
    started_at = _attr(current_run, "started_at", None) if current_run is not None else None
    started_at = started_at or _attr(task, "started_at", None) or created_at
    heartbeat_at = _run_clock(current_run, timeline, "heartbeat") or started_at
    progress_at = _run_clock(current_run, timeline, "progress") or started_at
    heartbeat_age = _age(heartbeat_at, now)
    progress_age = _age(progress_at, now)
    # ``CurrentRunProgress`` owns only live clocks and worker updates. The
    # overview's lifecycle classification stays task-wide: after a run closes,
    # the projection is empty while the timeline still records why the task
    # needs a restart.
    latest_kind = _attr(timeline[-1], "kind", "") if timeline else ""

    if status == "review":
        review = _last_event(timeline, {"review_requested"})
        reviewed_at = _attr(review, "created_at", None) or created_at
        return 1, _age(reviewed_at, now) or 0, "review", (
            f"{_active_index_title('🔎', title, status_card_url)}\n"
            f"🟡 In review: {format_elapsed_age(_age(reviewed_at, now) or 0)}"
        )

    dependency_wait = status == "todo" or (
        status == "blocked" and str(_attr(task, "block_kind", "") or "").lower() == "dependency"
    )
    if (dependency_wait or status in {"ready", "triage", "scheduled"}) and latest_kind not in {
        "crashed", "timed_out", "gave_up",
    }:
        wait_event = _last_event(timeline, {"blocked", "dependency_wait"})
        wait_at = _attr(wait_event, "created_at", None) or created_at
        wait_copy, _ = _dependency_wait_copy(parents) if dependency_wait else ("Queued", "")
        return 3, _age(wait_at, now) or 0, "queue", (
            f"{_active_index_title('⏳', title, status_card_url)}\n"
            f"🔗 {wait_copy}\n"
            f"⏱️ {_compact_elapsed(_age(wait_at, now))}"
        )

    warning = (
        latest_kind in {"crashed", "timed_out", "gave_up"}
        or status == "blocked" or status not in {"running", "review"}
    )
    if status == "running" and progress_age is not None and progress_age >= 15 * 60:
        warning = True
    if warning:
        event = _last_event(timeline, {"blocked", "crashed", "timed_out", "gave_up"})
        at = _attr(event, "created_at", None) or progress_at or created_at
        block_kind = str(_attr(task, "block_kind", "") or "").lower()
        if status == "running":
            detail = f"No new progress {_compact_age(progress_at, now)} · working {_compact_age(started_at, now)}"
        elif block_kind == "transient":
            detail = f"Temporary failure · {_compact_age(at, now)}"
        elif block_kind == "capability":
            detail = f"Manual help required · {_compact_age(at, now)}"
        else:
            detail = f"Reply needed to continue · {_compact_age(at, now)}"
        return 0, _age(at, now) or 0, "warning", f"{_active_index_title('⚠️', title, status_card_url)}\n🔴 {detail}"

    freshness = "🟢" if (heartbeat_age or 0) < 120 else "🟡" if (heartbeat_age or 0) < 900 else "🔴"
    progress = _compact_elapsed(progress_age if _run_clock(current_run, timeline, "progress") is not None else None)
    return 2, _age(started_at, now) or 0, "running", (
        f"{_active_index_title('🔫', title, status_card_url)}\n"
        f"{freshness} {_compact_elapsed(heartbeat_age)} · 🛠 {progress} · "
        f"⏱️ {_compact_elapsed(_age(started_at, now))}"
    )


def render_kanban_active_task_index(
    items: list[tuple[Any, ...]], *, now: int | None = None,
) -> str:
    """Render the durable board's compact projection for one chat container.

    Each item starts with a human-facing board name. Topics remain delivery
    routing only: they are not useful text in a chat-wide pinned overview.
    """
    now = int(time.time()) if now is None else int(now)
    if not items:
        return "📌 Active tasks\n\n✅ No active tasks"
    grouped: dict[str, list[tuple[int, int, str, str]]] = {}
    for item in items:
        board_name, task, timeline, parents = item[:4]
        current_run = item[4] if len(item) > 4 else None
        status_card_url = str(item[5] or "") if len(item) > 5 else ""
        grouped.setdefault(board_name, []).append(
            _active_index_item(task, timeline, parents, now, current_run, status_card_url)
        )
    sections = []
    for board_name in sorted(grouped, key=str.casefold):
        board_items = sorted(grouped[board_name], key=lambda item: (item[0], -item[1], item[3]))
        count = len(board_items)
        task_word = "task" if count == 1 else "tasks"
        sections.append(f"📌 {board_name} · {count} {task_word}\n\n" + "\n\n".join(item[3] for item in board_items))
    return "\n\n".join(sections)
