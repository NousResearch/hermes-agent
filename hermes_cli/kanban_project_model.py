"""Project-level helpers for the Hermes Kanban execution model.

Project Hub remains canonical; Kanban tasks carry execution metadata in event
payloads, run metadata, and lightweight body markers. These helpers are kept
pure-SQL / JSON so they can be reused by the Discord watcher, dashboard, DSR,
and janitor without importing gateway code.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Any, Iterable

PROJECT_STATUS = {"idea", "active", "blocked", "review", "done", "archived"}
PROJECT_BODY_MARKER_RE = re.compile(r"^Project Hub slug:\s*(?P<slug>[a-zA-Z0-9][a-zA-Z0-9_-]{0,100})\s*$", re.M)
PROJECT_TITLE_BODY_MARKER_RE = re.compile(r"^Project title:\s*(?P<title>.+?)\s*$", re.M)
THREAD_BODY_MARKER_RE = re.compile(r"^Discord thread id:\s*(?P<thread>[0-9]{6,})\s*$", re.M)
ROOT_BODY_MARKER_RE = re.compile(r"^Kanban root task id:\s*(?P<root>t_[a-f0-9]+|[A-Za-z0-9_-]+)\s*$", re.M)
STAGE_BODY_MARKER_RE = re.compile(r"^Kanban stage:\s*(?P<stage>.+?)\s*$", re.M)
RUN_KEY_BODY_MARKER_RE = re.compile(r"^Run key:\s*(?P<run_key>.+?)\s*$", re.M)

ROUTINE_EVENT_KINDS = {"created", "promoted", "claimed", "spawned", "heartbeat", "unblocked", "archived"}
THREAD_EVENT_KINDS = {"completed", "blocked", "failed", "crashed", "timed_out", "spawn_failed", "gave_up", "commented"}
PROJECT_EVENT_KINDS = {"project_kickoff", "project_stage_started", "project_stage_completed", "project_final_summary", "project_blocked", "project_review"}
DANGER_EVENT_KINDS = {"blocked", "failed", "crashed", "timed_out", "spawn_failed", "gave_up"}


def _json_loads(raw: Any, default: Any = None) -> Any:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return default


def _meta_from_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    meta = payload.get("metadata") or payload.get("project") or {}
    if isinstance(meta, str):
        meta = _json_loads(meta, {})
    return meta if isinstance(meta, dict) else {}


def _body_marker(body: str | None, regex: re.Pattern[str]) -> str | None:
    if not body:
        return None
    m = regex.search(body)
    if not m:
        return None
    return next(iter(m.groupdict().values())).strip()


def extract_task_project_metadata(task: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized project metadata from task body + event/run payload.

    Supported keys:
    - project_hub_slug
    - project_title
    - discord_thread_id
    - kanban_root_task_id
    - project_status
    - stage_name
    - execution_wave_id
    - dsr_visible
    """
    body = task.get("body") or ""
    payload = payload or {}
    meta = _meta_from_payload(payload).copy()
    out: dict[str, Any] = {}
    aliases = {
        "project_hub_slug": ("project_hub_slug", "project_slug", "hub_slug"),
        "project_title": ("project_title", "title"),
        "discord_thread_id": ("discord_thread_id", "thread_id"),
        "kanban_root_task_id": ("kanban_root_task_id", "root_task_id", "root_id"),
        "project_status": ("project_status", "status"),
        "stage_name": ("stage_name", "stage", "stage_id"),
        "execution_wave_id": ("execution_wave_id", "wave_id"),
        "run_key": ("run_key", "build_lane_run_key"),
        "dsr_visible": ("dsr_visible", "dsr_include"),
    }
    for target, keys in aliases.items():
        for key in keys:
            value = meta.get(key) or payload.get(key)
            if value not in (None, ""):
                out[target] = value
                break
    out.setdefault("project_hub_slug", _body_marker(body, PROJECT_BODY_MARKER_RE))
    out.setdefault("project_title", _body_marker(body, PROJECT_TITLE_BODY_MARKER_RE))
    out.setdefault("discord_thread_id", _body_marker(body, THREAD_BODY_MARKER_RE))
    out.setdefault("kanban_root_task_id", _body_marker(body, ROOT_BODY_MARKER_RE))
    out.setdefault("stage_name", _body_marker(body, STAGE_BODY_MARKER_RE))
    out.setdefault("run_key", _body_marker(body, RUN_KEY_BODY_MARKER_RE))
    if out.get("project_title") in (None, "") and out.get("project_hub_slug"):
        out["project_title"] = task.get("title") or out.get("project_hub_slug")
    if out.get("kanban_root_task_id") in (None, "") and out.get("project_hub_slug"):
        out["kanban_root_task_id"] = task.get("id")
    status = str(out.get("project_status") or "").strip().lower()
    if status and status not in PROJECT_STATUS:
        status = "active"
    if status:
        out["project_status"] = status
    if "dsr_visible" in out:
        out["dsr_visible"] = bool(out["dsr_visible"])
    return {k: v for k, v in out.items() if v not in (None, "")}


def project_metadata_markers(
    *,
    project_hub_slug: str,
    project_title: str | None = None,
    discord_thread_id: str | None = None,
    kanban_root_task_id: str | None = None,
    stage_name: str | None = None,
) -> str:
    lines = ["", "---", "Kanban project metadata:", f"Project Hub slug: {project_hub_slug}"]
    if project_title:
        lines.append(f"Project title: {project_title}")
    if discord_thread_id:
        lines.append(f"Discord thread id: {discord_thread_id}")
    if kanban_root_task_id:
        lines.append(f"Kanban root task id: {kanban_root_task_id}")
    if stage_name:
        lines.append(f"Kanban stage: {stage_name}")
    return "\n".join(lines) + "\n"


def task_lookup(con: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return dict(row) if row else {"id": task_id, "title": task_id, "body": "", "status": "unknown", "assignee": "unknown"}


def latest_run_metadata(con: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT metadata, summary, outcome FROM task_runs WHERE task_id=? ORDER BY COALESCE(ended_at, started_at) DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if not row:
        return {}
    meta = _json_loads(row["metadata"], {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    if row["summary"]:
        meta.setdefault("latest_summary", row["summary"])
    if row["outcome"]:
        meta.setdefault("latest_outcome", row["outcome"])
    return meta


def component_task_ids(con: sqlite3.Connection, task_id: str) -> set[str]:
    seen = {task_id}
    q = [task_id]
    while q:
        cur = q.pop(0)
        for row in con.execute("SELECT parent_id, child_id FROM task_links WHERE parent_id=? OR child_id=?", (cur, cur)).fetchall():
            parent, child = row["parent_id"], row["child_id"]
            for nxt in (parent, child):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return seen


def component_tasks(con: sqlite3.Connection, task_id: str) -> list[dict[str, Any]]:
    return [task_lookup(con, tid) for tid in sorted(component_task_ids(con, task_id))]


def component_root(con: sqlite3.Connection, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {t["id"] for t in tasks}
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = con.execute(f"SELECT child_id FROM task_links WHERE child_id IN ({placeholders})", tuple(ids)).fetchall()
    children = {r["child_id"] for r in rows}
    roots = [t for t in tasks if t["id"] not in children] or tasks
    roots.sort(key=lambda t: (-(t.get("priority") or 0), t.get("created_at") or 0, t.get("id") or ""))
    return roots[0]


def resolve_project_context(con: sqlite3.Connection, task_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    task = task_lookup(con, task_id)
    meta = extract_task_project_metadata(task, payload)
    tasks = component_tasks(con, task_id)
    root = component_root(con, tasks)
    if not meta.get("project_hub_slug"):
        for candidate in [root, *tasks]:
            candidate_meta = extract_task_project_metadata(candidate, payload if candidate.get("id") == task_id else None)
            if candidate_meta.get("project_hub_slug"):
                meta = {**candidate_meta, **meta}
                break
    if meta.get("project_hub_slug"):
        meta.setdefault("kanban_root_task_id", root.get("id") or task_id)
        meta.setdefault("project_title", root.get("title") or task.get("title") or meta["project_hub_slug"])
        meta.setdefault("project_status", map_project_status(tasks))
        meta["task_ids"] = sorted(t["id"] for t in tasks)
        return meta
    return {}


def map_project_status(tasks: Iterable[dict[str, Any]]) -> str:
    statuses = {str(t.get("status") or "").lower() for t in tasks}
    active = statuses - {"archived"}
    if not active:
        return "archived"
    if "blocked" in active:
        return "blocked"
    if "review" in active:
        return "review"
    if active and active <= {"done"}:
        return "done"
    if active & {"ready", "running", "todo", "scheduled", "triage"}:
        return "active"
    return "active"


def should_post_project_thread_event(kind: str, payload: dict[str, Any] | None = None) -> bool:
    payload = payload or {}
    if kind in PROJECT_EVENT_KINDS:
        return True
    if kind in ROUTINE_EVENT_KINDS:
        return False
    if kind in DANGER_EVENT_KINDS:
        return True
    if kind == "completed":
        meta = _meta_from_payload(payload)
        # Completion summaries are required for Kanban handoffs, but they are
        # not automatically Discord-worthy. Project-thread completion posts are
        # opt-in via explicit user-visible / DSR / final-project metadata so
        # routine leaf completions do not become thread spam.
        return bool(
            meta.get("user_visible_change")
            or meta.get("dsr_visible")
            or meta.get("dsr_include")
            or meta.get("project_final")
            or meta.get("project_completion")
        )
    if kind == "commented":
        text = str(payload.get("body") or payload.get("comment") or "").lower()
        return any(word in text for word in ("block", "approval", "review", "done", "complete", "milestone"))
    return False


def project_thread_key(project: dict[str, Any]) -> str:
    """Stable Discord thread-state key for a Project Hub run.

    Project Hub slugs are long-lived; Build Lane can create multiple runs for
    the same slug. Prefer an explicit run key, then the Kanban root id, so fresh
    starter-helper runs do not collapse into an older forum thread.
    """
    slug = str(project.get("project_hub_slug") or "unknown")
    run_key = project.get("run_key") or project.get("execution_wave_id")
    if run_key:
        return f"{slug}:{run_key}"
    root = project.get("kanban_root_task_id")
    if root:
        return f"{slug}:{root}"
    return slug



def format_project_thread_starter(project: dict[str, Any]) -> str:
    """Return the first Discord forum/thread message for a Project Hub run.

    The starter post is the only parent-channel-visible artifact for Discord
    forum channels, so keep it compact but structured enough for humans,
    project-thread consumers, and DSR readers to recover the Project Hub slug,
    Kanban root, and visibility contract without scraping later worker posts.
    """
    title = project.get("project_title") or project.get("project_hub_slug") or "Kanban project"
    slug = project.get("project_hub_slug") or "unknown"
    root = project.get("kanban_root_task_id") or "unknown"
    status = project.get("project_status") or "active"
    stage = project.get("stage_name") or project.get("stage")
    run_key = project.get("run_key") or project.get("execution_wave_id")
    task_ids = project.get("task_ids") or []
    lines = [
        f"Project thread: {title}",
        f"Project Hub: `{slug}`",
        f"Kanban root: `{root}`",
        f"Status: `{status}`",
    ]
    if stage:
        lines.append(f"Current stage: `{stage}`")
    if run_key:
        lines.append(f"Run key: `{run_key}`")
    if task_ids:
        lines.append(f"Tracked tasks: {len(task_ids)}")
    lines.append("DSR: project-linked updates with `dsr_visible`/`dsr_include` metadata are eligible for daily-status consumers.")
    return "\n".join(lines)[:1800]



def format_project_thread_update(task: dict[str, Any], kind: str, payload: dict[str, Any] | None, project: dict[str, Any]) -> str:
    payload = payload or {}
    title = task.get("title") or task.get("id") or "task"
    assignee = task.get("assignee") or "agent"
    meta = _meta_from_payload(payload)
    summary = payload.get("summary") or payload.get("result") or meta.get("dsr_summary") or meta.get("latest_summary") or ""
    stage = project.get("stage_name") or meta.get("stage_name") or meta.get("stage")
    worker_prefix = f"{assignee}: " if assignee and assignee not in {"worker", "x", "unknown", "unassigned"} else ""
    if kind == "completed":
        if meta.get("project_final") or meta.get("project_completion"):
            return f"✅ Final: {summary or title}"[:1800]
        lead = "✅ Done"
        if stage:
            lead += f" · {stage}"
        return f"{lead}: {worker_prefix}{title}\n{summary or 'Completed.'}"[:1800]
    if kind in DANGER_EVENT_KINDS:
        reason = payload.get("reason") or payload.get("error") or summary or "Needs attention."
        icon = "⛔" if kind == "blocked" else "⚠️"
        return f"{icon} {kind.replace('_', ' ').title()}: {title}\n{reason}"[:1800]
    if kind == "commented":
        body = payload.get("body") or payload.get("comment") or summary or "Comment added."
        return f"💬 Update: {title}\n{str(body)[:1200]}"[:1800]
    if kind in PROJECT_EVENT_KINDS:
        text = summary or payload.get("message") or title
        return f"📌 {kind.replace('_', ' ').title()}: {text}"[:1800]
    return f"• {title}: {summary or kind}"[:1800]


def project_rows(con: sqlite3.Connection, include_archived: bool = False) -> list[dict[str, Any]]:
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM tasks ORDER BY created_at ASC").fetchall()
    tasks = [dict(r) for r in rows]
    by_slug: dict[str, dict[str, Any]] = {}
    for task in tasks:
        meta = extract_task_project_metadata(task)
        if not meta.get("project_hub_slug"):
            continue
        slug = str(meta["project_hub_slug"])
        entry = by_slug.setdefault(slug, {
            "project_hub_slug": slug,
            "title": meta.get("project_title") or task.get("title") or slug,
            "project_hub_status": "active",
            "kanban_root_task_id": meta.get("kanban_root_task_id") or task.get("id"),
            "discord_thread_id": meta.get("discord_thread_id"),
            "tasks": [],
            "active_agents": [],
            "blockers": [],
            "latest_update": None,
            "next_step": None,
        })
        entry["tasks"].append(task)
        if meta.get("discord_thread_id"):
            entry["discord_thread_id"] = meta["discord_thread_id"]
        if meta.get("kanban_root_task_id"):
            entry["kanban_root_task_id"] = meta["kanban_root_task_id"]
    for slug, entry in list(by_slug.items()):
        tasks = entry["tasks"]
        status = map_project_status(tasks)
        if status == "archived" and not include_archived:
            del by_slug[slug]
            continue
        entry["project_hub_status"] = status
        active_agents = sorted({t.get("assignee") for t in tasks if t.get("status") in {"ready", "running", "todo", "scheduled", "triage"} and t.get("assignee")})
        blockers = [t for t in tasks if t.get("status") == "blocked"]
        entry["active_agents"] = active_agents
        entry["blockers"] = [{"id": t["id"], "title": t.get("title"), "assignee": t.get("assignee"), "last_failure_error": t.get("last_failure_error")} for t in blockers]
        latest_task = max(tasks, key=lambda t: max(t.get("completed_at") or 0, t.get("started_at") or 0, t.get("created_at") or 0)) if tasks else None
        if latest_task:
            ts = max(latest_task.get("completed_at") or 0, latest_task.get("started_at") or 0, latest_task.get("created_at") or 0)
            entry["latest_update"] = {"at": ts, "task_id": latest_task.get("id"), "title": latest_task.get("title"), "status": latest_task.get("status")}
        next_candidates = [t for t in tasks if t.get("status") in {"ready", "running", "todo", "scheduled", "triage", "blocked"}]
        if next_candidates:
            next_candidates.sort(key=lambda t: (-(t.get("priority") or 0), t.get("created_at") or 0))
            entry["next_step"] = next_candidates[0].get("title")
        entry["counts"] = {s: sum(1 for t in tasks if t.get("status") == s) for s in sorted({t.get("status") for t in tasks})}
        entry["task_count"] = len(tasks)
        entry.pop("tasks", None)
    return sorted(by_slug.values(), key=lambda p: ((p.get("project_hub_status") == "archived"), -(p.get("latest_update") or {}).get("at", 0), p.get("title") or ""))


def archive_completed_project_tasks(con: sqlite3.Connection, older_than_seconds: int = 48 * 3600, now_ts: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Silently archive completed project-linked tasks older than retention.

    Adds only `archived` events with `discord_silent=True`; the Discord watcher
    must ignore archived events regardless.
    """
    now_ts = now_ts or int(time.time())
    cutoff = now_ts - int(older_than_seconds)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM tasks WHERE status='done' AND completed_at IS NOT NULL AND completed_at < ?", (cutoff,)).fetchall()
    candidates = []
    for row in rows:
        task = dict(row)
        if extract_task_project_metadata(task):
            candidates.append(task)
    if dry_run:
        return {"archived": [], "candidates": [t["id"] for t in candidates], "cutoff": cutoff}
    archived = []
    with con:
        for task in candidates:
            con.execute("UPDATE tasks SET status='archived' WHERE id=? AND status='done'", (task["id"],))
            con.execute(
                "INSERT INTO task_events(task_id, run_id, kind, payload, created_at) VALUES (?, NULL, 'archived', ?, ?)",
                (task["id"], json.dumps({"reason": "project-retention-48h", "discord_silent": True}), now_ts),
            )
            archived.append(task["id"])
    return {"archived": archived, "candidates": [t["id"] for t in candidates], "cutoff": cutoff}


def dsr_project_activity(con: sqlite3.Connection, start_ts: int, end_ts: int, limit: int = 20) -> list[dict[str, Any]]:
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT t.*, r.summary AS run_summary, r.metadata AS run_metadata, r.outcome AS run_outcome
        FROM tasks t
        LEFT JOIN task_runs r ON r.id = (
            SELECT id FROM task_runs rr WHERE rr.task_id=t.id ORDER BY COALESCE(rr.ended_at, rr.started_at) DESC, rr.id DESC LIMIT 1
        )
        WHERE t.completed_at >= ? AND t.completed_at < ?
        ORDER BY t.completed_at DESC
        LIMIT ?
        """,
        (start_ts, end_ts, limit * 4),
    ).fetchall()
    out = []
    for row in rows:
        task = dict(row)
        payload_meta = _json_loads(task.get("run_metadata"), {}) or {}
        meta = extract_task_project_metadata(task, {"metadata": payload_meta})
        user_visible = bool(payload_meta.get("user_visible_change") or payload_meta.get("dsr_visible") or payload_meta.get("dsr_include") or payload_meta.get("project_final") or payload_meta.get("project_completion"))
        if not meta.get("project_hub_slug") and not user_visible:
            continue
        if not user_visible and task.get("status") == "done":
            # DSR should not become a worker attendance sheet.
            continue
        out.append({
            "project_hub_slug": meta.get("project_hub_slug"),
            "project_title": meta.get("project_title") or task.get("title"),
            "kanban_root_task_id": meta.get("kanban_root_task_id"),
            "stage_name": meta.get("stage_name"),
            "run_key": meta.get("run_key"),
            "discord_thread_id": meta.get("discord_thread_id"),
            "dsr_visible": bool(payload_meta.get("dsr_visible") or payload_meta.get("dsr_include")),
            "task_id": task.get("id"),
            "title": task.get("title"),
            "assignee": task.get("assignee"),
            "summary": payload_meta.get("dsr_summary") or task.get("run_summary") or task.get("result") or "completed",
            "completed_at": task.get("completed_at"),
            "metadata": payload_meta,
        })
        if len(out) >= limit:
            break
    return out


__all__ = [
    "PROJECT_STATUS",
    "extract_task_project_metadata",
    "project_metadata_markers",
    "resolve_project_context",
    "should_post_project_thread_event",
    "project_thread_key",
    "format_project_thread_starter",
    "format_project_thread_update",
    "project_rows",
    "archive_completed_project_tasks",
    "dsr_project_activity",
]
