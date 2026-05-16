"""Two-way sync between Hermes Kanban and the SoLoVision Notion Task Board.

This module is intentionally runnable as a quiet operational script:

    python -m hermes_cli.notion_kanban_sync --dry-run
    python -m hermes_cli.notion_kanban_sync --apply --limit 25
    python -m hermes_cli.notion_kanban_sync --apply --daemon --interval 180

No hard deletes are performed.  All audit artifacts are written under
``~/.hermes/reports/hermes-notion-sync/`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import requests

from hermes_constants import get_default_hermes_root
from hermes_cli import kanban_db

TASK_BOARD_DATABASE_ID = "8e85701f-81a6-490f-a859-5c0bc9e52827"
NOTION_READ_VERSION = "2025-09-03"
NOTION_WRITE_VERSION = "2022-06-28"
REPORT_SUBDIR = "hermes-notion-sync"
CANONICAL_NOTION_STATUSES = ("Triage", "Todo", "Ready", "Running", "Blocked", "Done", "Archived")

NOTION_TO_CANONICAL = {
    "triage": "Triage",
    "todo": "Todo",
    "to do": "Todo",
    "not started": "Todo",
    "ready": "Ready",
    "ready for creation": "Ready",
    "asset ready": "Ready",
    "running": "Running",
    "active": "Running",
    "in progress": "Running",
    "in review": "Running",
    "reviewable": "Running",
    "needs review": "Running",
    "blocked": "Blocked",
    "done": "Done",
    "completed": "Done",
    "complete": "Done",
    "cancelled": "Done",
    "canceled": "Done",
    "archived": "Archived",
}

HERMES_TO_NOTION = {
    "triage": "Triage",
    "todo": "Todo",
    "ready": "Ready",
    "running": "Running",
    "blocked": "Blocked",
    "done": "Done",
    "archived": "Archived",
}

NOTION_TO_HERMES = {
    "Triage": "triage",
    "Todo": "todo",
    "Ready": "ready",
    "Running": "running",
    "Blocked": "blocked",
    "Done": "done",
    "Archived": "archived",
}

DISPATCHER_OWNED_HERMES_STATUS = "running"

PRIORITY_TO_INT = {
    "urgent": 100,
    "critical": 100,
    "high": 75,
    "medium": 50,
    "normal": 50,
    "low": 25,
}

ROUTINE_WATCHDOG_RUN_RE = re.compile(r"\bstale[-\s]+session\s+watchdog\s+run\b", re.IGNORECASE)


@dataclass
class SyncStats:
    notion_pages_seen: int = 0
    notion_status_counts: dict[str, int] = field(default_factory=dict)
    canonical_status_counts: dict[str, int] = field(default_factory=dict)
    proposed_status_migrations: dict[str, int] = field(default_factory=dict)
    notion_pages_updated: int = 0
    notion_pages_would_update: int = 0
    notion_pages_created: int = 0
    notion_pages_would_create: int = 0
    hermes_tasks_seen: int = 0
    hermes_tasks_created: int = 0
    hermes_tasks_would_create: int = 0
    hermes_watchdog_reports_skipped: int = 0
    hermes_watchdog_report_pages_skipped: list[str] = field(default_factory=list)
    hermes_tasks_updated: int = 0
    hermes_tasks_would_update: int = 0
    hermes_to_notion_updates: int = 0
    hermes_to_notion_would_update: int = 0
    notion_to_hermes_updates: int = 0
    notion_to_hermes_would_update: int = 0
    comments_appended: int = 0
    comments_would_append: int = 0
    conflicts: int = 0
    errors: list[str] = field(default_factory=list)
    changed: bool = False


@dataclass
class NotionTask:
    page_id: str
    url: str
    title: str
    status: str
    canonical_status: str
    assigned_agent: str | None
    priority: str | None
    blockers: str | None
    notes: str | None
    source: str | None
    last_edited_time: str | None
    hermes_task_id: str | None
    hermes_status: str | None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_notion_time(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _plain_text(items: list[dict[str, Any]] | None) -> str:
    if not items:
        return ""
    return "".join(str(item.get("plain_text") or item.get("text", {}).get("content") or "") for item in items)


def _rich_text_prop(prop: dict[str, Any] | None) -> str:
    if not prop:
        return ""
    if prop.get("type") == "rich_text":
        return _plain_text(prop.get("rich_text"))
    if prop.get("type") == "title":
        return _plain_text(prop.get("title"))
    if prop.get("type") == "url":
        return prop.get("url") or ""
    return ""


def _select_name(prop: dict[str, Any] | None) -> str:
    if not prop:
        return ""
    if prop.get("type") == "select":
        val = prop.get("select")
        return val.get("name", "") if isinstance(val, dict) else ""
    if prop.get("type") == "status":
        val = prop.get("status")
        return val.get("name", "") if isinstance(val, dict) else ""
    return ""


def normalize_notion_status(status: str | None) -> str:
    key = (status or "").strip().casefold()
    return NOTION_TO_CANONICAL.get(key, "Triage")


def hermes_status_to_notion(status: str | None) -> str:
    return HERMES_TO_NOTION.get((status or "").strip().casefold(), "Triage")


def notion_status_to_hermes(status: str | None) -> str:
    return NOTION_TO_HERMES.get(normalize_notion_status(status), "triage")


def notion_status_to_safe_import_hermes(status: str | None) -> str:
    """Map Notion status for brand-new Hermes task creation.

    Notion's runtime-ish states (Running/In Progress/Active/etc.) are stale
    surprisingly often. A Hermes task may only become ``running`` via the
    dispatcher claim path, which also writes claim_lock, worker_pid,
    heartbeat/run metadata. Imported rows therefore enter the queue as
    ``ready`` so the dispatcher can claim them correctly.
    """
    mapped = notion_status_to_hermes(status)
    return "ready" if mapped == DISPATCHER_OWNED_HERMES_STATUS else mapped


def notion_running_ignored_message(notion_status: str | None, hermes_status: str) -> str:
    shown = notion_status or "(empty)"
    return (
        f"Notion status {shown!r} was ignored because Hermes dispatcher owns runtime state; "
        f"kept Hermes status {hermes_status!r}. Only the dispatcher may set a task to running "
        "because it creates/maintains claim_lock, worker_pid, heartbeat, and run metadata."
    )


def normalize_assignee(value: str | None, valid_profiles: set[str]) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    # Strip common labels and punctuation from Notion human text.
    candidate = raw.split(",", 1)[0].strip().lower()
    candidate = re.sub(r"[^a-z0-9_-]+", "-", candidate).strip("-")
    aliases = {
        "developer": "dev",
        "engineering": "dev",
        "head-of-engineering": "dev",
    }
    candidate = aliases.get(candidate, candidate)
    return candidate if candidate in valid_profiles else None


def priority_to_int(value: str | None) -> int:
    return PRIORITY_TO_INT.get((value or "").strip().casefold(), 0)


def notion_priority_from_int(value: int | None) -> str | None:
    priority = int(value or 0)
    if priority >= 75:
        return "High"
    if priority >= 50:
        return "Medium"
    if priority > 0:
        return "Low"
    return None


def _safe_rich_text(value: str, limit: int = 1900) -> list[dict[str, Any]]:
    text = (value or "")[:limit]
    return [{"text": {"content": text}}] if text else []


def _select_options(schema: dict[str, Any], name: str) -> set[str]:
    prop = schema.get("properties", {}).get(name, {})
    if prop.get("type") != "select":
        return set()
    return {str(opt.get("name") or "") for opt in prop.get("select", {}).get("options", []) if opt.get("name")}


def _matching_select_value(schema: dict[str, Any], name: str, candidate: str | None) -> str | None:
    raw = (candidate or "").strip()
    if not raw:
        return None
    options = _select_options(schema, name)
    if not options:
        return None
    by_casefold = {option.casefold(): option for option in options}
    return by_casefold.get(raw.casefold())


def _created_since_ts(created_since: str | None) -> float:
    if not created_since:
        return 0.0
    try:
        return float(created_since)
    except ValueError:
        return _parse_notion_time(created_since)


def is_completed_routine_watchdog_report(notion: NotionTask) -> bool:
    """True for completed per-run watchdog evidence rows, not actionable tasks.

    The stale-session watchdog may still create real follow-up/remediation
    Task Board rows (disk pressure, broken delivery, stale processes, human
    decisions).  Those rows must continue to sync into Kanban.  Only the
    routine run-summary artifact itself is suppressed, and only once it is
    already terminal in Notion and has no blocker text.
    """
    if notion.canonical_status not in {"Done", "Archived"}:
        return False
    if (notion.blockers or "").strip():
        return False
    return bool(ROUTINE_WATCHDOG_RUN_RE.search(notion.title or ""))


def is_routine_watchdog_hermes_task(task: kanban_db.Task) -> bool:
    if not ROUTINE_WATCHDOG_RUN_RE.search(task.title or ""):
        return False
    return task.status in {"done", "archived"}


def should_create_notion_page_for_hermes_task(
    task: kanban_db.Task,
    *,
    hermes_task_ids: set[str] | None,
    created_since_ts: float,
    state_last_run_ts: float,
) -> bool:
    if task.created_by == "notion-sync":
        return False
    if is_routine_watchdog_hermes_task(task):
        return False
    if hermes_task_ids and task.id in hermes_task_ids:
        return True
    if created_since_ts and float(task.created_at or 0) >= created_since_ts:
        return True
    if state_last_run_ts and float(task.created_at or 0) >= max(0.0, state_last_run_ts - 600.0):
        return True
    return False


def report_root(root: Path | None = None) -> Path:
    base = root or get_default_hermes_root()
    return base / "reports" / REPORT_SUBDIR


def load_notion_key() -> str:
    env = os.environ.get("NOTION_KEY") or os.environ.get("NOTION_API_KEY")
    if env and env.strip():
        return env.strip()
    candidates = [
        Path.home() / ".config" / "notion" / "api_key",
        get_default_hermes_root() / ".config" / "notion" / "api_key",
    ]
    for path in candidates:
        try:
            if path.exists():
                key = path.read_text(encoding="utf-8").strip()
                if key:
                    return key
        except OSError:
            continue
    raise RuntimeError("Notion API key not found in NOTION_KEY/NOTION_API_KEY or ~/.config/notion/api_key")


class NotionClient:
    def __init__(self, token: str, database_id: str = TASK_BOARD_DATABASE_ID, timeout: int = 30):
        self.database_id = database_id
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    def _request(self, method: str, url: str, *, version: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Notion-Version": version}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.session.request(method, url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code >= 400:
                    raise RuntimeError(f"Notion {method} {url} failed {resp.status_code}: {resp.text[:500]}")
                if not resp.text:
                    return {}
                return resp.json()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Notion {method} {url} failed after retries: {last_error}")

    def retrieve_database(self) -> dict[str, Any]:
        # Database schema reads must use the stable pre-data-source API shape;
        # Notion-Version 2025-09-03 returns an empty database-level
        # ``properties`` object for split data sources.
        return self._request(
            "GET",
            f"https://api.notion.com/v1/databases/{self.database_id}",
            version=NOTION_WRITE_VERSION,
        )

    def ensure_properties(self, *, dry_run: bool, prune_status_options: bool = False) -> dict[str, Any]:
        db = self.retrieve_database()
        props = db.get("properties", {})
        updates: dict[str, Any] = {}

        status = props.get("Status", {})
        if status.get("type") != "select":
            raise RuntimeError("Notion Task Board Status property is not a select")
        existing_options = status.get("select", {}).get("options", [])
        existing = {opt.get("name") for opt in existing_options}
        missing_statuses = [name for name in CANONICAL_NOTION_STATUSES if name not in existing]
        extra_statuses = [name for name in existing if name and name not in CANONICAL_NOTION_STATUSES]
        if prune_status_options and (missing_statuses or extra_statuses):
            by_name = {opt.get("name"): opt for opt in existing_options}
            options = []
            for name in CANONICAL_NOTION_STATUSES:
                opt = dict(by_name.get(name) or {"name": name})
                opt["name"] = name
                options.append(opt)
            updates["Status"] = {"select": {"options": options}}
        elif missing_statuses:
            options = existing_options + [{"name": name} for name in missing_statuses]
            updates["Status"] = {"select": {"options": options}}

        desired = {
            "Hermes Task ID": {"rich_text": {}},
            "Last Synced At": {"date": {}},
            "Sync Source": {"rich_text": {}},
            "Sync Error": {"rich_text": {}},
        }
        for name, spec in desired.items():
            if name not in props:
                updates[name] = spec

        retired = []
        if "Hermes Status" in props:
            if "Legacy Hermes Status" not in props:
                retired_name = "Legacy Hermes Status"
            else:
                retired_name = "Retired Hermes Status"
                suffix = 2
                while retired_name in props:
                    retired_name = f"Retired Hermes Status {suffix}"
                    suffix += 1
            updates["Hermes Status"] = {"name": retired_name}
            retired.append(f"Hermes Status -> {retired_name}")

        if updates and not dry_run:
            self._request(
                "PATCH",
                f"https://api.notion.com/v1/databases/{self.database_id}",
                version=NOTION_WRITE_VERSION,
                payload={"properties": updates},
            )
        return {"missing_statuses": missing_statuses, "extra_statuses": extra_statuses, "properties_added": [k for k in updates if k not in {"Status", "Hermes Status"}], "retired_properties": retired}

    def query_tasks(self, *, page_size: int = 100, since: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"page_size": min(max(page_size, 1), 100)}
        if since:
            payload["filter"] = {"timestamp": "last_edited_time", "last_edited_time": {"on_or_after": since}}
        rows: list[dict[str, Any]] = []
        while True:
            data = self._request(
                "POST",
                f"https://api.notion.com/v1/databases/{self.database_id}/query",
                version=NOTION_WRITE_VERSION,
                payload=payload,
            )
            rows.extend(data.get("results", []))
            if limit and len(rows) >= limit:
                return rows[:limit]
            if not data.get("has_more"):
                return rows
            payload["start_cursor"] = data.get("next_cursor")

    def update_page_properties(self, page_id: str, properties: dict[str, Any]) -> None:
        self._request(
            "PATCH",
            f"https://api.notion.com/v1/pages/{page_id}",
            version=NOTION_WRITE_VERSION,
            payload={"properties": properties},
        )

    def create_page_for_hermes_task(self, task: kanban_db.Task, *, schema: dict[str, Any]) -> dict[str, Any]:
        props_schema = schema.get("properties", {})
        title_prop = "Task" if "Task" in props_schema else "Name"
        notes_parts = [
            "Imported from Hermes Kanban by Hermes-Notion sync.",
            f"Hermes Task ID: {task.id}",
            f"Created by: {task.created_by or '(unknown)'}",
        ]
        if task.workspace_kind or task.workspace_path:
            notes_parts.append(f"Workspace: {task.workspace_kind or '(unknown)'} {task.workspace_path or ''}".strip())
        if task.body:
            notes_parts.append("Body:\n" + task.body[:1200])

        properties: dict[str, Any] = {
            title_prop: {"title": _safe_rich_text(task.title or task.id, 1900)},
            "Status": {"select": {"name": hermes_status_to_notion(task.status)}},
            "Notes": {"rich_text": _safe_rich_text("\n".join(notes_parts), 1900)},
            "Source": {"rich_text": _safe_rich_text(f"Hermes Kanban sync: {task.id}", 1900)},
            **notion_properties_for_sync(hermes_task_id=task.id, sync_source="hermes"),
        }

        priority = _matching_select_value(schema, "Priority", notion_priority_from_int(task.priority))
        if priority:
            properties["Priority"] = {"select": {"name": priority}}

        agent = _matching_select_value(schema, "Assigned Agent", (task.assignee or "").replace("_", "-").title())
        if not agent and task.assignee:
            agent = _matching_select_value(schema, "Assigned Agent", task.assignee)
        if agent:
            properties["Assigned Agent"] = {"select": {"name": agent}}

        payload = {"parent": {"database_id": self.database_id}, "properties": properties}
        try:
            return self._request("POST", "https://api.notion.com/v1/pages", version=NOTION_WRITE_VERSION, payload=payload)
        except RuntimeError as exc:
            # Optional selects can fail if Notion schema options changed between the schema read
            # and create. Retry once with only required/safe text/status/sync fields.
            if "select" not in str(exc).casefold():
                raise
            properties.pop("Priority", None)
            properties.pop("Assigned Agent", None)
            payload = {"parent": {"database_id": self.database_id}, "properties": properties}
            return self._request("POST", "https://api.notion.com/v1/pages", version=NOTION_WRITE_VERSION, payload=payload)

    def append_activity(self, page_id: str, text: str) -> None:
        self._request(
            "PATCH",
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            version=NOTION_READ_VERSION,
            payload={"children": [{"paragraph": {"rich_text": [{"text": {"content": text[:1900]}}]}}]},
        )


def parse_notion_task(page: dict[str, Any]) -> NotionTask:
    props = page.get("properties", {})
    title = _rich_text_prop(props.get("Task")) or _rich_text_prop(props.get("Name")) or "Untitled Notion task"
    status = _select_name(props.get("Status"))
    return NotionTask(
        page_id=page.get("id", ""),
        url=page.get("url", ""),
        title=title,
        status=status,
        canonical_status=normalize_notion_status(status),
        assigned_agent=(
            _select_name(props.get("Assigned Agent"))
            or _rich_text_prop(props.get("Assigned Agent"))
            or _rich_text_prop(props.get("Assigned To"))
            or _select_name(props.get("Assigned To"))
            or None
        ),
        priority=_select_name(props.get("Priority")) or _rich_text_prop(props.get("Priority")) or None,
        blockers=_rich_text_prop(props.get("Blockers")) or None,
        notes=_rich_text_prop(props.get("Notes")) or None,
        source=_rich_text_prop(props.get("Source")) or None,
        last_edited_time=page.get("last_edited_time"),
        hermes_task_id=_rich_text_prop(props.get("Hermes Task ID")) or None,
        hermes_status=(
            _rich_text_prop(props.get("Legacy Hermes Status"))
            or _rich_text_prop(props.get("Hermes Status"))
            or None
        ),
    )


def notion_page_id_from_task(task: kanban_db.Task) -> str | None:
    key = task.idempotency_key or ""
    if key.startswith("notion:"):
        return key.split(":", 1)[1]
    body = task.body or ""
    match = re.search(r"Notion Page ID:\s*([0-9a-fA-F-]{32,36})", body)
    return match.group(1) if match else None


def task_last_event_ts(conn: sqlite3.Connection, task_id: str) -> float:
    row = conn.execute("SELECT MAX(created_at) AS ts FROM task_events WHERE task_id = ?", (task_id,)).fetchone()
    return float(row["ts"] or 0)


def add_kanban_comment(conn: sqlite3.Connection, task_id: str, body: str, author: str = "notion-sync") -> None:
    now = int(time.time())
    with kanban_db.write_txn(conn):
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
            (task_id, author, body, now),
        )
        kanban_db._append_event(conn, task_id, "commented", {"author": author})


def set_kanban_status(conn: sqlite3.Connection, task_id: str, status: str, *, reason: str) -> bool:
    if status not in kanban_db.VALID_STATUSES:
        raise ValueError(f"invalid Hermes status: {status}")
    now = int(time.time())
    with kanban_db.write_txn(conn):
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row or row["status"] == status:
            return False
        if status == DISPATCHER_OWNED_HERMES_STATUS:
            raise ValueError(
                "notion sync may not set Hermes status to running; "
                "dispatcher claim path owns runtime state"
            )
        updates = "status = ?"
        params: list[Any] = [status]
        if status == "done":
            updates += ", completed_at = COALESCE(completed_at, ?), claim_lock = NULL, claim_expires = NULL, worker_pid = NULL"
            params.append(now)
        if status in {"triage", "todo", "ready", "blocked"}:
            updates += ", claim_lock = NULL, claim_expires = NULL, worker_pid = NULL"
        params.append(task_id)
        conn.execute(f"UPDATE tasks SET {updates} WHERE id = ?", tuple(params))
        kanban_db._append_event(conn, task_id, "notion_sync_status", {"from": row["status"], "to": status, "reason": reason})
        return True


def set_kanban_assignee_priority(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    assignee: str | None,
    priority: int | None,
    reason: str,
) -> bool:
    changed = False
    with kanban_db.write_txn(conn):
        row = conn.execute("SELECT assignee, priority FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return False
        sets: list[str] = []
        params: list[Any] = []
        if assignee is not None and row["assignee"] != assignee:
            sets.append("assignee = ?")
            params.append(assignee)
        if priority is not None and int(row["priority"] or 0) != int(priority):
            sets.append("priority = ?")
            params.append(int(priority))
        if sets:
            params.append(task_id)
            conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", tuple(params))
            kanban_db._append_event(conn, task_id, "notion_sync_fields", {"assignee": assignee, "priority": priority, "reason": reason})
            changed = True
    return changed


def notion_properties_for_sync(
    *,
    status: str | None = None,
    hermes_task_id: str | None = None,
    sync_source: str = "sync-engine",
    sync_error: str = "",
) -> dict[str, Any]:
    props: dict[str, Any] = {
        "Last Synced At": {"date": {"start": _now_iso()}},
        "Sync Source": {"rich_text": [{"text": {"content": sync_source}}]},
        "Sync Error": {"rich_text": [{"text": {"content": sync_error[:1900]}}] if sync_error else []},
    }
    if status:
        props["Status"] = {"select": {"name": status}}
    if hermes_task_id:
        props["Hermes Task ID"] = {"rich_text": [{"text": {"content": hermes_task_id}}]}
    return props


def make_task_body(notion: NotionTask) -> str:
    parts = [
        "Imported from Notion Task Board by Hermes-Notion sync.",
        f"Notion Page ID: {notion.page_id}",
        f"Notion URL: {notion.url}",
        f"Original Notion Status: {notion.status or '(empty)'}",
    ]
    if notion_status_to_hermes(notion.status) == DISPATCHER_OWNED_HERMES_STATUS:
        parts.append(
            "Import note: Notion runtime status was not imported as Hermes running; "
            "created as ready so the dispatcher can claim it and create runtime metadata."
        )
    if notion.assigned_agent:
        parts.append(f"Assigned Agent: {notion.assigned_agent}")
    if notion.blockers:
        parts.append(f"Blockers: {notion.blockers}")
    if notion.notes:
        parts.append(f"Notes: {notion.notes}")
    if notion.source:
        parts.append(f"Source: {notion.source}")
    return "\n".join(parts)


def valid_profiles() -> set[str]:
    profiles = set(kanban_db.list_profiles_on_disk())
    profiles.update({"halo", "dev"})
    return profiles


def load_state(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"pages": {}, "tasks": {}, "last_run_at": None}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_report(root: Path, name: str, payload: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _schema_snapshot(db: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, prop in db.get("properties", {}).items():
        entry: dict[str, Any] = {"type": prop.get("type")}
        if prop.get("type") == "select":
            entry["options"] = [opt.get("name") for opt in prop.get("select", {}).get("options", [])]
        snapshot[name] = entry
    return snapshot


def _page_backup_records(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for page in pages:
        props = page.get("properties", {})
        task = parse_notion_task(page)
        records.append({
            "page_id": task.page_id,
            "url": task.url,
            "title": task.title,
            "status": task.status,
            "canonical_status": task.canonical_status,
            "hermes_task_id": task.hermes_task_id,
            "hermes_status_old": _rich_text_prop(props.get("Hermes Status")) or None,
            "legacy_hermes_status": _rich_text_prop(props.get("Legacy Hermes Status")) or None,
            "last_edited_time": task.last_edited_time,
        })
    return records


class NotionKanbanSync:
    def __init__(
        self,
        *,
        notion: NotionClient,
        board: str | None = None,
        report_dir: Path | None = None,
        state_path: Path | None = None,
    ):
        self.notion = notion
        self.board = board
        self.report_dir = report_dir or report_root()
        self.state_path = state_path or self.report_dir / "state.json"
        self.state = load_state(self.state_path)

    def run_once(
        self,
        *,
        dry_run: bool,
        limit: int | None = None,
        since: str | None = None,
        status_migration: bool = False,
        status_migration_only: bool = False,
        prune_status_options: bool = False,
        max_creates: int | None = None,
        hermes_task_ids: set[str] | None = None,
        created_since: str | None = None,
        quiet: bool = False,
    ) -> tuple[SyncStats, Path]:
        stats = SyncStats()
        db_schema = self.notion.retrieve_database()
        raw_pages = self.notion.query_tasks(limit=limit, since=since)
        notion_tasks = [parse_notion_task(page) for page in raw_pages if not page.get("archived")]
        if hermes_task_ids:
            notion_tasks = [
                task for task in notion_tasks
                if task.hermes_task_id in hermes_task_ids or (task.source or "") in {f"Hermes Kanban sync: {task_id}" for task_id in hermes_task_ids}
            ]
        stats.notion_pages_seen = len(notion_tasks)
        stats.notion_status_counts = dict(Counter(task.status or "(empty)" for task in notion_tasks))
        stats.canonical_status_counts = dict(Counter(task.canonical_status for task in notion_tasks))
        stats.proposed_status_migrations = dict(Counter(
            f"{task.status or '(empty)'} -> {task.canonical_status}"
            for task in notion_tasks
            if (task.status or "") != task.canonical_status
        ))

        backup_path = None
        if not dry_run:
            backup_name = f"backup-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            backup_path = write_report(self.report_dir, backup_name, {
                "generated_at": _now_iso(),
                "database_id": self.notion.database_id,
                "schema": _schema_snapshot(db_schema),
                "pages": _page_backup_records(raw_pages),
            })

        ensure = self.notion.ensure_properties(dry_run=dry_run, prune_status_options=prune_status_options)
        if ensure.get("missing_statuses") or ensure.get("properties_added") or ensure.get("retired_properties") or (prune_status_options and ensure.get("extra_statuses")):
            stats.changed = True

        if status_migration_only:
            for notion_task in notion_tasks:
                if notion_task.status == notion_task.canonical_status:
                    continue
                if dry_run:
                    stats.notion_pages_would_update += 1
                else:
                    self.notion.update_page_properties(
                        notion_task.page_id,
                        notion_properties_for_sync(status=notion_task.canonical_status, hermes_task_id=notion_task.hermes_task_id),
                    )
                    stats.notion_pages_updated += 1
                stats.changed = True
            report_payload = {
                "generated_at": _now_iso(),
                "dry_run": dry_run,
                "status_migration_only": True,
                "board": self.board or kanban_db.get_current_board(),
                "database_id": self.notion.database_id,
                "ensure": ensure,
                "schema": _schema_snapshot(db_schema),
                "backup_path": str(backup_path) if backup_path else None,
                "stats": asdict(stats),
            }
            report_name = f"{'dry-run' if dry_run else 'sync'}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            report_path = write_report(self.report_dir, report_name, report_payload)
            if not quiet or dry_run or stats.changed or stats.errors:
                print(json.dumps({"report": str(report_path), "stats": asdict(stats)}, indent=2, sort_keys=True))
            return stats, report_path

        profiles = valid_profiles()
        with kanban_db.connect(board=self.board) as conn:
            created_task_ids: set[str] = set()
            tasks = kanban_db.list_tasks(conn, include_archived=True)
            stats.hermes_tasks_seen = len(tasks)
            by_notion_page = {notion_page_id_from_task(task): task for task in tasks if notion_page_id_from_task(task)}
            by_task_id = {task.id: task for task in tasks}

            for notion_task in notion_tasks:
                self._sync_notion_task(
                    conn,
                    notion_task,
                    profiles,
                    by_notion_page,
                    by_task_id,
                    dry_run=dry_run,
                    status_migration=status_migration,
                    max_creates=max_creates,
                    created_task_ids=created_task_ids,
                    stats=stats,
                )

            # Refresh task list after any Notion-created tasks, then push Hermes runtime state to Notion.
            # Pair by the embedded/idempotency Notion page id when present, and fall back to
            # the Notion row's Hermes Task ID property. Some older or manually paired tasks
            # have no Notion Page ID in Hermes, but the Notion row still points at the Hermes
            # task; Hermes remains canonical for runtime state in both cases.
            tasks = kanban_db.list_tasks(conn, include_archived=True)
            by_page_id = {task.page_id: task for task in notion_tasks}
            by_hermes_task_id = {task.hermes_task_id: task for task in notion_tasks if task.hermes_task_id}
            by_hermes_source = {task.source: task for task in notion_tasks if task.source and task.source.startswith("Hermes Kanban sync: ")}
            state_last_run_ts = _parse_notion_time(self.state.get("last_run_at"))
            min_created_ts = _created_since_ts(created_since)
            for task in tasks:
                if task.id in created_task_ids:
                    continue
                if hermes_task_ids and task.id not in hermes_task_ids:
                    continue
                page_id = notion_page_id_from_task(task)
                notion_task = by_page_id.get(page_id) if page_id else None
                if notion_task is None:
                    notion_task = by_hermes_task_id.get(task.id)
                if notion_task is None:
                    notion_task = by_hermes_source.get(f"Hermes Kanban sync: {task.id}")
                if notion_task is None:
                    if not should_create_notion_page_for_hermes_task(
                        task,
                        hermes_task_ids=hermes_task_ids,
                        created_since_ts=min_created_ts,
                        state_last_run_ts=state_last_run_ts,
                    ):
                        continue
                    if dry_run:
                        stats.notion_pages_would_create += 1
                        stats.hermes_to_notion_would_update += 1
                        stats.changed = True
                        continue
                    created_page = self.notion.create_page_for_hermes_task(task, schema=db_schema)
                    created_notion_task = parse_notion_task(created_page)
                    by_page_id[created_notion_task.page_id] = created_notion_task
                    by_hermes_task_id[task.id] = created_notion_task
                    if created_notion_task.source:
                        by_hermes_source[created_notion_task.source] = created_notion_task
                    now = _now_iso()
                    self.state.setdefault("pages", {})[created_notion_task.page_id] = {"last_synced_at": now, "hermes_task_id": task.id}
                    self.state.setdefault("tasks", {})[task.id] = {"last_synced_at": now, "notion_page_id": created_notion_task.page_id}
                    stats.notion_pages_created += 1
                    stats.notion_pages_updated += 1
                    stats.hermes_to_notion_updates += 1
                    stats.changed = True
                    continue
                self._sync_hermes_task_to_notion(
                    conn,
                    task,
                    notion_task,
                    dry_run=dry_run,
                    stats=stats,
                )

        report_payload = {
            "generated_at": _now_iso(),
            "dry_run": dry_run,
            "board": self.board or kanban_db.get_current_board(),
            "database_id": self.notion.database_id,
            "ensure": ensure,
            "schema": _schema_snapshot(db_schema),
            "backup_path": str(backup_path) if backup_path else None,
            "stats": asdict(stats),
        }
        report_name = f"{'dry-run' if dry_run else 'sync'}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        report_path = write_report(self.report_dir, report_name, report_payload)
        if not dry_run:
            self.state["last_run_at"] = _now_iso()
            save_state(self.state_path, self.state)
        if not quiet or dry_run or stats.changed or stats.errors:
            print(json.dumps({"report": str(report_path), "stats": asdict(stats)}, indent=2, sort_keys=True))
        return stats, report_path

    def _sync_notion_task(
        self,
        conn: sqlite3.Connection,
        notion_task: NotionTask,
        profiles: set[str],
        by_notion_page: dict[str | None, kanban_db.Task],
        by_task_id: dict[str, kanban_db.Task],
        *,
        dry_run: bool,
        status_migration: bool,
        max_creates: int | None,
        created_task_ids: set[str],
        stats: SyncStats,
    ) -> None:
        if is_completed_routine_watchdog_report(notion_task):
            stats.hermes_watchdog_reports_skipped += 1
            stats.hermes_watchdog_report_pages_skipped.append(notion_task.page_id)
            return

        existing = None
        if notion_task.hermes_task_id:
            existing = by_task_id.get(notion_task.hermes_task_id)
        if existing is None:
            existing = by_notion_page.get(notion_task.page_id)

        if existing is None:
            if max_creates is not None and stats.hermes_tasks_created >= max_creates and not dry_run:
                return
            assignee = normalize_assignee(notion_task.assigned_agent, profiles) or "halo"
            triage = notion_task.canonical_status == "Triage"
            hermes_status = notion_status_to_safe_import_hermes(notion_task.status)
            if dry_run:
                stats.hermes_tasks_would_create += 1
            else:
                task_id = kanban_db.create_task(
                    conn,
                    title=notion_task.title,
                    body=make_task_body(notion_task),
                    assignee=assignee,
                    created_by="notion-sync",
                    priority=priority_to_int(notion_task.priority),
                    triage=triage,
                    idempotency_key=f"notion:{notion_task.page_id}",
                )
                created = kanban_db.get_task(conn, task_id)
                if created and created.status != hermes_status:
                    set_kanban_status(conn, task_id, hermes_status, reason="initial Notion import")
                import_sync_error = ""
                import_status = notion_task.canonical_status if status_migration else None
                if notion_status_to_hermes(notion_task.status) == DISPATCHER_OWNED_HERMES_STATUS:
                    import_sync_error = notion_running_ignored_message(notion_task.status, hermes_status)
                    import_status = hermes_status_to_notion(hermes_status)
                self.notion.update_page_properties(
                    notion_task.page_id,
                    notion_properties_for_sync(
                        status=import_status,
                        hermes_task_id=task_id,
                        sync_error=import_sync_error,
                    ),
                )
                by_task_id[task_id] = kanban_db.get_task(conn, task_id)  # type: ignore[assignment]
                created_task_ids.add(task_id)
                now = _now_iso()
                self.state.setdefault("pages", {})[notion_task.page_id] = {"last_synced_at": now, "hermes_task_id": task_id}
                self.state.setdefault("tasks", {})[task_id] = {"last_synced_at": now, "notion_page_id": notion_task.page_id}
                stats.hermes_tasks_created += 1
                stats.notion_pages_updated += 1
            stats.changed = True
            return

        # Existing paired task: apply safe Notion -> Hermes field/status updates.
        state_page = self.state.setdefault("pages", {}).get(notion_task.page_id, {})
        last_sync_ts = _parse_notion_time(state_page.get("last_synced_at"))
        notion_changed = _parse_notion_time(notion_task.last_edited_time) > last_sync_ts if last_sync_ts else True
        hermes_changed = task_last_event_ts(conn, existing.id) > last_sync_ts if last_sync_ts else True
        target_hermes_status = notion_status_to_hermes(notion_task.status)

        # Archived Hermes tasks are terminal runtime state. A stale Notion row
        # (often Status=Done / Hermes Status=done from pre-archive syncs) must
        # not pull the task back out of the archive column; the later
        # Hermes -> Notion pass will make the Notion archive state explicit.
        if existing.status == "archived" and target_hermes_status != "archived":
            return

        if target_hermes_status == DISPATCHER_OWNED_HERMES_STATUS:
            # Never let Notion move an existing task into Hermes runtime state.
            # Running must come from the dispatcher claim path so claim/run
            # metadata stay coherent. If Hermes is already running, leave it
            # alone; the Hermes -> Notion pass remains allowed to publish that.
            if existing.status == DISPATCHER_OWNED_HERMES_STATUS:
                return
            if notion_changed:
                stats.conflicts += 1
                msg = notion_running_ignored_message(notion_task.status, existing.status)
                if dry_run:
                    stats.comments_would_append += 1
                else:
                    add_kanban_comment(conn, existing.id, msg)
                    self.notion.update_page_properties(
                        notion_task.page_id,
                        notion_properties_for_sync(
                            status=hermes_status_to_notion(existing.status),
                            sync_error=msg,
                            hermes_task_id=existing.id,
                        ),
                    )
                    now = _now_iso()
                    self.state.setdefault("pages", {})[notion_task.page_id] = {"last_synced_at": now, "hermes_task_id": existing.id}
                    self.state.setdefault("tasks", {})[existing.id] = {"last_synced_at": now, "notion_page_id": notion_task.page_id}
                    stats.comments_appended += 1
                    stats.notion_pages_updated += 1
                stats.changed = True
            return

        if notion_changed and hermes_changed and existing.status != target_hermes_status:
            stats.conflicts += 1
            msg = f"Conflict: Notion status {notion_task.status!r} and Hermes status {existing.status!r} both changed since last sync; Hermes runtime state kept."
            if dry_run:
                stats.comments_would_append += 1
            else:
                add_kanban_comment(conn, existing.id, msg)
                self.notion.update_page_properties(notion_task.page_id, notion_properties_for_sync(sync_error=msg, hermes_task_id=existing.id))
                now = _now_iso()
                self.state.setdefault("pages", {})[notion_task.page_id] = {"last_synced_at": now, "hermes_task_id": existing.id}
                self.state.setdefault("tasks", {})[existing.id] = {"last_synced_at": now, "notion_page_id": notion_task.page_id}
                stats.comments_appended += 1
                stats.notion_pages_updated += 1
            stats.changed = True
            return

        if notion_changed:
            status_changed = existing.status != target_hermes_status
            assignee = normalize_assignee(notion_task.assigned_agent, profiles)
            priority = priority_to_int(notion_task.priority) if notion_task.priority else None
            if dry_run:
                if status_changed or assignee or priority is not None:
                    stats.notion_to_hermes_would_update += 1
                    stats.changed = True
            else:
                did = False
                if status_changed:
                    did = set_kanban_status(conn, existing.id, target_hermes_status, reason="Notion Task Board status changed") or did
                did = set_kanban_assignee_priority(conn, existing.id, assignee=assignee, priority=priority, reason="Notion Task Board field changed") or did
                if did:
                    stats.notion_to_hermes_updates += 1
                    stats.changed = True
                needs_notion_update = (
                    did
                    or notion_task.hermes_task_id != existing.id
                    or (status_migration and notion_task.status != notion_task.canonical_status)
                )
                if needs_notion_update:
                    self.notion.update_page_properties(
                        notion_task.page_id,
                        notion_properties_for_sync(
                            hermes_task_id=existing.id,
                            status=notion_task.canonical_status if status_migration and notion_task.status != notion_task.canonical_status else None,
                        ),
                    )
                    stats.notion_pages_updated += 1
                    now = _now_iso()
                    self.state.setdefault("pages", {})[notion_task.page_id] = {"last_synced_at": now, "hermes_task_id": existing.id}
                    self.state.setdefault("tasks", {})[existing.id] = {"last_synced_at": now, "notion_page_id": notion_task.page_id}
        elif status_migration and notion_task.status != notion_task.canonical_status:
            if dry_run:
                stats.notion_pages_would_update += 1
            else:
                self.notion.update_page_properties(notion_task.page_id, notion_properties_for_sync(status=notion_task.canonical_status, hermes_task_id=existing.id))
                now = _now_iso()
                self.state.setdefault("pages", {})[notion_task.page_id] = {"last_synced_at": now, "hermes_task_id": existing.id}
                self.state.setdefault("tasks", {})[existing.id] = {"last_synced_at": now, "notion_page_id": notion_task.page_id}
                stats.notion_pages_updated += 1
            stats.changed = True

    def _sync_hermes_task_to_notion(
        self,
        conn: sqlite3.Connection,
        task: kanban_db.Task,
        notion_task: NotionTask,
        *,
        dry_run: bool,
        stats: SyncStats,
    ) -> None:
        desired = hermes_status_to_notion(task.status)
        if (
            notion_task.canonical_status == desired
            and notion_task.hermes_task_id == task.id
        ):
            return
        state_task = self.state.setdefault("tasks", {}).get(task.id, {})
        last_sync_ts = _parse_notion_time(state_task.get("last_synced_at"))
        hermes_changed = task_last_event_ts(conn, task.id) > last_sync_ts if last_sync_ts else True
        if not hermes_changed:
            return
        if dry_run:
            stats.hermes_to_notion_would_update += 1
        else:
            self.notion.update_page_properties(
                notion_task.page_id,
                notion_properties_for_sync(status=desired, hermes_task_id=task.id),
            )
            self.notion.append_activity(notion_task.page_id, f"Hermes sync: {task.id} moved to {task.status} at {_now_iso()}.")
            stats.hermes_to_notion_updates += 1
            stats.notion_pages_updated += 1
            stats.comments_appended += 1
            self.state.setdefault("tasks", {})[task.id] = {"last_synced_at": _now_iso(), "notion_page_id": notion_task.page_id}
        stats.changed = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync Hermes Kanban with the SoLoVision Notion Task Board")
    parser.add_argument("--apply", action="store_true", help="perform writes; default is dry-run")
    parser.add_argument("--dry-run", action="store_true", help="force dry-run even if --apply is absent")
    parser.add_argument("--database-id", default=TASK_BOARD_DATABASE_ID)
    parser.add_argument("--board", default=None)
    parser.add_argument("--limit", type=int, default=None, help="limit Notion pages processed (useful for safe samples)")
    parser.add_argument("--since", default=None, help="only query Notion rows edited on/after this ISO timestamp")
    parser.add_argument("--created-since", default=None, help="only create Notion pages for unpaired Hermes tasks created on/after this Unix timestamp or ISO time")
    parser.add_argument("--status-migration", action="store_true", help="rewrite legacy Notion statuses to canonical select values")
    parser.add_argument("--status-migration-only", action="store_true", help="only rewrite Notion Status select values; do not create or mutate Hermes tasks")
    parser.add_argument("--prune-status-options", action="store_true", help="replace Notion Status select options with only canonical Hermes lifecycle values")
    parser.add_argument("--max-creates", type=int, default=None, help="cap new Hermes tasks per apply run; useful for cron-safe batched backfill")
    parser.add_argument("--hermes-task-id", action="append", default=[], help="only process Notion rows linked to this Hermes task id; repeat for a limited targeted apply")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--interval", type=int, default=180)
    parser.add_argument("--quiet", action="store_true", help="suppress no-change output in apply mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dry_run = not args.apply or args.dry_run
    notion = NotionClient(load_notion_key(), database_id=args.database_id)
    sync = NotionKanbanSync(
        notion=notion,
        board=args.board,
        report_dir=Path(args.report_dir).expanduser() if args.report_dir else None,
        state_path=Path(args.state_path).expanduser() if args.state_path else None,
    )
    while True:
        stats, _ = sync.run_once(
            dry_run=dry_run,
            limit=args.limit,
            since=args.since,
            status_migration=args.status_migration or args.status_migration_only,
            status_migration_only=args.status_migration_only,
            prune_status_options=args.prune_status_options,
            max_creates=args.max_creates,
            hermes_task_ids=set(args.hermes_task_id) or None,
            created_since=args.created_since,
            quiet=args.quiet,
        )
        if not args.daemon:
            return 2 if stats.errors else 0
        time.sleep(max(args.interval, 30))


if __name__ == "__main__":
    raise SystemExit(main())
