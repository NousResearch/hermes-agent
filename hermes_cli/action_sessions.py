"""Cross-profile, read-only Action Session projection.

Kanban ``tasks.session_id`` is the registration boundary: an ordinary session is
never promoted into the active-action view merely because it has a live process,
recent activity, or an unexpired turn lease.  Registered active tasks bring their
session descendants (subagents) with them.  Every SQLite connection is opened in
read-only/query-only mode; this module deliberately exposes no control path.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import quote

_ACTIVE_TASK_STATUSES = frozenset({"triage", "todo", "scheduled", "ready", "running", "blocked", "review"})
_EXCLUDED_SOURCES = frozenset({"cron", "webhook", "monitor", "tool"})
_GATEWAY_SOURCES = frozenset({
    "telegram", "discord", "slack", "whatsapp", "signal", "matrix", "imessage",
    "bluebubbles", "sms", "email", "teams", "feishu", "dingtalk", "wecom", "weixin",
    "qqbot", "mattermost", "google_chat", "yuanbao", "homeassistant",
})
_TERMINAL_SOURCES = frozenset({"cli", "tui", "acp", "api_server"})
_ACTION_ROOT_SOURCES = _GATEWAY_SOURCES | (_TERMINAL_SOURCES - {"api_server"}) | frozenset({"desktop"})
_LIVE_LOG_TAIL_BYTES = 16 * 1024
_PHASE_BY_TASK_STATUS = {
    "triage": "planning",
    "todo": "waiting",
    "scheduled": "waiting",
    "ready": "waiting",
    "running": "execution",
    "blocked": "blocked",
    "review": "awaiting_review",
}


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"invalid SQLite identifier: {name!r}")
    return f'"{name}"'


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute("PRAGMA table_info(" + _identifier(table) + ")")}
    except sqlite3.DatabaseError:
        return set()


def _select_available(
    conn: sqlite3.Connection,
    table: str,
    wanted: Iterable[str],
    *,
    where: str = "",
    params: tuple[Any, ...] = (),
    order: str = "",
) -> list[dict[str, Any]]:
    existing = _columns(conn, table)
    if not existing:
        return []
    selected = [name for name in wanted if name in existing]
    if not selected:
        return []
    sql = "SELECT " + ", ".join(_identifier(name) for name in selected) + " FROM " + _identifier(table)
    if where:
        sql += f" WHERE {where}"
    if order:
        sql += f" ORDER BY {order}"
    return [dict(row) for row in conn.execute(sql, params)]


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _default_profile_homes() -> dict[str, Path]:
    from hermes_cli.profiles import _get_default_hermes_home, _iter_named_profile_dirs

    homes = {"default": _get_default_hermes_home()}
    homes.update({path.name: path for path in _iter_named_profile_dirs()})
    return homes


def _default_kanban_paths() -> dict[str, Path]:
    from hermes_cli.kanban_db import kanban_db_path, list_boards

    paths: dict[str, Path] = {}
    for board in list_boards(include_archived=False):
        slug = str(board.get("slug") or "default")
        path = Path(board.get("db_path") or kanban_db_path(slug))
        if path.is_file():
            paths[slug] = path
    default = kanban_db_path("default")
    if default.is_file():
        paths.setdefault("default", default)
    return paths


def _task_rows(board: str, path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with _connect_ro(path) as conn:
        columns = _columns(conn, "tasks")
        if "status" not in columns or "session_id" not in columns:
            return []
        placeholders = ",".join("?" for _ in _ACTIVE_TASK_STATUSES)
        wanted = (
            "id", "title", "assignee", "status", "created_at", "started_at", "completed_at",
            "workspace_kind", "workspace_path", "branch_name", "session_id", "model_override",
            "provider_override", "current_step_key", "block_kind", "last_failure_error", "current_run_id",
        )
        rows = _select_available(
            conn,
            "tasks",
            wanted,
            where=f"status IN ({placeholders}) AND session_id IS NOT NULL AND session_id != ''",
            params=tuple(sorted(_ACTIVE_TASK_STATUSES)),
            order="COALESCE(started_at, created_at) DESC",
        )
        run_columns = _columns(conn, "task_runs")
        for row in rows:
            row["board"] = board
            run_id = row.get("current_run_id")
            if run_id is not None and run_columns:
                run = _select_available(
                    conn,
                    "task_runs",
                    ("id", "profile", "status", "started_at", "ended_at", "last_heartbeat_at", "summary", "metadata", "error"),
                    where="id = ?",
                    params=(run_id,),
                )
                row["run"] = run[0] if run else None
        return rows


def _profile_snapshot(profile: str, home: Path, now: float) -> dict[str, Any]:
    path = home / "state.db"
    empty = {"sessions": {}, "children": {}, "leases": {}, "delegations": {}}
    if not path.is_file():
        return empty
    with _connect_ro(path) as conn:
        sessions = _select_available(
            conn,
            "sessions",
            (
                "id", "source", "model", "model_config", "parent_session_id", "started_at", "ended_at",
                "end_reason", "title", "session_key", "chat_id", "chat_type", "thread_id", "origin_json", "cwd",
                "git_branch", "git_repo_root", "profile_name", "last_activity_at",
                "last_activity_description",
            ),
        )
        message_activity: dict[str, float] = {}
        message_columns = _columns(conn, "messages")
        if {"session_id", "timestamp"}.issubset(message_columns):
            for row in conn.execute(
                'SELECT "session_id", MAX("timestamp") AS "latest" FROM "messages" GROUP BY "session_id"'
            ):
                if row["session_id"] is not None and row["latest"] is not None:
                    message_activity[str(row["session_id"])] = float(row["latest"])
        for row in sessions:
            session_id = str(row.get("id") or "")
            observed_activity = []
            if row.get("last_activity_at") is not None:
                observed_activity.append(float(row["last_activity_at"]))
            if session_id in message_activity:
                observed_activity.append(message_activity[session_id])
            row["_lineage_activity_at"] = (
                max(observed_activity) if observed_activity else float(row.get("started_at") or 0)
            )
        session_map = {str(row["id"]): row for row in sessions}
        children: dict[str, list[dict[str, Any]]] = {}
        for row in sessions:
            parent = row.get("parent_session_id")
            if parent:
                children.setdefault(str(parent), []).append(row)
        leases = {
            str(row["conversation_id"]): row
            for row in _select_available(
                conn,
                "session_turn_leases",
                ("conversation_id", "holder", "acquired_at", "expires_at"),
                where="expires_at > ?",
                params=(now,),
            )
        }
        delegations: dict[str, list[dict[str, Any]]] = {}
        for row in _select_available(
            conn,
            "async_delegations",
            (
                "delegation_id", "parent_session_id", "state", "dispatched_at", "updated_at",
                "completed_at", "task_json", "result_json",
            ),
            order="dispatched_at ASC",
        ):
            parent = row.get("parent_session_id")
            if parent:
                delegations.setdefault(str(parent), []).append(row)
        return {"sessions": session_map, "children": children, "leases": leases, "delegations": delegations}


def _descendants(root_ids: Iterable[str], children: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Return only delegated descendants rooted in the chosen continuation lineage."""
    lineage_ids = {root_id for root_id in root_ids if root_id}
    found: list[dict[str, Any]] = []
    queue = [row for root_id in lineage_ids for row in children.get(root_id, ())]
    seen = set(lineage_ids)
    while queue:
        row = queue.pop(0)
        session_id = str(row.get("id") or "")
        if not session_id or session_id in seen:
            continue
        seen.add(session_id)
        parent_id = str(row.get("parent_session_id") or "")
        config = _json_object(row.get("model_config"))
        delegated = row.get("source") == "subagent" or config.get("_delegate_from") == parent_id
        if delegated and str(row.get("source") or "") not in _EXCLUDED_SOURCES:
            found.append(row)
            queue.extend(children.get(session_id, ()))
    return sorted(found, key=lambda row: float(row.get("started_at") or 0))


def _explicit_fork(row: Mapping[str, Any]) -> bool:
    if row.get("source") == "tool":
        return True
    config = _json_object(row.get("model_config"))
    markers = (config.get("_branched_from"), config.get("_delegate_from"), config.get("_reset_from"))
    return any(marker is not None for marker in markers)


def _continuation_lineage(
    root_id: str,
    sessions: Mapping[str, Mapping[str, Any]],
    children: Mapping[str, list[dict[str, Any]]],
) -> list[Mapping[str, Any]]:
    """Return compression continuations without crossing explicit fork edges."""
    root = sessions.get(root_id)
    if root is None:
        return []
    lineage: list[Mapping[str, Any]] = [root]
    current = root
    seen = {root_id}
    for _ in range(100):
        if current.get("end_reason") != "compression":
            break
        parent_id = str(current.get("id") or "")
        candidates = [child for child in children.get(parent_id, ()) if not _explicit_fork(child)]
        if not candidates:
            break

        def preference(row: Mapping[str, Any]) -> tuple[int, float, float, str]:
            lifecycle_rank = 2 if row.get("end_reason") == "compression" else (1 if row.get("ended_at") is None else 0)
            activity_value = row.get("_lineage_activity_at")
            activity = float(activity_value) if activity_value is not None else float(row.get("started_at") or 0)
            return lifecycle_rank, activity, float(row.get("started_at") or 0), str(row.get("id") or "")

        current = max(candidates, key=preference)
        child_id = str(current.get("id") or "")
        if not child_id or child_id in seen:
            break
        seen.add(child_id)
        lineage.append(current)
    return lineage


def _lease_key(session_id: str, sessions: Mapping[str, Mapping[str, Any]]) -> str:
    """Mirror SessionDB's compression-lineage lease key without opening a writer."""
    current = sessions.get(session_id)
    seen = {session_id}
    while current:
        parent_id = str(current.get("parent_session_id") or "")
        if not parent_id or parent_id in seen or _explicit_fork(current):
            break
        parent = sessions.get(parent_id)
        if not parent or parent.get("end_reason") != "compression":
            break
        seen.add(parent_id)
        current = parent
    return str(current.get("id") or session_id) if current else session_id


def _last_live_line(delegation_id: str, live_root: Path) -> str:
    """Read a bounded tail from this delegation's profile-owned live logs."""
    root = live_root.resolve()
    if not delegation_id or Path(delegation_id).name != delegation_id or delegation_id in {".", ".."}:
        return ""
    directory = (root / delegation_id).resolve()
    if not directory.is_relative_to(root):
        return ""
    try:
        paths = sorted(directory.glob("task-*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return ""
    for path in paths:
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(directory):
                continue
            with resolved.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - _LIVE_LOG_TAIL_BYTES))
                chunk = handle.read(_LIVE_LOG_TAIL_BYTES)
            lines = chunk.decode("utf-8", errors="replace").splitlines()
            if size > _LIVE_LOG_TAIL_BYTES and lines:
                lines.pop(0)  # the bounded tail may begin in the middle of a line
        except OSError:
            continue
        for line in reversed(lines):
            if " | " not in line:
                continue
            left, text = line.split(" | ", 1)
            role = left.split()[-1] if left.split() else "activity"
            return f"{role} | {text.strip()}"
    return ""


def _owner(root: Mapping[str, Any], store_profile: str) -> tuple[str, str, dict[str, str]]:
    source = str(root.get("source") or "unknown")
    profile = str(root.get("profile_name") or store_profile or "default")
    if source in _GATEWAY_SOURCES:
        route = {
            "platform": source,
            "session_key": str(root.get("session_key") or ""),
            "chat_id": str(root.get("chat_id") or ""),
            "thread_id": str(root.get("thread_id") or ""),
        }
        return "gateway", profile, {key: value for key, value in route.items() if value}
    if source == "desktop":
        return "desktop", profile, {"surface": "desktop", "session_id": str(root.get("id") or "")}
    if source in _TERMINAL_SOURCES:
        return "terminal", profile, {"surface": source, "session_id": str(root.get("id") or "")}
    return "session", profile, {"surface": source, "session_id": str(root.get("id") or "")}


def _lease_state(root: Mapping[str, Any], lease: Optional[Mapping[str, Any]], now: float, stale_after: float) -> str:
    if lease and float(lease.get("expires_at") or 0) > now:
        return "leased"
    activity = float(root.get("last_activity_at") or root.get("started_at") or 0)
    if activity and now - activity <= stale_after:
        return "recent_unleased"
    return "stale"


def _child_cards(
    descendants: list[dict[str, Any]], delegations: list[dict[str, Any]], now: float,
    stale_after: float, live_root: Path,
) -> list[dict[str, Any]]:
    """Project child sessions and delegation units without guessing a 1:1 link."""
    cards: list[dict[str, Any]] = []
    for row in descendants:
        cards.append({
            "kind": "session",
            "session_id": str(row.get("id") or ""),
            "source": str(row.get("source") or ""),
            "model": row.get("model"),
            "state": "ended" if row.get("ended_at") is not None else _lease_state(row, None, now, stale_after),
            "current_activity": str(row.get("last_activity_description") or ""),
            "started_at": row.get("started_at"),
            "activity_at": row.get("last_activity_at") or row.get("started_at"),
        })
    for delegation in delegations:
        delegation_id = str(delegation.get("delegation_id") or "")
        task = _json_object(delegation.get("task_json"))
        cards.append({
            "kind": "delegation",
            "session_id": "",
            "delegation_id": delegation_id,
            "source": "subagent",
            "state": str(delegation.get("state") or "unknown"),
            "model": task.get("model"),
            "current_activity": _last_live_line(delegation_id, live_root),
            "started_at": delegation.get("dispatched_at"),
            "activity_at": delegation.get("updated_at") or delegation.get("dispatched_at"),
        })
    return sorted(cards, key=lambda card: float(card.get("started_at") or 0))


def collect_active_actions(
    *,
    profile_homes: Optional[Mapping[str, Path]] = None,
    kanban_paths: Optional[Mapping[str, Path]] = None,
    now: Optional[float] = None,
    stale_after_seconds: float = 900,
) -> list[dict[str, Any]]:
    """Build the active-action projection without opening any writable handle."""
    observed_at = time.time() if now is None else float(now)
    homes = {name: Path(path) for name, path in (profile_homes or _default_profile_homes()).items()}
    boards = {name: Path(path) for name, path in (kanban_paths or _default_kanban_paths()).items()}
    snapshots = {name: _profile_snapshot(name, home, observed_at) for name, home in homes.items()}
    session_index: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for profile, snapshot in snapshots.items():
        for session_id, row in snapshot["sessions"].items():
            session_index.setdefault(session_id, []).append((profile, row))

    cards: list[dict[str, Any]] = []
    for board, path in boards.items():
        for task in _task_rows(board, path):
            session_id = str(task.get("session_id") or "")
            candidates = session_index.get(session_id, [])
            run = task.get("run") or {}
            preferred_profiles = [str(task.get("assignee") or ""), str(run.get("profile") or "")]
            located = next((item for item in candidates if item[0] in preferred_profiles), None)
            if located is None and len(candidates) == 1:
                located = candidates[0]
            if located is None:
                continue  # duplicate ids without profile evidence are intentionally ambiguous
            store_profile, root = located
            source = str(root.get("source") or "")
            if source not in _ACTION_ROOT_SOURCES:
                continue
            snapshot = snapshots[store_profile]
            owner_kind, owner_profile, owner_route = _owner(root, store_profile)
            lineage = _continuation_lineage(session_id, snapshot["sessions"], snapshot["children"])
            current_root = lineage[-1] if lineage else root
            lineage_ids = {str(row.get("id") or "") for row in lineage}
            lineage_delegations = [
                delegation
                for lineage_id in lineage_ids
                for delegation in snapshot["delegations"].get(lineage_id, [])
            ]
            descendants = _descendants(lineage_ids, snapshot["children"])
            child_cards = _child_cards(
                descendants,
                lineage_delegations,
                observed_at,
                stale_after_seconds,
                homes[store_profile] / "cache" / "delegation" / "live",
            )
            activity_candidates = [
                (float(current_root.get("last_activity_at") or 0),
                 str(current_root.get("last_activity_description") or "")),
                *[(float(child.get("activity_at") or 0), str(child.get("current_activity") or ""))
                  for child in child_cards if child.get("current_activity")],
            ]
            current_activity = max(activity_candidates, default=(0, ""))[1] or "idle"
            blocker = task.get("block_kind") or task.get("last_failure_error") or run.get("error") or None
            started_at = float(task.get("started_at") or task.get("created_at") or root.get("started_at") or observed_at)
            model = task.get("model_override") or current_root.get("model") or root.get("model")
            lease = snapshot["leases"].get(_lease_key(str(current_root.get("id") or session_id), snapshot["sessions"]))
            activity_state = _lease_state(current_root, lease, observed_at, stale_after_seconds)
            task_status = str(task.get("status") or "unknown")
            cards.append({
                "action_id": str(task.get("id") or ""),
                "registration_kind": "kanban_session_link",
                "board": board,
                "title": str(task.get("title") or root.get("title") or session_id),
                "phase": str(task.get("current_step_key") or _PHASE_BY_TASK_STATUS.get(task_status, task_status)),
                "phase_source": "kanban_step" if task.get("current_step_key") else "kanban_status",
                "task_status": task_status,
                "step_key": task.get("current_step_key"),
                "run_id": task.get("current_run_id"),
                "run_status": run.get("status"),
                "run_started_at": run.get("started_at"),
                "run_ended_at": run.get("ended_at"),
                "run_last_heartbeat_at": run.get("last_heartbeat_at"),
                "run_summary": _bounded_redacted(run.get("summary")),
                "actor": str(task.get("assignee") or run.get("profile") or owner_profile),
                "model": model,
                "current_activity": current_activity,
                "workspace": task.get("workspace_path") or current_root.get("cwd") or root.get("cwd")
                    or current_root.get("git_repo_root") or root.get("git_repo_root"),
                "branch": task.get("branch_name") or current_root.get("git_branch") or root.get("git_branch"),
                "started_at": started_at,
                "elapsed_seconds": max(0, observed_at - started_at),
                "blocker": blocker,
                "session_id": session_id,
                "session_ended": current_root.get("ended_at") is not None,
                "activity_state": activity_state,
                "lease_state": activity_state,
                "lease_expires_at": (lease or {}).get("expires_at"),
                "owner_kind": owner_kind,
                "owner_profile": owner_profile,
                "owner_route": owner_route,
                "children": child_cards,
                "warnings": ["Temporary registration source: Kanban tasks.session_id (GAR-25 pending)."],
                "observed_at": observed_at,
            })
    return sorted(cards, key=lambda card: (-float(card["started_at"]), card["action_id"]))


def _bounded_redacted(value: Any, limit: int = 500) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(str(value), force=True) or ""
    except Exception:  # fail closed: run summaries are untrusted persisted text
        return "[summary withheld: redaction unavailable]"
    return text if len(text) <= limit else text[:limit] + f" …(+{len(text) - limit} chars)"


def _safe_inline(value: Any, limit: int = 500) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    text = "".join(" " if unicodedata.category(char) in {"Cc", "Cf", "Cs"} else char for char in text)
    text = " ".join(text.split())
    return text if not limit or len(text) <= limit else text[: limit - 3] + "..."


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def render_action_cards(cards: Iterable[Mapping[str, Any]], *, now: Optional[float] = None) -> str:
    """Render a compact tmux-friendly card stream (no cursor movement or mutation)."""
    rendered: list[str] = []
    for card in cards:
        children = list(card.get("children") or [])
        child_summary = ", ".join(
            f"{_safe_inline(child.get('delegation_id') or child.get('session_id') or '?')}:{_safe_inline(child.get('state') or '?')}"
            for child in children
        ) or "—"
        owner_route = _safe_inline(json.dumps(
            card.get("owner_route") or {}, ensure_ascii=False, separators=(",", ":")))
        rendered.extend([
            f"┌─ {_safe_inline(card.get('action_id'))} · {_safe_inline(card.get('title'))}",
            f"│ phase: {_safe_inline(card.get('phase'))}   task/run: "
                f"{_safe_inline(card.get('task_status') or '—')}/{_safe_inline(card.get('run_status') or '—')}",
            f"│ actor/model: {_safe_inline(card.get('actor'))}/{_safe_inline(card.get('model') or '—')}",
            f"│ activity: {_safe_inline(card.get('current_activity') or 'idle')} "
                f"[{_safe_inline(card.get('activity_state') or card.get('lease_state'))}]   "
                f"session: {'ended' if card.get('session_ended') else 'live'}",
            f"│ workspace: {_safe_inline(card.get('workspace') or '—')}   "
                f"branch: {_safe_inline(card.get('branch') or '—')}",
            f"│ elapsed: {_duration(float(card.get('elapsed_seconds') or 0))}   "
                f"blocker: {_safe_inline(card.get('blocker') or '—')}",
            f"│ owner: {_safe_inline(card.get('owner_kind'))}:{_safe_inline(card.get('owner_profile'))} "
                f"route: {owner_route}",
            f"│ children: {child_summary}",
            "└",
        ])
    return "\n".join(rendered) if rendered else "No active Action Sessions."
