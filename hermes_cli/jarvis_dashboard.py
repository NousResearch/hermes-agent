"""Read-only Jarvis cockpit dashboard aggregation helpers.

The dashboard page should render safe operational summaries only: counts,
statuses, task titles, source labels, and documented product notes.  This module
intentionally avoids env/config secret files, raw logs, session bodies, and
mutation APIs.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

from hermes_cli import kanban_db

JARVIS_BOARD = "jarvis-dashboard"
PRODUCT_BOARDS = ("clubhub", "cast-and-tag")
DEFAULT_OBSIDIAN_VAULT_PATH = "/home/aolson/ObsidianMemory"
FALLBACK_OBSIDIAN_VAULT_PATH = str(Path.home() / "Documents" / "Obsidian Vault")
OPEN_STATUSES = {"triage", "todo", "scheduled", "ready", "running", "blocked", "review"}
REVIEW_STATUSES = {"ready", "review"}
SECRET_KEY_PARTS = (
    "token",
    "api_key",
    "apikey",
    "password",
    "secret",
    "authorization",
    "cookie",
    "private_key",
    "client_secret",
)

PRODUCT_SOURCES = {
    "clubhub": {
        "slug": "clubhub",
        "name": "ClubHub",
        "board": "clubhub",
        "status_path": "/home/aolson/products/clubhub-session-planner/docs/status.md",
        "charter_path": "/home/aolson/product-agents/clubhub-session-planner/AGENT_CHARTER.md",
        "summary": "Engineering readiness, branch-lineage reconciliation, and Facilities Scheduler workflow review.",
        "approval_note": "Deploy, restart, production, permission, and user-communication actions require explicit approval.",
        "role_prefixes": {
            "clubhub": "Coordinator",
            "clubhubdev": "Rivet / development",
            "clubhubqa": "Hawkeye / QA",
            "clubhubdesign": "Pixel / design",
            "clubhubproduct": "Product",
            "clubhubsecurity": "Security",
            "clubhubsupport": "Support",
        },
    },
    "cast-and-tag": {
        "slug": "cast-and-tag",
        "name": "Cast & Tag",
        "board": "cast-and-tag",
        "status_path": "/home/aolson/products/cast-and-tag/docs/status.md",
        "charter_path": "/home/aolson/product-agents/cast-and-tag/AGENT_CHARTER.md",
        "summary": "Product-agent operating base setup with official-source and PostgreSQL trust boundaries.",
        "approval_note": "Production, publication, regulatory, financial, secret, and permission changes require explicit approval.",
        "trust_rule": "Official agency sources are authoritative; PostgreSQL is application truth; memory is context only.",
        "role_prefixes": {
            "castandtag": "Coordinator",
            "castandtagbob": "Bob / development",
            "castandtagjen": "Jen / content",
            "castandtagqa": "Scout / QA",
            "castandtagdesign": "Canvas / design",
        },
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_ok(label: str, kind: str, path: str | None = None, endpoint: str | None = None) -> dict[str, Any]:
    return {"label": label, "kind": kind, "path": path, "endpoint": endpoint, "status": "ok"}


def _source_unavailable(
    label: str,
    kind: str,
    reason: str,
    path: str | None = None,
    endpoint: str | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "kind": kind,
        "path": path,
        "endpoint": endpoint,
        "status": "unavailable",
        "error": _sanitize_error(reason),
    }


def _sanitize_error(reason: str) -> str:
    text = str(reason or "unavailable")
    for part in SECRET_KEY_PARTS:
        text = re.sub(
            rf"{re.escape(part)}\s*[:=]\s*\S+",
            "redacted=[redacted]",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(re.escape(part), "redacted", text, flags=re.IGNORECASE)
    text = re.sub(r"bearer\s+\S+", "Bearer [redacted]", text, flags=re.IGNORECASE)
    return text[:240]


def _read_markdown_sections(path_text: str) -> tuple[dict[str, Any], dict[str, list[str]]]:
    path = Path(path_text)
    if not path.exists():
        return _source_unavailable(path.name, "file", "file not found", path_text), {}
    if not path.is_file():
        return _source_unavailable(path.name, "file", "source is not a file", path_text), {}
    try:
        text = path.read_text(encoding="utf-8")
    except PermissionError:
        return _source_unavailable(path.name, "file", "file is not readable", path_text), {}
    except OSError as exc:
        return _source_unavailable(path.name, "file", str(exc), path_text), {}

    sections: dict[str, list[str]] = {"metadata": []}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("last updated:"):
            sections["metadata"].append(line)
            continue
        if line.startswith("## "):
            current = line[3:].strip().lower()
            sections.setdefault(current, [])
            continue
        if current is None or not line:
            continue
        if line.startswith(("- ", "* ")):
            sections[current].append(line[2:].strip())
        elif line[:2].isdigit() and ". " in line[:5]:
            sections[current].append(line.split(". ", 1)[1].strip())
        elif not line.startswith("#"):
            sections[current].append(line)
    return _source_ok(path.name, "file", path_text), sections


def _first_section_line(sections: dict[str, list[str]], name: str) -> str | None:
    values = sections.get(name.lower()) or []
    return values[0] if values else None


def _section_items(sections: dict[str, list[str]], name: str, limit: int = 3) -> list[str]:
    return (sections.get(name.lower()) or [])[:limit]


def _attention_action_label(status: str | None, block_kind: str | None) -> str:
    normalized_status = str(status or "").lower()
    normalized_kind = str(block_kind or "").lower()
    if normalized_status == "blocked":
        if normalized_kind == "needs_input":
            return "Answer needed"
        if normalized_kind == "capability":
            return "Access needed"
        if normalized_kind == "transient":
            return "Retry/diagnose"
        return "Read blocker note"
    if normalized_status == "review":
        return "Review changes"
    if normalized_status == "ready":
        return "Ready for next agent"
    if normalized_status == "running":
        return "Monitor running worker"
    return "Monitor status"


def _bounded_attention_reason(raw: str | None, *, limit: int = 160) -> str | None:
    if not raw:
        return None
    text = " ".join(str(raw).replace("\n", " ").split())
    if not text:
        return None
    text = _sanitize_error(text)
    return text[:limit]


def _attention_from_task_history(
    task: kanban_db.Task,
    events: list[kanban_db.Event],
    comments: list[kanban_db.Comment],
) -> tuple[str | None, int | None]:
    for event in reversed(events):
        if event.kind in {"blocked", "dependency_wait", "block_loop_detected"} and isinstance(event.payload, dict):
            reason = _bounded_attention_reason(event.payload.get("reason"))
            if reason:
                return reason, event.created_at
    for comment in reversed(comments):
        body = comment.body or ""
        lowered = body.lower()
        if "review-required:" in lowered or "blocked" in lowered or "needs input" in lowered:
            reason = _bounded_attention_reason(body)
            if reason:
                return reason, comment.created_at
    return None, task.started_at or task.created_at


def _task_summary(
    task: kanban_db.Task,
    *,
    board: str,
    events: list[kanban_db.Event] | None = None,
    comments: list[kanban_db.Comment] | None = None,
) -> dict[str, Any]:
    attention_reason, attention_since = _attention_from_task_history(task, events or [], comments or [])
    block_kind = getattr(task, "block_kind", None)
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "assignee": task.assignee,
        "priority": task.priority,
        "board": board,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "block_kind": block_kind,
        "attention_reason": attention_reason,
        "attention_action": _attention_action_label(task.status, block_kind),
        "attention_since": attention_since,
        "task_href": f"/plugins/kanban?board={quote(board, safe='')}&task={quote(task.id, safe='')}",
        "last_failure_error": _sanitize_error(task.last_failure_error) if task.last_failure_error else None,
    }


def _collect_board(board: str, *, limit_open: int = 12) -> dict[str, Any]:
    return _collect_board_unpinned(board, limit_open=limit_open)


def _kanban_db_path_for_board(board: str) -> Path:
    """Resolve a board DB without consulting process-global board env vars."""
    if board == getattr(kanban_db, "DEFAULT_BOARD", "default"):
        return kanban_db.kanban_home() / "kanban.db"
    return kanban_db.board_dir(board) / "kanban.db"


def _collect_board_unpinned(board: str, *, limit_open: int = 12) -> dict[str, Any]:
    if not kanban_db.board_exists(board):
        return {
            "board": board,
            "available": False,
            "source": _source_unavailable(f"Kanban board {board}", "kanban", "board does not exist", endpoint=f"board:{board}"),
            "counts": {},
            "open_tasks": [],
            "blocked_tasks": [],
            "review_tasks": [],
            "blocked_kind_counts": {},
            "review_status_counts": {},
            "assignee_counts": {},
        }

    conn = kanban_db.connect(db_path=_kanban_db_path_for_board(board))
    try:
        tasks = kanban_db.list_tasks(conn, include_archived=False)
        open_tasks = [task for task in tasks if task.status in OPEN_STATUSES]
        task_histories: dict[str, tuple[list[kanban_db.Event], list[kanban_db.Comment]]] = {}
        for task in open_tasks:
            task_histories[task.id] = (kanban_db.list_events(conn, task.id), kanban_db.list_comments(conn, task.id))
    finally:
        conn.close()

    counts: dict[str, int] = {}
    assignee_counts: dict[str, dict[str, int]] = {}
    for task in tasks:
        counts[task.status] = counts.get(task.status, 0) + 1
        if task.assignee:
            bucket = assignee_counts.setdefault(task.assignee, {})
            bucket[task.status] = bucket.get(task.status, 0) + 1

    blocked_tasks = [task for task in tasks if task.status == "blocked"]
    review_tasks = [task for task in tasks if task.status in REVIEW_STATUSES]
    blocked_kind_counts: dict[str, int] = {}
    for task in blocked_tasks:
        kind = getattr(task, "block_kind", None) or "unknown"
        blocked_kind_counts[kind] = blocked_kind_counts.get(kind, 0) + 1
    review_status_counts: dict[str, int] = {}
    for task in review_tasks:
        review_status_counts[task.status] = review_status_counts.get(task.status, 0) + 1

    def summarize(task: kanban_db.Task) -> dict[str, Any]:
        events, comments = task_histories.get(task.id, ([], []))
        return _task_summary(task, board=board, events=events, comments=comments)

    return {
        "board": board,
        "available": True,
        "source": _source_ok(f"Kanban board {board}", "kanban", endpoint=f"board:{board}"),
        "counts": counts,
        "open_tasks": [summarize(task) for task in open_tasks[:limit_open]],
        "blocked_tasks": [summarize(task) for task in blocked_tasks[:8]],
        "review_tasks": [summarize(task) for task in review_tasks[:8]],
        "blocked_kind_counts": blocked_kind_counts,
        "review_status_counts": review_status_counts,
        "assignee_counts": assignee_counts,
    }


def _product_health(blockers: list[str], board: dict[str, Any], source_status: str) -> str:
    if source_status != "ok" or not board.get("available"):
        return "unknown"
    if board.get("counts", {}).get("blocked", 0) > 0:
        return "blocked"
    if blockers or board.get("counts", {}).get("review", 0) > 0:
        return "attention"
    return "ok"


def _build_product_card(slug: str, board: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = PRODUCT_SOURCES[slug]
    status_source, sections = _read_markdown_sections(cfg["status_path"])
    charter_path = Path(cfg["charter_path"])
    charter_source = (
        _source_ok(f"{cfg['name']} charter", "file", cfg["charter_path"])
        if charter_path.is_file()
        else _source_unavailable(f"{cfg['name']} charter", "file", "file not found", cfg["charter_path"])
    )
    blockers = _section_items(sections, "Known blockers / unknowns", 4)
    priorities = _section_items(sections, "Active priorities", 4)
    next_actions = _section_items(sections, "Next recommended work", 4)
    safety_notes = _section_items(sections, "Safety notes", 3)
    phase = _first_section_line(sections, "Current phase") or cfg["summary"]
    last_updated = None
    for item in sections.get("metadata", []):
        if item.lower().startswith("last updated:"):
            last_updated = item.split(":", 1)[1].strip()
            break
    health = _product_health(blockers, board, status_source["status"])
    freshness = _product_freshness(last_updated, status_source, charter_source, board)
    blocker_summary, approval_summary = _product_summaries(board, cfg["approval_note"])
    owner_action = _owner_action_from_product(board, next_actions)
    primary_cta = _product_primary_cta(owner_action, freshness, board, cfg["status_path"])
    card = {
        "slug": cfg["slug"],
        "name": cfg["name"],
        "health": health,
        "summary": cfg["summary"],
        "phase": phase,
        "last_updated": last_updated,
        "priorities": priorities,
        "next_actions": next_actions,
        "safety_notes": safety_notes,
        "blockers": blockers,
        "approval_note": cfg["approval_note"],
        "trust_rule": cfg.get("trust_rule"),
        "owner_action": owner_action,
        "primary_cta": primary_cta,
        "freshness": freshness,
        "blocker_summary": blocker_summary,
        "approval_summary": approval_summary,
        "board": {
            "slug": cfg["board"],
            "available": board.get("available", False),
            "counts": board.get("counts", {}),
            "blocked_count": board.get("counts", {}).get("blocked", 0),
            "review_count": sum(board.get("counts", {}).get(status, 0) for status in REVIEW_STATUSES),
            "open_tasks": board.get("open_tasks", [])[:5],
            "blocked_tasks": board.get("blocked_tasks", [])[:5],
            "review_tasks": board.get("review_tasks", [])[:5],
            "blocked_kind_counts": board.get("blocked_kind_counts", {}),
            "review_status_counts": board.get("review_status_counts", {}),
        },
        "links": [
            {"label": "Status doc", "href": cfg["status_path"]},
            {"label": "Agent charter", "href": cfg["charter_path"]},
            {"label": "Kanban board", "href": f"/plugins/kanban?board={quote(cfg['board'], safe='')}"},
        ],
        "charter_path": cfg["charter_path"],
        "status_path": cfg["status_path"],
    }
    return card, [status_source, charter_source, board["source"]]



def _safe_text(raw: str | None, *, limit: int = 160) -> str | None:
    if not raw:
        return None
    text = " ".join(str(raw).replace("\n", " ").split())
    if not text:
        return None
    return _sanitize_error(text)[:limit]


def _parse_status_date(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _product_freshness(
    last_updated: str | None,
    status_source: dict[str, Any],
    charter_source: dict[str, Any],
    board: dict[str, Any],
    *,
    as_of: date | None = None,
) -> dict[str, Any]:
    today = as_of or datetime.now(timezone.utc).date()
    parsed = _parse_status_date(last_updated)
    age_days = (today - parsed).days if parsed else None
    source_rows = [
        {
            "label": "Status doc",
            "status": status_source.get("status", "unknown"),
            "href": status_source.get("path"),
            "message": status_source.get("error"),
        },
        {
            "label": "Board",
            "status": "ok" if board.get("available") else "unavailable",
            "href": f"/plugins/kanban?board={board.get('board')}",
            "message": None if board.get("available") else (board.get("source") or {}).get("error"),
        },
        {
            "label": "Charter",
            "status": charter_source.get("status", "unknown"),
            "href": charter_source.get("path"),
            "message": charter_source.get("error"),
        },
    ]
    if status_source.get("status") != "ok" or not board.get("available"):
        status = "unavailable"
        message = "Source unavailable — verify board and status docs before acting."
    elif age_days is None:
        status = "unknown"
        message = "Status doc update date unknown — verify before acting."
    elif age_days <= 3:
        status = "fresh"
        message = f"Status doc fresh · {age_days}d old."
    elif age_days <= 7:
        status = "aging"
        message = f"Status doc aging · {age_days}d old · verify soon."
    else:
        status = "stale"
        message = f"Status doc stale · {age_days}d old · verify before acting."
    return {
        "status": status,
        "last_updated": last_updated,
        "age_days": age_days,
        "message": message,
        "sources": source_rows,
    }


def _task_action_kind(task: dict[str, Any]) -> str:
    status = str(task.get("status") or "").lower()
    block_kind = str(task.get("block_kind") or "").lower()
    if status == "review" or (status == "blocked" and block_kind == "needs_input"):
        return "approval"
    if status == "ready":
        return "ready"
    if status == "blocked" and block_kind == "capability":
        return "access"
    if status == "blocked" and block_kind == "transient":
        return "retry"
    if status == "blocked":
        return "blocker"
    return "monitor"


def _owner_action_from_product(board: dict[str, Any], next_actions: list[str]) -> dict[str, Any]:
    tasks = list(board.get("blocked_tasks") or []) + list(board.get("review_tasks") or [])
    priority = {("blocked", "needs_input"): 0, ("review", ""): 1, ("ready", ""): 2, ("blocked", "capability"): 3, ("blocked", "transient"): 4, ("blocked", ""): 5}

    def rank(task: dict[str, Any]) -> tuple[int, int, int]:
        status = str(task.get("status") or "").lower()
        block_kind = str(task.get("block_kind") or "").lower()
        pri = priority.get((status, block_kind), priority.get((status, ""), 9))
        since = int(task.get("attention_since") or task.get("created_at") or 0)
        return pri, since, -int(task.get("priority") or 0)

    candidates = [task for task in tasks if rank(task)[0] < 9]
    if candidates:
        task = sorted(candidates, key=rank)[0]
        return {
            "kind": _task_action_kind(task),
            "label": task.get("attention_action") or "Open task",
            "task_id": task.get("id"),
            "title": _safe_text(task.get("title"), limit=140) or "Open owner-visible task",
            "reason": _safe_text(task.get("attention_reason"), limit=160),
            "age_label": None,
            "href": task.get("task_href"),
            "source": "kanban",
        }
    if next_actions:
        return {
            "kind": "next_work",
            "label": "Next safe work",
            "task_id": None,
            "title": _safe_text(next_actions[0], limit=180) or "Review product status doc",
            "reason": None,
            "age_label": None,
            "href": None,
            "source": "status_doc",
        }
    return {
        "kind": "monitor",
        "label": "Monitor",
        "task_id": None,
        "title": "No owner action surfaced by board or status doc.",
        "reason": None,
        "age_label": None,
        "href": f"/plugins/kanban?board={board.get('board')}",
        "source": "kanban",
    }


def _product_primary_cta(
    owner_action: dict[str, Any],
    freshness: dict[str, Any],
    board: dict[str, Any],
    status_path: str,
) -> dict[str, str]:
    """Choose one navigation-only CTA for the visible product state.

    The kanban dashboard currently supports board and task query params, but not
    a documented status filter query. For blocked-board fallback we keep the
    label action-specific while using the safe board route rather than inventing
    an unsupported filter URL.
    """
    board_slug = str(board.get("board") or "")
    board_href = f"/plugins/kanban?board={quote(board_slug, safe='')}"
    action_label = str(owner_action.get("label") or "").lower()
    action_kind = str(owner_action.get("kind") or "").lower()
    action_href = owner_action.get("href")

    if action_href and (
        action_kind == "approval"
        or action_label in {"answer needed", "review changes"}
    ):
        return {"label": "Open approval task", "href": str(action_href), "kind": "approval_task"}

    if freshness.get("status") in {"stale", "unavailable"}:
        return {"label": "Open source doc", "href": status_path, "kind": "source_doc"}

    blocked_count = int((board.get("counts") or {}).get("blocked", 0) or 0)
    if blocked_count > 0:
        return {"label": "Open blocked board", "href": board_href, "kind": "blocked_board"}

    return {"label": "Open board", "href": board_href, "kind": "board"}


def _product_summaries(board: dict[str, Any], approval_note: str) -> tuple[dict[str, Any], dict[str, Any]]:
    kind_counts = board.get("blocked_kind_counts") or {}
    review_counts = board.get("review_status_counts") or {}
    blocker_summary = {
        "total": int(board.get("counts", {}).get("blocked", 0) or 0),
        "needs_input": int(kind_counts.get("needs_input", 0) or 0),
        "capability": int(kind_counts.get("capability", 0) or 0),
        "transient": int(kind_counts.get("transient", 0) or 0),
        "unknown": int(kind_counts.get("unknown", 0) or 0),
        "examples": (board.get("blocked_tasks") or [])[:2],
    }
    approval_summary = {
        "total": sum(int(review_counts.get(status, 0) or 0) for status in REVIEW_STATUSES),
        "review": int(review_counts.get("review", 0) or 0),
        "ready": int(review_counts.get("ready", 0) or 0),
        "examples": (board.get("review_tasks") or [])[:2],
        "approval_note": approval_note,
    }
    return blocker_summary, approval_summary

def _role_for_profile(profile: str) -> tuple[str, str]:
    if profile == "default":
        return "Jarvis", "Command-center coordinator"
    for cfg in PRODUCT_SOURCES.values():
        role = cfg["role_prefixes"].get(profile)
        if role:
            return cfg["name"], role
    return "Hermes", "Agent profile"


def _build_agent_profiles(status: dict[str, Any], boards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = [str(p) for p in status.get("profiles") or []]
    if not profiles:
        profiles = ["default"]
    out = []
    for name in profiles:
        product, role = _role_for_profile(name)
        task_counts: dict[str, int] = {}
        blocked_tasks: list[dict[str, Any]] = []
        open_tasks: list[dict[str, Any]] = []
        for board in boards.values():
            for state, count in board.get("assignee_counts", {}).get(name, {}).items():
                task_counts[state] = task_counts.get(state, 0) + count
            for task in board.get("blocked_tasks", []):
                if task.get("assignee") == name:
                    blocked_tasks.append(task)
            for task in board.get("open_tasks", []):
                if task.get("assignee") == name:
                    open_tasks.append(task)
        if task_counts.get("blocked"):
            state = "blocked"
        elif task_counts.get("running"):
            state = "running"
        elif task_counts.get("ready") or task_counts.get("review"):
            state = "ready"
        elif sum(task_counts.values()) > 0:
            state = "queued"
        else:
            state = "idle"
        out.append(
            {
                "name": name,
                "product": product,
                "role": role,
                "state": state,
                "task_counts": task_counts,
                "blocked_count": len(blocked_tasks),
                "open_count": len(open_tasks),
                "blocked_tasks": blocked_tasks[:3],
                "open_tasks": open_tasks[:3],
                "needs_attention": state in {"blocked", "ready"},
            }
        )
    return out


def _summarize_cron(jobs: list[dict[str, Any]] | None) -> dict[str, Any]:
    if jobs is None:
        return {"available": False, "total": 0, "enabled": 0, "paused": 0, "recent_failures": 0, "local_only": 0}
    paused = 0
    failures = 0
    local_only = 0
    enabled = 0
    for job in jobs:
        if job.get("enabled"):
            enabled += 1
        state = str(job.get("state") or "").lower()
        if state in {"paused", "disabled"} or not job.get("enabled", True):
            paused += 1
        if str(job.get("last_status") or "").lower() in {"failed", "error", "timeout"}:
            failures += 1
        deliver = str(job.get("deliver") or "origin").lower()
        if deliver in {"", "origin", "local"}:
            local_only += 1
    return {
        "available": True,
        "total": len(jobs),
        "enabled": enabled,
        "paused": paused,
        "recent_failures": failures,
        "local_only": local_only,
    }


def _note_title(path: Path) -> str:
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
            line = raw.strip()
            if line.startswith("# "):
                return line[2:].strip()[:120] or path.stem
    except OSError:
        pass
    return path.stem


def _vault_note_summary(path: Path, vault_root: Path) -> dict[str, Any]:
    rel = path.relative_to(vault_root).as_posix()
    stat = path.stat()
    return {
        "title": _note_title(path),
        "relative_path": rel,
        "href": f"/files?path={quote(str(path), safe='')}",
        "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _obsidian_vault_insights(vault_root: Path) -> dict[str, Any]:
    notes = [path for path in vault_root.rglob("*.md") if path.is_file() and ".obsidian" not in path.parts]
    notes.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    command_center = vault_root / "00 Command Center"
    decisions = vault_root / "Decisions"
    products = vault_root / "Products"
    return {
        "note_count": len(notes),
        "decision_count": len(list(decisions.glob("*.md"))) if decisions.is_dir() else 0,
        "product_note_count": len(list(products.rglob("*.md"))) if products.is_dir() else 0,
        "recent_notes": [_vault_note_summary(path, vault_root) for path in notes[:6]],
        "quick_links": [
            {"label": "Owner Cockpit", "href": f"/files?path={quote(str(command_center / 'Owner Cockpit.md'), safe='')}"},
            {"label": "Approvals Queue", "href": f"/files?path={quote(str(command_center / 'Approvals Queue.md'), safe='')}"},
            {"label": "Agent Ops", "href": f"/files?path={quote(str(command_center / 'Agent Operations Dashboard.md'), safe='')}"},
            {"label": "Weekly Review", "href": f"/files?path={quote(str(command_center / 'Weekly Business Review.md'), safe='')}"},
        ],
    }


def _memory_vault_status() -> tuple[dict[str, Any], dict[str, Any]]:
    configured = os.environ.get("OBSIDIAN_VAULT_PATH")
    candidates: list[tuple[str, str]] = []
    if configured:
        candidates.append((configured, "OBSIDIAN_VAULT_PATH"))
    candidates.extend(
        [
            (DEFAULT_OBSIDIAN_VAULT_PATH, "fallback"),
            (FALLBACK_OBSIDIAN_VAULT_PATH, "fallback"),
        ]
    )

    seen: set[str] = set()
    for raw_path, source in candidates:
        if raw_path in seen:
            continue
        seen.add(raw_path)
        path = Path(raw_path).expanduser()
        if path.is_dir():
            resolved = str(path)
            vault = {
                "configured": True,
                "status": "available",
                "label": "Obsidian Memory",
                "path": resolved,
                "source": source,
                "href": f"/files?path={quote(resolved, safe='')}",
                "message": "Obsidian memory vault is available for read-only browsing.",
                **_obsidian_vault_insights(path),
            }
            return {"obsidian": vault}, _source_ok("Obsidian Memory", "file", resolved)

    missing_path = configured or DEFAULT_OBSIDIAN_VAULT_PATH
    vault = {
        "configured": False,
        "status": "setup_needed",
        "label": "Obsidian Memory",
        "path": missing_path if missing_path else None,
        "source": "not_configured" if not configured else "OBSIDIAN_VAULT_PATH",
        "href": "/files",
        "message": "Set OBSIDIAN_VAULT_PATH to enable vault browsing.",
        "note_count": 0,
        "decision_count": 0,
        "product_note_count": 0,
        "recent_notes": [],
        "quick_links": [],
    }
    return {"obsidian": vault}, _source_unavailable(
        "Obsidian Memory",
        "file",
        "vault path not configured or missing",
        missing_path,
    )


def assert_no_secret_keys(payload: Any, path: str = "$") -> None:
    """Fail closed if the overview payload grows a secret-bearing key."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(part in lowered for part in SECRET_KEY_PARTS):
                raise ValueError(f"Jarvis overview contains disallowed key at {path}.{key}")
            assert_no_secret_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            assert_no_secret_keys(value, f"{path}[{idx}]")


def build_jarvis_overview(
    status: dict[str, Any],
    system_stats: dict[str, Any] | None = None,
    cron_jobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    boards = {board: _collect_board(board) for board in (JARVIS_BOARD, *PRODUCT_BOARDS)}
    memory_vault, memory_source = _memory_vault_status()
    products = []
    sources = [
        _source_ok("Dashboard status", "endpoint", endpoint="/api/status"),
        _source_ok("System stats", "endpoint", endpoint="/api/system/stats"),
        _source_ok("Cron jobs", "endpoint", endpoint="/api/cron/jobs"),
        memory_source,
    ]
    for slug in PRODUCT_BOARDS:
        product, product_sources = _build_product_card(slug, boards[slug])
        products.append(product)
        sources.extend(product_sources)

    components = status.get("components") or {}
    platform_component = components.get("platforms") or {}
    service_health = {
        "overall": status.get("overall") or "unknown",
        "gateway": components.get("gateway") or {"status": "unknown", "state": status.get("gateway_state")},
        "dashboard": components.get("dashboard") or {"status": "unknown"},
        "storage": components.get("storage") or {"status": "unknown"},
        "platforms": platform_component,
        "system": {
            "cpu_percent": (system_stats or {}).get("cpu_percent"),
            "memory_percent": ((system_stats or {}).get("memory") or {}).get("percent"),
            "disk_percent": ((system_stats or {}).get("disk") or {}).get("percent"),
            "uptime_seconds": (system_stats or {}).get("uptime_seconds"),
            "psutil": (system_stats or {}).get("psutil"),
        },
        "cron": _summarize_cron(cron_jobs),
    }

    overview = {
        "generated_at": _now_iso(),
        "refresh_after_seconds": 15,
        "agent_status": {
            "overall": status.get("overall") or "unknown",
            "gateway_state": status.get("gateway_state") or ("running" if status.get("gateway_running") else "stopped"),
            "active_agents": int(status.get("active_agents") or 0),
            "active_sessions": int(status.get("active_sessions") or 0),
            "auth_required": bool(status.get("auth_required")),
            "connected_platforms": int(platform_component.get("connected") or 0),
            "configured_platforms": int(platform_component.get("configured") or len(status.get("gateway_platforms") or {})),
            "profiles": _build_agent_profiles(status, boards),
            "components": components,
        },
        "todos": boards[JARVIS_BOARD].get("open_tasks", []),
        "products": products,
        "memory_vault": memory_vault,
        "service_health": service_health,
        "sources": sources,
    }
    assert_no_secret_keys(overview)
    return overview
