"""Deterministic daily consolidation for the local subconscious provider."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .store import SubconsciousStore, default_ttl_days

_DURABLE_HINTS = re.compile(
    r"\b(prefers?|remember|do not|don't|always|never|uses?|project|topic|rule|protocol|workflow|approved|decision)\b",
    re.IGNORECASE,
)
_PROCEDURAL_HINTS = re.compile(r"\b(run|command|workflow|steps?|procedure|skill|debug|test|deploy)\b", re.IGNORECASE)
_NOISE_HINTS = re.compile(
    r"("
    r"^\s*[\{\[]|"
    r'"(?:output|exit_code|tool_calls_made|success|error|job_id|targets)"\s*:|'
    r"^\s*(?:#\s*)?Cron Job:|"
    r"<persisted-output>|"
    r"\[CONTEXT COMPACTION|"
    r"Review the conversation above and update|"
    r"Script path must be relative|"
    r"📚\s*skill_view|💻\s*terminal|⏰\s*cronjob|"
    r"/Users/xbr/.+\.(?:jsonl|sqlite3|py|md)"
    r")",
    re.IGNORECASE,
)
_MAX_ITEM_CHARS = 900


def _state_db_path(hermes_home: str | Path) -> Path:
    return Path(hermes_home).expanduser() / "state.db"


def _iter_recent_messages(hermes_home: str | Path, limit: int = 120) -> list[dict[str, Any]]:
    import sqlite3

    db_path = _state_db_path(hermes_home)
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT m.session_id, m.role, m.content, m.timestamp, s.title, s.source AS platform
            FROM messages m
            LEFT JOIN sessions s ON s.id = m.session_id
            WHERE m.content IS NOT NULL AND TRIM(m.content) != ''
            ORDER BY m.timestamp DESC, m.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _compact(text: str, max_chars: int = _MAX_ITEM_CHARS) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _is_noisy_message(msg: dict[str, Any], content: str) -> bool:
    """Skip raw tool/system artifacts so reflection keeps human-useful memory."""
    role = str(msg.get("role") or "").lower()
    if role == "tool":
        return True
    return bool(_NOISE_HINTS.search(content or ""))


def run_consolidation(
    hermes_home: str | Path,
    *,
    db_path: str | Path | None = None,
    run_key: str | None = None,
    dry_run: bool = False,
    message_limit: int = 120,
) -> dict[str, Any]:
    """Extract lightweight structured memories from recent session history.

    This is intentionally deterministic and stdlib-only: no LLM calls, no network,
    and safe for no-agent cron jobs.  It records candidates conservatively so a
    human/agent can later refine them into built-in memory or skills.
    """
    hermes_home = Path(hermes_home).expanduser()
    db_path = Path(db_path) if db_path else hermes_home / "subconscious" / "subconscious.db"
    store = SubconsciousStore(db_path)
    run_key = run_key or datetime.now(timezone.utc).strftime("daily-%Y-%m-%d")
    stats = {"messages_scanned": 0, "working": 0, "episodic": 0, "semantic": 0, "procedural": 0, "dry_run": dry_run}

    try:
        if not dry_run and not store.begin_run(run_key):
            stats["skipped"] = "already_ran"
            return {"success": True, "status": "skipped", "run_key": run_key, "stats": stats}

        messages = list(reversed(_iter_recent_messages(hermes_home, limit=message_limit)))
        stats["messages_scanned"] = len(messages)
        seen: set[tuple[str, str]] = set()

        for msg in messages:
            content = _compact(str(msg.get("content") or ""))
            if not content:
                continue
            if _is_noisy_message(msg, content):
                continue
            session_id = str(msg.get("session_id") or "")
            source = f"session:{session_id}" if session_id else "session"
            lower = content.lower()

            layer = None
            tags = ["reflection"]
            if any(k in lower for k in ("blocker", "pending", "todo", "next step", "unresolved", "жду", "нужно")):
                layer = "working"
                tags.append("open-loop")
            elif _PROCEDURAL_HINTS.search(content):
                layer = "procedural"
                tags.append("workflow")
            elif _DURABLE_HINTS.search(content):
                layer = "semantic"
                tags.append("fact-candidate")
            elif msg.get("role") == "assistant" and len(content) > 120:
                layer = "episodic"
                tags.append("summary-candidate")

            if not layer:
                continue
            key = (layer, content)
            if key in seen:
                continue
            seen.add(key)
            stats[layer] += 1
            if not dry_run:
                store.add_memory(
                    layer,
                    content,
                    summary=_compact(content, 180),
                    tags=tags,
                    source=source,
                    session_id=session_id,
                    confidence=0.45 if layer in {"semantic", "procedural"} else 0.4,
                    ttl_days=default_ttl_days(layer),
                    metadata={"role": msg.get("role"), "platform": msg.get("platform"), "title": msg.get("title")},
                )

        if not dry_run:
            stats["metrics"] = store.capture_metrics_snapshot()
            stats["expire"] = store.expire_stale_memories()
            stats["conflicts"] = store.detect_conflicts()
            store.finish_run(run_key, status="completed", stats=stats)
        return {"success": True, "status": "completed", "run_key": run_key, "stats": stats, "db_path": str(db_path)}
    except Exception as exc:
        if not dry_run:
            store.finish_run(run_key, status="failed", stats=stats, error=str(exc))
        return {"success": False, "status": "failed", "run_key": run_key, "stats": stats, "error": str(exc), "db_path": str(db_path)}
    finally:
        store.close()
