"""Build local context for the scheduled self-improvement audit.

The cron pre-run script imports this module directly, so keep it dependency-light and
safe to run outside a live agent process.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # PyYAML is part of Hermes' normal runtime, but keep a safe fallback.
    import yaml
except Exception:  # pragma: no cover - only used in stripped environments
    yaml = None  # type: ignore[assignment]


_ERROR_RE = re.compile(r"\b(error|failed|exception|traceback|modulenotfounderror|importerror)\b", re.I)
_TRACEBACK_CAUSE_RE = re.compile(
    r"(?P<kind>[A-Za-z_][\w.]*?(?:Error|Exception))(?::\s*(?P<msg>.*))?$"
)


@dataclass(frozen=True)
class AuditWindow:
    now: float
    since: float
    lookback_hours: int


def _load_config(hermes_home: Path) -> dict[str, Any]:
    config_path = hermes_home / "config.yaml"
    if not config_path.exists() or yaml is None:
        return {}
    try:
        loaded = yaml.safe_load(config_path.read_text(errors="replace"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        return {"_config_error": str(exc)}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _state_stats(hermes_home: Path, window: AuditWindow) -> dict[str, Any]:
    db_path = hermes_home / "state.db"
    if not db_path.exists():
        return {"db_path": str(db_path), "exists": False}

    stats: dict[str, Any] = {"db_path": str(db_path), "exists": True}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        with conn:
            if _table_exists(conn, "sessions"):
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS sessions,
                           COALESCE(SUM(message_count), 0) AS messages,
                           COALESCE(SUM(tool_call_count), 0) AS tool_calls
                    FROM sessions
                    WHERE started_at >= ?
                    """,
                    (window.since,),
                ).fetchone()
                stats.update(
                    {
                        "sessions_last_window": int(row["sessions"] or 0),
                        "messages_last_window": int(row["messages"] or 0),
                        "tool_calls_last_window": int(row["tool_calls"] or 0),
                    }
                )
                source_rows = conn.execute(
                    """
                    SELECT COALESCE(source, 'unknown') AS source, COUNT(*) AS count
                    FROM sessions
                    WHERE started_at >= ?
                    GROUP BY source
                    ORDER BY count DESC
                    LIMIT 8
                    """,
                    (window.since,),
                ).fetchall()
                stats["top_sources"] = [dict(row) for row in source_rows]
            if _table_exists(conn, "messages"):
                err_row = conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM messages
                    WHERE timestamp >= ?
                      AND (LOWER(COALESCE(content, '')) LIKE '%error%'
                           OR LOWER(COALESCE(content, '')) LIKE '%failed%'
                           OR LOWER(COALESCE(content, '')) LIKE '%exception%'
                           OR LOWER(COALESCE(content, '')) LIKE '%traceback%')
                    """,
                    (window.since,),
                ).fetchone()
                stats["error_like_messages_last_window"] = int(err_row["count"] or 0)
    except Exception as exc:
        stats["read_error"] = str(exc)
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass
    return stats


def _log_stats(hermes_home: Path, window: AuditWindow, max_tail_bytes: int = 750_000) -> dict[str, Any]:
    logs_dir = hermes_home / "logs"
    if not logs_dir.exists():
        return {"exists": False, "files": []}

    files: list[dict[str, Any]] = []
    causes: Counter[str] = Counter()
    for path in sorted(logs_dir.glob("*.log")):
        try:
            stat = path.stat()
            # Logs are append-only enough for mtime to be a useful cheap window gate.
            recent = stat.st_mtime >= window.since
            text = path.read_text(errors="replace")[-max_tail_bytes:] if recent else ""
        except Exception as exc:
            files.append({"name": path.name, "read_error": str(exc)})
            continue

        tracebacks = text.count("Traceback")
        error_terms = len(_ERROR_RE.findall(text))
        for line in text.splitlines():
            stripped = line.strip()
            if "ModuleNotFoundError" in stripped or "ImportError" in stripped:
                causes[stripped[-220:]] += 1
                continue
            match = _TRACEBACK_CAUSE_RE.search(stripped)
            if match:
                msg = (match.group("msg") or "").strip()
                causes[f"{match.group('kind')}: {msg[:140]}"] += 1
        files.append(
            {
                "name": path.name,
                "recent": recent,
                "tracebacks_tail": tracebacks,
                "error_terms_tail": error_terms,
                "size_bytes": stat.st_size,
                "mtime": int(stat.st_mtime),
            }
        )
    return {"exists": True, "files": files, "top_causes": causes.most_common(8)}


def _skill_dirs(hermes_home: Path) -> list[Path]:
    roots = [hermes_home / "skills"]
    repo_skills = hermes_home / "hermes-agent" / "skills"
    if repo_skills.exists():
        roots.append(repo_skills)
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for skill_md in root.rglob("SKILL.md"):
            if ".git" not in skill_md.parts:
                found.append(skill_md.parent)
    return found


def _stale_skills(hermes_home: Path, stale_days: int, now: float, limit: int = 20) -> list[dict[str, Any]]:
    cutoff = now - stale_days * 86400
    stale: list[dict[str, Any]] = []
    for skill_dir in _skill_dirs(hermes_home):
        skill_md = skill_dir / "SKILL.md"
        try:
            mtime = skill_md.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            stale.append(
                {
                    "name": skill_dir.name,
                    "path": str(skill_md),
                    "days_old": int((now - mtime) // 86400),
                }
            )
    stale.sort(key=lambda item: item["days_old"], reverse=True)
    return stale[:limit]


def _format_context(payload: dict[str, Any]) -> str:
    cfg = payload["config"]
    memory = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}
    self_imp = cfg.get("self_improvement", {}) if isinstance(cfg.get("self_improvement"), dict) else {}
    skills_cfg = cfg.get("skills", {}) if isinstance(cfg.get("skills"), dict) else {}
    state = payload["state"]
    logs = payload["logs"]
    stale = payload["stale_skills"]

    lines = [
        "# Self-improvement audit context",
        f"- generated_at_epoch: {payload['generated_at_epoch']}",
        f"- lookback_hours: {payload['lookback_hours']}",
        "",
        "## Config signals",
        f"- memory.provider: {memory.get('provider', '<unset>')}",
        f"- memory.nudge_interval: {memory.get('nudge_interval', '<unset>')}",
        f"- memory.flush_min_turns: {memory.get('flush_min_turns', '<unset>')}",
        f"- self_improvement.enabled: {self_imp.get('enabled', '<unset>')}",
        f"- self_improvement.deterministic_triggers: {self_imp.get('deterministic_triggers', '<unset>')}",
        f"- self_improvement.recent_learning_overlay: {self_imp.get('recent_learning_overlay', '<unset>')}",
        f"- skills.creation_nudge_interval: {skills_cfg.get('creation_nudge_interval', '<unset>')}",
        "",
        "## Activity window",
        f"- sessions: {state.get('sessions_last_window', 0)}",
        f"- messages: {state.get('messages_last_window', 0)}",
        f"- tool_calls: {state.get('tool_calls_last_window', 0)}",
        f"- error_like_messages: {state.get('error_like_messages_last_window', 0)}",
    ]
    if state.get("read_error"):
        lines.append(f"- state_db_read_error: {state['read_error']}")
    if state.get("top_sources"):
        lines.append("- top_sources: " + json.dumps(state["top_sources"], ensure_ascii=False))

    lines.extend(["", "## Log signals"])
    for item in logs.get("files", []):
        if item.get("recent") or item.get("read_error"):
            lines.append(f"- {item.get('name')}: tracebacks={item.get('tracebacks_tail', 0)} error_terms={item.get('error_terms_tail', 0)} read_error={item.get('read_error', '')}")
    if logs.get("top_causes"):
        lines.append("- top_causes:")
        for cause, count in logs["top_causes"]:
            lines.append(f"  - {count}x {cause}")

    lines.extend(["", f"## Stale skills ({payload['stale_skill_days']}+ days)"])
    if stale:
        for item in stale:
            lines.append(f"- {item['name']}: {item['days_old']}d old ({item['path']})")
    else:
        lines.append("- none")

    # Do not append raw config/session JSON here. This context is delivered into
    # cron model prompts and may be echoed to Discord; keep it to sanitized
    # aggregate signals only.
    return "\n".join(lines)


def build_self_improvement_audit_context(hermes_home: Path | str | None = None) -> str:
    """Return markdown context for the nightly self-improvement audit cron.

    The output is intentionally compact and read-only: config signals, activity
    counts, recent log error clusters, and stale skill candidates.
    """

    home = Path(hermes_home).expanduser() if hermes_home is not None else Path.home() / ".hermes"
    config = _load_config(home)
    self_imp = config.get("self_improvement", {}) if isinstance(config.get("self_improvement"), dict) else {}
    lookback_hours = int(self_imp.get("audit_lookback_hours") or 24)
    stale_skill_days = int(self_imp.get("audit_stale_skill_days") or 90)
    now = time.time()
    window = AuditWindow(now=now, since=now - lookback_hours * 3600, lookback_hours=lookback_hours)
    payload = {
        "generated_at_epoch": int(now),
        "hermes_home": str(home),
        "lookback_hours": lookback_hours,
        "stale_skill_days": stale_skill_days,
        "config": config,
        "state": _state_stats(home, window),
        "logs": _log_stats(home, window),
        "stale_skills": _stale_skills(home, stale_skill_days, now),
    }
    return _format_context(payload)


__all__ = ["build_self_improvement_audit_context"]
