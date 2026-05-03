from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli.config import load_config
from hermes_constants import get_hermes_home

MEMORY_WARN_RATIO = 0.85
MEMORY_CRIT_RATIO = 0.90
USER_WARN_RATIO = 0.80
USER_CRIT_RATIO = 0.90
LOW_TRUST_THRESHOLD = 0.30
LOW_TRUST_LIMIT = 20
SENSITIVE_LIMIT = 20
TOP_ENTITIES_LIMIT = 20
MUTABLE_FACT_BUCKET_LIMIT = 10
MUTABLE_FACT_BUCKETS = (
    ("7-29d", 7, 29),
    ("30-89d", 30, 89),
    ("90+d", 90, None),
)
MUTABLE_KEYWORDS = (
    "current",
    "active",
    "version",
    "port",
    "path",
    "url",
    "endpoint",
    "service",
    "enabled",
    "disabled",
    "config",
    "model",
    "provider",
    "token",
    "key",
    "secret",
    "password",
)

_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.I), "private key block"),
    (
        "api_key",
        re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|gh[pous]_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{20,})\b"),
        "known API/token prefix",
    ),
    (
        "auth_token",
        re.compile(r"\b(?:access[_-]?token|refresh[_-]?token|id[_-]?token|bearer|oauth|session[_-]?token)\b", re.I),
        "token/auth keyword",
    ),
    (
        "password",
        re.compile(r"\b(?:password|passwd|pwd)\b", re.I),
        "password keyword",
    ),
    (
        "secret",
        re.compile(r"\b(?:secret|client[_-]?secret|api[_-]?secret|app[_-]?secret)\b", re.I),
        "secret keyword",
    ),
    (
        "cookie_or_session",
        re.compile(r"\b(?:cookie|session[_-]?id|set-cookie)\b", re.I),
        "cookie/session keyword",
    ),
    (
        "ssh_key",
        re.compile(r"\bssh[-_ ]?key\b", re.I),
        "SSH key keyword",
    ),
]


def _read_config_limits() -> tuple[int, int]:
    config = load_config()
    memory_cfg = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}
    memory_limit = int(memory_cfg.get("memory_char_limit", 2200))
    user_limit = int(memory_cfg.get("user_char_limit", 1375))
    return memory_limit, user_limit


def _sqlite_uri(path: Path) -> str:
    resolved = path.expanduser().resolve()
    return f"{resolved.as_uri()}?mode=ro"


def _open_readonly_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(_sqlite_uri(path), uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1", (table,)
    ).fetchone()
    return row is not None


def _ratio_status(ratio: float, warn: float, crit: float) -> str:
    if ratio >= crit:
        return "critical"
    if ratio >= warn:
        return "warning"
    return "ok"


def _merge_status(current: str, candidate: str) -> str:
    order = {"ok": 0, "warning": 1, "critical": 2}
    return candidate if order.get(candidate, 0) > order.get(current, 0) else current


def _file_stat(path: Path, limit: int, warn_ratio: float, crit_ratio: float) -> dict[str, Any]:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    ratio = (size / limit * 100.0) if limit else 0.0
    status = "critical" if not exists else _ratio_status(size / limit if limit else 0.0, warn_ratio, crit_ratio)
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": size,
        "limit_bytes": limit,
        "percent_used": round(ratio, 2),
        "status": status,
    }


def _count_skill_docs(skills_root: Path) -> int:
    if not skills_root.is_dir():
        return 0
    count = 0
    for md in skills_root.rglob("SKILL.md"):
        md_str = str(md)
        if "/.hub/" in md_str or "/.git/" in md_str:
            continue
        count += 1
    return count


def _classify_sensitive(content: str) -> Optional[tuple[str, str]]:
    for kind, pattern, reason in _SENSITIVE_PATTERNS:
        if pattern.search(content):
            return kind, reason

    # Generic fallback: high-risk credential-looking strings accompanied by a secret keyword.
    lowered = content.lower()
    if any(k in lowered for k in ("token", "secret", "password", "bearer", "apikey", "api key")):
        if re.search(r"\b[A-Za-z0-9/+_=:-]{24,}\b", content):
            return "credential_like", "credential keyword plus long opaque string"
    return None


def _field_keyword_matches(*values: str) -> list[str]:
    haystack = " ".join(v.lower() for v in values if v)
    matches: list[str] = []
    for keyword in MUTABLE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", haystack):
            matches.append(keyword)
    return matches


def _parse_sqlite_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_days_from_timestamp(value: Any, *, now: Optional[datetime] = None) -> Optional[int]:
    ts = _parse_sqlite_timestamp(value)
    if ts is None:
        return None
    now = now or datetime.utcnow()
    delta = now - ts.replace(tzinfo=None)
    return max(delta.days, 0)


def _build_stale_bucket_items(rows: list[sqlite3.Row], bucket: tuple[str, int, Optional[int]], *, now: Optional[datetime] = None) -> dict[str, Any]:
    label, min_days, max_days = bucket
    now = now or datetime.utcnow()
    items: list[dict[str, Any]] = []
    for row in rows:
        age_days = _age_days_from_timestamp(row["updated_at"] or row["created_at"], now=now)
        if age_days is None or age_days < min_days:
            continue
        if max_days is not None and age_days > max_days:
            continue
        keywords = _field_keyword_matches(
            str(row["content"] or ""),
            str(row["category"] or ""),
            str(row["tags"] or ""),
        )
        if not keywords:
            continue
        items.append(
            {
                "fact_id": int(row["fact_id"]),
                "age_days": age_days,
                "keywords": keywords[:4],
                "fields": [
                    field
                    for field, value in (
                        ("content", row["content"]),
                        ("category", row["category"]),
                        ("tags", row["tags"]),
                    )
                    if _field_keyword_matches(str(value or ""))
                ],
            }
        )
    items.sort(key=lambda item: (-item["age_days"], item["fact_id"]))
    return {
        "bucket": label,
        "min_age_days": min_days,
        "max_age_days": max_days,
        "count": len(items),
        "items": items[:MUTABLE_FACT_BUCKET_LIMIT],
    }


def _collect_top_entities_report(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "entities") or not _table_exists(conn, "fact_entities"):
        return {"count": TOP_ENTITIES_LIMIT, "items": []}

    rows = conn.execute(
        """
        SELECT e.entity_id, e.name, COUNT(*) AS fact_count
        FROM entities e
        JOIN fact_entities fe ON fe.entity_id = e.entity_id
        GROUP BY e.entity_id, e.name
        ORDER BY fact_count DESC, e.entity_id ASC
        LIMIT ?
        """,
        (TOP_ENTITIES_LIMIT,),
    ).fetchall()
    return {
        "count": TOP_ENTITIES_LIMIT,
        "items": [
            {
                "entity_id": int(row["entity_id"]),
                "entity": str(row["name"] or ""),
                "fact_count": int(row["fact_count"]),
            }
            for row in rows
        ],
    }


def _collect_stale_mutable_facts_report(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "facts"):
        return {"total": 0, "buckets": []}

    rows = conn.execute(
        "SELECT fact_id, content, category, tags, created_at, updated_at FROM facts ORDER BY fact_id ASC"
    ).fetchall()
    buckets = [
        _build_stale_bucket_items(rows, bucket)
        for bucket in MUTABLE_FACT_BUCKETS
    ]
    return {
        "total": sum(bucket["count"] for bucket in buckets),
        "buckets": buckets,
    }


def _collect_fact_store_report(home: Path, limit: int = SENSITIVE_LIMIT) -> dict[str, Any]:
    db_path = home / "memory_store.db"
    report: dict[str, Any] = {
        "path": str(db_path),
        "exists": db_path.exists(),
        "status": "warning" if not db_path.exists() else "ok",
        "facts": 0,
        "entities": 0,
        "facts_without_entity": 0,
        "facts_without_entity_pct": 0.0,
        "low_trust_threshold": LOW_TRUST_THRESHOLD,
        "low_trust_facts": {"count": 0, "items": []},
        "top_entities": {"count": TOP_ENTITIES_LIMIT, "items": []},
        "stale_mutable_facts": {"total": 0, "buckets": []},
        "sensitive_facts": [],
    }

    if not db_path.exists():
        return report

    try:
        conn = _open_readonly_sqlite(db_path)
    except sqlite3.Error as exc:
        report["status"] = "critical"
        report["error"] = f"sqlite_open_failed: {exc.__class__.__name__}"
        return report

    try:
        if _table_exists(conn, "facts"):
            report["facts"] = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
            low_rows = conn.execute(
                """
                SELECT fact_id, trust_score
                FROM facts
                WHERE trust_score < ?
                ORDER BY trust_score ASC, fact_id ASC
                LIMIT ?
                """,
                (LOW_TRUST_THRESHOLD, LOW_TRUST_LIMIT),
            ).fetchall()
            report["low_trust_facts"] = {
                "count": int(
                    conn.execute(
                        "SELECT COUNT(*) FROM facts WHERE trust_score < ?",
                        (LOW_TRUST_THRESHOLD,),
                    ).fetchone()[0]
                ),
                "items": [
                    {"fact_id": int(row["fact_id"]), "trust_score": float(row["trust_score"])}
                    for row in low_rows
                ],
            }
        else:
            report["status"] = _merge_status(report["status"], "warning")
            report["error"] = "missing_table:facts"
            return report

        if _table_exists(conn, "entities"):
            report["entities"] = int(conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0])
        else:
            report["status"] = _merge_status(report["status"], "warning")

        if _table_exists(conn, "fact_entities"):
            report["facts_without_entity"] = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM facts f
                    LEFT JOIN fact_entities fe ON fe.fact_id = f.fact_id
                    WHERE fe.entity_id IS NULL
                    """
                ).fetchone()[0]
            )
            if report["facts"]:
                report["facts_without_entity_pct"] = round(
                    report["facts_without_entity"] / report["facts"] * 100.0,
                    2,
                )
        else:
            report["status"] = _merge_status(report["status"], "warning")

        report["top_entities"] = _collect_top_entities_report(conn)
        report["stale_mutable_facts"] = _collect_stale_mutable_facts_report(conn)

        if _table_exists(conn, "facts"):
            sensitive: list[dict[str, Any]] = []
            rows = conn.execute(
                "SELECT fact_id, content, category, trust_score FROM facts ORDER BY fact_id ASC"
            )
            for row in rows:
                classified = _classify_sensitive(str(row["content"] or ""))
                if not classified:
                    continue
                kind, reason = classified
                sensitive.append(
                    {
                        "fact_id": int(row["fact_id"]),
                        "type": kind,
                        "reason": reason,
                        "category": str(row["category"] or "general"),
                        "trust_score": float(row["trust_score"]),
                    }
                )
                if len(sensitive) >= limit:
                    break
            report["sensitive_facts"] = sensitive
            if sensitive:
                report["status"] = "critical"

        low_count = report.get("low_trust_facts", {}).get("count", 0)
        if low_count:
            report["status"] = _merge_status(report["status"], "warning")
        if report.get("facts_without_entity_pct", 0.0) >= 40.0:
            report["status"] = _merge_status(report["status"], "warning")

    except sqlite3.Error as exc:
        report["status"] = "critical"
        report["error"] = f"sqlite_query_failed: {exc.__class__.__name__}"
    finally:
        conn.close()

    return report


def _collect_session_search_report(home: Path) -> dict[str, Any]:
    db_path = home / "state.db"
    report: dict[str, Any] = {
        "path": str(db_path),
        "exists": db_path.exists(),
        "status": "warning" if not db_path.exists() else "ok",
        "sessions": 0,
        "messages": 0,
    }

    if not db_path.exists():
        return report

    try:
        conn = _open_readonly_sqlite(db_path)
    except sqlite3.Error as exc:
        report["status"] = "critical"
        report["error"] = f"sqlite_open_failed: {exc.__class__.__name__}"
        return report

    try:
        if _table_exists(conn, "sessions"):
            report["sessions"] = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        else:
            report["status"] = _merge_status(report["status"], "warning")
            report["error"] = "missing_table:sessions"
            return report

        if _table_exists(conn, "messages"):
            report["messages"] = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        else:
            report["status"] = _merge_status(report["status"], "warning")
            report["error"] = "missing_table:messages"
    except sqlite3.Error as exc:
        report["status"] = "critical"
        report["error"] = f"sqlite_query_failed: {exc.__class__.__name__}"
    finally:
        conn.close()

    return report


def _collect_skills_report(home: Path) -> dict[str, Any]:
    skills_root = home / "skills"
    count = _count_skill_docs(skills_root)
    status = "ok" if skills_root.is_dir() else "warning"
    return {
        "path": str(skills_root),
        "exists": skills_root.is_dir(),
        "status": status,
        "skill_docs": count,
    }


def build_memory_doctor_report() -> dict[str, Any]:
    home = get_hermes_home()
    memory_limit, user_limit = _read_config_limits()
    memory_path = home / "memories" / "MEMORY.md"
    user_path = home / "memories" / "USER.md"

    memory = _file_stat(memory_path, memory_limit, MEMORY_WARN_RATIO, MEMORY_CRIT_RATIO)
    user = _file_stat(user_path, user_limit, USER_WARN_RATIO, USER_CRIT_RATIO)
    fact_store = _collect_fact_store_report(home)
    session_search = _collect_session_search_report(home)
    skills = _collect_skills_report(home)

    overall = "ok"
    for section in (memory, user, fact_store, session_search, skills):
        overall = _merge_status(overall, section.get("status", "ok"))

    return {
        "mode": "dry-run",
        "overall_status": overall,
        "memory": memory,
        "user": user,
        "fact_store": fact_store,
        "session_search": session_search,
        "skills": skills,
    }


def format_memory_doctor_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Hermes memory doctor (dry-run)")
    lines.append(f"Overall status: {report.get('overall_status', 'unknown')}")
    lines.append("")

    def _fmt_file(name: str, section: dict[str, Any], warn_note: str = "") -> None:
        lines.append(name)
        lines.append(f"  path: {section.get('path')}")
        lines.append(
            f"  size: {section.get('size_bytes', 0):,} / {section.get('limit_bytes', 0):,} bytes"
        )
        lines.append(f"  used: {section.get('percent_used', 0.0):.2f}%")
        lines.append(f"  status: {section.get('status', 'unknown')}")
        if warn_note:
            lines.append(f"  note: {warn_note}")
        lines.append("")

    _fmt_file("MEMORY.md", report["memory"])
    _fmt_file("USER.md", report["user"])

    fact = report["fact_store"]
    lines.append("fact_store")
    lines.append(f"  path: {fact.get('path')}")
    lines.append(f"  status: {fact.get('status', 'unknown')}")
    if fact.get("error"):
        lines.append(f"  error: {fact['error']}")
    lines.append(f"  facts: {fact.get('facts', 0)}")
    lines.append(f"  entities: {fact.get('entities', 0)}")
    lines.append(
        f"  facts_without_entity: {fact.get('facts_without_entity', 0)} ({fact.get('facts_without_entity_pct', 0.0):.2f}%)"
    )
    low = fact.get("low_trust_facts", {})
    lines.append(f"  low_trust_facts(<{LOW_TRUST_THRESHOLD:.2f}): {low.get('count', 0)}")
    items = low.get("items", [])[:10]
    for item in items:
        lines.append(
            f"    - fact_id={item.get('fact_id')} trust={item.get('trust_score'):.2f}"
        )
    sens = fact.get("sensitive_facts", [])
    lines.append(f"  possible_sensitive_facts: {len(sens)}")
    for item in sens[:10]:
        lines.append(
            f"    - fact_id={item.get('fact_id')} type={item.get('type')} reason={item.get('reason')}"
        )
    lines.append("")

    top_entities = fact.get("top_entities", {})
    lines.append("top_entities")
    lines.append(f"  limit: {top_entities.get('count', TOP_ENTITIES_LIMIT)}")
    lines.append(f"  items: {len(top_entities.get('items', []))}")
    for item in top_entities.get("items", [])[:TOP_ENTITIES_LIMIT]:
        lines.append(
            f"    - entity={item.get('entity')} fact_count={item.get('fact_count')} entity_id={item.get('entity_id')}"
        )
    lines.append("")

    stale = fact.get("stale_mutable_facts", {})
    lines.append("stale_mutable_facts")
    lines.append(f"  total: {stale.get('total', 0)}")
    for bucket in stale.get("buckets", []):
        label = bucket.get("bucket")
        lines.append(
            f"  {label}: {bucket.get('count', 0)} (age {bucket.get('min_age_days')}..{bucket.get('max_age_days') or '+'} days)"
        )
        for item in bucket.get("items", [])[:MUTABLE_FACT_BUCKET_LIMIT]:
            lines.append(
                f"    - fact_id={item.get('fact_id')} age≈{item.get('age_days')}d keywords={','.join(item.get('keywords', []))} fields={','.join(item.get('fields', []))}"
            )
    lines.append("")

    session = report["session_search"]
    lines.append("session_search")
    lines.append(f"  path: {session.get('path')}")
    lines.append(f"  status: {session.get('status', 'unknown')}")
    if session.get("error"):
        lines.append(f"  error: {session['error']}")
    lines.append(f"  sessions: {session.get('sessions', 0)}")
    lines.append(f"  messages: {session.get('messages', 0)}")
    lines.append("")

    skills = report["skills"]
    lines.append("skills")
    lines.append(f"  path: {skills.get('path')}")
    lines.append(f"  status: {skills.get('status', 'unknown')}")
    lines.append(f"  skill_docs: {skills.get('skill_docs', 0)}")
    lines.append("")

    return "\n".join(lines)


def cmd_memory_doctor(args: argparse.Namespace) -> dict[str, Any]:
    report = build_memory_doctor_report()
    if getattr(args, "json", False):
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_memory_doctor_report(report))
    return report
