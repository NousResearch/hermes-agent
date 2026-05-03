from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_cli.memory_doctor import build_memory_doctor_report
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

_ARTIFACT_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*§\s*$"),
    re.compile(r"^\s*[-=]{3,}\s*$"),
    re.compile(r"^\s*#\s*(?:temporary|draft|scratch|artifact)\b.*$", re.I),
    re.compile(r"^\s*(?:temporary|temp|draft|scratch)(?:\s+note)?\s*:.*$", re.I),
    re.compile(r"^\s*\[(?:draft|temp|wip|todo)\]\s*$", re.I),
)


def _timestamp_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _action(
    *,
    scope: str,
    target_type: str,
    target_id: str | int,
    proposed_action: str,
    reason: str,
    severity: str,
    confidence: str,
    age_days: int | None = None,
    keywords: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "scope": scope,
        "target_type": target_type,
        "target_id": target_id,
        "proposed_action": proposed_action,
        "reason": reason,
        "severity": severity,
        "confidence": confidence,
    }
    if age_days is not None:
        item["age_days"] = age_days
    if keywords:
        item["keywords"] = keywords
    if evidence:
        item["evidence"] = evidence
    return item


def _severity_for_status(status: str) -> str:
    return {"ok": "low", "warning": "medium", "critical": "high"}.get(status, "low")


def _confidence_for_status(status: str) -> str:
    return {"ok": "high", "warning": "medium", "critical": "medium"}.get(status, "low")


def _ratio_label(percent_used: float, warn: float, crit: float) -> str:
    if percent_used >= crit:
        return "critical"
    if percent_used >= warn:
        return "warning"
    return "ok"


def _is_artifact_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern.match(stripped) for pattern in _ARTIFACT_LINE_PATTERNS)


def _compact_text(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    had_trailing_newline = text.endswith("\n")
    compacted: list[str] = []
    seen_exact: set[str] = set()
    removed_duplicates = 0
    removed_artifacts = 0
    removed_blank_runs = 0

    for line in lines:
        normalized = line.rstrip("\r")
        if _is_artifact_line(normalized):
            removed_artifacts += 1
            continue
        if normalized == "":
            if compacted and compacted[-1] == "":
                removed_duplicates += 1
                continue
            compacted.append("")
            continue
        if normalized in seen_exact:
            removed_duplicates += 1
            continue
        seen_exact.add(normalized)
        compacted.append(normalized)

    while compacted and compacted[0] == "":
        compacted.pop(0)
        removed_blank_runs += 1
    while compacted and compacted[-1] == "":
        compacted.pop()
        removed_blank_runs += 1

    new_text = "\n".join(compacted)
    if had_trailing_newline and new_text:
        new_text += "\n"

    return {
        "text": new_text,
        "changed": new_text != text,
        "removed_duplicates": removed_duplicates,
        "removed_artifacts": removed_artifacts,
        "removed_blank_runs": removed_blank_runs,
        "old_bytes": len(text.encode("utf-8")),
        "new_bytes": len(new_text.encode("utf-8")),
        "old_hash": _hash_text(text),
        "new_hash": _hash_text(new_text),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.bak.{_timestamp_slug()}")


def _checkpoint_path() -> Path:
    return get_hermes_home() / "logs" / "checkpoints"


def _write_checkpoint_record(payload: dict[str, Any]) -> str:
    checkpoint_dir = _checkpoint_path()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"memory-maintain-apply-safe-{_timestamp_slug()}.json"
    checkpoint_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(checkpoint_file)


def _memory_file_report(
    *,
    label: str,
    path: Path,
    percent_used: float,
    warn_ratio: float,
    crit_ratio: float,
) -> dict[str, Any]:
    status = _ratio_label(percent_used / 100.0, warn_ratio, crit_ratio)
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "percent_used": round(percent_used, 2),
        "status": status,
    }


def _apply_safe_to_memory_file(
    *,
    label: str,
    path: Path,
    percent_used: float,
    warn_ratio: float,
    crit_ratio: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "threshold_action": "NOOP",
        "applied": False,
        "reason": "below_safe_threshold",
        "percent_used": round(percent_used, 2),
        "status": _ratio_label(percent_used / 100.0, warn_ratio, crit_ratio),
        "backup_path": "",
        "checkpoint_path": "",
        "before_bytes": 0,
        "after_bytes": 0,
        "removed_duplicates": 0,
        "removed_artifacts": 0,
        "removed_blank_runs": 0,
        "before_hash": "",
        "after_hash": "",
    }

    if percent_used < crit_ratio * 100.0:
        if percent_used >= warn_ratio * 100.0:
            result["reason"] = "warning_only"
        return result

    if not path.exists():
        result["reason"] = "missing_file"
        result["status"] = "critical"
        return result

    original_text = path.read_text(encoding="utf-8")
    compacted = _compact_text(original_text)
    result.update(
        {
            "before_bytes": compacted["old_bytes"],
            "after_bytes": compacted["old_bytes"],
            "before_hash": compacted["old_hash"],
            "after_hash": compacted["old_hash"],
        }
    )

    if not compacted["changed"]:
        result["reason"] = "no_safe_candidates"
        return result

    backup = _backup_path(path)
    shutil.copy2(path, backup)
    _atomic_write_text(path, compacted["text"])

    result.update(
        {
            "threshold_action": "APPLY",
            "applied": True,
            "reason": "safe_compaction_applied",
            "backup_path": str(backup),
            "before_bytes": compacted["old_bytes"],
            "after_bytes": compacted["new_bytes"],
            "removed_duplicates": compacted["removed_duplicates"],
            "removed_artifacts": compacted["removed_artifacts"],
            "removed_blank_runs": compacted["removed_blank_runs"],
            "before_hash": compacted["old_hash"],
            "after_hash": compacted["new_hash"],
        }
    )
    return result


def _collect_top_entities_report(conn: Any) -> dict[str, Any]:
    from sqlite3 import Connection

    if not isinstance(conn, Connection):
        return {"count": TOP_ENTITIES_LIMIT, "items": []}
    has_entities = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name IN ('entities', 'fact_entities') LIMIT 1"
    ).fetchone()
    if not has_entities:
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


def _build_stale_bucket_items(rows: list[Any], bucket: tuple[str, int, int | None], *, now: datetime | None = None) -> dict[str, Any]:
    from sqlite3 import Row

    label, min_days, max_days = bucket
    now = now or datetime.utcnow()
    items: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Row):
            continue
        updated_at = row["updated_at"] or row["created_at"]
        try:
            ts = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
        except ValueError:
            continue
        age_days = max((now - ts.replace(tzinfo=None)).days, 0)
        if age_days < min_days:
            continue
        if max_days is not None and age_days > max_days:
            continue
        haystack = " ".join(str(row[field] or "") for field in ("content", "category", "tags")).lower()
        keywords = [keyword for keyword in MUTABLE_KEYWORDS if re.search(rf"\b{re.escape(keyword)}\b", haystack)]
        if not keywords:
            continue
        items.append(
            {
                "fact_id": int(row["fact_id"]),
                "age_days": age_days,
                "keywords": keywords[:4],
                "fields": [field for field in ("content", "category", "tags") if str(row[field] or "") and any(
                    re.search(rf"\b{re.escape(keyword)}\b", str(row[field]).lower()) for keyword in MUTABLE_KEYWORDS
                )],
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


def _collect_stale_mutable_facts_report(conn: Any) -> dict[str, Any]:
    from sqlite3 import Connection

    if not isinstance(conn, Connection):
        return {"total": 0, "buckets": []}
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts' LIMIT 1").fetchone():
        return {"total": 0, "buckets": []}
    rows = conn.execute(
        "SELECT fact_id, content, category, tags, created_at, updated_at FROM facts ORDER BY fact_id ASC"
    ).fetchall()
    buckets = [_build_stale_bucket_items(rows, bucket) for bucket in MUTABLE_FACT_BUCKETS]
    return {"total": sum(bucket["count"] for bucket in buckets), "buckets": buckets}


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
    except Exception as exc:  # pragma: no cover - defensive
        report["status"] = "critical"
        report["error"] = f"sqlite_open_failed: {exc.__class__.__name__}"
        return report

    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts' LIMIT 1").fetchone():
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
                    conn.execute("SELECT COUNT(*) FROM facts WHERE trust_score < ?", (LOW_TRUST_THRESHOLD,)).fetchone()[0]
                ),
                "items": [
                    {"fact_id": int(row["fact_id"]), "trust_score": float(row["trust_score"])}
                    for row in low_rows
                ],
            }
        else:
            report["status"] = "warning"
            report["error"] = "missing_table:facts"
            return report

        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='entities' LIMIT 1").fetchone():
            report["entities"] = int(conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0])
        else:
            report["status"] = _severity_for_status("warning")

        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='fact_entities' LIMIT 1").fetchone():
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
                report["facts_without_entity_pct"] = round(report["facts_without_entity"] / report["facts"] * 100.0, 2)
        else:
            report["status"] = _severity_for_status("warning")

        report["top_entities"] = _collect_top_entities_report(conn)
        report["stale_mutable_facts"] = _collect_stale_mutable_facts_report(conn)

        sensitive: list[dict[str, Any]] = []
        for row in conn.execute("SELECT fact_id, content, category, trust_score FROM facts ORDER BY fact_id ASC"):
            content = str(row["content"] or "")
            if not content:
                continue
            lowered = content.lower()
            if any(token in lowered for token in ("token", "secret", "password", "bearer", "api key", "apikey")) and re.search(
                r"\b[A-Za-z0-9/+_=:-]{24,}\b", content
            ):
                sensitive.append(
                    {
                        "fact_id": int(row["fact_id"]),
                        "type": "credential_like",
                        "reason": "credential keyword plus long opaque string",
                        "category": str(row["category"] or "general"),
                        "trust_score": float(row["trust_score"]),
                    }
                )
            elif re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", content, re.I):
                sensitive.append(
                    {
                        "fact_id": int(row["fact_id"]),
                        "type": "private_key",
                        "reason": "private key block",
                        "category": str(row["category"] or "general"),
                        "trust_score": float(row["trust_score"]),
                    }
                )
            if len(sensitive) >= limit:
                break
        report["sensitive_facts"] = sensitive
        if sensitive:
            report["status"] = "critical"

        if report.get("low_trust_facts", {}).get("count", 0) and report["status"] == "ok":
            report["status"] = "warning"
        if report.get("facts_without_entity_pct", 0.0) >= 40.0 and report["status"] == "ok":
            report["status"] = "warning"
    except Exception as exc:  # pragma: no cover - defensive
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
    except Exception as exc:  # pragma: no cover - defensive
        report["status"] = "critical"
        report["error"] = f"sqlite_open_failed: {exc.__class__.__name__}"
        return report

    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions' LIMIT 1").fetchone():
            report["sessions"] = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        else:
            report["status"] = "warning"
            report["error"] = "missing_table:sessions"
            return report

        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages' LIMIT 1").fetchone():
            report["messages"] = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        else:
            report["status"] = "warning"
            report["error"] = "missing_table:messages"
    except Exception as exc:  # pragma: no cover - defensive
        report["status"] = "critical"
        report["error"] = f"sqlite_query_failed: {exc.__class__.__name__}"
    finally:
        conn.close()

    return report


def _collect_skills_report(home: Path) -> dict[str, Any]:
    skills_root = home / "skills"
    count = 0
    if skills_root.is_dir():
        for md in skills_root.rglob("SKILL.md"):
            md_str = str(md)
            if "/.hub/" in md_str or "/.git/" in md_str:
                continue
            count += 1
    return {
        "path": str(skills_root),
        "exists": skills_root.is_dir(),
        "status": "ok" if skills_root.is_dir() else "warning",
        "skill_docs": count,
    }


def _build_action_matrix(doctor: dict[str, Any]) -> list[dict[str, Any]]:
    fact_store = doctor.get("fact_store", {})
    memory = doctor.get("memory", {})
    user = doctor.get("user", {})
    session = doctor.get("session_search", {})
    skills = doctor.get("skills", {})

    actions: list[dict[str, Any]] = []

    memory_status = str(memory.get("status", "ok"))
    actions.append(
        _action(
            scope="memory",
            target_type="memory_file",
            target_id="MEMORY.md",
            proposed_action="NOOP" if memory_status == "ok" else "REVALIDATE",
            reason="below_threshold" if memory_status == "ok" else "review_warning",
            severity=_severity_for_status(memory_status),
            confidence=_confidence_for_status(memory_status),
            evidence={"status": memory_status, "percent_used": memory.get("percent_used", 0.0)},
        )
    )

    user_status = str(user.get("status", "ok"))
    actions.append(
        _action(
            scope="user",
            target_type="memory_file",
            target_id="USER.md",
            proposed_action="NOOP" if user_status == "ok" else "REVALIDATE",
            reason="below_threshold" if user_status == "ok" else "review_warning",
            severity=_severity_for_status(user_status),
            confidence=_confidence_for_status(user_status),
            evidence={"status": user_status, "percent_used": user.get("percent_used", 0.0)},
        )
    )

    for item in fact_store.get("sensitive_facts", []):
        actions.append(
            _action(
                scope="fact_store",
                target_type="fact",
                target_id=f"fact:{item.get('fact_id')}",
                proposed_action="REDACT_ON_OUTPUT_ONLY",
                reason="sensitive_fact",
                severity="high",
                confidence="high",
                keywords=[str(item.get("type", ""))],
                evidence={"category": item.get("category"), "trust_score": item.get("trust_score")},
            )
        )

    for bucket in fact_store.get("stale_mutable_facts", {}).get("buckets", []):
        bucket_label = str(bucket.get("bucket", ""))
        for item in bucket.get("items", []):
            actions.append(
                _action(
                    scope="fact_store",
                    target_type="fact",
                    target_id=f"fact:{item.get('fact_id')}",
                    proposed_action="REVALIDATE",
                    reason="stale_mutable_fact",
                    severity="medium",
                    confidence="medium",
                    age_days=item.get("age_days"),
                    keywords=list(item.get("keywords", [])),
                    evidence={"bucket": bucket_label, "fields": item.get("fields", [])},
                )
            )

    if fact_store.get("facts_without_entity", 0):
        actions.append(
            _action(
                scope="fact_store",
                target_type="fact_group",
                target_id="facts_without_entity",
                proposed_action="REVALIDATE",
                reason="missing_entity_links",
                severity="medium",
                confidence="medium",
                evidence={"count": fact_store.get("facts_without_entity", 0)},
            )
        )

    if fact_store.get("low_trust_facts", {}).get("count", 0):
        for item in fact_store.get("low_trust_facts", {}).get("items", []):
            actions.append(
                _action(
                    scope="fact_store",
                    target_type="fact",
                    target_id=f"fact:{item.get('fact_id')}",
                    proposed_action="REVALIDATE",
                    reason="low_trust_fact",
                    severity="medium",
                    confidence="medium",
                    evidence={"trust_score": item.get("trust_score")},
                )
            )

    actions.append(
        _action(
            scope="session_search",
            target_type="session_store",
            target_id="session_search",
            proposed_action="KEEP_IN_SESSION_SEARCH_ONLY",
            reason="historical_record",
            severity="low",
            confidence="high",
            evidence={"sessions": session.get("sessions", 0), "messages": session.get("messages", 0)},
        )
    )
    actions.append(
        _action(
            scope="skills",
            target_type="skill_store",
            target_id="skills",
            proposed_action="NOOP",
            reason="not_analyzed_yet",
            severity="low",
            confidence="high",
            evidence={"skill_docs": skills.get("skill_docs", 0)},
        )
    )
    return actions


def _apply_safe_execution(doctor: dict[str, Any]) -> dict[str, Any]:
    execution: dict[str, Any] = {
        "mode": "apply-safe",
        "applied": False,
        "files": [],
        "checkpoint_path": "",
        "post_doctor_status": "",
    }
    home = get_hermes_home()
    files = (
        ("MEMORY.md", doctor.get("memory", {}), MEMORY_WARN_RATIO, MEMORY_CRIT_RATIO),
        ("USER.md", doctor.get("user", {}), USER_WARN_RATIO, USER_CRIT_RATIO),
    )
    applied_any = False
    file_results: list[dict[str, Any]] = []

    for label, file_report, warn_ratio, crit_ratio in files:
        percent_used = float(file_report.get("percent_used", 0.0) or 0.0)
        file_path_text = str(file_report.get("path", "") or "")
        result = {
            "label": label,
            "path": file_path_text,
            "threshold_action": "NOOP",
            "applied": False,
            "reason": "below_threshold",
            "percent_used": round(percent_used, 2),
            "status": str(file_report.get("status", "ok")),
            "backup_path": "",
            "checkpoint_path": "",
            "before_bytes": int(file_report.get("size_bytes", 0) or 0),
            "after_bytes": int(file_report.get("size_bytes", 0) or 0),
            "removed_duplicates": 0,
            "removed_artifacts": 0,
            "removed_blank_runs": 0,
            "before_hash": "",
            "after_hash": "",
        }
        if percent_used < warn_ratio * 100.0:
            result["reason"] = "below_threshold"
        elif percent_used < crit_ratio * 100.0:
            result["reason"] = "warning_only"
        else:
            file_path = Path(file_path_text)
            if file_path.exists():
                apply_result = _apply_safe_to_memory_file(
                    label=label,
                    path=file_path,
                    percent_used=percent_used,
                    warn_ratio=warn_ratio,
                    crit_ratio=crit_ratio,
                )
                result.update(apply_result)
                if apply_result["applied"]:
                    applied_any = True
            else:
                result["reason"] = "missing_file"
                result["status"] = "critical"
        file_results.append(result)

    execution["files"] = file_results
    execution["applied"] = applied_any

    if applied_any:
        checkpoint_payload = {
            "mode": "apply-safe",
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "files": file_results,
            "pre_doctor_status": doctor.get("overall_status", "unknown"),
        }
        execution["checkpoint_path"] = _write_checkpoint_record(checkpoint_payload)
        post_doctor = build_memory_doctor_report()
        execution["post_doctor_status"] = str(post_doctor.get("overall_status", "unknown"))
        execution["post_doctor"] = {
            "overall_status": post_doctor.get("overall_status", "unknown"),
            "memory": post_doctor.get("memory", {}),
            "user": post_doctor.get("user", {}),
        }
    return execution


def build_memory_maintain_report(apply_safe: bool = False) -> dict[str, Any]:
    doctor = build_memory_doctor_report()
    fact_store = doctor.get("fact_store", {})
    session = doctor.get("session_search", {})
    skills = doctor.get("skills", {})

    actions = _build_action_matrix(doctor)
    counts = Counter(action["proposed_action"] for action in actions)
    overall_status = str(doctor.get("overall_status", "ok"))
    if any(action["severity"] == "high" for action in actions):
        overall_status = "critical"
    elif overall_status == "ok" and any(action["severity"] == "medium" for action in actions):
        overall_status = "warning"

    execution = {
        "mode": "apply-safe" if apply_safe else "dry-run",
        "applied": False,
        "files": [],
        "checkpoint_path": "",
        "post_doctor_status": "",
    }
    post_doctor: dict[str, Any] | None = None
    if apply_safe:
        execution = _apply_safe_execution(doctor)
        if execution.get("applied"):
            post_doctor = execution.get("post_doctor") if isinstance(execution.get("post_doctor"), dict) else None
            if post_doctor:
                overall_status = str(post_doctor.get("overall_status", overall_status))

    summary = {
        "memory": {"status": doctor.get("memory", {}).get("status", "ok"), "percent_used": doctor.get("memory", {}).get("percent_used", 0.0)},
        "user": {"status": doctor.get("user", {}).get("status", "ok"), "percent_used": doctor.get("user", {}).get("percent_used", 0.0)},
        "fact_store": {
            "sensitive_facts": len(fact_store.get("sensitive_facts", [])),
            "stale_mutable_facts": fact_store.get("stale_mutable_facts", {}).get("total", 0),
            "facts_without_entity": fact_store.get("facts_without_entity", 0),
            "low_trust_facts": fact_store.get("low_trust_facts", {}).get("count", 0),
            "top_entities": len(fact_store.get("top_entities", {}).get("items", [])),
        },
        "session_search": {"sessions": session.get("sessions", 0), "messages": session.get("messages", 0)},
        "skills": {"skill_docs": skills.get("skill_docs", 0)},
        "proposal_counts": dict(counts),
        "apply_safe": {
            "enabled": apply_safe,
            "applied": execution.get("applied", False),
            "checkpoint_path": execution.get("checkpoint_path", ""),
            "post_doctor_status": execution.get("post_doctor_status", ""),
            "files": execution.get("files", []),
        },
    }

    warnings = ["dry-run: no changes applied"] if not apply_safe else []
    if fact_store.get("sensitive_facts"):
        warnings.append("sensitive facts are redacted in output only")
    if apply_safe and not execution.get("applied"):
        warnings.append("apply-safe: no eligible changes were applied")
    if apply_safe and execution.get("applied"):
        warnings.append("apply-safe: post-doctor verification completed")

    report: dict[str, Any] = {
        "mode": "apply-safe" if apply_safe else "dry-run",
        "overall_status": overall_status,
        "summary": summary,
        "actions": actions,
        "warnings": warnings,
        "doctor": doctor,
        "execution": execution,
    }
    if post_doctor is not None:
        report["post_doctor"] = post_doctor
    return report


def format_memory_maintain_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    mode = report.get("mode", "dry-run")
    lines.append(f"Hermes memory maintain ({mode})")
    lines.append(f"Overall status: {report.get('overall_status', 'unknown')}")
    lines.append("")

    summary = report.get("summary", {}) if isinstance(report.get("summary", {}), dict) else {}
    lines.append("Summary")
    memory = summary.get("memory", {})
    user = summary.get("user", {})
    fact_store = summary.get("fact_store", {})
    session = summary.get("session_search", {})
    skills = summary.get("skills", {})
    lines.append(f"  MEMORY.md: {memory.get('status', 'unknown')} ({memory.get('percent_used', 0.0):.2f}%)")
    lines.append(f"  USER.md: {user.get('status', 'unknown')} ({user.get('percent_used', 0.0):.2f}%)")
    lines.append(
        "  fact_store: sensitive={sensitive} stale={stale} missing_entity={missing} low_trust={low_trust} top_entities={top_entities}".format(
            sensitive=fact_store.get("sensitive_facts", 0),
            stale=fact_store.get("stale_mutable_facts", 0),
            missing=fact_store.get("facts_without_entity", 0),
            low_trust=fact_store.get("low_trust_facts", 0),
            top_entities=fact_store.get("top_entities", 0),
        )
    )
    lines.append(f"  session_search: sessions={session.get('sessions', 0)} messages={session.get('messages', 0)}")
    lines.append(f"  skills: skill_docs={skills.get('skill_docs', 0)}")
    lines.append("")

    lines.append("Actions")
    for action in report.get("actions", []):
        extra: list[str] = []
        if action.get("age_days") is not None:
            extra.append(f"age≈{action['age_days']}d")
        if action.get("keywords"):
            extra.append(f"keywords={','.join(action['keywords'])}")
        evidence = action.get("evidence")
        if evidence:
            extra.append(f"evidence={evidence}")
        extra_text = f" [{' ; '.join(extra)}]" if extra else ""
        lines.append(
            f"  - scope={action.get('scope')} target={action.get('target_id')} action={action.get('proposed_action')} reason={action.get('reason')} severity={action.get('severity')} confidence={action.get('confidence')}{extra_text}"
        )
    lines.append("")

    execution = report.get("execution", {}) if isinstance(report.get("execution", {}), dict) else {}
    if mode == "apply-safe":
        lines.append("Apply-safe execution")
        lines.append(f"  applied: {execution.get('applied', False)}")
        for item in execution.get("files", []):
            lines.append(
                "  - {label}: {threshold_action} reason={reason} status={status} before={before_bytes} after={after_bytes} duplicates={removed_duplicates} artifacts={removed_artifacts}".format(
                    label=item.get("label"),
                    threshold_action=item.get("threshold_action"),
                    reason=item.get("reason"),
                    status=item.get("status"),
                    before_bytes=item.get("before_bytes", 0),
                    after_bytes=item.get("after_bytes", 0),
                    removed_duplicates=item.get("removed_duplicates", 0),
                    removed_artifacts=item.get("removed_artifacts", 0),
                )
            )
            if item.get("backup_path"):
                lines.append(f"      backup: {item.get('backup_path')}")
            if item.get("checkpoint_path"):
                lines.append(f"      checkpoint: {item.get('checkpoint_path')}")
        if execution.get("checkpoint_path"):
            lines.append(f"  checkpoint_record: {execution.get('checkpoint_path')}")
        if execution.get("post_doctor_status"):
            lines.append(f"  post_doctor_status: {execution.get('post_doctor_status')}")
        lines.append("")

    lines.append("Best next action")
    best = next((a for a in report.get("actions", []) if a.get("severity") == "high"), None)
    if best is None:
        best = next((a for a in report.get("actions", []) if a.get("proposed_action") != "NOOP"), None)
    if best:
        lines.append(f"  Review {best.get('scope')} proposal for {best.get('target_id')} ({best.get('proposed_action')})")
    else:
        lines.append("  No changes needed")
    lines.append("")

    if report.get("warnings"):
        lines.append("Warnings")
        for warning in report.get("warnings", []):
            lines.append(f"  - {warning}")
        lines.append("")

    lines.append("No changes applied." if not execution.get("applied") else "Safe changes applied.")
    return "\n".join(lines)


def cmd_memory_maintain(args: argparse.Namespace) -> dict[str, Any]:
    apply_safe = bool(getattr(args, "apply_safe", False))
    report = build_memory_maintain_report(apply_safe=apply_safe)
    if getattr(args, "json", False):
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_memory_maintain_report(report))
    return report
