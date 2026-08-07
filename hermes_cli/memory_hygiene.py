"""Built-in memory hygiene helpers.

This module keeps Hermes' built-in MEMORY.md / USER.md deliberately simple in
prompt form while adding operator-grade audit metadata and linting around it.
The sidecar metadata never enters the system prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.memory_tool import ENTRY_DELIMITER

_MEMORY_FILES = {
    "memory": ("MEMORY.md", 2200),
    "user": ("USER.md", 1375),
}
_METADATA_FILE = "metadata.json"

_DATED_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
_TASK_OUTCOME_RE = re.compile(
    r"\b(fixed|shipped|deployed|completed|merged|submitted|phase \d+ done)\b",
    re.IGNORECASE,
)
_ARTIFACT_ID_RE = re.compile(
    r"\b(PR\s*#?\d+|issue\s*#?\d+|commit\s+[0-9a-f]{7,40}|[0-9a-f]{7,40}\b|session_id\b|@session:)\b",
    re.IGNORECASE,
)
_OBSIDIAN_PATH_RE = re.compile(r"\b\d{2}\s+(Projects|People|Decisions|Operating System|Research|Agent Memory)/", re.IGNORECASE)


def memory_dir(hermes_home: Path | None = None) -> Path:
    """Return the profile-scoped built-in memories directory."""
    return (hermes_home or get_hermes_home()) / "memories"


def _read_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    return [entry.strip() for entry in raw.split(ENTRY_DELIMITER) if entry.strip()]


def _entry_id(target: str, content: str) -> str:
    digest = hashlib.sha256(f"{target}\0{content}".encode("utf-8")).hexdigest()
    return digest[:16]


def classify_entry(content: str) -> str:
    """Best-effort category for a compact memory entry."""
    text = content.lower()
    if any(k in text for k in ["comms:", "discord", "telegram", "desktop", "external sends"]):
        return "communication"
    if any(k in text for k in ["style:", "prefers", "dislikes", "expectations", "identity:", "call him"]):
        return "preference"
    if any(k in text for k in ["ori", "tde", "edgy", "mission control", "ventures"]):
        return "project"
    if any(k in text for k in ["hermes/dev", "mac mini", "tailscale", "gateway", "oauth", "vault", "supabase"]):
        return "environment"
    if any(k in text for k in ["bjj", "jiu jitsu", "pilates", "chiba", "primal", "gisdio"]):
        return "health"
    if any(k in text for k in ["finance", "ynab", "net worth", "debt", "equity"]):
        return "finance"
    if any(k in text for k in ["travel", "hotel", "arrival-card", "train", "suitcase"]):
        return "travel"
    if any(k in text for k in ["models:", "gpt", "grok", "moa", "openai-codex"]):
        return "modeling"
    if any(k in text for k in ["skill", "workflow", "verify", "procedure", "source-of-truth"]):
        return "workflow"
    return "general"


def lint_entry(content: str, *, target: str) -> list[dict[str, str]]:
    """Return memory-quality issues for one entry.

    Lints are intentionally conservative: they flag patterns that made Charles'
    memory noisy in practice while leaving compact durable facts alone.
    """
    issues: list[dict[str, str]] = []
    stripped = content.strip()
    lower = stripped.lower()

    if len(stripped) > 280:
        issues.append({
            "code": "too_long",
            "severity": "warn",
            "message": "Entry is long enough that it may belong in Obsidian with a compact pointer here.",
        })

    artifact_match = _ARTIFACT_ID_RE.search(stripped)

    if _DATED_RE.search(stripped) or (_TASK_OUTCOME_RE.search(stripped) and artifact_match):
        issues.append({
            "code": "task_log",
            "severity": "warn",
            "message": "Looks like dated task progress; prefer session_search or an Obsidian project log.",
        })

    if artifact_match:
        issues.append({
            "code": "artifact_id",
            "severity": "warn",
            "message": "Contains PR/issue/commit/session-style IDs that tend to go stale.",
        })

    if "discord" in lower and "no discord" not in lower and "does not use discord" not in lower:
        issues.append({
            "code": "stale_discord",
            "severity": "error",
            "message": "Mentions Discord without the current no-Discord preference.",
        })

    if target == "memory" and _OBSIDIAN_PATH_RE.search(stripped) and len(stripped) > 220:
        issues.append({
            "code": "obsidian_candidate",
            "severity": "info",
            "message": "This looks like long-form Obsidian context; keep only the pointer in Hermes memory.",
        })

    return issues


def collect_entries(hermes_home: Path | None = None) -> list[dict[str, Any]]:
    """Read both built-in memory stores into entry records."""
    base = memory_dir(hermes_home)
    records: list[dict[str, Any]] = []
    for target, (filename, _limit) in _MEMORY_FILES.items():
        for index, content in enumerate(_read_entries(base / filename), start=1):
            records.append({
                "id": _entry_id(target, content),
                "target": target,
                "index": index,
                "content": content,
                "category": classify_entry(content),
                "issues": lint_entry(content, target=target),
                "chars": len(content),
            })
    return records


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            return data
    except Exception:
        pass
    return {"version": 1, "entries": {}}


def refresh_metadata(
    hermes_home: Path | None = None,
    entries: list[dict[str, Any]] | None = None,
    *,
    now: str | None = None,
) -> dict[str, Any]:
    """Refresh the prompt-free metadata sidecar for built-in memory entries."""
    home = hermes_home or get_hermes_home()
    base = memory_dir(home)
    base.mkdir(parents=True, exist_ok=True)
    path = base / _METADATA_FILE
    previous = _load_metadata(path)
    previous_entries = previous.get("entries", {}) if isinstance(previous.get("entries"), dict) else {}
    timestamp = now or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    source_entries = entries if entries is not None else collect_entries(home)

    updated: dict[str, Any] = {
        "version": 1,
        "updated_at": timestamp,
        "note": "Sidecar metadata for built-in Hermes memory. Not injected into the system prompt.",
        "entries": {},
    }
    for record in source_entries:
        entry_id = record.get("id") or _entry_id(record["target"], record["content"])
        prior = previous_entries.get(entry_id, {}) if isinstance(previous_entries, dict) else {}
        updated["entries"][entry_id] = {
            "target": record["target"],
            "index": record.get("index"),
            "category": record.get("category") or classify_entry(record.get("content", "")),
            "chars": len(record.get("content", "")),
            "first_seen": prior.get("first_seen") or timestamp,
            "last_seen": timestamp,
            "issue_codes": [issue["code"] for issue in record.get("issues", [])],
            "canonical_link": prior.get("canonical_link", ""),
            "confidence": prior.get("confidence", ""),
            "review_after": prior.get("review_after", ""),
        }

    path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return updated


def audit_memory(hermes_home: Path | None = None, *, write_metadata: bool = False) -> dict[str, Any]:
    """Audit built-in Hermes memory stores and optionally refresh sidecar metadata."""
    home = hermes_home or get_hermes_home()
    base = memory_dir(home)
    entries = collect_entries(home)
    stores: dict[str, Any] = {}

    for target, (filename, limit) in _MEMORY_FILES.items():
        path = base / filename
        target_entries = [entry for entry in entries if entry["target"] == target]
        content = ENTRY_DELIMITER.join(entry["content"] for entry in target_entries)
        chars = len(content) if target_entries else 0
        stores[target] = {
            "file": str(path),
            "exists": path.exists(),
            "chars": chars,
            "limit": limit,
            "usage_pct": round((chars / limit) * 100, 1) if limit else 0,
            "entry_count": len(target_entries),
            "entries": target_entries,
        }

    total_issues = sum(len(entry["issues"]) for entry in entries)
    report: dict[str, Any] = {
        "hermes_home": str(home),
        "stores": stores,
        "summary": {
            "entry_count": len(entries),
            "total_issues": total_issues,
            "stores_over_80_pct": [name for name, store in stores.items() if store["usage_pct"] > 80],
        },
    }
    if write_metadata:
        metadata = refresh_metadata(home, entries)
        report["metadata_path"] = str(base / _METADATA_FILE)
        report["metadata_entries"] = len(metadata.get("entries", {}))
    return report


def format_audit_report(report: dict[str, Any]) -> str:
    """Format an audit report for terminal output."""
    lines = ["Built-in memory audit", "─" * 40]
    summary = report.get("summary", {})
    lines.append(f"Entries: {summary.get('entry_count', 0)}")
    lines.append(f"Issues:  {summary.get('total_issues', 0)}")
    if summary.get("stores_over_80_pct"):
        lines.append(f"Over 80%: {', '.join(summary['stores_over_80_pct'])}")
    lines.append("")

    for target in ["memory", "user"]:
        if target not in report["stores"]:
            continue
        store = report["stores"][target]
        lines.append(f"{target}: {store['chars']:,}/{store['limit']:,} chars ({store['usage_pct']}%), {store['entry_count']} entries")
        for entry in store["entries"]:
            issue_text = ""
            if entry["issues"]:
                issue_text = " — " + ", ".join(issue["code"] for issue in entry["issues"])
            preview = entry["content"].replace("\n", " ")
            if len(preview) > 96:
                preview = preview[:93] + "..."
            lines.append(f"  {entry['index']:>2}. [{entry['category']}] {preview}{issue_text}")
        lines.append("")

    if report.get("metadata_path"):
        lines.append(f"Metadata refreshed: {report['metadata_path']}")
    return "\n".join(lines).rstrip() + "\n"
