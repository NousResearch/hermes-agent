"""Append-only project continuity records.

These records preserve compact project-level breadcrumbs across physical chat
rotation without storing full transcripts or raw tool output.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home


def _project_key(packet: Dict[str, Any]) -> str:
    return str(packet.get("git_repo_root") or packet.get("cwd") or "").strip()


def _safe_project_filename(project_key: str) -> str:
    normalized = project_key.replace("\\", "/").strip().strip("/")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
    safe = safe.replace(":", "_").strip("._-")
    return safe[:160] or "unknown-project"


def append_project_continuity_record(
    packet: Dict[str, Any],
    *,
    event: str = "compression",
    summary: str = "",
    reports_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Append a compact continuity record for a project-bound packet.

    Returns the JSONL path written, or ``None`` when the packet has no project
    key.  The record intentionally contains only metadata and curated candidate
    lists, not raw conversation text.
    """

    project_key = _project_key(packet)
    if not project_key:
        return None

    candidates = packet.get("durable_candidates") or {}
    base_dir = reports_dir or (get_hermes_home() / "reports" / "project-continuity")
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{_safe_project_filename(project_key)}.jsonl"

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "summary": summary,
        "session_id": packet.get("session_id", ""),
        "old_session_id": packet.get("old_session_id", ""),
        "lineage_root_id": packet.get("lineage_root_id", ""),
        "platform": packet.get("platform", ""),
        "compression_count": packet.get("compression_count", 0),
        "project": {
            "cwd": packet.get("cwd", ""),
            "git_repo_root": packet.get("git_repo_root", ""),
            "git_branch": packet.get("git_branch", ""),
        },
        "title": packet.get("title", ""),
        "project_facts": list(candidates.get("project_facts") or []),
        "decisions": list(candidates.get("decisions") or []),
        "open_threads": list(candidates.get("open_threads") or []),
        "procedures": list(candidates.get("procedures") or []),
        "artifacts": list(candidates.get("artifacts") or []),
        "do_not_carry": list(candidates.get("do_not_carry") or []),
    }

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path
