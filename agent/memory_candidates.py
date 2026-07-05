"""Governed memory candidate queue and exact memory-wiki retrieval helpers."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from hermes_constants import get_hermes_home
from agent.memory_wiki import build_memory_wiki_index, select_memory_context

MemoryTarget = Literal["memory", "user"]


@dataclass(frozen=True)
class MemoryCandidate:
    candidate_id: str
    target: MemoryTarget
    content: str
    status: str
    path: Path
    payload: dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _queue_dir() -> Path:
    return get_hermes_home() / "memory-candidates"


def _candidate_path(candidate_id: str) -> Path:
    safe = str(candidate_id)
    if not re.fullmatch(r"mem-\d{8}T\d{6}-\d{6}", safe):
        raise ValueError("invalid memory candidate id")
    return _queue_dir() / f"{safe}.json"


def _target_file(target: str) -> Path:
    if target == "memory":
        return get_hermes_home() / "memories" / "MEMORY.md"
    if target == "user":
        return get_hermes_home() / "memories" / "USER.md"
    raise ValueError("target must be 'memory' or 'user'")


def _new_id() -> str:
    return datetime.now(timezone.utc).strftime("mem-%Y%m%dT%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_payload(candidate_id: str) -> tuple[Path, dict[str, Any]]:
    path = _candidate_path(candidate_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload


def _shape(path: Path, payload: dict[str, Any]) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=str(payload["candidate_id"]),
        target=payload["target"],
        content=str(payload["content"]),
        status=str(payload.get("status", "staged")),
        path=path,
        payload=payload,
    )


def stage_memory_candidate(
    *,
    target: MemoryTarget,
    content: str,
    source: dict[str, Any] | None = None,
    rationale: str = "",
) -> MemoryCandidate:
    """Stage a memory write candidate without mutating active memory files."""
    content_clean = str(content).strip()
    if "\n§\n" in content_clean:
        raise ValueError("content cannot contain memory entry separator")
    _target_file(target)  # validates target
    candidate_id = _new_id()
    path = _candidate_path(candidate_id)
    payload = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "target": target,
        "content": content_clean,
        "status": "staged",
        "created_at": _now_iso(),
        "source": dict(source or {}),
        "rationale": rationale,
    }
    _write_payload(path, payload)
    return _shape(path, payload)


def load_memory_candidate(candidate_id: str) -> MemoryCandidate:
    path, payload = _load_payload(candidate_id)
    return _shape(path, payload)


def promote_memory_candidate(candidate_id: str) -> MemoryCandidate:
    """Append a staged memory candidate to MEMORY.md or USER.md."""
    candidate = load_memory_candidate(candidate_id)
    if candidate.status == "promoted":
        return candidate
    if candidate.status != "staged":
        raise ValueError("candidate is not staged")
    target_path = _target_file(candidate.target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    parts = [part.strip() for part in existing.split("\n§\n") if part.strip()]
    appended = candidate.content not in parts
    appended_suffix = ""
    if appended:
        sep = "\n§\n" if existing.strip() else ""
        appended_suffix = sep + candidate.content
        target_path.write_text(existing + appended_suffix, encoding="utf-8")
    payload = dict(candidate.payload)
    payload["status"] = "promoted"
    payload["promoted_at"] = _now_iso()
    payload["promote_action"] = "appended" if appended else "noop_duplicate"
    payload["promoted_suffix"] = appended_suffix
    _write_payload(candidate.path, payload)
    return _shape(candidate.path, payload)


def rollback_memory_candidate(candidate_id: str) -> MemoryCandidate:
    """Rollback a promoted append when the candidate is still the last entry."""
    candidate = load_memory_candidate(candidate_id)
    if candidate.status == "rolled_back":
        return candidate
    if candidate.status != "promoted":
        raise ValueError("candidate is not promoted")
    target_path = _target_file(candidate.target)
    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    suffix = str(candidate.payload.get("promoted_suffix") or "")
    if not suffix:
        payload = dict(candidate.payload)
        payload["status"] = "rolled_back"
        payload["rolled_back_at"] = _now_iso()
        _write_payload(candidate.path, payload)
        return _shape(candidate.path, payload)
    if not existing.endswith(suffix):
        raise ValueError("cannot rollback: promoted content is not the last memory entry")
    target_path.write_text(existing[: -len(suffix)], encoding="utf-8")
    payload = dict(candidate.payload)
    payload["status"] = "rolled_back"
    payload["rolled_back_at"] = _now_iso()
    _write_payload(candidate.path, payload)
    return _shape(candidate.path, payload)


def reject_memory_candidate(candidate_id: str, *, reason: str = "") -> MemoryCandidate:
    candidate = load_memory_candidate(candidate_id)
    payload = dict(candidate.payload)
    payload["status"] = "rejected"
    payload["rejected_at"] = _now_iso()
    payload["rejection_reason"] = str(reason)
    _write_payload(candidate.path, payload)
    return _shape(candidate.path, payload)


def memory_wiki_find(query: str, *, max_chars: int = 4000) -> list[dict[str, Any]]:
    """Find relevant memory-wiki entries by query."""
    return list(select_memory_context(query, max_chars=max_chars).get("entries") or [])


def memory_wiki_read(entry_id: str) -> dict[str, Any]:
    """Read one exact memory-wiki entry by stable id."""
    for entry in build_memory_wiki_index().get("entries") or []:
        if entry.get("id") == entry_id:
            return entry
    raise KeyError(entry_id)


def memory_wiki_retrieve(query: str, *, max_chars: int = 1200) -> dict[str, Any]:
    """Retrieve a fenced, budgeted memory-wiki context block for a query."""
    return select_memory_context(query, max_chars=max_chars)


def memory_wiki_grep(pattern: str) -> list[dict[str, Any]]:
    """Case-insensitive regex grep across memory-wiki entries."""
    rx = re.compile(pattern, re.IGNORECASE)
    return [entry for entry in build_memory_wiki_index().get("entries") or [] if rx.search(str(entry.get("text") or ""))]
