"""Import OpenClaw-style context files into formal Memory v2 core records.

This module is intentionally deterministic and local-only. It reads profile
context files from a source Hermes home, writes formal ``CoreMemoryRecord`` YAML
records under a target Hermes home's ``memory_v2/core/`` tree, and preserves one
canonical ``SourceRef`` per imported file.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .schemas import CoreMemoryRecord, SourceRef, utc_now_iso
from .store import MemoryV2Store

CONTEXT_FILES: Tuple[str, ...] = (
    "SOUL.md",
    "IDENTITY.md",
    "AGENTS.md",
    "USER.md",
    "TOOLS.md",
    "MEMORY.md",
    "memories/USER.md",
    "memories/MEMORY.md",
)

_CATEGORY_BY_FILE = {
    "SOUL.md": "assistant_identity",
    "IDENTITY.md": "assistant_identity",
    "AGENTS.md": "operating_rule",
    "USER.md": "user",
    "TOOLS.md": "environment",
    "MEMORY.md": "operating_rule",
    "memories/USER.md": "user",
    "memories/MEMORY.md": "operating_rule",
}

_SOURCE_ID_BY_STEM = {
    "SOUL.md": "source_context_soul",
    "IDENTITY.md": "source_context_identity",
    "AGENTS.md": "source_context_agents",
    "USER.md": "source_context_user",
    "TOOLS.md": "source_context_tools",
    "MEMORY.md": "source_context_memory",
    "memories/USER.md": "source_context_memories_user",
    "memories/MEMORY.md": "source_context_memories_memory",
}

_PRIORITY_BY_CATEGORY = {
    "user": 0.95,
    "assistant_identity": 0.9,
    "operating_rule": 0.86,
    "environment": 0.82,
}


def import_core_memory_from_context_files(
    *,
    target_hermes_home: str | Path,
    source_hermes_home: str | Path | None = None,
    context_files: Iterable[str] = CONTEXT_FILES,
    max_records_per_file: int = 12,
) -> Dict[str, Any]:
    """Import context files into a target profile's Memory v2 core store.

    Args:
        target_hermes_home: Hermes home/profile that receives ``memory_v2`` data.
        source_hermes_home: Hermes home/profile to read context files from. Defaults
            to target, which is useful after copying files into a test profile.
        context_files: Relative context file paths to import.
        max_records_per_file: Hard cap to keep the core prompt small.

    Returns:
        JSON-serializable import report.
    """

    target_home = Path(target_hermes_home).expanduser().resolve()
    source_home = Path(source_hermes_home or target_home).expanduser().resolve()
    if max_records_per_file < 1:
        raise ValueError("max_records_per_file must be at least 1")

    store = MemoryV2Store(target_home / "memory_v2")
    store.initialize()

    imported_files: List[str] = []
    skipped_files: List[str] = []
    record_ids: List[str] = []
    source_ids: List[str] = []

    for rel_path in context_files:
        rel = str(rel_path).strip().replace("\\", "/")
        if not rel:
            continue
        path = (source_home / rel).resolve()
        if not _is_relative_to(path, source_home) or not path.is_file():
            skipped_files.append(rel)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        statements = _extract_statements(text, rel, limit=max_records_per_file)
        if not statements:
            skipped_files.append(rel)
            continue

        source = _source_ref_for_file(rel, source_home, text)
        store.write_source_ref(source)
        source_ids.append(source.id)
        category = _category_for_file(rel)
        priority = _PRIORITY_BY_CATEGORY[category]
        for statement in statements:
            record = CoreMemoryRecord(
                id=_record_id(category, statement),
                category=category,
                statement=statement,
                priority=priority,
                confidence=0.9,
                updated_at=utc_now_iso(),
                source_refs=[source.id],
                tags=["imported_context", _safe_slug(rel.rsplit("/", 1)[-1].removesuffix(".md"))],
            )
            store.write_core_memory_record(record)
            record_ids.append(record.id)
        imported_files.append(rel)

    return {
        "success": True,
        "target_hermes_home": str(target_home),
        "source_hermes_home": str(source_home),
        "imported_files": imported_files,
        "skipped_files": skipped_files,
        "sources_written": len(set(source_ids)),
        "records_written": len(set(record_ids)),
        "source_ids": sorted(set(source_ids)),
        "record_ids": sorted(set(record_ids)),
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _category_for_file(rel_path: str) -> str:
    return _CATEGORY_BY_FILE.get(rel_path, "operating_rule")


def _source_ref_for_file(rel_path: str, source_home: Path, text: str) -> SourceRef:
    source_id = _SOURCE_ID_BY_STEM.get(rel_path) or f"source_context_{_safe_slug(rel_path)}"
    title = f"Profile context: {rel_path}"
    quote = _first_nonempty_line(text)[:500]
    return SourceRef(
        id=source_id,
        type="file",
        uri=f"file:{rel_path}",
        title=title,
        observed_at=utc_now_iso(),
        quote=quote or None,
    )


def _extract_statements(text: str, rel_path: str, *, limit: int) -> List[str]:
    normalized = text.replace("\r\n", "\n")
    candidates: List[str] = []
    if "§" in normalized:
        candidates.extend(part.strip() for part in normalized.split("§"))
    else:
        for line in normalized.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("[summary") or stripped.startswith("[... truncated"):
                continue
            if stripped in {"---", "</file>"}:
                continue
            if stripped.startswith("-"):
                candidates.append(stripped.lstrip("- ").strip())
            elif _looks_like_plain_statement(stripped):
                candidates.append(stripped)

    statements: List[str] = []
    seen = set()
    for candidate in candidates:
        statement = _clean_statement(candidate)
        if not statement or len(statement) < 12:
            continue
        key = statement.lower()
        if key in seen:
            continue
        seen.add(key)
        statements.append(statement)
        if len(statements) >= limit:
            break
    return statements


def _looks_like_plain_statement(text: str) -> bool:
    if len(text) > 500:
        return False
    lowered = text.lower()
    return any(
        lowered.startswith(prefix)
        for prefix in (
            "prefers ",
            "dylan ",
            "hermes ",
            "be ",
            "do ",
            "do not ",
            "never ",
            "use ",
            "treat ",
            "when ",
            "current ",
            "host",
            "ffmpeg",
        )
    )


def _clean_statement(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = text.strip(" -\t")
    if text.startswith("§"):
        text = text.lstrip("§ ")
    return text[:500]


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("# ").strip()
        if stripped:
            return stripped
    return ""


def _record_id(category: str, statement: str) -> str:
    digest = hashlib.sha256(f"{category}\n{statement}".encode("utf-8")).hexdigest()[:12]
    return f"core_{_safe_slug(category)}_{digest}"


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "context"
