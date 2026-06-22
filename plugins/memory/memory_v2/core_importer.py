"""Import profile context files into formal Memory v2 core records.

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
    core_budget: int | None = None,
    category_minimums: Dict[str, int] | None = None,
    archive_pruned: bool = False,
) -> Dict[str, Any]:
    """Import context files into a target profile's Memory v2 core store.

    Args:
        target_hermes_home: Hermes home/profile that receives ``memory_v2`` data.
        source_hermes_home: Hermes home/profile to read context files from. Defaults
            to target, which is useful after copying files into a test profile.
        context_files: Relative context file paths to import.
        max_records_per_file: Hard cap for extraction from each file before global pruning.
        core_budget: Optional global prompt-core budget. When set, imported candidates are scored and only the top records are written as core.
        category_minimums: Optional per-category minimum counts protected during pruning when enough candidates exist.
        archive_pruned: If true, write pruned records to ``inbox/core_import_pruned.jsonl`` as archive-only evidence.

    Returns:
        JSON-serializable import report.
    """

    target_home = Path(target_hermes_home).expanduser().resolve()
    source_home = Path(source_hermes_home or target_home).expanduser().resolve()
    if max_records_per_file < 1:
        raise ValueError("max_records_per_file must be at least 1")
    if core_budget is not None and core_budget < 1:
        raise ValueError("core_budget must be at least 1 when provided")

    store = MemoryV2Store(target_home / "memory_v2")
    store.initialize()

    imported_files: List[str] = []
    skipped_files: List[str] = []
    source_ids: List[str] = []
    candidates: List[CoreMemoryRecord] = []

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
        for statement in statements:
            priority = _score_statement(statement, category)
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
            candidates.append(record)
        imported_files.append(rel)

    selected, pruned, pruned_reasons = _select_core_records(
        candidates,
        core_budget=core_budget,
        category_minimums=category_minimums or {},
    )
    import_categories = {getattr(record.category, "value", str(record.category)) for record in candidates}
    _rewrite_core_categories(store, selected, import_categories=import_categories)
    if archive_pruned:
        for record in pruned:
            store._append_jsonl(
                store.inbox_dir / "core_import_pruned.jsonl",
                {
                    "id": record.id,
                    "decision": "archive_only",
                    "reason": pruned_reasons[record.id],
                    "record": record.to_dict(),
                    "created_at": utc_now_iso(),
                },
            )

    record_ids = [record.id for record in selected]
    return {
        "success": True,
        "target_hermes_home": str(target_home),
        "source_hermes_home": str(source_home),
        "imported_files": imported_files,
        "skipped_files": skipped_files,
        "sources_written": len(set(source_ids)),
        "records_seen": len(candidates),
        "records_written": len(set(record_ids)),
        "records_pruned": len(pruned),
        "archive_only_written": len(pruned) if archive_pruned else 0,
        "pruned_reasons": pruned_reasons,
        "source_ids": sorted(set(source_ids)),
        "record_ids": sorted(set(record_ids)),
    }



def _select_core_records(
    candidates: List[CoreMemoryRecord],
    *,
    core_budget: int | None,
    category_minimums: Dict[str, int],
) -> tuple[List[CoreMemoryRecord], List[CoreMemoryRecord], Dict[str, str]]:
    deduped: Dict[str, CoreMemoryRecord] = {}
    for record in candidates:
        existing = deduped.get(record.id)
        if existing is None or record.priority > existing.priority:
            deduped[record.id] = record
    records = list(deduped.values())
    if core_budget is None or len(records) <= core_budget:
        return sorted(records, key=_record_sort_key), [], {}

    selected_ids: set[str] = set()
    for category, minimum in category_minimums.items():
        if minimum <= 0:
            continue
        category_records = [record for record in records if record.category.value == category]
        for record in sorted(category_records, key=_record_sort_key)[:minimum]:
            if len(selected_ids) < core_budget:
                selected_ids.add(record.id)
    for record in sorted(records, key=_record_sort_key):
        if len(selected_ids) >= core_budget:
            break
        selected_ids.add(record.id)

    selected = [record for record in records if record.id in selected_ids]
    pruned = [record for record in records if record.id not in selected_ids]
    pruned_reasons = {
        record.id: f"score={record.priority:.3f}; outside core_budget={core_budget}; archived_only"
        for record in pruned
    }
    return sorted(selected, key=_record_sort_key), sorted(pruned, key=_record_sort_key), pruned_reasons


def _record_sort_key(record: CoreMemoryRecord) -> tuple[float, str, str]:
    return (-record.priority, record.category.value, record.id)


def _rewrite_core_categories(store: MemoryV2Store, records: List[CoreMemoryRecord], *, import_categories: set[str]) -> None:
    by_category: Dict[str, List[CoreMemoryRecord]] = {}
    for record in records:
        by_category.setdefault(record.category.value, []).append(record)
    for category in import_categories:
        preserved = [
            record
            for record in store.list_core_memory_records(category=category)
            if "imported_context" not in record.tags
        ]
        category_records = sorted([*preserved, *by_category.get(category, [])], key=_record_sort_key)
        path = store._core_category_path(category)
        if category_records:
            store._atomic_write_yaml(
                path,
                {
                    "version": 1,
                    "category": category,
                    "records": [record.to_dict() for record in category_records],
                },
            )
        elif path.exists():
            path.unlink()


def _score_statement(statement: str, category: str) -> float:
    text = statement.lower()
    score = _PRIORITY_BY_CATEGORY.get(category, 0.75)
    high_signal_markers = {
        "prefers": 0.08,
        "wants": 0.08,
        "source-grounded": 0.07,
        "low-compute": 0.07,
        "gated": 0.06,
        "spec/eval": 0.06,
        "memory": 0.04,
        "voice-to-voice": 0.06,
        "discord": 0.04,
        "truth": 0.05,
        "private": 0.05,
        "external actions": 0.05,
        "ffmpeg": 0.04,
        "installed": 0.03,
    }
    low_signal_markers = {
        "temporary": -0.2,
        "yesterday": -0.18,
        "lunch": -0.18,
        "random": -0.15,
        "#general": -0.18,
        "channel id": -0.08,
        "ids that": -0.08,
    }
    for marker, delta in high_signal_markers.items():
        if marker in text:
            score += delta
    for marker, delta in low_signal_markers.items():
        if marker in text:
            score += delta
    if text.startswith("#"):
        score -= 0.25
    if len(statement) > 280:
        score -= 0.05
    if len(statement) < 28:
        score -= 0.04
    return max(0.0, min(1.0, round(score, 3)))

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
    category = _category_for_file(rel_path)
    statements.sort(key=lambda statement: (-_score_statement(statement, category), statement.lower()))
    return statements[:limit]


def _looks_like_plain_statement(text: str) -> bool:
    if len(text) > 500:
        return False
    lowered = text.lower()
    return any(
        lowered.startswith(prefix)
        for prefix in (
            "prefers ",
            "user ",
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
            "tool ",
            "example-cli",
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
