#!/usr/bin/env python3
"""Dry-run/apply cleanup for duplicate priority-tagged Hindsight memories."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    _memory_context_identifier,
    _dedupe_key_for_result,
    _memory_document_id,
    _memory_recency,
    _memory_result_id,
    _memory_result_tags,
    _memory_result_text,
    _normalize_retain_tags,
    _utc_timestamp,
)


DEFAULT_PRIORITY_TAGS = ["persona", "etat-interne", "pouls", "persona-state"]
DEFAULT_DEDUPE_CONTEXT_TAGS = ["persona-state", "etat-interne", "pouls", "persona"]


def _record_from_mapping(item: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**item)


def _cleanup_dedupe_key_for_result(record: Any, context_tags: list[str]) -> str:
    tags = _memory_result_tags(record)
    matched_tag = next((tag for tag in context_tags if tag in tags), "")
    if not matched_tag:
        return ""
    context_id = _memory_context_identifier(record)
    if context_id:
        return f"{matched_tag}:context:{context_id}"
    return _dedupe_key_for_result(record, context_tags)


def load_records(path: Path | None, priority_tags: list[str]) -> list[Any]:
    if path is None:
        provider = HindsightMemoryProvider()
        provider.initialize(session_id="hindsight-cleanup-persona-duplicates")
        query = "persona etat-interne pouls inner_state"
        recall_kwargs = provider._build_recall_kwargs(query)
        recall_kwargs["tags"] = priority_tags
        recall_kwargs["tags_match"] = "any"
        response = provider._run_hindsight_operation(lambda client: client.arecall(**recall_kwargs))
        return list(response.results or [])
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--input-json must contain a JSON array")
    return [_record_from_mapping(item) for item in data if isinstance(item, dict)]


def choose_duplicate_actions(records: list[Any], context_tags: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[Any, int]]] = {}
    for index, record in enumerate(records):
        key = _cleanup_dedupe_key_for_result(record, context_tags)
        if key:
            groups.setdefault(key, []).append((record, index))

    actions: list[dict[str, Any]] = []
    for key, group in groups.items():
        if len(group) < 2:
            continue

        def _sort_key(item: tuple[Any, int]) -> tuple[int, str, int]:
            record, index = item
            recency = _memory_recency(record)
            return (1 if recency else 0, recency.isoformat() if recency else "", -index)

        canonical, _canonical_index = max(group, key=_sort_key)
        canonical_memory_id = _memory_result_id(canonical)
        canonical_document_id = _memory_document_id(canonical)
        for duplicate, _index in group:
            if duplicate is canonical:
                continue
            actions.append(
                {
                    "status": "superseded",
                    "memory_id": _memory_result_id(duplicate) or None,
                    "document_id": _memory_document_id(duplicate) or None,
                    "query": _memory_result_text(duplicate)[:300],
                    "reason": "duplicate priority-tagged memory superseded by newer canonical memory",
                    "correction": _memory_result_text(canonical),
                    "tags": _normalize_retain_tags(["hygiene-memoire", *_memory_result_tags(duplicate)]),
                    "superseded_by_memory_id": canonical_memory_id or None,
                    "superseded_by_document_id": canonical_document_id or None,
                    "dedupe_key": key,
                    "created_at": _utc_timestamp(),
                }
            )
    return actions


def _existing_hygiene_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            keys.add((
                str(item.get("status") or ""),
                str(item.get("memory_id") or ""),
                str(item.get("document_id") or ""),
            ))
    return keys


@contextlib.contextmanager
def _hygiene_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


def apply_hygiene_actions(path: Path, actions: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _hygiene_lock(path):
        backup = path.with_suffix(path.suffix + f".bak.{_utc_timestamp().replace(':', '')}")
        if path.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        existing_keys = _existing_hygiene_keys(path)
        new_items = [
            action
            for action in actions
            if (
                str(action.get("status") or ""),
                str(action.get("memory_id") or ""),
                str(action.get("document_id") or ""),
            ) not in existing_keys
        ]
        if not new_items:
            return 0
        with path.open("a", encoding="utf-8") as f:
            for item in new_items:
                f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        return len(new_items)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, help="Read fake/fixture Hindsight records from a JSON array")
    parser.add_argument("--hygiene-path", type=Path, default=Path(os.environ.get("HERMES_MEMORY_HYGIENE_PATH", "/workspace/projects/persona/memory_hygiene.jsonl")))
    parser.add_argument("--priority-tags", default=",".join(DEFAULT_PRIORITY_TAGS))
    parser.add_argument("--dedupe-context-tags", default=",".join(DEFAULT_DEDUPE_CONTEXT_TAGS))
    parser.add_argument("--apply", action="store_true", help="Append superseded entries to memory_hygiene.jsonl")
    args = parser.parse_args()

    priority_tags = _normalize_retain_tags(args.priority_tags) or DEFAULT_PRIORITY_TAGS
    records = load_records(args.input_json, priority_tags)
    context_tags = _normalize_retain_tags(args.dedupe_context_tags) or DEFAULT_DEDUPE_CONTEXT_TAGS
    actions = choose_duplicate_actions(records, context_tags)
    applied = apply_hygiene_actions(args.hygiene_path, actions) if args.apply else 0
    report = {
        "mode": "apply" if args.apply else "dry-run",
        "records": len(records),
        "duplicate_actions": len(actions),
        "applied": applied,
        "hygiene_path": str(args.hygiene_path),
        "actions": actions,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
