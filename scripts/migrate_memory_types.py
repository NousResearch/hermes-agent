#!/usr/bin/env python3
"""Suggest or apply built-in memory type classifications.

This migrates Hermes' file-backed MEMORY.md / USER.md stores by writing
type metadata into the sidecar JSON files used by tools/memory_tool.py.
The original memory content files are left unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Iterable, List

from tools.memory_tool import DEFAULT_MEMORY_TYPE, MemoryStore, _infer_memory_type

TARGETS = ("memory", "user")


def _iter_targets(selected: str) -> Iterable[str]:
    if selected == "all":
        return TARGETS
    return (selected,)


def _suggest_type(target: str, content: str) -> Dict[str, str]:
    text = content.lower()

    feedback_markers = (
        "don't", "do not", "avoid", "instead", "reply", "respond",
        "response", "summary", "format", "remember this",
    )
    reference_markers = (
        "http://", "https://", "docs", "documentation", "reference",
        "api", "spec", "manual", "readme", "link",
    )
    project_markers = (
        "project", "repo", "repository", "codebase", "database", "postgres",
        "sqlite", "fastapi", "pytest", "service", "deploy", "branch",
        ".py", ".ts", "tests/", "src/", "file ", "path ",
    )
    user_markers = (
        "name", "timezone", "prefers", "prefer", "likes", "role",
        "works as", "i am", "my ", "user ",
    )

    if any(marker in text for marker in feedback_markers):
        return {"suggested_type": "feedback", "reason": "matched feedback/instruction markers"}
    if target == "user":
        return {"suggested_type": "user", "reason": "entry came from the user profile store"}
    if any(marker in text for marker in reference_markers):
        return {"suggested_type": "reference", "reason": "matched reference/docs markers"}
    if any(marker in text for marker in project_markers):
        return {"suggested_type": "project", "reason": "matched project/environment markers"}
    if any(marker in text for marker in user_markers):
        return {"suggested_type": "user", "reason": "matched user/profile markers"}
    return {"suggested_type": DEFAULT_MEMORY_TYPE, "reason": "no strong heuristic match"}


def build_report(store: MemoryStore, target: str = "all") -> Dict[str, object]:
    return _build_report(store, target=target, auto_classify=False)


def _build_report(
    store: MemoryStore,
    target: str = "all",
    *,
    auto_classify: bool = False,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "success": True,
        "mode": "dry-run",
        "auto_classify": auto_classify,
        "targets": {},
        "summary": {
            "total_entries": 0,
            "typed_entries": 0,
            "uncategorized_entries": 0,
            "suggested_updates": 0,
        },
    }

    targets = report["targets"]
    summary = report["summary"]

    for current_target in _iter_targets(target):
        result = store.read(current_target)
        if not result.get("success"):
            return {
                "success": False,
                "mode": "dry-run",
                "error": result.get("error", f"Failed to read {current_target} memories."),
            }

        typed_entries: List[Dict[str, str]] = result["typed_entries"]
        suggestions: List[Dict[str, str]] = []
        typed_count = 0
        uncategorized_count = 0

        for item in typed_entries:
            current_type = item["type"]
            if current_type == DEFAULT_MEMORY_TYPE:
                uncategorized_count += 1
            else:
                typed_count += 1

            if auto_classify and current_type == DEFAULT_MEMORY_TYPE:
                suggested_type = _infer_memory_type(item["content"])
                reason = "auto-classify via _infer_memory_type()"
            else:
                suggestion = _suggest_type(current_target, item["content"])
                suggested_type = suggestion["suggested_type"]
                reason = suggestion["reason"]
            suggestions.append({
                "content": item["content"],
                "current_type": current_type,
                "suggested_type": suggested_type,
                "reason": reason,
            })

        targets[current_target] = {
            "total_entries": len(typed_entries),
            "typed_entries": typed_count,
            "uncategorized_entries": uncategorized_count,
            "suggestions": suggestions,
        }

        summary["total_entries"] += len(typed_entries)
        summary["typed_entries"] += typed_count
        summary["uncategorized_entries"] += uncategorized_count
        summary["suggested_updates"] += sum(
            1
            for item in suggestions
            if item["current_type"] == DEFAULT_MEMORY_TYPE
            and item["suggested_type"] != DEFAULT_MEMORY_TYPE
        )

    return report


def apply_report(store: MemoryStore, report: Dict[str, object]) -> Dict[str, object]:
    apply_summary = {
        "applied_updates": 0,
        "per_target": {},
    }

    for target, target_report in report.get("targets", {}).items():
        suggestions = target_report.get("suggestions", [])
        updates = {
            item["content"]: item["suggested_type"]
            for item in suggestions
            if item["current_type"] == DEFAULT_MEMORY_TYPE
            and item["suggested_type"] != DEFAULT_MEMORY_TYPE
        }
        result = store.bulk_update_types(target, updates)
        apply_summary["per_target"][target] = result
        if result.get("success"):
            apply_summary["applied_updates"] += result.get("applied", 0)

    return {
        "success": True,
        "mode": "apply",
        "summary": {
            **report.get("summary", {}),
            **apply_summary,
        },
        "targets": report.get("targets", {}),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Suggest or apply type metadata for Hermes built-in memory entries."
    )
    parser.add_argument(
        "--target",
        choices=("memory", "user", "all"),
        default="all",
        help="Which built-in memory store to inspect.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print suggested type updates without applying them (default behavior).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply suggested non-uncategorized types to the sidecar metadata files.",
    )
    parser.add_argument(
        "--auto-classify",
        action="store_true",
        help="Use tools.memory_tool._infer_memory_type() to suggest types for uncategorized entries.",
    )
    args = parser.parse_args(argv)

    store = MemoryStore()
    store.load_from_disk()
    report = _build_report(store, target=args.target, auto_classify=args.auto_classify)

    if not report.get("success"):
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 1

    output = report
    if args.apply:
        output = apply_report(store, report)

    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
