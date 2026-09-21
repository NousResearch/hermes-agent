#!/usr/bin/env python3
"""Compact, text-only observer for pending SahilBlog approvals.

This program has no delivery side effects. It reconciles the JSONL tracker with
MDX approval state, then prints one deterministic, paged batch for an external
wrapper to deliver.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "blog_topics" / "pending_approvals.jsonl"
DEFAULT_PAGE_SIZE = 10
MAX_LINES = 15
_APPROVED_RE = re.compile(r"^approved\s*:\s*(?:true|['\"]true['\"])\s*$", re.IGNORECASE | re.MULTILINE)
_ABSOLUTE_PATH_RE = re.compile(r"(?<!\S)/(?:[^\s|]+)")


def _display_text(value: object) -> str:
    """Compact metadata without allowing tracker content to disclose paths."""
    return _ABSOLUTE_PATH_RE.sub("<path>", " ".join(str(value or "").split()))


def _read_entries(tracker: Path) -> list[dict[str, Any]]:
    if not tracker.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in tracker.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _atomic_write_entries(tracker: Path, entries: list[dict[str, Any]]) -> None:
    tracker.parent.mkdir(parents=True, exist_ok=True)
    temporary = tracker.with_name(f".{tracker.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
            encoding="utf-8",
        )
        os.replace(temporary, tracker)
    finally:
        temporary.unlink(missing_ok=True)


def _is_approved_mdx(entry: dict[str, Any]) -> bool:
    mdx_path = entry.get("mdx_path")
    if not isinstance(mdx_path, str) or not mdx_path:
        return False
    try:
        return bool(_APPROVED_RE.search(Path(mdx_path).read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


def reconcile_tracker(*, tracker: Path = TRACKER) -> list[dict[str, Any]]:
    """Remove tracker rows whose corresponding MDX has already been approved."""
    entries = _read_entries(tracker)
    retained = [entry for entry in entries if not _is_approved_mdx(entry)]
    if retained != entries:
        _atomic_write_entries(tracker, retained)
    return retained


def load_pending(*, tracker: Path = TRACKER) -> list[dict[str, Any]]:
    """Return reconciled pending entries in a stable review order."""
    entries = reconcile_tracker(tracker=tracker)
    pending = [entry for entry in entries if entry.get("status") == "pending"]
    return sorted(pending, key=lambda entry: (str(entry.get("created_at", "")), str(entry.get("slug", ""))))


def build_message(entries: list[dict[str, Any]], *, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> str:
    """Build <=15 lines of text-only approval status; never expose local paths."""
    if not entries:
        return "[SILENT]"
    if page < 1 or page_size < 1 or page_size > MAX_LINES - 2:
        raise ValueError("page must be positive and page_size must keep output within 15 lines")

    ordered = sorted(entries, key=lambda entry: (str(entry.get("created_at", "")), str(entry.get("slug", ""))))
    total_pages = (len(ordered) + page_size - 1) // page_size
    if page > total_pages:
        return "[SILENT]"
    start = (page - 1) * page_size
    current = ordered[start:start + page_size]
    lines = [f"Blog approvals {page}/{total_pages} ({len(entries)} pending)"]
    for entry in current:
        slug = _display_text(entry.get("slug")) or "?"
        title = _display_text(entry.get("title")) or "Untitled"
        stream = _display_text(entry.get("stream")) or "?"
        tier = _display_text(entry.get("tier")) or "?"
        lines.append(f"- {slug} | {title} [{stream}/{tier}]")
    lines.append("Reply: !approve <slug> | !reject <slug> <reason> | !amend <slug> <notes>")
    return "\n".join(lines)


def observe(*, tracker: Path = TRACKER, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> str:
    """Reconcile then render one page, without sending media or invoking providers."""
    return build_message(load_pending(tracker=tracker), page=page, page_size=page_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a compact SahilBlog approval page")
    parser.add_argument("--tracker", type=Path, default=TRACKER)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    args = parser.parse_args(argv)
    try:
        print(observe(tracker=args.tracker, page=args.page, page_size=args.page_size))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
