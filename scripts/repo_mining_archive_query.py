#!/usr/bin/env python3
"""Tiny query helper for repo_mining_archive_index JSON files.

Examples:
  python scripts/repo_mining_archive_query.py
  python scripts/repo_mining_archive_query.py --priority high --status synthesized
  python scripts/repo_mining_archive_query.py --category memory --has-url --format paths
  python scripts/repo_mining_archive_query.py --q orchestration --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
LATEST_INDEX_ALIAS = "repo_mining_archive_index_latest.json"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _dated_index_candidates(reports_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in reports_dir.glob("repo_mining_archive_index_*.json")
        if p.name != LATEST_INDEX_ALIAS
    )


def _default_index_path() -> Path:
    reports_dir = _workspace_root() / "reports"
    alias = reports_dir / LATEST_INDEX_ALIAS
    if alias.exists():
        return alias

    candidates = _dated_index_candidates(reports_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No index files found under {reports_dir} (expected repo_mining_archive_index_*.json)."
        )
    return candidates[-1]


def _entry_name(entry: dict[str, Any]) -> str:
    return str(entry.get("repo_name") or entry.get("name") or "")


def _entry_categories(entry: dict[str, Any]) -> list[str]:
    categories = entry.get("category")
    if categories is None:
        categories = entry.get("categories")

    if categories is None:
        return []
    if isinstance(categories, list):
        return [str(c) for c in categories]
    return [str(categories)]


def _entry_repo_slug(entry: dict[str, Any]) -> str:
    repo = entry.get("repo")
    if isinstance(repo, dict) and repo.get("slug"):
        return str(repo.get("slug"))
    return ""


def _entry_repo_url(entry: dict[str, Any]) -> str:
    url = entry.get("repo_url")
    if url:
        return str(url)
    repo = entry.get("repo")
    if isinstance(repo, dict) and repo.get("url"):
        return str(repo.get("url"))
    return ""


def _entry_report_paths(entry: dict[str, Any]) -> list[str]:
    report_paths = entry.get("report_paths") or []
    if isinstance(report_paths, list):
        return [str(p) for p in report_paths]
    return [str(report_paths)]


def _load_entries(index_path: Path) -> list[dict[str, Any]]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Invalid index: entries must be a list")
    return [e for e in entries if isinstance(e, dict)]


def _load_entries_with_alias_fallback(index_path: Path) -> tuple[Path, list[dict[str, Any]], str | None]:
    try:
        return index_path, _load_entries(index_path), None
    except Exception as exc:  # noqa: BLE001
        if index_path.name != LATEST_INDEX_ALIAS:
            raise

        candidates = _dated_index_candidates(index_path.parent)
        if not candidates:
            raise

        fallback_path = candidates[-1]
        fallback_entries = _load_entries(fallback_path)
        warning = (
            f"warning: failed to load {index_path.name} ({exc}); "
            f"falling back to {fallback_path.name}"
        )
        return fallback_path, fallback_entries, warning


def _ci(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _contains_any(haystack_items: list[str], needles: list[str]) -> bool:
    haystack_set = {_ci(x) for x in haystack_items}
    return any(n in haystack_set for n in needles)


def _matches(entry: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.priority:
        if _ci(entry.get("priority")) not in args.priority:
            return False

    if args.status:
        if _ci(entry.get("status")) not in args.status:
            return False

    if args.model:
        if _ci(entry.get("model")) not in args.model:
            return False

    if args.category:
        if not _contains_any(_entry_categories(entry), args.category):
            return False

    if args.repo:
        repo_blob = " ".join([_entry_name(entry), _entry_repo_slug(entry)]).lower()
        if args.repo not in repo_blob:
            return False

    if args.has_url and not _entry_repo_url(entry):
        return False

    if args.missing_url and _entry_repo_url(entry):
        return False

    if args.q:
        categories = _entry_categories(entry)
        reports = _entry_report_paths(entry)
        search_blob = "\n".join(
            [
                _entry_name(entry),
                _entry_repo_slug(entry),
                str(entry.get("notes") or ""),
                " ".join(str(c) for c in categories),
                " ".join(str(p) for p in reports),
            ]
        ).lower()
        if args.q not in search_blob:
            return False

    return True


def _sort_key(entry: dict[str, Any]) -> tuple[int, str]:
    priority = _ci(entry.get("priority"))
    return (PRIORITY_ORDER.get(priority, 99), _ci(_entry_name(entry)))


def _render_table(entries: list[dict[str, Any]]) -> str:
    headers = ["name", "priority", "status", "model", "reports", "categories", "url"]

    rows: list[list[str]] = []
    for e in entries:
        categories = _entry_categories(e)
        report_paths = _entry_report_paths(e)
        row = [
            _entry_name(e) or "-",
            str(e.get("priority") or "-"),
            str(e.get("status") or "-"),
            str(e.get("model") or "-"),
            str(len(report_paths) if isinstance(report_paths, list) else 0),
            ",".join(str(c) for c in categories) if categories else "-",
            "yes" if _entry_repo_url(e) else "no",
        ]
        rows.append(row)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = min(max(widths[i], len(value)), 54)

    def clip(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "…"

    lines = []
    header_line = " | ".join(clip(h, widths[i]).ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append(" | ".join(clip(v, widths[i]).ljust(widths[i]) for i, v in enumerate(row)))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query repo mining archive index")
    parser.add_argument("--index", type=Path, help="Path to index JSON file")
    parser.add_argument("--priority", action="append", default=[], help="Filter priority (repeatable)")
    parser.add_argument("--status", action="append", default=[], help="Filter status (repeatable)")
    parser.add_argument("--model", action="append", default=[], help="Filter exact model (repeatable)")
    parser.add_argument("--category", action="append", default=[], help="Filter category membership (repeatable)")
    parser.add_argument(
        "--repo",
        default="",
        help="Case-insensitive substring filter on entry name and repo slug",
    )
    parser.add_argument(
        "--q",
        default="",
        help="Case-insensitive text search across name/slug/notes/categories/report_paths",
    )

    url_group = parser.add_mutually_exclusive_group()
    url_group.add_argument("--has-url", action="store_true", help="Only entries with a repo URL")
    url_group.add_argument("--missing-url", action="store_true", help="Only entries without a repo URL")

    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows (0 = no limit)")
    parser.add_argument("--format", choices=["table", "json", "paths"], default="table")
    parser.add_argument("--summary", action="store_true", help="Print total/matched counts to stderr")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.priority = [_ci(x) for x in args.priority]
    args.status = [_ci(x) for x in args.status]
    args.model = [_ci(x) for x in args.model]
    args.category = [_ci(x) for x in args.category]
    args.repo = _ci(args.repo)
    args.q = _ci(args.q)

    try:
        index_path = args.index if args.index else _default_index_path()
        if args.index:
            entries = _load_entries(index_path)
            warning = None
        else:
            index_path, entries, warning = _load_entries_with_alias_fallback(index_path)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if warning:
        print(warning, file=sys.stderr)

    filtered = [e for e in entries if _matches(e, args)]
    filtered.sort(key=_sort_key)

    if args.limit and args.limit > 0:
        filtered = filtered[: args.limit]

    if args.summary:
        print(
            f"index={index_path} total={len(entries)} matched={len(filtered)}",
            file=sys.stderr,
        )

    if args.format == "json":
        print(json.dumps(filtered, ensure_ascii=False, indent=2))
    elif args.format == "paths":
        for e in filtered:
            for path in _entry_report_paths(e):
                print(path)
    else:
        print(_render_table(filtered))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
