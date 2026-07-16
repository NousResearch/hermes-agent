#!/usr/bin/env python3
"""Unified social-memory sync: gateway sessions -> Ebbinghaus -> Obsidian wiki.

Orchestrates:
1. Discord / LINE / LINE-personal / Telegram gateway sessions (``state.db``)
2. Optional X (lm-twitterer) activity traces
3. Ebbinghaus SQLite consolidation (``ebbinghaus_memory.db``)
4. Sanitized Obsidian export via ``memory_llm_wiki`` when a vault is discoverable

Incremental mode stores per-source watermarks under ``HERMES_HOME`` so cron /
Task Scheduler runs only import new sessions.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.memory.social_ebbinghaus_sync import (  # noqa: E402
    DEFAULT_SOURCES,
    SocialMemorySync,
    hermes_home,
)

DEFAULT_INDEX_NAME = "memory_sync_last_index.json"
LEGACY_INDEX_NAME = "last_index.txt"


def _default_index_path() -> Path:
    return hermes_home() / DEFAULT_INDEX_NAME


def load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "sources": {}, "last_run_at": None}
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return {"version": 1, "sources": {}, "last_run_at": None}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Legacy plain timestamp in last_index.txt
        try:
            ts = float(raw.splitlines()[0].strip())
        except ValueError:
            return {"version": 1, "sources": {}, "last_run_at": None}
        return {"version": 1, "sources": {"*": {"last_started_at": ts}}, "last_run_at": ts}
    if isinstance(data, (int, float)):
        ts = float(data)
        return {"version": 1, "sources": {"*": {"last_started_at": ts}}, "last_run_at": ts}
    if isinstance(data, dict):
        data.setdefault("version", 1)
        data.setdefault("sources", {})
        return data
    return {"version": 1, "sources": {}, "last_run_at": None}


def save_index(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _watermark_for_sources(index: dict[str, Any], sources: Sequence[str]) -> float | None:
    per_source = index.get("sources") or {}
    stamps: list[float] = []
    for source in sources:
        entry = per_source.get(source) or {}
        ts = entry.get("last_started_at")
        if ts is not None:
            stamps.append(float(ts))
    wildcard = (per_source.get("*") or {}).get("last_started_at")
    if wildcard is not None:
        stamps.append(float(wildcard))
    return min(stamps) if stamps else None


def _latest_started_at(state_db: Path, sources: Sequence[str]) -> dict[str, float]:
    if not state_db.exists() or not sources:
        return {}
    placeholders = ",".join("?" for _ in sources)
    query = f"""
        SELECT lower(source) AS source, MAX(started_at) AS last_started_at
        FROM sessions
        WHERE lower(source) IN ({placeholders})
        GROUP BY lower(source)
    """
    try:
        with sqlite3.connect(state_db) as con:
            rows = con.execute(query, tuple(s.lower() for s in sources)).fetchall()
    except sqlite3.Error:
        return {}
    return {str(src): float(ts) for src, ts in rows if ts is not None}


def _export_obsidian(*, dry_run: bool, max_ebbinghaus: int) -> dict[str, Any]:
    try:
        from plugins.memory_llm_wiki.core import handle_export
    except Exception as exc:  # pragma: no cover - optional plugin surface
        return {"success": False, "skipped": True, "error": f"memory_llm_wiki unavailable: {exc}"}
    payload = handle_export(
        {
            "include_curated": True,
            "include_ebbinghaus": True,
            "max_ebbinghaus": max_ebbinghaus,
            "dry_run": dry_run,
        }
    )
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"success": False, "error": "invalid export payload", "raw": payload}


def run_sync(
    *,
    sources: Sequence[str],
    max_sessions: int,
    max_x_events: int,
    sleep: bool,
    incremental: bool,
    index_path: Path,
    export_obsidian: bool,
    obsidian_dry_run: bool,
    max_ebbinghaus_export: int,
    state_db: Path | None,
    memory_db: Path | None,
) -> dict[str, Any]:
    index = load_index(index_path)
    min_started_at = _watermark_for_sources(index, sources) if incremental else None

    sync = SocialMemorySync(
        state_db=state_db,
        memory_db=memory_db,
        sources=sources,
        min_started_at=min_started_at,
    )
    social_result = sync.run(
        max_sessions=max_sessions,
        max_x_events=max_x_events,
        sleep=sleep,
    )
    social_result["incremental"] = incremental
    social_result["index_path"] = str(index_path)

    obsidian_result: dict[str, Any] | None = None
    if export_obsidian:
        obsidian_result = _export_obsidian(dry_run=obsidian_dry_run, max_ebbinghaus=max_ebbinghaus_export)

    if incremental and not obsidian_dry_run:
        latest = _latest_started_at(sync.state_db, sources)
        per_source = index.setdefault("sources", {})
        for source, ts in latest.items():
            prev = (per_source.get(source) or {}).get("last_started_at")
            per_source[source] = {"last_started_at": max(float(prev or 0), ts)}
        index["last_run_at"] = time.time()
        save_index(index_path, index)

    return {
        "success": True,
        "social": social_result,
        "obsidian": obsidian_result,
        "index_updated": incremental and not obsidian_dry_run,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated gateway session sources (default: line,line-personal,discord,telegram)",
    )
    parser.add_argument("--max-sessions", type=int, default=80)
    parser.add_argument("--max-x-events", type=int, default=80)
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--full", action="store_true", help="Ignore incremental watermark and rescan recent sessions")
    parser.add_argument(
        "--index-file",
        type=Path,
        default=None,
        help=f"Watermark file (default: ~/.hermes/{DEFAULT_INDEX_NAME}; legacy: ./last_index.txt if present)",
    )
    parser.add_argument("--no-obsidian", action="store_true", help="Skip Obsidian wiki export")
    parser.add_argument("--dry-run", action="store_true", help="Plan Obsidian export without writing vault files")
    parser.add_argument("--max-ebbinghaus-export", type=int, default=120)
    parser.add_argument("--state-db", type=Path, default=None)
    parser.add_argument("--memory-db", type=Path, default=None)
    return parser


def resolve_index_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    legacy = REPO_ROOT / LEGACY_INDEX_NAME
    if legacy.exists():
        return legacy
    return _default_index_path()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    sources = tuple(part.strip() for part in args.sources.split(",") if part.strip())
    result = run_sync(
        sources=sources,
        max_sessions=args.max_sessions,
        max_x_events=args.max_x_events,
        sleep=not args.no_sleep,
        incremental=not args.full,
        index_path=resolve_index_path(args.index_file),
        export_obsidian=not args.no_obsidian,
        obsidian_dry_run=args.dry_run,
        max_ebbinghaus_export=args.max_ebbinghaus_export,
        state_db=args.state_db,
        memory_db=args.memory_db,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
