"""Daily consolidation/report command for Memory v2.

This module provides a cheap deterministic daily maintenance pass: run the
rule-based candidate consolidator, snapshot counts/open loops, and write both a
machine-readable report and an episodic daily summary.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .consolidation import RuleBasedConsolidator
from .index import MemoryV2Index
from .schemas import utc_now_iso
from .store import MemoryV2Store

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def run_daily_consolidation_report(
    store: MemoryV2Store,
    index: MemoryV2Index,
    *,
    date: str | None = None,
    recent_raw_limit: int = 20,
) -> dict[str, Any]:
    """Run daily Memory v2 consolidation and persist report artifacts.

    Returns a JSON-serializable payload. Paths are relative to ``memory_v2/`` so
    rendered outputs do not expose local profile paths unnecessarily.
    """
    report_date = _normalized_date(date)
    before_counts = _counts(store)
    consolidation = RuleBasedConsolidator().consolidate(store, index).to_dict()
    after_counts = _counts(store)

    report: dict[str, Any] = {
        "success": True,
        "kind": "daily_memory_consolidation_report",
        "date": report_date,
        "created_at": utc_now_iso(),
        "before_counts": before_counts,
        "after_counts": after_counts,
        "consolidation": consolidation,
        "open_loops": store.list_open_loops(status="open"),
        "recent_raw_event_ids": [str(event.get("id")) for event in store.read_raw_events(limit=recent_raw_limit)],
        "report_path": f"reports/daily_consolidation/{report_date}.json",
        "daily_episode_path": f"episodic/daily/{report_date}.yaml",
    }

    _write_daily_episode(store, report)
    _write_json(store.base_dir / report["report_path"], report)
    return report


def _counts(store: MemoryV2Store) -> dict[str, int]:
    return {
        "raw_events": store.count_raw_events(),
        "candidates": len(store.list_candidates()),
        "pending_candidates": store.count_pending_candidates(),
        "rejected_candidates": store.count_rejected_candidates(),
        "memory_items": len(store.list_memory_items()),
        "project_cards": len(store.list_project_cards()),
        "open_loops": len(store.list_open_loops(status="open")),
        "source_refs": len(store.list_source_refs()),
        "session_archives": len(store.list_session_archives()),
    }


def _write_daily_episode(store: MemoryV2Store, report: dict[str, Any]) -> None:
    payload = {
        "version": 1,
        "kind": "daily_memory_consolidation",
        "date": report["date"],
        "created_at": report["created_at"],
        "report_path": report["report_path"],
        "before_counts": report["before_counts"],
        "after_counts": report["after_counts"],
        "consolidation": report["consolidation"],
        "open_loops": report["open_loops"],
        "recent_raw_event_ids": report["recent_raw_event_ids"],
    }
    _write_yaml(store.base_dir / report["daily_episode_path"], payload)


def _normalized_date(date: str | None) -> str:
    if date is None or not str(date).strip():
        return datetime.now(timezone.utc).date().isoformat()
    text = str(date).strip()
    if not _DATE_RE.match(text):
        raise ValueError("date must be YYYY-MM-DD")
    # Validate calendar correctness, not just shape.
    datetime.strptime(text, "%Y-%m-%d")
    return text


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Memory v2 daily consolidation/report for one profile.")
    parser.add_argument("--hermes-home", required=True, help="Profile home containing memory_v2/.")
    parser.add_argument("--date", default=None, help="Report date as YYYY-MM-DD. Defaults to current UTC date.")
    parser.add_argument("--session-id", default="daily-consolidation", help="Session id used when initializing the index.")
    args = parser.parse_args(argv)

    hermes_home = Path(args.hermes_home).expanduser().resolve()
    store = MemoryV2Store(hermes_home / "memory_v2")
    store.initialize()
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.rebuild_from_store(store)
    report = run_daily_consolidation_report(store, index, date=args.date)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
