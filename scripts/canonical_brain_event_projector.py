#!/usr/bin/env python3
"""Build Canonical Brain projections from the append-only event log.

This projector is deliberately mechanical: it folds structured event types and
fields only. It never scans summaries/transcripts for keywords and never assigns
business category, priority, resolver, risk, or next action.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
from typing import Any, Mapping

from gateway.canonical_brain_projection import fold_case_events


def read_events(events_path: pathlib.Path, *, limit: int) -> list[dict[str, Any]]:
    """Read a bounded export produced by the privileged writer service.

    This projector intentionally has no database/helper/secret path.  The
    writer service owns database reads and may write an atomic JSON export for
    this pure folding process.  The export is derived state, never authority;
    Canonical Brain's append-only log remains the source of truth.
    """

    with events_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, Mapping):
        rows = value.get("events")
    else:
        rows = value
    if not isinstance(rows, list):
        raise ValueError("writer event export must contain an events array")
    if len(rows) > limit:
        raise ValueError("writer event export exceeds the projector limit")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("writer event export rows must be objects")
        normalized.append(dict(row))
    return normalized


def projection_documents(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cases = fold_case_events(rows)
    created_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    route_backs = [
        {
            "case_id": case["case_id"],
            **case["route_back"],
            "latest_event_at": case["latest_event_at"],
        }
        for case in cases
        if case["route_back"]["latest_event_type"]
    ]
    return {
        "cases.json": {
            "schema": "canonical_brain.projection.cases.v3",
            "created_at": created_at,
            "items": cases,
        },
        "route_backs.json": {
            "schema": "canonical_brain.projection.route_backs.v3",
            "created_at": created_at,
            "items": route_backs,
        },
        "index.json": {
            "schema": "canonical_brain.projection.index.v3",
            "created_at": created_at,
            "source": "canonical_event_log",
            "event_count": len(rows),
            "case_count": len(cases),
            "files": ["cases.json", "route_backs.json"],
            "semantic_classifier": False,
        },
    }


def write_documents(output_dir: pathlib.Path, documents: dict[str, dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        target = output_dir / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events-json",
        type=pathlib.Path,
        required=True,
        help="atomic event export produced by muncho-canonical-writer",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=200_000)
    args = parser.parse_args()
    if not args.events_json.is_file():
        raise SystemExit("privileged Canonical writer event export unavailable")
    if args.limit < 1 or args.limit > 1_000_000:
        raise SystemExit("--limit must be between 1 and 1000000")
    rows = read_events(args.events_json, limit=args.limit)
    documents = projection_documents(rows)
    write_documents(args.output_dir, documents)
    print(json.dumps(documents["index.json"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
