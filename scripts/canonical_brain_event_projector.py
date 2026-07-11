#!/usr/bin/env python3
"""Build Canonical Brain projections from the append-only event log.

This projector is deliberately mechanical: it folds structured event types and
fields only. It never scans summaries/transcripts for keywords and never assigns
business category, priority, resolver, risk, or next action.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import pathlib
from typing import Any

from gateway.canonical_brain_projection import fold_case_events


DEFAULT_HELPER = pathlib.Path(
    "/opt/adventico-ai-platform/canonical-brain/bin/cloud_sql_synthetic_write_gate.py"
)
EVENT_TABLE = "canonical_event_log"


def _load_helper(path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location("canonical_brain_cloud_sql_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Canonical Brain Cloud SQL helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_events(helper_path: pathlib.Path, *, limit: int) -> list[dict[str, Any]]:
    helper = _load_helper(helper_path)
    password = helper.get_secret_value()
    try:
        sock = helper.connect(password)
        try:
            result = helper.query(sock, f"""
SELECT event_id::text, schema_version, event_type, case_id, occurred_at::text,
       source, actor, subject, evidence, decision, status, next_action, safety, payload
FROM {EVENT_TABLE}
WHERE event_type <> 'runtime.lease.renewed'
ORDER BY occurred_at DESC, event_id DESC
LIMIT {int(limit)};
""")
            rows = result.get("rows", []) if isinstance(result, dict) else []
        finally:
            try:
                sock.close()
            except Exception:
                pass
    finally:
        password = ""

    columns = [
        "event_id", "schema_version", "event_type", "case_id", "occurred_at",
        "source", "actor", "subject", "evidence", "decision", "status",
        "next_action", "safety", "payload",
    ]
    return [
        row if isinstance(row, dict) else dict(zip(columns, row))
        for row in rows
        if isinstance(row, (dict, list, tuple))
    ]


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
    parser.add_argument("--helper", type=pathlib.Path, default=DEFAULT_HELPER)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=200_000)
    args = parser.parse_args()
    if not args.helper.is_file():
        raise SystemExit("Canonical Brain Cloud SQL helper unavailable")
    if args.limit < 1 or args.limit > 1_000_000:
        raise SystemExit("--limit must be between 1 and 1000000")
    rows = read_events(args.helper, limit=args.limit)
    documents = projection_documents(rows)
    write_documents(args.output_dir, documents)
    print(json.dumps(documents["index.json"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
