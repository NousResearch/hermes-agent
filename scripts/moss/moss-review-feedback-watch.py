#!/usr/bin/env python3
"""Read-only entry point for fixture/API review feedback observation.

Without an explicit --fixture it emits a dry-run observation and performs no
network call.  Production scheduling/credentials are intentionally out of
scope for this foundation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moss_review_feedback import (  # noqa: E402
    atomic_write_json, build_action_plan, classify_records, dedupe_records, feedback_paths,
    load_json, normalise_feedback, promote_records,
)


def process_fixture(payload: dict) -> dict:
    paths = feedback_paths()
    state = load_json(paths["state"], {"seen": []})
    records = dedupe_records(normalise_feedback(payload), state)
    classified = classify_records(records)
    promoted = promote_records(classified)
    plan = build_action_plan(classified)
    ledger = load_json(paths["ledger"], {"events": []})
    atomic_write_json(paths["state"], {"seen": state.get("seen", []) + [row["key"] for row in records]})
    atomic_write_json(paths["queue"], {"pending": promoted, "action_plan": plan})
    # Observation is not resolution: only persist_resolution may add ledger events.
    atomic_write_json(paths["ledger"], {"events": list(ledger.get("events", []))})
    return {"new_count": len(records), "classified": classified, "promoted": promoted, "action_plan": plan}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if not args.fixture:
        print("dry-run: no fixture supplied; no GitHub command executed")
        return 0
    result = process_fixture(json.loads(args.fixture.read_text(encoding="utf-8")))
    print(f"Mossy review feedback: {result['new_count']} new item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
