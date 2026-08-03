#!/usr/bin/env python3
"""Classify Sahil's PR review feedback via moss_review_feedback module.

Reads /tmp/pr-feedback-payloads.json, runs full pipeline:
  normalise -> dedupe -> classify -> refine -> Gate2 -> promote
Outputs JSON summary to stdout.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure module importable from scripts dir
sys_path = str(Path(__file__).resolve().parent)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from scripts.moss_review_feedback import (  # type: ignore
    MAINTAINER_AUTHORS,
    PROMOTABLE,
    classify_records,
    dedupe_records,
    load_json,
    normalise_feedback,
    promote_records,
)

KNOWN_APPROVERS = {"tonydwb", "odathetkan", "alt-glitch"}


def is_sweeper_template(body: str) -> bool:
    return "<!-- hermes-sweeper:" in body


def is_approval_review(author: str, body: str, review_state: str | None) -> bool:
    lowered = body.lower()
    return (
        author.lower() in KNOWN_APPROVERS
        and any(marker in lowered for marker in ("lgtm", "approved", "verdict:"))
    )


def apply_refinements(records: list[dict[str, any]]) -> list[dict[str, any]]:
    refined = []
    for record in records:
        body = str(record.get("body", ""))
        author = str(record.get("author", ""))
        review_state = record.get("review_state")

        if is_sweeper_template(body):
            record["classification"] = "reply_only"
            record["refinement"] = "sweeper_template"
        elif is_approval_review(author, body, review_state):
            record["classification"] = "reply_only"
            record["refinement"] = "approver_reclassification"
        refined.append(record)
    return refined


def load_dedup_state() -> dict[str, any]:
    paths = {
        "state": Path.home() / ".hermes" / "data" / "moss-review-feedback-state.json",
    }
    return load_json(paths["state"], {"seen": []})


def save_dedup_state(state: dict[str, any], records: list[dict[str, any]]) -> None:
    seen = list(state.get("seen", []))
    for record in records:
        key = record.get("key")
        if key and key not in seen:
            seen.append(key)
    state["seen"] = seen
    paths = {
        "state": Path.home() / ".hermes" / "data" / "moss-review-feedback-state.json",
    }
    from scripts.moss_review_feedback import atomic_write_json  # type: ignore
    atomic_write_json(paths["state"], state)


def main() -> None:
    payload_path = Path("/tmp/pr-feedback-payloads.json")
    if not payload_path.exists():
        print(json.dumps({"error": "missing payloads", "path": str(payload_path)}))
        sys.exit(1)

    payloads = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payloads, list):
        payloads = [payloads]

    all_raw: list[dict[str, any]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        records = normalise_feedback(payload)
        all_raw.extend(records)

    state = load_dedup_state()
    deduped = dedupe_records(all_raw, state)
    classified = classify_records(deduped)
    refined = apply_refinements(classified)
    promoted = promote_records(refined)

    counts: dict[str, int] = {}
    for record in refined:
        cls = record.get("classification", "unknown")
        counts[cls] = counts.get(cls, 0) + 1

    result = {
        "total_raw": len(all_raw),
        "after_dedupe": len(deduped),
        "after_refinement": len(refined),
        "counts": counts,
        "promoted_count": len(promoted),
        "promoted": promoted,
    }

    # Update dedup state with all processed keys
    save_dedup_state(state, all_raw)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
