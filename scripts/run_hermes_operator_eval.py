"""Validate Hermes operator eval definitions and optional SFT coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(value)
    return rows


def validate_eval_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    seen_ids = set()
    for index, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            errors.append(f"row {index}: missing id")
        elif row_id in seen_ids:
            errors.append(f"row {index}: duplicate id {row_id}")
        else:
            seen_ids.add(row_id)
        if not isinstance(row.get("prompt"), str) or not row.get("prompt"):
            errors.append(f"row {index}: missing prompt")
        if not isinstance(row.get("tags"), list) or not row.get("tags"):
            errors.append(f"row {index}: missing tags")
        if not isinstance(row.get("expected_behaviors"), list) or not row.get("expected_behaviors"):
            errors.append(f"row {index}: missing expected_behaviors")
    return errors


def corpus_tag_coverage(eval_rows: list[dict[str, Any]], sft_rows: list[dict[str, Any]]) -> dict[str, bool]:
    available = set()
    for row in sft_rows:
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            tags = metadata.get("tags")
            if isinstance(tags, list):
                available.update(str(tag) for tag in tags)
    required = set()
    for row in eval_rows:
        tags = row.get("tags")
        if isinstance(tags, list):
            required.update(str(tag) for tag in tags)
    return {tag: tag in available for tag in sorted(required)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Hermes operator evals.")
    parser.add_argument("--evals", type=Path, default=Path("evals/hermes_operator_eval.jsonl"))
    parser.add_argument("--sft", type=Path)
    args = parser.parse_args(argv)

    eval_rows = _read_jsonl(args.evals)
    errors = validate_eval_rows(eval_rows)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"validated {len(eval_rows)} eval row(s)")

    if args.sft:
        sft_rows = _read_jsonl(args.sft)
        coverage = corpus_tag_coverage(eval_rows, sft_rows)
        missing = [tag for tag, present in coverage.items() if not present]
        print(f"SFT rows: {len(sft_rows)}")
        print("tag coverage: " + ", ".join(f"{tag}={'yes' if ok else 'no'}" for tag, ok in coverage.items()))
        if missing:
            print("missing tags: " + ", ".join(missing))
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
