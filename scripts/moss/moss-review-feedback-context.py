#!/usr/bin/env python3
"""Render an observation-only review-feedback context for a future push gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moss_review_feedback import build_pre_push_report, feedback_paths, load_json  # noqa: E402


def render_context(classified: list[dict], promoted: list[dict]) -> str:
    report = build_pre_push_report(classified, promoted, {"routine_patch": "tests-first"})
    lines = ["Mossy review feedback", f"Promoted: {len(promoted)}", f"Push blocked: {report['blocked']}"]
    for row in classified:
        lines.append(f"- [{row['classification']}] {row['author']}: {row['body']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.json:
        data = json.loads(args.json.read_text(encoding="utf-8"))
        print(render_context(data.get("classified", []), data.get("promoted", [])))
        return 0
    queue = load_json(feedback_paths()["queue"], {"pending": []})
    print(render_context(queue.get("pending", []), queue.get("pending", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
