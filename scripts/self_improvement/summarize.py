#!/usr/bin/env python3
"""Summarize Hermes self-improvement telemetry without raw transcripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from hermes_constants import get_hermes_home

DEFAULT_ROOT = get_hermes_home() / "ops" / "self-improvement-log"
TASK_RUNS_FILENAME = "task_runs.jsonl"
MEMORY_AUDIT_FILENAME = "memory_context_audit.jsonl"

THRESHOLDS = {
    "input_tokens_high": 200_000,
    "cache_read_tokens_high": 2_000_000,
    "api_calls_high": 40,
    "duplicate_skill_view_count": 2,
    "repeated_cronjob_count": 2,
    "memory_context_candidate_count": 1,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _sum_int(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key) or 0) for row in rows)


def _tool_count(task_run: dict[str, Any], tool_name: str) -> int:
    for item in task_run.get("tool_stats") or []:
        if not isinstance(item, str) or ":" not in item:
            continue
        name, count = item.rsplit(":", 1)
        if name == tool_name:
            try:
                return int(count)
            except ValueError:
                return 0
    return 0


def review_flags(latest_task: dict[str, Any] | None, latest_memory_audit: dict[str, Any] | None) -> list[str]:
    if latest_task is None:
        return []
    flags: list[str] = []
    if int(latest_task.get("input_tokens") or 0) >= THRESHOLDS["input_tokens_high"]:
        flags.append("high_input_tokens")
    if int(latest_task.get("cache_read_tokens") or 0) >= THRESHOLDS["cache_read_tokens_high"]:
        flags.append("high_cache_read_tokens")
    if int(latest_task.get("api_call_count") or 0) >= THRESHOLDS["api_calls_high"]:
        flags.append("high_api_calls")
    if _tool_count(latest_task, "skill_view") >= THRESHOLDS["duplicate_skill_view_count"]:
        flags.append("duplicate_skill_view")
    if _tool_count(latest_task, "cronjob") >= THRESHOLDS["repeated_cronjob_count"]:
        flags.append("repeated_cronjob_list")
    if latest_memory_audit and int(latest_memory_audit.get("candidate_count") or 0) >= THRESHOLDS["memory_context_candidate_count"]:
        flags.append("memory_context_noise")
    return flags


def build_summary(root: Path) -> dict[str, Any]:
    task_runs = read_jsonl(root / TASK_RUNS_FILENAME)
    memory_audits = read_jsonl(root / MEMORY_AUDIT_FILENAME)
    latest_task = task_runs[-1] if task_runs else None
    latest_memory_audit = memory_audits[-1] if memory_audits else None

    telemetry_totals = {
        "input_tokens": _sum_int(task_runs, "input_tokens"),
        "output_tokens": _sum_int(task_runs, "output_tokens"),
        "reasoning_tokens": _sum_int(task_runs, "reasoning_tokens"),
        "cache_read_tokens": _sum_int(task_runs, "cache_read_tokens"),
        "cache_write_tokens": _sum_int(task_runs, "cache_write_tokens"),
        "api_calls": _sum_int(task_runs, "api_call_count"),
        "tool_calls": _sum_int(task_runs, "tool_call_count"),
    }
    return {
        "schema_version": 1,
        "kind": "self_improvement_summary",
        "root": str(root),
        "task_runs": len(task_runs),
        "memory_context_audits": len(memory_audits),
        "latest_task_session": latest_task.get("session_id") if latest_task else "",
        "telemetry_totals": telemetry_totals,
        "largest_context_items": list((latest_task or {}).get("largest_context_items") or [])[:8],
        "tool_stats": list((latest_task or {}).get("tool_stats") or []),
        "memory_context": {
            "candidate_count": int((latest_memory_audit or {}).get("candidate_count") or 0),
            "candidates_preview": [
                {
                    "reasons": candidate.get("reasons") or [],
                    "content_chars": len(str(candidate.get("content") or "")),
                    "memory_id": candidate.get("memory_id") or "",
                }
                for candidate in list((latest_memory_audit or {}).get("candidates") or [])[:5]
                if isinstance(candidate, dict)
            ],
        },
        "review_flags": review_flags(latest_task, latest_memory_audit),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Self-improvement log directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(build_summary(Path(args.root)), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
