#!/usr/bin/env python3
"""Silent cron wrapper for Torben pattern mining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torben_job_contract import torben_home
from torben_open_loops import load_loops
from torben_pattern_miner import mine_patterns, write_pattern_proposals


JOB_NAME = "torben-pattern-miner"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def collect_pattern_events(profile_home: Path | None = None) -> list[dict[str, Any]]:
    home = profile_home or torben_home()
    state = home / "state"
    events: list[dict[str, Any]] = []
    for row in load_loops(state / "torben-open-loops.csv"):
        events.append({"source": "open_loop", "text": row.item, "confidence": 0.75})
    for row in _read_jsonl(state / "torben-capture-gate-retained.jsonl"):
        if row.get("text"):
            events.append({"source": "capture_gate", "text": row["text"], "confidence": row.get("confidence", 0.7)})
    for row in _read_jsonl(state / "torben-action-ledger.jsonl"):
        summary = row.get("summary")
        if summary:
            events.append({"source": "action_ledger", "text": summary, "confidence": 0.8})
    for row in _read_jsonl(state / "torben-capture-confirmations.jsonl"):
        text = str(row.get("text") or "")
        prefix = "Captured loop #"
        if text.startswith(prefix) and ": " in text:
            events.append({"source": "signal_capture", "text": text.split(": ", 1)[1], "confidence": 0.85})
    return events


def run_pattern_miner(profile_home: Path | None = None) -> dict[str, Any]:
    home = profile_home or torben_home()
    payload = mine_patterns(events=collect_pattern_events(home))
    write_pattern_proposals(home / "state" / "torben-pattern-proposals.json", payload)
    return payload


def main() -> int:
    run_pattern_miner()
    return 0


if __name__ == "__main__":
    from torben_job_contract import run_job

    raise SystemExit(run_job(JOB_NAME, main))
