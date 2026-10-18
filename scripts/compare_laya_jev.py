#!/usr/bin/env python3
"""Compare Jev and Laya shadow-router JSONL records by Hermes task_id."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        if task_id:
            out[task_id] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jev", type=Path)
    parser.add_argument("laya", type=Path)
    args = parser.parse_args()
    jev, laya = load(args.jev), load(args.laya)
    ids = sorted(set(jev) & set(laya))
    comparable = [i for i in ids if jev[i].get("ok") and laya[i].get("ok")]
    agree = [i for i in comparable if jev[i].get("choice") == laya[i].get("choice")]
    jev_ms = [float(jev[i]["latency_ms"]) for i in comparable if jev[i].get("latency_ms") is not None]
    laya_ms = [float(laya[i]["latency_ms"]) for i in comparable if laya[i].get("latency_ms") is not None]
    print(json.dumps({
        "jev_records": len(jev),
        "laya_records": len(laya),
        "paired_records": len(ids),
        "comparable_successes": len(comparable),
        "choice_agreement": round(len(agree) / len(comparable), 4) if comparable else None,
        "jev_mean_latency_ms": round(mean(jev_ms), 1) if jev_ms else None,
        "laya_mean_latency_ms": round(mean(laya_ms), 1) if laya_ms else None,
        "disagreements": [
            {"task_id": i, "jev": jev[i].get("choice"), "laya": laya[i].get("choice")}
            for i in comparable if jev[i].get("choice") != laya[i].get("choice")
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
