#!/usr/bin/env python3
"""Langfuse experiment gate.

Default and --self-check stay offline. --live-read calls the public experiments
API when LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL are
set. No OpenBao. No provider key. Not part of ci.yaml all-checks-pass.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "tests" / "ci" / "langfuse_baselines"
BASELINE_PATH = BASELINE_DIR / "baseline.json"


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    name = data.get("dataset")
    if not isinstance(name, str) or not name.startswith("hermes-"):
        raise SystemExit("baseline dataset name mismatch")
    floor = data.get("min_score")
    if not isinstance(floor, (int, float)):
        raise SystemExit("baseline min_score missing")
    if not isinstance(data.get("score_name"), str) or not isinstance(data.get("experiment_prefix"), str):
        raise SystemExit("baseline fields missing")
    return data


def baseline_paths(argv) -> list[Path]:
    if "--baseline" in argv:
        i = argv.index("--baseline")
        return [Path(argv[i + 1])]
    return sorted(BASELINE_DIR.glob("baseline*.json"))


def decide(score, floor: float) -> tuple[int, str]:
    if not isinstance(score, (int, float)):
        return 1, "MISSING score"
    if float(score) < float(floor):
        return 1, f"REGRESSION score={score} baseline={floor}"
    return 0, f"PASS score={score} baseline={floor}"


def _self_check() -> None:
    assert decide(1.0, 1.0)[0] == 0
    assert decide(0.5, 1.0)[0] == 1
    assert decide(None, 1.0)[0] == 1
    baseline = load_baseline()
    assert baseline["min_score"] == 1.0
    assert baseline["score_name"] == "hermes_prompt_ok_avg"
    assert baseline["experiment_prefix"] == "hermes-offline"
    for path in baseline_paths([]):
        load_baseline(path)
    print("self-check ok")


def _score_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _attached(experiment, name: str):
    for score in experiment.scores or []:
        value = _score_number(getattr(score, "value", None))
        if getattr(score, "name", None) == name and value is not None:
            return value
    return None


def _item_mean(items, name: str):
    vals = []
    for item in items or []:
        for score in getattr(item, "scores", None) or []:
            value = _score_number(getattr(score, "value", None))
            if getattr(score, "name", None) == name and value is not None:
                vals.append(value)
    if not vals:
        return None
    return sum(vals) / len(vals)


def live_read() -> int:
    """Read latest experiment scores. Writes nothing."""
    missing = [key for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL") if not os.environ.get(key)]
    if missing:
        print("BLOCKED missing", ",".join(missing))
        return 2
    os.environ.setdefault("LANGFUSE_HOST", os.environ["LANGFUSE_BASE_URL"])
    from langfuse import get_client

    lf = get_client()
    try:
        codes = []
        now = datetime.now(timezone.utc)
        for path in baseline_paths([]):
            baseline = load_baseline(path)
            dataset = lf.api.datasets.get(baseline["dataset"])
            latest = None
            for _ in range(6):
                resp = lf.api.experiments.list(
                    from_start_time=now - timedelta(days=7),
                    dataset_id=dataset.id,
                    fields="core,scores",
                    limit=20,
                )
                rows = [row for row in resp.data if (row.name or "").startswith(baseline["experiment_prefix"])]
                rows.sort(key=lambda row: row.end_time, reverse=True)
                if rows:
                    latest = rows[0]
                    if _attached(latest, baseline["score_name"]) is not None:
                        break
                time.sleep(2)
            if latest is None:
                print("MISSING experiment", path.name)
                codes.append(1)
                continue
            value = _attached(latest, baseline["score_name"])
            source = "attached"
            if value is None:
                items = lf.api.experiments.list_items(
                    from_start_time=now - timedelta(days=7),
                    experiment_id=latest.id,
                    fields="core,scores",
                    limit=50,
                )
                value = _item_mean(items.data, baseline["score_name"])
                source = "item_avg"
            code, message = decide(value, baseline["min_score"])
            print(path.name, message, "source", source)
            print("experiment_id", latest.id, "name", latest.name, "items", latest.item_count)
            codes.append(code)
        return 1 if any(codes) else 0
    finally:
        lf.shutdown()


def main(argv) -> int:
    _self_check()
    if "--live-read" in argv:
        return live_read()
    if "--self-check" in argv or argv == []:
        return 0
    print("pass --live-read to query Langfuse, or --self-check")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
