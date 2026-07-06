#!/usr/bin/env python3
"""Validation gate for Torben auto-capture candidates."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torben_capture import capture_auto_candidate
from torben_pattern_miner import DEFAULT_SCORE_THRESHOLD, DEFAULT_SUPPORT_THRESHOLD, redact_secrets, score_candidate


SCHEMA = "torben.capture-gate.v1"


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def gate_auto_capture_candidate(
    *,
    text: str,
    source: str,
    source_id: str,
    support_count: int,
    confidence: float,
    tracker_path: Path,
    dedupe_path: Path,
    retained_path: Path,
    domain: str = "admin",
    support_threshold: int = DEFAULT_SUPPORT_THRESHOLD,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    now: datetime | None = None,
) -> dict[str, Any]:
    score = score_candidate(
        support_count=support_count,
        confidence=confidence,
        support_threshold=support_threshold,
        score_threshold=score_threshold,
    )
    safe_text = redact_secrets(text)
    if not score["passes"]:
        retained = {
            "schema": SCHEMA,
            "created_at": _iso(now),
            "status": "withheld",
            "reason": "below_validation_threshold",
            "text": safe_text,
            "source": source,
            "source_id": source_id,
            **score,
        }
        _append_jsonl(retained_path, retained)
        return retained
    capture = capture_auto_candidate(
        text=safe_text,
        source=source,
        source_id=source_id,
        score=score["validation_score"],
        threshold=score_threshold,
        tracker_path=tracker_path,
        dedupe_path=dedupe_path,
        domain=domain,
    )
    return {
        "schema": SCHEMA,
        "created_at": _iso(now),
        "status": capture["status"],
        "capture": capture,
        **score,
    }


def _default_state_path(name: str) -> Path:
    home = os.getenv("HERMES_HOME")
    root = Path(home).expanduser() if home else Path(".")
    return root / "state" / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", default=None)
    parser.add_argument("--dedupe", default=None)
    parser.add_argument("--retained", default=None)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--support-count", type=int, required=True)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--domain", default="admin")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("text", nargs="+")
    args = parser.parse_args(argv)

    payload = gate_auto_capture_candidate(
        text=" ".join(args.text),
        source=args.source,
        source_id=args.source_id,
        support_count=args.support_count,
        confidence=args.confidence,
        tracker_path=Path(args.tracker) if args.tracker else _default_state_path("torben-open-loops.csv"),
        dedupe_path=Path(args.dedupe) if args.dedupe else _default_state_path("torben-capture-dedupe.json"),
        retained_path=Path(args.retained) if args.retained else _default_state_path("torben-capture-gate-retained.jsonl"),
        domain=args.domain,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
