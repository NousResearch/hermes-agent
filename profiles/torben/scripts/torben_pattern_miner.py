#!/usr/bin/env python3
"""Algorithmic Torben pattern miner with review-gated proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "torben.pattern-miner.v1"
DEFAULT_SUPPORT_THRESHOLD = 3
DEFAULT_SCORE_THRESHOLD = 0.7
SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|xox[baprs]-[A-Za-z0-9\-]{8,}|ghp_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,}|"
    r"access_token\s*[:=]\s*['\"]?[^'\"\s]+|refresh_token\s*[:=]\s*['\"]?[^'\"\s]+|"
    r"password\s*[:=]\s*['\"]?[^'\"\s]+)",
    re.I,
)


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def redact_secrets(text: str) -> str:
    return SECRET_RE.sub("[REDACTED]", text)


def normalize_action(text: str) -> str:
    cleaned = redact_secrets(_clean_text(text).lower())
    cleaned = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<date>", cleaned)
    cleaned = re.sub(r"\b\d+\b", "<num>", cleaned)
    return cleaned


def proposal_id(action_key: str) -> str:
    return "pattern-" + hashlib.sha256(action_key.encode("utf-8")).hexdigest()[:12]


def validation_score(*, support_count: int, confidence: float, support_threshold: int = DEFAULT_SUPPORT_THRESHOLD) -> float:
    support_component = min(max(support_count, 0) / max(support_threshold + 1, 1), 1.0)
    confidence_component = min(max(float(confidence), 0.0), 1.0)
    return round((support_component * 0.6) + (confidence_component * 0.4), 3)


def score_candidate(
    *,
    support_count: int,
    confidence: float,
    support_threshold: int = DEFAULT_SUPPORT_THRESHOLD,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> dict[str, Any]:
    score = validation_score(support_count=support_count, confidence=confidence, support_threshold=support_threshold)
    return {
        "support_count": support_count,
        "confidence": round(float(confidence), 3),
        "validation_score": score,
        "support_threshold": support_threshold,
        "score_threshold": score_threshold,
        "passes": support_count >= support_threshold and score >= score_threshold,
        "review_gated": True,
    }


def mine_patterns(
    *,
    events: list[dict[str, Any]],
    support_threshold: int = DEFAULT_SUPPORT_THRESHOLD,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    now: datetime | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        text = _clean_text(event.get("text") or event.get("action") or "")
        if not text:
            continue
        groups.setdefault(normalize_action(text), []).append(event)

    proposals: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    for action_key, items in sorted(groups.items()):
        support_count = len(items)
        confidence_values = [float(item.get("confidence", 1.0)) for item in items]
        confidence = sum(confidence_values) / max(len(confidence_values), 1)
        score = score_candidate(
            support_count=support_count,
            confidence=confidence,
            support_threshold=support_threshold,
            score_threshold=score_threshold,
        )
        source_set = sorted({_clean_text(item.get("source") or "unknown") for item in items})
        sanitized_examples = [redact_secrets(_clean_text(item.get("text") or item.get("action") or "")) for item in items[:3]]
        base = {
            "schema": SCHEMA,
            "id": proposal_id(action_key),
            "type": "automation_candidate",
            "action_key": action_key,
            "summary": f"Automation candidate: {sanitized_examples[0] if sanitized_examples else action_key}",
            "support_count": support_count,
            "source_count": len(source_set),
            "sources": source_set,
            "examples": sanitized_examples,
            "created_at": _iso(now),
            **score,
        }
        if score["passes"]:
            proposals.append({**base, "status": "review_gated"})
        else:
            withheld.append({**base, "status": "withheld"})
    return {
        "schema": SCHEMA,
        "generated_at": _iso(now),
        "support_threshold": support_threshold,
        "score_threshold": score_threshold,
        "review_gated": True,
        "proposals": proposals,
        "withheld": withheld,
    }


def write_pattern_proposals(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def proposals_for_weekly_reset(payload: dict[str, Any]) -> list[dict[str, Any]]:
    proposals = payload.get("proposals") or []
    items: list[dict[str, Any]] = []
    for proposal in proposals:
        if not isinstance(proposal, dict) or proposal.get("status") != "review_gated":
            continue
        items.append(
            {
                "handle": proposal.get("id"),
                "summary": proposal.get("summary"),
                "risk_class": "low",
                "category": "pattern_miner",
                "status": "review_gated",
                "support_count": proposal.get("support_count"),
                "validation_score": proposal.get("validation_score"),
            }
        )
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-json", required=True)
    parser.add_argument("--output")
    parser.add_argument("--support-threshold", type=int, default=DEFAULT_SUPPORT_THRESHOLD)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = mine_patterns(
        events=json.loads(args.events_json),
        support_threshold=args.support_threshold,
        score_threshold=args.score_threshold,
    )
    if args.output:
        write_pattern_proposals(Path(args.output), payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
