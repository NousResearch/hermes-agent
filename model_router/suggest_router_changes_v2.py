from __future__ import annotations

import argparse
import json
from typing import Any

from analyze_telemetry import (
    join_decisions_with_feedback,
    latest_feedback_by_request_id,
    load_events,
    split_events,
)


def segment_key(row: dict[str, Any], fields: list[str]) -> str:
    parts = []
    for field in fields:
        parts.append(f"{field}={row.get(field) or 'unknown'}")
    return " | ".join(parts)


def aggregate_segments(joined: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}

    for row in joined:
        key = segment_key(row, fields)
        bucket = buckets.setdefault(
            key,
            {
                "fields": {field: row.get(field) for field in fields},
                "total": 0,
                "with_feedback": 0,
                "success": 0,
                "bad_fit": 0,
                "failed": 0,
                "fallback_used": 0,
                "abandoned": 0,
                "ratings": [],
                "mismatch_actual_model": 0,
            },
        )

        bucket["total"] += 1
        if row["feedback_present"]:
            bucket["with_feedback"] += 1

        outcome = row.get("outcome")
        if outcome in ("success", "bad_fit", "failed", "fallback_used", "abandoned"):
            bucket[outcome] += 1
        elif row.get("fallback_used"):
            bucket["fallback_used"] += 1

        rating = row.get("user_rating")
        if isinstance(rating, int):
            bucket["ratings"].append(rating)

        actual_model = row.get("actual_model_used")
        primary_model = row.get("primary_model")
        if actual_model and primary_model and actual_model != primary_model:
            bucket["mismatch_actual_model"] += 1

    result = {}
    for key, bucket in buckets.items():
        ratings = bucket.pop("ratings")
        total = bucket["total"]
        with_feedback = bucket["with_feedback"]
        result[key] = {
            **bucket,
            "feedback_coverage_rate": round(with_feedback / total, 4) if total else 0.0,
            "success_rate": round(bucket["success"] / with_feedback, 4) if with_feedback else None,
            "fallback_rate": round(bucket["fallback_used"] / with_feedback, 4) if with_feedback else None,
            "mismatch_rate": round(bucket["mismatch_actual_model"] / with_feedback, 4) if with_feedback else None,
            "average_rating": round(sum(ratings) / len(ratings), 3) if ratings else None,
        }
    return result


def segmented_suggestions(
    joined: list[dict[str, Any]],
    *,
    min_feedback_samples: int = 4,
    high_fallback_rate: float = 0.35,
    low_success_rate: float = 0.60,
    low_rating: float = 3.2,
    high_mismatch_rate: float = 0.25,
) -> list[dict[str, Any]]:
    suggestions = []

    segment_defs = [
        ["primary_model", "task_type"],
        ["primary_model", "priority"],
        ["task_type", "priority"],
        ["task_type", "quota"],
    ]

    for fields in segment_defs:
        if joined and any(field not in joined[0] for field in fields):
            continue

        segments = aggregate_segments(joined, fields)

        for seg_key, stats in segments.items():
            with_feedback = stats["with_feedback"]
            if with_feedback < min_feedback_samples:
                continue

            success_rate = stats["success_rate"]
            fallback_rate = stats["fallback_rate"]
            avg_rating = stats["average_rating"]
            mismatch_rate = stats["mismatch_rate"]

            if fallback_rate is not None and fallback_rate > high_fallback_rate:
                suggestions.append(
                    {
                        "type": "segment_high_fallback",
                        "segment": stats["fields"],
                        "severity": "high" if fallback_rate > 0.5 else "medium",
                        "message": f"בסגמנט {seg_key} יש fallback_rate גבוה ({fallback_rate:.2%}). כדאי להקשיח routing או לבחור מודל ראשי אחר עבור המקרה הזה.",
                    }
                )

            if success_rate is not None and success_rate < low_success_rate:
                suggestions.append(
                    {
                        "type": "segment_low_success",
                        "segment": stats["fields"],
                        "severity": "high" if success_rate < 0.4 else "medium",
                        "message": f"בסגמנט {seg_key} יש success_rate נמוך ({success_rate:.2%}). זה נראה כמו כלל ניתוב בעייתי.",
                    }
                )

            if avg_rating is not None and avg_rating < low_rating:
                suggestions.append(
                    {
                        "type": "segment_low_rating",
                        "segment": stats["fields"],
                        "severity": "medium",
                        "message": f"בסגמנט {seg_key} יש דירוג ממוצע נמוך ({avg_rating:.2f}). שווה לבדוק איכות התאמה.",
                    }
                )

            if mismatch_rate is not None and mismatch_rate > high_mismatch_rate:
                suggestions.append(
                    {
                        "type": "segment_high_mismatch",
                        "segment": stats["fields"],
                        "severity": "medium",
                        "message": f"בסגמנט {seg_key} יש mismatch_rate גבוה ({mismatch_rate:.2%}). כנראה המודל הראשוני לא נכון לעיתים קרובות.",
                    }
                )

    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    suggestions.sort(key=lambda item: (severity_order.get(item["severity"], 99), item["type"], json.dumps(item["segment"], sort_keys=True)))

    if not suggestions:
        return [
            {
                "type": "no_segment_action",
                "segment": {},
                "severity": "info",
                "message": "לא נמצאו כרגע חריגות חזקות ברמת סגמנט. צריך עוד feedback או שהתצורה הנוכחית סבירה.",
            }
        ]

    return suggestions


def format_suggestions(items: list[dict[str, Any]]) -> str:
    lines = ["== Segmented Router Suggestions =="]
    for item in items:
        lines.append("")
        lines.append(f"- [{item['severity']}] {item['type']}")
        lines.append(f"  segment: {item['segment']}")
        lines.append(f"  {item['message']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suggest segmented router policy changes from telemetry")
    parser.add_argument("log_path", help="Path to telemetry JSONL log")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-feedback-samples", type=int, default=4)
    parser.add_argument("--high-fallback-rate", type=float, default=0.35)
    parser.add_argument("--low-success-rate", type=float, default=0.60)
    parser.add_argument("--low-rating", type=float, default=3.2)
    parser.add_argument("--high-mismatch-rate", type=float, default=0.25)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    events = load_events(args.log_path)
    decisions, feedbacks = split_events(events)
    latest_feedback = latest_feedback_by_request_id(feedbacks)
    joined = join_decisions_with_feedback(decisions, latest_feedback)

    suggestions = segmented_suggestions(
        joined,
        min_feedback_samples=args.min_feedback_samples,
        high_fallback_rate=args.high_fallback_rate,
        low_success_rate=args.low_success_rate,
        low_rating=args.low_rating,
        high_mismatch_rate=args.high_mismatch_rate,
    )

    if args.json:
        print(json.dumps(suggestions, ensure_ascii=False, indent=2))
    else:
        print(format_suggestions(suggestions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
