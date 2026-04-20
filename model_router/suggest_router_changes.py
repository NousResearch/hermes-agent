from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_telemetry import load_events, summarize


def suggest_from_summary(
    summary: dict[str, Any],
    *,
    min_feedback_samples: int = 5,
    high_fallback_rate: float = 0.35,
    low_success_rate: float = 0.60,
    low_rating: float = 3.2,
    high_mismatch_rate: float = 0.25,
    low_usage_threshold: int = 2,
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []

    by_model = summary.get("joined", {}).get("by_primary_model", {})
    by_task = summary.get("joined", {}).get("by_task_type", {})
    primary_models = summary.get("primary_models", {})

    for model, stats in by_model.items():
        total = stats.get("total", 0)
        with_feedback = stats.get("with_feedback", 0)
        success_rate = stats.get("success_rate")
        fallback_rate = stats.get("fallback_rate")
        avg_rating = stats.get("average_rating")
        mismatch_count = stats.get("mismatch_actual_model", 0)
        mismatch_rate = (mismatch_count / with_feedback) if with_feedback else None

        if with_feedback >= min_feedback_samples:
            if fallback_rate is not None and fallback_rate > high_fallback_rate:
                suggestions.append(
                    {
                        "type": "high_fallback_rate",
                        "scope": "primary_model",
                        "target": model,
                        "severity": "high" if fallback_rate > 0.5 else "medium",
                        "message": f"המודל {model} מציג fallback_rate גבוה ({fallback_rate:.2%}). כדאי לצמצם ניתוב אליו למשימות גבוליות או להקשיח תנאי כניסה.",
                    }
                )

            if success_rate is not None and success_rate < low_success_rate:
                suggestions.append(
                    {
                        "type": "low_success_rate",
                        "scope": "primary_model",
                        "target": model,
                        "severity": "high" if success_rate < 0.4 else "medium",
                        "message": f"המודל {model} מציג success_rate נמוך ({success_rate:.2%}). כדאי לבדוק באילו סוגי משימות הוא נכשל ולהפחית שימוש ראשי.",
                    }
                )

            if avg_rating is not None and avg_rating < low_rating:
                suggestions.append(
                    {
                        "type": "low_rating",
                        "scope": "primary_model",
                        "target": model,
                        "severity": "medium",
                        "message": f"המודל {model} מקבל דירוג ממוצע נמוך ({avg_rating:.2f}). שווה לבדוק אם הוא נבחר למשימות לא מתאימות.",
                    }
                )

            if mismatch_rate is not None and mismatch_rate > high_mismatch_rate:
                suggestions.append(
                    {
                        "type": "high_mismatch_rate",
                        "scope": "primary_model",
                        "target": model,
                        "severity": "medium",
                        "message": f"המודל {model} מוחלף בפועל לעיתים קרובות ({mismatch_rate:.2%} mismatch). כנראה הבחירה הראשונית לא אופטימלית.",
                    }
                )

    for task_type, stats in by_task.items():
        with_feedback = stats.get("with_feedback", 0)
        success_rate = stats.get("success_rate")
        fallback_rate = stats.get("fallback_rate")
        avg_rating = stats.get("average_rating")

        if with_feedback >= min_feedback_samples:
            if success_rate is not None and success_rate < low_success_rate:
                suggestions.append(
                    {
                        "type": "weak_task_routing",
                        "scope": "task_type",
                        "target": task_type,
                        "severity": "high" if success_rate < 0.4 else "medium",
                        "message": f"בקטגוריית {task_type} יש success_rate נמוך ({success_rate:.2%}). כדאי לבדוק אם המודל הראשי לקטגוריה הזו נכון.",
                    }
                )

            if fallback_rate is not None and fallback_rate > high_fallback_rate:
                suggestions.append(
                    {
                        "type": "task_type_high_fallback",
                        "scope": "task_type",
                        "target": task_type,
                        "severity": "medium",
                        "message": f"בקטגוריית {task_type} יש fallback_rate גבוה ({fallback_rate:.2%}). ייתכן שהראוטר בוחר מודל ראשוני חלש מדי.",
                    }
                )

            if avg_rating is not None and avg_rating < low_rating:
                suggestions.append(
                    {
                        "type": "task_type_low_rating",
                        "scope": "task_type",
                        "target": task_type,
                        "severity": "medium",
                        "message": f"בקטגוריית {task_type} הדירוג הממוצע נמוך ({avg_rating:.2f}). שווה לבדוק התאמת מודל או policy.",
                    }
                )

    for model, count in primary_models.items():
        if count <= low_usage_threshold:
            suggestions.append(
                {
                    "type": "low_usage_model",
                    "scope": "primary_model",
                    "target": model,
                    "severity": "low",
                    "message": f"המודל {model} כמעט לא נבחר ({count} פעמים). ייתכן שהחוק שמוביל אליו מיותר או לא נגיש בפועל.",
                }
            )

    if not suggestions:
        suggestions.append(
            {
                "type": "no_action",
                "scope": "global",
                "target": "router",
                "severity": "info",
                "message": "לא נמצאו כרגע דפוסים חזקים שמצדיקים שינוי policy. כדאי לאסוף עוד נתונים.",
            }
        )

    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    suggestions.sort(key=lambda item: (severity_order.get(item["severity"], 99), item["type"], item["target"]))
    return suggestions


def format_suggestions(suggestions: list[dict[str, Any]]) -> str:
    lines = ["== Router Change Suggestions =="]
    for item in suggestions:
        lines.append("")
        lines.append(f"- [{item['severity']}] {item['type']} / {item['scope']} / {item['target']}")
        lines.append(f"  {item['message']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suggest router policy changes from telemetry")
    parser.add_argument("log_path", help="Path to telemetry JSONL log")
    parser.add_argument("--json", action="store_true", help="Output suggestions as JSON")
    parser.add_argument("--min-feedback-samples", type=int, default=5)
    parser.add_argument("--high-fallback-rate", type=float, default=0.35)
    parser.add_argument("--low-success-rate", type=float, default=0.60)
    parser.add_argument("--low-rating", type=float, default=3.2)
    parser.add_argument("--high-mismatch-rate", type=float, default=0.25)
    parser.add_argument("--low-usage-threshold", type=int, default=2)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    events = load_events(Path(args.log_path))
    summary = summarize(events)
    suggestions = suggest_from_summary(
        summary,
        min_feedback_samples=args.min_feedback_samples,
        high_fallback_rate=args.high_fallback_rate,
        low_success_rate=args.low_success_rate,
        low_rating=args.low_rating,
        high_mismatch_rate=args.high_mismatch_rate,
        low_usage_threshold=args.low_usage_threshold,
    )

    if args.json:
        print(json.dumps(suggestions, ensure_ascii=False, indent=2))
    else:
        print(format_suggestions(suggestions))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
