from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_events(path: str | Path) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    events = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def safe_get(d: dict[str, Any], *keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def split_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = []
    feedbacks = []
    for event in events:
        event_type = event.get("event_type", "decision")
        if event_type == "feedback":
            feedbacks.append(event)
        else:
            decisions.append(event)
    return decisions, feedbacks


def latest_feedback_by_request_id(feedbacks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fb in feedbacks:
        request_id = fb.get("request_id")
        if request_id:
            grouped[request_id].append(fb)

    latest = {}
    for request_id, items in grouped.items():
        items_sorted = sorted(items, key=lambda row: row.get("timestamp", ""))
        latest[request_id] = items_sorted[-1]
    return latest


def join_decisions_with_feedback(
    decisions: list[dict[str, Any]], latest_feedback: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    joined = []
    for decision_event in decisions:
        request_id = decision_event.get("request_id")
        feedback = latest_feedback.get(request_id)
        joined.append(
            {
                "request_id": request_id,
                "timestamp": decision_event.get("timestamp"),
                "task_type": safe_get(decision_event, "input", "task_type"),
                "priority": safe_get(decision_event, "input", "priority"),
                "quota": safe_get(decision_event, "input", "quota"),
                "primary_model": safe_get(decision_event, "decision", "primary_model"),
                "reviewer": safe_get(decision_event, "decision", "reviewer"),
                "feedback_present": feedback is not None,
                "outcome": feedback.get("outcome") if feedback else None,
                "actual_model_used": feedback.get("actual_model_used") if feedback else None,
                "fallback_used": feedback.get("fallback_used") if feedback else False,
                "user_rating": feedback.get("user_rating") if feedback else None,
            }
        )
    return joined


def aggregate_joined(joined: list[dict[str, Any]], key_name: str) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}

    for row in joined:
        key = row.get(key_name) or "unknown"
        bucket = buckets.setdefault(
            key,
            {
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
            "average_rating": round(sum(ratings) / len(ratings), 3) if ratings else None,
        }
    return result


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    decisions, feedbacks = split_events(events)
    latest_feedback = latest_feedback_by_request_id(feedbacks)
    joined = join_decisions_with_feedback(decisions, latest_feedback)

    request_ids = set()
    timestamps = []
    primary_models = Counter()
    task_types = Counter()
    priorities = Counter()
    reviewers = Counter()
    override_signals = Counter()
    outcomes = Counter()
    actual_models = Counter()
    ratings = []
    feedback_request_ids = set()

    for event in decisions:
        request_id = event.get("request_id")
        if request_id:
            request_ids.add(request_id)
        ts = event.get("timestamp")
        if ts:
            timestamps.append(ts)
        primary = safe_get(event, "decision", "primary_model")
        if primary:
            primary_models[primary] += 1
        task_type = safe_get(event, "input", "task_type")
        if task_type:
            task_types[task_type] += 1
        priority = safe_get(event, "input", "priority")
        if priority:
            priorities[priority] += 1
        reviewer = safe_get(event, "decision", "reviewer")
        if reviewer:
            reviewers[reviewer] += 1
        trace = safe_get(event, "decision", "trace", default=[]) or []
        for item in trace:
            if "normalize: has_code/has_logs -> coding" in item:
                override_signals["normalized_to_coding"] += 1
            if "override: quota=" in item:
                override_signals["quota_override"] += 1
            if "override: privacy=" in item or "privacy=sensitive + batch" in item:
                override_signals["privacy_override"] += 1
            if "override: fast + trivial + low" in item:
                override_signals["fast_trivial_override"] += 1
            if "constraint: high priority forbids cheap primary" in item:
                override_signals["high_priority_constraint"] += 1

    for fb in feedbacks:
        request_id = fb.get("request_id")
        if request_id:
            feedback_request_ids.add(request_id)
            request_ids.add(request_id)
        outcome = fb.get("outcome")
        if outcome:
            outcomes[outcome] += 1
        actual_model = fb.get("actual_model_used")
        if actual_model:
            actual_models[actual_model] += 1
        rating = fb.get("user_rating")
        if isinstance(rating, int):
            ratings.append(rating)

    return {
        "total_decisions": len(decisions),
        "total_feedback_events": len(feedbacks),
        "unique_request_ids": len(request_ids),
        "time_range": {
            "first": min(timestamps) if timestamps else None,
            "last": max(timestamps) if timestamps else None,
        },
        "primary_models": dict(primary_models),
        "task_types": dict(task_types),
        "priorities": dict(priorities),
        "reviewers": dict(reviewers),
        "override_signals": dict(override_signals),
        "feedback": {
            "unique_requests_with_feedback": len(feedback_request_ids),
            "outcomes": dict(outcomes),
            "actual_models_used": dict(actual_models),
            "average_rating": (sum(ratings) / len(ratings)) if ratings else None,
        },
        "joined": {
            "total_joined_rows": len(joined),
            "by_primary_model": aggregate_joined(joined, "primary_model"),
            "by_task_type": aggregate_joined(joined, "task_type"),
        },
    }


def format_report(summary: dict[str, Any]) -> str:
    lines = []
    lines.append("== Router Telemetry Summary ==")
    lines.append(f"Total decisions: {summary['total_decisions']}")
    lines.append(f"Total feedback events: {summary['total_feedback_events']}")
    lines.append(f"Unique request IDs: {summary['unique_request_ids']}")
    lines.append(f"Time range: {summary['time_range']['first']} -> {summary['time_range']['last']}")

    def add_ranked_section(title: str, data: dict[str, Any]):
        lines.append("")
        lines.append(f"{title}:")
        if not data:
            lines.append("  (none)")
            return
        for key, value in sorted(data.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  - {key}: {value}")

    add_ranked_section("Primary models", summary["primary_models"])
    add_ranked_section("Task types", summary["task_types"])
    add_ranked_section("Priorities", summary["priorities"])
    add_ranked_section("Reviewers", summary["reviewers"])
    add_ranked_section("Override signals", summary["override_signals"])
    add_ranked_section("Feedback outcomes", summary["feedback"]["outcomes"])
    add_ranked_section("Actual models used", summary["feedback"]["actual_models_used"])
    lines.append("")
    lines.append(f"Average rating: {summary['feedback']['average_rating']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze router telemetry JSONL")
    parser.add_argument("log_path", help="Path to telemetry JSONL log")
    parser.add_argument("--json", action="store_true", help="Print raw JSON summary")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    events = load_events(args.log_path)
    summary = summarize(events)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
