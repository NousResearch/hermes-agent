from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_telemetry import (
    join_decisions_with_feedback,
    latest_feedback_by_request_id,
    split_events,
    summarize,
)
from log_router_feedback import build_feedback_event
from propose_config_patch import build_patch_proposal, load_yaml
from route_cli import build_parser as build_route_parser, maybe_log_decision
from model_router import (
    Mode,
    Privacy,
    Priority,
    Quota,
    RouterInput,
    Speed,
    TaskType,
    load_default_config,
    route_model,
)
from suggest_router_changes import suggest_from_summary
from suggest_router_changes_v2 import segmented_suggestions
from telemetry import append_jsonl
from validate_router_config import validate_router_config


def run_route(log_path: Path, request_id: str, *, task_type: str, mode: str = "draft", priority: str = "medium", privacy: str = "normal", quota: str = "normal", speed: str = "normal", has_code: bool = False, has_logs: bool = False) -> dict:
    config = load_default_config()
    router_input = RouterInput(
        task_type=TaskType(task_type),
        mode=Mode(mode),
        priority=Priority(priority),
        privacy=Privacy(privacy),
        quota=Quota(quota),
        speed=Speed(speed),
        has_code=has_code,
        has_logs=has_logs,
    )
    decision = route_model(router_input, config)

    args = build_route_parser().parse_args([
        "--task-type", task_type,
        "--mode", mode,
        "--priority", priority,
        "--privacy", privacy,
        "--quota", quota,
        "--speed", speed,
        "--log-path", str(log_path),
        "--request-id", request_id,
        *( ["--has-code"] if has_code else [] ),
        *( ["--has-logs"] if has_logs else [] ),
    ])
    maybe_log_decision(args, config, router_input, decision)

    return {
        "request_id": request_id,
        "primary_model": decision.primary_model.value,
        "reviewer": decision.reviewer.value if decision.reviewer else None,
        "trace": decision.trace,
    }


def add_feedback(log_path: Path, request_id: str, *, outcome: str, actual_model_used: str | None = None, fallback_used: bool = False, user_rating: int | None = None, notes: str | None = None) -> None:
    args = Namespace(
        request_id=request_id,
        outcome=outcome,
        actual_model_used=actual_model_used,
        fallback_used=fallback_used,
        user_rating=user_rating,
        notes=notes,
    )
    append_jsonl(log_path, build_feedback_event(args))


def build_demo_output(log_path: Path) -> dict:
    root = Path(__file__).resolve().parent.parent
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    routes = [
        run_route(log_path, "req-demo-1", task_type="coding", mode="execute", priority="high", has_code=True),
        run_route(log_path, "req-demo-2", task_type="chat", priority="low", quota="critical"),
        run_route(log_path, "req-demo-3", task_type="batch", privacy="local_only"),
    ]

    add_feedback(log_path, "req-demo-1", outcome="success", actual_model_used="gpt-5.4", user_rating=5, notes="coding flow worked well")
    add_feedback(log_path, "req-demo-2", outcome="bad_fit", actual_model_used="claude-sonnet-4.6", fallback_used=True, user_rating=2, notes="deepseek היה חלש לשיחה הזאת")
    add_feedback(log_path, "req-demo-3", outcome="success", actual_model_used="ollama", user_rating=4, notes="privacy rule נשמר")

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize(events)
    suggestions = suggest_from_summary(summary, min_feedback_samples=1, low_usage_threshold=0)

    decisions_raw, feedbacks_raw = split_events(events)
    latest_feedback = latest_feedback_by_request_id(feedbacks_raw)
    joined_rows = join_decisions_with_feedback(decisions_raw, latest_feedback)
    segment_suggestions = segmented_suggestions(joined_rows, min_feedback_samples=1)

    config = load_yaml(root / "router_config.yaml")
    patch_result = build_patch_proposal(config, segment_suggestions)
    validation = validate_router_config(patch_result["proposed_config"])

    return {
        "log_path": str(log_path),
        "routes": routes,
        "summary": {
            "total_decisions": summary["total_decisions"],
            "total_feedback_events": summary["total_feedback_events"],
            "primary_models": summary["primary_models"],
            "feedback_outcomes": summary["feedback"]["outcomes"],
        },
        "global_suggestions": suggestions,
        "segment_suggestions": segment_suggestions,
        "patch_generated_count": patch_result["generated_count"],
        "patch_generated_overrides": patch_result["generated_overrides"],
        "patched_config_valid": validation["valid"],
        "patched_config_errors": validation["errors"],
        "patched_config_warnings": validation["warnings"],
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    output = build_demo_output(root / "sample_data" / "demo_router_log.jsonl")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
