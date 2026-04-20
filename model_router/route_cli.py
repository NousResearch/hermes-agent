from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from telemetry import append_jsonl, build_event


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operational model router CLI")
    parser.add_argument("--task-type", required=True, choices=[item.value for item in TaskType])
    parser.add_argument("--mode", default=Mode.DRAFT.value, choices=[item.value for item in Mode])
    parser.add_argument("--priority", default=Priority.MEDIUM.value, choices=[item.value for item in Priority])
    parser.add_argument("--privacy", default=Privacy.NORMAL.value, choices=[item.value for item in Privacy])
    parser.add_argument("--quota", default=Quota.NORMAL.value, choices=[item.value for item in Quota])
    parser.add_argument("--speed", default=Speed.NORMAL.value, choices=[item.value for item in Speed])
    parser.add_argument("--has-code", action="store_true")
    parser.add_argument("--has-logs", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--log-path", help="Append telemetry event to JSONL file")
    parser.add_argument("--request-id", help="Optional request id for telemetry correlation")
    parser.add_argument("--no-log", action="store_true", help="Disable logging even if log-path was provided")
    return parser


def format_text(decision, show_trace: bool) -> str:
    lines = [
        f"primary_model: {decision.primary_model.value}",
        f"fallback_models: {[item.value for item in decision.fallback_models]}",
        f"reviewer: {decision.reviewer.value if decision.reviewer else None}",
        f"reason: {decision.reason}",
    ]
    if show_trace:
        lines.append("trace:")
        lines.extend([f"  - {item}" for item in decision.trace])
    return "\n".join(lines)


def maybe_log_decision(args, config, router_input, decision) -> None:
    if args.no_log or not args.log_path:
        return
    event = build_event(
        router_version=config.router_version,
        config_path=config.config_path,
        request_input=router_input,
        decision=decision,
        request_id=args.request_id,
    )
    append_jsonl(Path(args.log_path), event)


def main() -> int:
    args = build_parser().parse_args()
    config = load_default_config()

    router_input = RouterInput(
        task_type=TaskType(args.task_type),
        mode=Mode(args.mode),
        priority=Priority(args.priority),
        privacy=Privacy(args.privacy),
        quota=Quota(args.quota),
        speed=Speed(args.speed),
        has_code=args.has_code,
        has_logs=args.has_logs,
    )
    decision = route_model(router_input, config)
    maybe_log_decision(args, config, router_input, decision)

    if args.primary_only:
        print(decision.primary_model.value)
        return 0

    if args.json:
        print(
            json.dumps(
                {
                    "primary_model": decision.primary_model.value,
                    "fallback_models": [item.value for item in decision.fallback_models],
                    "reviewer": decision.reviewer.value if decision.reviewer else None,
                    "reason": decision.reason,
                    "trace": decision.trace,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(format_text(decision, args.show_trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
