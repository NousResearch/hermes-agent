from __future__ import annotations

import argparse
from pathlib import Path

from telemetry import append_jsonl, utc_now_iso


VALID_OUTCOMES = {"success", "bad_fit", "fallback_used", "failed", "abandoned"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append router feedback event to telemetry log")
    parser.add_argument("--log-path", required=True, help="Path to telemetry JSONL log")
    parser.add_argument("--request-id", required=True, help="Request ID to attach feedback to")
    parser.add_argument("--outcome", required=True, choices=sorted(VALID_OUTCOMES))
    parser.add_argument("--actual-model-used", help="Model actually used in the end")
    parser.add_argument("--fallback-used", action="store_true", help="Mark that fallback was used")
    parser.add_argument("--user-rating", type=int, choices=[1, 2, 3, 4, 5], help="Optional 1-5 rating")
    parser.add_argument("--notes", help="Optional freeform notes")
    return parser


def build_feedback_event(args: argparse.Namespace) -> dict:
    return {
        "timestamp": utc_now_iso(),
        "event_type": "feedback",
        "request_id": args.request_id,
        "outcome": args.outcome,
        "fallback_used": bool(args.fallback_used),
        "actual_model_used": args.actual_model_used,
        "user_rating": args.user_rating,
        "notes": args.notes,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    event = build_feedback_event(args)
    append_jsonl(Path(args.log_path), event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
