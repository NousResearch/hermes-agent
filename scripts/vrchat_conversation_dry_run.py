"""Run a dry-run multimodal VRChat conversation proof."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.openclaw.vrchat_conversation import run_multimodal_conversation_dry_run  # noqa: E402
from tools.openclaw.vrchat_observations import parse_jsonl_observation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan a dry-run multimodal VRChat conversation turn without live actuation."
    )
    parser.add_argument("--profile", default="", help="Optional VRChat autonomy profile JSON path.")
    parser.add_argument(
        "--observation-json",
        action="append",
        default=[],
        help="Inline JSON observation. May be supplied more than once.",
    )
    parser.add_argument(
        "--stdin-jsonl",
        action="store_true",
        help="Read additional observation JSONL from stdin.",
    )
    parser.add_argument("--decision-json", default="", help="Optional structured decision JSON override.")
    parser.add_argument("--persist-observations", action="store_true", help="Persist normalized observations.")
    parser.add_argument("--queue", default="", help="Optional observation queue JSONL path.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_multimodal_conversation_dry_run(
        profile_path=args.profile or None,
        observations=_load_observations(args) or None,
        decision=_load_decision(args.decision_json),
        persist_observations=args.persist_observations,
        queue_path=args.queue or None,
        output_path=args.output or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


def _load_observations(args: argparse.Namespace) -> list[dict]:
    observations: list[dict] = []
    for raw in args.observation_json:
        parsed = parse_jsonl_observation(raw)
        if not parsed["success"]:
            raise SystemExit(f"invalid --observation-json: {parsed['error']}")
        observations.append(parsed["observation"])
    if args.stdin_jsonl:
        for line in sys.stdin:
            if not line.strip():
                continue
            parsed = parse_jsonl_observation(line)
            if not parsed["success"]:
                raise SystemExit(f"invalid stdin observation: {parsed['error']}")
            observations.append(parsed["observation"])
    return observations


def _load_decision(raw: str) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --decision-json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("invalid --decision-json: decision must be an object")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
