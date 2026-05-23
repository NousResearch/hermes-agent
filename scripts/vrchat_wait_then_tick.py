"""Wait for VRChat readiness, then run one gated heartbeat tick."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.openclaw.vrchat_autonomy import LIVE_ACTUATION_ACK  # noqa: E402
from tools.openclaw.vrchat_observations import parse_jsonl_observation  # noqa: E402
from tools.openclaw.vrchat_preflight import wait_for_readiness_then_tick  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for read-only readiness, then run one safe profile heartbeat tick."
    )
    parser.add_argument("--profile", default="", help="Optional VRChat autonomy profile JSON path.")
    parser.add_argument("--voicevox-url", default="http://127.0.0.1:50021")
    parser.add_argument("--harness-url", default="http://127.0.0.1:18794")
    parser.add_argument("--audio-output-device", default="", help="Virtual cable playback device to verify.")
    parser.add_argument("--require-harness", action="store_true", help="Require Hypura harness readiness.")
    parser.add_argument("--queue", default="", help="Optional observation queue JSONL path.")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--max-snapshots", type=int, default=25)
    parser.add_argument("--no-persist-heartbeat", action="store_true", help="Do not persist heartbeat state.")
    parser.add_argument(
        "--allow-live-profile",
        action="store_true",
        help="Allow a non-dry-run profile when live ACK is also supplied.",
    )
    parser.add_argument("--live-ack", default="", help="Exact live acknowledgement for non-dry-run profiles.")
    parser.add_argument("--print-live-ack", action="store_true", help="Print the exact live acknowledgement and exit.")
    parser.add_argument("--emergency-stop", action="store_true", help="Disable loop state and perform no actuation.")
    parser.add_argument(
        "--observation-json",
        action="append",
        default=[],
        help="Inline JSON observation. May be supplied more than once.",
    )
    parser.add_argument(
        "--stdin-jsonl",
        action="store_true",
        help="Read additional observation JSONL from stdin before waiting.",
    )
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_live_ack:
        print(LIVE_ACTUATION_ACK)
        return 0

    result = wait_for_readiness_then_tick(
        profile_path=args.profile or None,
        observations=_load_observations(args),
        voicevox_url=args.voicevox_url,
        harness_url=args.harness_url,
        audio_output_device=args.audio_output_device or None,
        require_harness=args.require_harness,
        queue_path=args.queue or None,
        timeout_sec=args.timeout_sec,
        interval_sec=args.interval_sec,
        max_snapshots=args.max_snapshots,
        persist_heartbeat=not args.no_persist_heartbeat,
        allow_live_profile=args.allow_live_profile,
        live_ack=args.live_ack,
        emergency_stop=args.emergency_stop,
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


if __name__ == "__main__":
    raise SystemExit(main())
