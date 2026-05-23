"""Run a read-only VRChat/VOICEVOX runtime doctor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.openclaw.vrchat_preflight import build_runtime_doctor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose VRChat and VOICEVOX readiness mismatches without live actuation."
    )
    parser.add_argument("--profile", default="", help="Optional VRChat autonomy profile JSON path.")
    parser.add_argument("--voicevox-url", default="http://127.0.0.1:50021")
    parser.add_argument("--harness-url", default="http://127.0.0.1:18794")
    parser.add_argument("--audio-output-device", default="", help="Virtual cable playback device to verify.")
    parser.add_argument("--require-harness", action="store_true")
    parser.add_argument("--queue", default="", help="Optional observation queue JSONL path.")
    parser.add_argument("--no-audio-devices", action="store_true")
    parser.add_argument("--max-audio-devices", type=int, default=20)
    parser.add_argument(
        "--operator-reported-vrchat",
        action="store_true",
        help="Record that the operator reports VRChat is already running.",
    )
    parser.add_argument(
        "--operator-reported-voicevox",
        action="store_true",
        help="Record that the operator reports VOICEVOX is already running.",
    )
    parser.add_argument("--voicevox-probe-timeout", type=float, default=1.0)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_runtime_doctor(
        profile_path=args.profile or None,
        voicevox_url=args.voicevox_url,
        harness_url=args.harness_url,
        audio_output_device=args.audio_output_device or None,
        require_harness=args.require_harness,
        queue_path=args.queue or None,
        include_audio_devices=not args.no_audio_devices,
        max_audio_devices=args.max_audio_devices,
        operator_reported_vrchat=args.operator_reported_vrchat,
        operator_reported_voicevox=args.operator_reported_voicevox,
        voicevox_probe_timeout=args.voicevox_probe_timeout,
        output_path=args.output or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
