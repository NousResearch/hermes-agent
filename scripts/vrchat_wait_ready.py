"""Poll read-only VRChat autonomy readiness until ready or timeout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.openclaw.vrchat_preflight import wait_for_readiness  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for read-only VRChat autonomy readiness without live actuation."
    )
    parser.add_argument("--profile", default="", help="Optional VRChat autonomy profile JSON path.")
    parser.add_argument("--voicevox-url", default="http://127.0.0.1:50021")
    parser.add_argument("--harness-url", default="http://127.0.0.1:18794")
    parser.add_argument("--audio-output-device", default="", help="Virtual cable playback device to verify.")
    parser.add_argument("--require-harness", action="store_true", help="Require Hypura harness readiness.")
    parser.add_argument("--queue", default="", help="Optional observation queue JSONL path.")
    parser.add_argument(
        "--include-audio-devices",
        action="store_true",
        help="List output-capable audio devices on each poll. Default: disabled for cheaper polling.",
    )
    parser.add_argument("--max-audio-devices", type=int, default=20)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--max-snapshots", type=int, default=25)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = wait_for_readiness(
        profile_path=args.profile or None,
        voicevox_url=args.voicevox_url,
        harness_url=args.harness_url,
        audio_output_device=args.audio_output_device or None,
        require_harness=args.require_harness,
        queue_path=args.queue or None,
        include_audio_devices=args.include_audio_devices,
        max_audio_devices=args.max_audio_devices,
        timeout_sec=args.timeout_sec,
        interval_sec=args.interval_sec,
        max_snapshots=args.max_snapshots,
        output_path=args.output or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
