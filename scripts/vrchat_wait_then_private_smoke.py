"""Wait for readiness, then prepare or run gated private VRChat smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.openclaw.vrchat_autonomy import LIVE_ACTUATION_ACK  # noqa: E402
from tools.openclaw.vrchat_smoke import wait_then_private_smoke  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for read-only readiness, then prepare gated private VRChat smoke."
    )
    parser.add_argument("--profile", default="", help="Optional VRChat autonomy profile JSON path.")
    parser.add_argument("--voicevox-url", default="http://127.0.0.1:50021")
    parser.add_argument("--harness-url", default="http://127.0.0.1:18794")
    parser.add_argument("--require-harness", action="store_true", help="Require Hypura harness readiness.")
    parser.add_argument("--audio-output-device", default="", help="Virtual cable playback device to verify.")
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
    parser.add_argument("--chatbox-text", default="Hermes VRChat private smoke test.")
    parser.add_argument("--speak-text", default="Hermes smoke test.")
    parser.add_argument("--avatar-action", default="")
    parser.add_argument(
        "--allow-live-smoke",
        action="store_true",
        help="Attempt live private smoke only if readiness, profile, and ACK gates pass.",
    )
    parser.add_argument("--live-ack", default="", help="Exact acknowledgement required for --allow-live-smoke.")
    parser.add_argument("--print-live-ack", action="store_true", help="Print the exact acknowledgement and exit.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_live_ack:
        print(LIVE_ACTUATION_ACK)
        return 0

    result = wait_then_private_smoke(
        profile_path=args.profile or None,
        voicevox_url=args.voicevox_url,
        harness_url=args.harness_url,
        require_harness=args.require_harness,
        audio_output_device=args.audio_output_device or None,
        queue_path=args.queue or None,
        include_audio_devices=args.include_audio_devices,
        max_audio_devices=args.max_audio_devices,
        timeout_sec=args.timeout_sec,
        interval_sec=args.interval_sec,
        max_snapshots=args.max_snapshots,
        chatbox_text=args.chatbox_text,
        speak_text=args.speak_text,
        avatar_action=args.avatar_action,
        allow_live_smoke=args.allow_live_smoke,
        live_ack=args.live_ack,
        output_path=args.output or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
