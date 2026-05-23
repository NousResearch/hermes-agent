"""Run staged private-instance smoke checks for Hermes VRChat autonomy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.openclaw.vrchat_autonomy import LIVE_ACTUATION_ACK  # noqa: E402
from tools.openclaw.vrchat_smoke import prepare_private_smoke, run_private_smoke  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hermes VRChat private smoke check.")
    parser.add_argument("--profile", default="", help="Optional VRChat autonomy profile JSON path.")
    parser.add_argument("--voicevox-url", default="http://127.0.0.1:50021")
    parser.add_argument("--harness-url", default="http://127.0.0.1:18794")
    parser.add_argument("--require-harness", action="store_true")
    parser.add_argument("--audio-output-device", default="", help="Virtual cable playback device to verify.")
    parser.add_argument("--chatbox-text", default="Hermes VRChat private smoke test.")
    parser.add_argument("--speak-text", default="Hermes smoke test.")
    parser.add_argument("--avatar-action", default="")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Evaluate live-smoke gates and build a dry-run plan without live execution.",
    )
    parser.add_argument("--live", action="store_true", help="Attempt live actuation if every safety gate passes.")
    parser.add_argument("--live-ack", default="", help="Exact acknowledgement required for --live.")
    parser.add_argument(
        "--print-live-ack",
        action="store_true",
        help="Print the exact acknowledgement string and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_live_ack:
        print(LIVE_ACTUATION_ACK)
        return 0
    common_args = {
        "profile_path": args.profile or None,
        "voicevox_url": args.voicevox_url,
        "harness_url": args.harness_url,
        "require_harness": args.require_harness,
        "audio_output_device": args.audio_output_device or None,
        "chatbox_text": args.chatbox_text,
        "speak_text": args.speak_text,
        "avatar_action": args.avatar_action,
    }
    if args.prepare_only:
        result = prepare_private_smoke(**common_args, live_ack=args.live_ack)
    else:
        result = run_private_smoke(**common_args, live=args.live, live_ack=args.live_ack)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") or result.get("code") in {"DRY_RUN_SMOKE_DONE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
