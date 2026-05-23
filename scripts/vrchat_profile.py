"""Prepare or inspect a local VRChat autonomy profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.openclaw.vrchat_autonomy import LIVE_ACTUATION_ACK, load_autonomy_profile  # noqa: E402
from tools.openclaw.vrchat_profile import prepare_autonomy_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or inspect a local VRChat autonomy profile."
    )
    parser.add_argument("--profile", default="", help="Optional VRChat autonomy profile JSON path.")
    parser.add_argument("--show", action="store_true", help="Load and validate the profile without writing.")
    parser.add_argument("--print-live-ack", action="store_true", help="Print the exact live acknowledgement.")
    parser.add_argument("--disabled", action="store_true", help="Write the profile with enabled=false.")
    parser.add_argument(
        "--mode",
        choices=["observe", "private_test", "trusted_instance", "public"],
        default="private_test",
    )
    parser.add_argument("--audio-output-device", default="CABLE Input")
    parser.add_argument("--vrchat-microphone-device", default="CABLE Output")
    parser.add_argument("--require-harness", action="store_true")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--no-chatbox", action="store_true")
    parser.add_argument("--allow-movement", action="store_true")
    parser.add_argument("--allow-interrupt", action="store_true")
    parser.add_argument("--speaker", type=int, default=8)
    parser.add_argument("--persona", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--arm-live", action="store_true")
    parser.add_argument("--live-ack", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_live_ack:
        print(LIVE_ACTUATION_ACK)
        return 0

    if args.show:
        result = load_autonomy_profile(args.profile or None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["success"] else 1

    result = prepare_autonomy_profile(
        profile_path=args.profile or None,
        enabled=not args.disabled,
        mode=args.mode,
        audio_output_device=args.audio_output_device,
        vrchat_microphone_device=args.vrchat_microphone_device,
        require_harness=args.require_harness,
        allow_voice=not args.no_voice,
        allow_chatbox=not args.no_chatbox,
        allow_movement=args.allow_movement,
        allow_interrupt=args.allow_interrupt,
        voicevox_speaker=args.speaker,
        persona=args.persona,
        task=args.task,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        arm_live=args.arm_live,
        live_ack=args.live_ack,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
