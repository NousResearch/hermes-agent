"""Collect a read-only VRChat autonomy preflight evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.openclaw.vrchat_preflight import build_preflight_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect read-only evidence before a VRChat autonomy private smoke test."
    )
    parser.add_argument("--profile", default="", help="Optional VRChat autonomy profile JSON path.")
    parser.add_argument("--voicevox-url", default="http://127.0.0.1:50021")
    parser.add_argument("--harness-url", default="http://127.0.0.1:18794")
    parser.add_argument("--audio-output-device", default="", help="Virtual cable playback device to verify.")
    parser.add_argument("--require-harness", action="store_true", help="Require Hypura harness readiness.")
    parser.add_argument("--queue", default="", help="Optional observation queue JSONL path.")
    parser.add_argument(
        "--no-audio-devices",
        action="store_true",
        help="Skip output-capable audio device enumeration.",
    )
    parser.add_argument("--max-audio-devices", type=int, default=20)
    parser.add_argument(
        "--include-voicevox-synthesis",
        action="store_true",
        help="Probe VOICEVOX audio_query/synthesis without playback.",
    )
    parser.add_argument(
        "--voicevox-synthesis-text",
        default="\u30c6\u30b9\u30c8",
        help="Short text for the no-playback VOICEVOX synthesis probe.",
    )
    parser.add_argument("--voicevox-synthesis-speaker", type=int, default=None)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_preflight_bundle(
        profile_path=args.profile or None,
        voicevox_url=args.voicevox_url,
        harness_url=args.harness_url,
        audio_output_device=args.audio_output_device or None,
        require_harness=args.require_harness,
        queue_path=args.queue or None,
        include_audio_devices=not args.no_audio_devices,
        max_audio_devices=args.max_audio_devices,
        include_voicevox_synthesis=args.include_voicevox_synthesis,
        voicevox_synthesis_text=args.voicevox_synthesis_text,
        voicevox_synthesis_speaker=args.voicevox_synthesis_speaker,
        output_path=args.output or None,
    )
    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
