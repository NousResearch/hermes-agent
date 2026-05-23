"""Run a read-only completion audit for the VRChat autonomy goal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.openclaw.vrchat_completion_audit import build_completion_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit current evidence for the Hermes VRChat Neuro-style autonomy goal."
    )
    parser.add_argument("--profile", default="", help="Optional VRChat autonomy profile JSON path.")
    parser.add_argument("--voicevox-url", default="http://127.0.0.1:50021")
    parser.add_argument("--harness-url", default="http://127.0.0.1:18794")
    parser.add_argument("--audio-output-device", default="", help="Virtual cable playback device to verify.")
    parser.add_argument("--require-harness", action="store_true")
    parser.add_argument("--queue", default="", help="Optional observation queue JSONL path.")
    parser.add_argument("--include-audio-devices", action="store_true")
    parser.add_argument(
        "--skip-voicevox-synthesis",
        action="store_true",
        help="Skip the no-playback VOICEVOX audio_query/synthesis probe.",
    )
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_completion_audit(
        profile_path=args.profile or None,
        voicevox_url=args.voicevox_url,
        harness_url=args.harness_url,
        audio_output_device=args.audio_output_device or None,
        require_harness=args.require_harness,
        queue_path=args.queue or None,
        include_audio_devices=args.include_audio_devices,
        include_voicevox_synthesis=not args.skip_voicevox_synthesis,
        output_path=args.output or None,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
