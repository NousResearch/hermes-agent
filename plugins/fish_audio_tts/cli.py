from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import status_payload, synthesize_text


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def register_cli(subparser) -> None:
    actions = subparser.add_subparsers(dest="fish_audio_tts_action")
    actions.add_parser("status", help="Show Fish Audio provider readiness.")

    synth_parser = actions.add_parser("synthesize", help="Synthesize speech with Fish Audio.")
    synth_parser.add_argument("--text", help="Text to synthesize.")
    synth_parser.add_argument("--input-path", help="Read text from a UTF-8 file.")
    synth_parser.add_argument("--output-path", help="Destination audio path.")
    synth_parser.add_argument("--format", choices=["mp3", "wav", "opus", "pcm"], default="mp3")
    synth_parser.add_argument("--voice", help="Fish Audio reference_id voice model.")
    synth_parser.add_argument("--model", default=None, help="Fish Audio model header.")
    synth_parser.add_argument("--speed", type=float, default=None, help="Speech speed from 0.5 to 2.0.")
    subparser.set_defaults(func=fish_audio_tts_command)


def fish_audio_tts_command(args: Any) -> int:
    action = getattr(args, "fish_audio_tts_action", None)
    if action == "status":
        _print_json(status_payload())
        return 0
    if action == "synthesize":
        text = Path(args.input_path).read_text(encoding="utf-8") if args.input_path else args.text
        if not text:
            print("Provide --text or --input-path.")
            return 2
        _print_json(
            synthesize_text(
                text=text,
                output_path=args.output_path,
                voice=args.voice,
                model=args.model,
                output_format=args.format,
                speed=args.speed,
            )
        )
        return 0
    print("usage: hermes fish-audio-tts {status,synthesize}")
    return 2
