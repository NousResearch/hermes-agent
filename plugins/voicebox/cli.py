from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import ensure_hakua_profile, list_profiles, settings, status_payload, synthesize_text, transcribe_audio


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def register_cli(subparser) -> None:
    actions = subparser.add_subparsers(dest="voicebox_action")

    actions.add_parser("status", help="Show Voicebox server and provider status.")
    actions.add_parser("profiles", help="List Voicebox voice profiles.")
    import_parser = actions.add_parser(
        "import-hakua",
        help="Import Irodori hakua.ogg into Voicebox as a cloned profile.",
    )
    import_parser.add_argument(
        "--reference-text",
        default=None,
        help="Transcript/reference text for the Hakua sample.",
    )
    actions.add_parser(
        "import-hakua",
        help="Import Irodori hakua.ogg into Voicebox as a cloned profile.",
    )

    synth_parser = actions.add_parser("synthesize", help="Synthesize speech via Voicebox /speak.")
    synth_parser.add_argument("--text", help="Text to synthesize.")
    synth_parser.add_argument("--input-path", help="Read text from a UTF-8 file.")
    synth_parser.add_argument("--output-path", help="Destination audio path.")
    synth_parser.add_argument("--voice", default=None, help="Voice profile name or id.")
    synth_parser.add_argument("--model", default=None, help="Voicebox engine id.")
    synth_parser.add_argument("--language", default=None, help="Language code.")
    synth_parser.add_argument(
        "--personality",
        action="store_true",
        help="Rewrite text in the profile personality before TTS.",
    )

    transcribe_parser = actions.add_parser("transcribe", help="Transcribe audio via Voicebox /transcribe.")
    transcribe_parser.add_argument("--audio-path", required=True, help="Audio file to transcribe.")
    transcribe_parser.add_argument("--language", default=None)
    transcribe_parser.add_argument("--model", default=None)

    subparser.set_defaults(func=voicebox_command)


def voicebox_command(args: Any) -> int:
    action = getattr(args, "voicebox_action", None)
    if action == "status":
        return _cmd_status(args)
    if action == "profiles":
        return _cmd_profiles(args)
    if action == "import-hakua":
        return _cmd_import_hakua(args)
    if action == "synthesize":
        return _cmd_synthesize(args)
    if action == "transcribe":
        return _cmd_transcribe(args)
    print("usage: hermes voicebox {status,profiles,import-hakua,synthesize,transcribe}")
    return 2


def _cmd_status(args: Any) -> int:
    del args
    _print_json(status_payload())
    return 0


def _cmd_profiles(args: Any) -> int:
    del args
    _print_json({"profiles": list_profiles()})
    return 0


def _cmd_import_hakua(args: Any) -> int:
    del args
    _print_json(ensure_hakua_profile())
    return 0


def _read_text(args: Any) -> str:
    if getattr(args, "text", None):
        return str(args.text)
    input_path = getattr(args, "input_path", None)
    if input_path:
        return Path(input_path).expanduser().read_text(encoding="utf-8")
    print("Provide --text or --input-path.")
    raise SystemExit(2)


def _cmd_synthesize(args: Any) -> int:
    text = _read_text(args)
    result = synthesize_text(
        text=text,
        output_path=getattr(args, "output_path", None),
        voice=getattr(args, "voice", None),
        model=getattr(args, "model", None),
        language=getattr(args, "language", None),
        personality=True if getattr(args, "personality", False) else None,
    )
    _print_json(result)
    return 0


def _cmd_transcribe(args: Any) -> int:
    result = transcribe_audio(
        audio_path=args.audio_path,
        language=getattr(args, "language", None),
        model=getattr(args, "model", None),
    )
    _print_json(result)
    return 0
