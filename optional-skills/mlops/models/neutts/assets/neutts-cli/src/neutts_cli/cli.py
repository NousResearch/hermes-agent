from __future__ import annotations

import argparse
import json
import sys

from .config import AppConfig
from .core import (
    KNOWN_MODELS,
    doctor_report,
    list_voices,
    load_voice,
    platform_notes,
    run_install,
    save_voice,
    synthesize,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone CLI for local NeuTTS workflows"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser(
        "install", help="Install NeuTTS into the current Python environment"
    )
    install_parser.add_argument(
        "--llama",
        action="store_true",
        help="Install llama-cpp-python support via neutts[llama]",
    )
    install_parser.add_argument(
        "--onnx",
        action="store_true",
        help="Install ONNX decoder support via neutts[onnx]",
    )
    install_parser.add_argument(
        "--all", action="store_true", help="Install all upstream NeuTTS extras"
    )
    install_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the install command without running it",
    )

    subparsers.add_parser("doctor", help="Inspect NeuTTS CLI environment")
    subparsers.add_parser(
        "list-models", help="Show known official NeuTTS model repositories"
    )
    subparsers.add_parser("list-voices", help="Show local voice profiles")

    add_voice_parser = subparsers.add_parser(
        "add-voice", help="Save a local voice profile from a reference sample"
    )
    add_voice_parser.add_argument("name", help="Voice profile name")
    add_voice_parser.add_argument(
        "--ref-audio", required=True, help="Reference WAV file"
    )
    add_voice_parser.add_argument(
        "--ref-text", help="Transcript text for the reference audio"
    )
    add_voice_parser.add_argument(
        "--ref-text-file",
        help="Path to a text file containing the reference transcript",
    )
    add_voice_parser.add_argument(
        "--language", default="unknown", help="Optional language tag"
    )

    synth_parser = subparsers.add_parser(
        "synth", help="Synthesize speech to a WAV file"
    )
    synth_parser.add_argument(
        "--text", nargs="+", required=True, help="Text to synthesize"
    )
    synth_parser.add_argument("--voice", help="Saved voice profile name")
    synth_parser.add_argument(
        "--ref-audio", help="Reference audio path when not using --voice"
    )
    synth_parser.add_argument(
        "--ref-text", help="Reference transcript when not using --voice"
    )
    synth_parser.add_argument("--out", default="out.wav", help="Output WAV file path")

    config_parser = subparsers.add_parser(
        "config", help="View or update default synthesis settings"
    )
    config_parser.add_argument("--backbone-repo")
    config_parser.add_argument("--backbone-device")
    config_parser.add_argument("--codec-repo")
    config_parser.add_argument("--codec-device")
    config_parser.add_argument("--sample-rate", type=int)
    config_parser.add_argument(
        "--default-voice",
        help="Voice profile name to use when --voice is omitted from synth",
    )

    return parser


def _read_ref_text(args: argparse.Namespace) -> str:
    if args.ref_text:
        return args.ref_text.strip()
    if args.ref_text_file:
        with open(args.ref_text_file, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    raise ValueError("Provide either --ref-text or --ref-text-file")


def _normalize_text_arg(value: str | list[str]) -> str:
    if isinstance(value, list):
        return " ".join(value).strip()
    return value.strip()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "install":
            commands = run_install(args.llama, args.onnx, args.all, args.dry_run)
            print(
                json.dumps(
                    {
                        "commands": commands,
                        "notes": platform_notes(),
                        "dry_run": args.dry_run,
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "doctor":
            print(json.dumps(doctor_report(), indent=2))
            return 0

        if args.command == "list-models":
            print(json.dumps(KNOWN_MODELS, indent=2))
            return 0

        if args.command == "list-voices":
            profiles = [profile.__dict__ for profile in list_voices()]
            print(json.dumps(profiles, indent=2))
            return 0

        if args.command == "add-voice":
            metadata_path = save_voice(
                name=args.name,
                ref_audio=args.ref_audio,
                ref_text=_read_ref_text(args),
                language=args.language,
            )
            profile = load_voice(args.name)
            print(
                json.dumps(
                    {"saved": str(metadata_path), "voice": profile.__dict__}, indent=2
                )
            )
            return 0

        if args.command == "synth":
            output = synthesize(
                text=_normalize_text_arg(args.text),
                out=args.out,
                voice=args.voice,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
            )
            print(json.dumps({"output": str(output)}, indent=2))
            return 0

        if args.command == "config":
            config = AppConfig.load()
            changed = False
            for field in (
                "backbone_repo",
                "backbone_device",
                "codec_repo",
                "codec_device",
                "sample_rate",
                "default_voice",
            ):
                value = getattr(args, field, None)
                if value is not None:
                    setattr(config, field, value)
                    changed = True
            if changed:
                config.save()
            print(json.dumps(config.__dict__, indent=2))
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
