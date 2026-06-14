#!/usr/bin/env python3
"""Validate a voice-backed Hermes command TTS provider.

By default this script writes a temporary Hermes config that points a command
provider at `voice say --format ogg-opus`, runs Hermes' real
text_to_speech_tool entry point, and verifies the resulting audio is
WhatsApp-ready Ogg/Opus.

Pass `--use-existing-config` to validate a deployed HERMES_HOME without
rewriting config.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_TEXT = "Hermes voice command provider smoke test."


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_executable(value: str, *, label: str) -> str:
    if "/" in value:
        path = Path(value).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"{label} is not executable: {path}")
        return str(path.resolve())

    found = shutil.which(value)
    if not found:
        raise SystemExit(f"{label} not found on PATH: {value}")
    return found


def build_config(*, provider: str, voice_bin: str, voice: str, speed: str, timeout: float) -> str:
    command = (
        f"{shlex.quote(voice_bin)} say --format ogg-opus "
        "--input-file {input_path} --output {output_path} "
        "--voice {voice} --speed {speed}"
    )
    return (
        "tts:\n"
        f"  provider: {provider}\n"
        "  providers:\n"
        f"    {provider}:\n"
        "      type: command\n"
        f"      command: {json.dumps(command)}\n"
        "      output_format: ogg\n"
        "      voice_compatible: true\n"
        f"      voice: {json.dumps(voice)}\n"
        f"      speed: {json.dumps(speed)}\n"
        f"      timeout: {timeout:g}\n"
        "      max_text_length: 2000\n"
    )


def write_isolated_config(
    hermes_home: Path,
    *,
    provider: str,
    voice_bin: str,
    voice: str,
    speed: str,
    timeout: float,
    force: bool,
) -> Path:
    hermes_home.mkdir(parents=True, exist_ok=True)
    config_path = hermes_home / "config.yaml"
    if config_path.exists() and not force:
        raise SystemExit(
            f"{config_path} already exists; pass --force to overwrite it"
        )
    config_path.write_text(
        build_config(
            provider=provider,
            voice_bin=voice_bin,
            voice=voice,
            speed=speed,
            timeout=timeout,
        ),
        encoding="utf-8",
    )
    return config_path


def default_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def existing_config_path(hermes_home: Path) -> Path:
    config_path = hermes_home / "config.yaml"
    if not config_path.is_file():
        raise SystemExit(
            f"{config_path} does not exist; remove --use-existing-config or pass --hermes-home"
        )
    return config_path


def run_tts_tool(
    *,
    hermes_home: Path,
    text: str,
    output_path: str | None,
    platform: str,
) -> dict[str, Any]:
    os.environ["HERMES_HOME"] = str(hermes_home)
    os.environ["HERMES_SESSION_PLATFORM"] = platform
    sys.path.insert(0, str(repo_root()))

    from tools.tts_tool import text_to_speech_tool

    raw = text_to_speech_tool(text, output_path=output_path)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"text_to_speech_tool returned invalid JSON: {raw}") from exc

    if not result.get("success"):
        raise SystemExit(
            "text_to_speech_tool failed: "
            + json.dumps(result, ensure_ascii=False, indent=2)
        )
    return result


def parse_ffprobe(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values


def probe_audio(path: Path, *, ffprobe_bin: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_ffprobe(completed.stdout)


def validate_result(
    result: dict[str, Any],
    probe: dict[str, str],
    *,
    expected_provider: str,
) -> None:
    path = Path(str(result.get("file_path") or ""))
    media_tag = str(result.get("media_tag") or "")
    failures: list[str] = []

    if result.get("provider") != expected_provider:
        failures.append(
            f"expected provider {expected_provider}, got {result.get('provider')!r}"
        )
    if result.get("voice_compatible") is not True:
        failures.append("expected voice_compatible=true")
    if not media_tag.startswith("[[audio_as_voice]]\nMEDIA:"):
        failures.append("expected media_tag to start with [[audio_as_voice]] and MEDIA:")
    if path.suffix.lower() != ".ogg":
        failures.append(f"expected .ogg output, got {path}")
    if not path.is_file() or path.stat().st_size == 0:
        failures.append(f"expected non-empty output file at {path}")
    if probe.get("codec_name") != "opus":
        failures.append(f"expected codec_name=opus, got {probe.get('codec_name')!r}")
    if probe.get("sample_rate") != "48000":
        failures.append(f"expected sample_rate=48000, got {probe.get('sample_rate')!r}")
    if probe.get("channels") != "1":
        failures.append(f"expected channels=1, got {probe.get('channels')!r}")

    if failures:
        raise SystemExit("validation failed:\n- " + "\n- ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Hermes TTS against a command provider backed by "
            "`voice say --format ogg-opus` and verify Ogg/Opus output."
        )
    )
    parser.add_argument("--voice-bin", default=os.environ.get("VOICE_BIN", "voice"))
    parser.add_argument("--ffprobe-bin", default=os.environ.get("FFPROBE_BIN", "ffprobe"))
    parser.add_argument("--hermes-home", type=Path)
    parser.add_argument(
        "--use-existing-config",
        action="store_true",
        help=(
            "Use the existing config.yaml under --hermes-home or HERMES_HOME. "
            "The default mode writes an isolated temporary config."
        ),
    )
    parser.add_argument("--keep-home", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-path")
    parser.add_argument(
        "--provider",
        default="kokoro",
        help=(
            "Provider name to write in isolated mode, and expected result "
            "provider in --use-existing-config mode."
        ),
    )
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--platform", default="whatsapp")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffprobe_bin = resolve_executable(args.ffprobe_bin, label="ffprobe")

    if args.use_existing_config:
        temporary_home = False
        hermes_home = (
            args.hermes_home.expanduser().resolve()
            if args.hermes_home is not None
            else default_hermes_home().resolve()
        )
        config_path = existing_config_path(hermes_home)
        mode = "existing"
    else:
        voice_bin = resolve_executable(args.voice_bin, label="voice binary")
        temporary_home = args.hermes_home is None
        hermes_home = (
            Path(tempfile.mkdtemp(prefix="hermes-voice-command-tts."))
            if temporary_home
            else args.hermes_home.expanduser().resolve()
        )
        config_path = write_isolated_config(
            hermes_home,
            provider=args.provider,
            voice_bin=voice_bin,
            voice=args.voice,
            speed=args.speed,
            timeout=args.timeout,
            force=args.force or temporary_home,
        )
        mode = "isolated"

    try:
        result = run_tts_tool(
            hermes_home=hermes_home,
            text=args.text,
            output_path=args.output_path,
            platform=args.platform,
        )
        audio_path = Path(str(result["file_path"]))
        probe = probe_audio(audio_path, ffprobe_bin=ffprobe_bin)
        validate_result(result, probe, expected_provider=args.provider)
        retained = bool(args.output_path) or not (temporary_home and not args.keep_home)
        print(
            json.dumps(
                {
                    "success": True,
                    "mode": mode,
                    "hermes_home": str(hermes_home),
                    "config_path": str(config_path),
                    "retained": retained,
                    "file_path": (
                        str(audio_path)
                        if retained
                        else "<temporary; pass --keep-home or --output-path to retain>"
                    ),
                    "provider": result.get("provider"),
                    "voice_compatible": result.get("voice_compatible"),
                    "media_tag_prefix": str(result.get("media_tag", "")).splitlines()[0],
                    "codec_name": probe.get("codec_name"),
                    "sample_rate": probe.get("sample_rate"),
                    "channels": probe.get("channels"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        if temporary_home and not args.keep_home:
            shutil.rmtree(hermes_home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
