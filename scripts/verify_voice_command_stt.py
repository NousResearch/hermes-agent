#!/usr/bin/env python3
"""Validate a voice-backed Hermes command STT provider.

By default this script writes a temporary Hermes config that points
``stt.providers.voice`` at ``voice stream-transcribe --quiet {input_path}``,
generates a small WAV fixture with ``voice say``, runs Hermes'
``transcribe_audio`` entry point, and verifies the transcript/provider.

Pass ``--use-existing-config`` to validate a deployed HERMES_HOME without
rewriting config.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any


DEFAULT_TEXT = "hello world"
DEFAULT_EXPECT_WORDS = ("hello", "world")


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


def build_config(*, provider: str, voice_bin: str, timeout: float) -> str:
    command = f"{shlex.quote(voice_bin)} stream-transcribe --quiet {{input_path}}"
    return (
        "stt:\n"
        "  enabled: true\n"
        f"  provider: {provider}\n"
        "  providers:\n"
        f"    {provider}:\n"
        "      type: command\n"
        f"      command: {json.dumps(command)}\n"
        "      format: txt\n"
        f"      timeout: {timeout:g}\n"
    )


def write_isolated_config(
    hermes_home: Path,
    *,
    provider: str,
    voice_bin: str,
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
        build_config(provider=provider, voice_bin=voice_bin, timeout=timeout),
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


def generate_audio(
    *,
    voice_bin: str,
    text: str,
    output_path: Path,
    voice: str,
    speed: str,
    timeout: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        voice_bin,
        "--quiet",
        "say",
        "--format",
        "wav",
        "--output",
        str(output_path),
        "--voice",
        voice,
        "--speed",
        speed,
        text,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(
            f"voice say failed with exit code {completed.returncode}"
            + (f": {detail[:1000]}" if detail else "")
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise SystemExit(f"voice say did not create a non-empty WAV: {output_path}")


def run_transcription_tool(*, hermes_home: Path, audio_path: Path) -> dict[str, Any]:
    os.environ["HERMES_HOME"] = str(hermes_home)
    sys.path.insert(0, str(repo_root()))

    from tools.transcription_tools import transcribe_audio

    result = transcribe_audio(str(audio_path))
    if not result.get("success"):
        raise SystemExit(
            "transcribe_audio failed: "
            + json.dumps(result, ensure_ascii=False, indent=2)
        )
    return result


def transcript_text(result: dict[str, Any]) -> str:
    return str(result.get("transcript") or result.get("text") or "")


def validate_result(
    result: dict[str, Any],
    *,
    expected_provider: str,
    expected_words: list[str],
) -> None:
    transcript = transcript_text(result)
    transcript_lower = transcript.lower()
    failures: list[str] = []

    if result.get("provider") != expected_provider:
        failures.append(
            f"expected provider {expected_provider}, got {result.get('provider')!r}"
        )
    if not transcript.strip():
        failures.append("expected a non-empty transcript")
    for word in expected_words:
        if word.lower() not in transcript_lower:
            failures.append(f"expected transcript to contain {word!r}: {transcript!r}")

    if failures:
        raise SystemExit("validation failed:\n- " + "\n- ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Hermes STT against a command provider backed by "
            "`voice stream-transcribe --quiet` and verify the transcript."
        )
    )
    parser.add_argument("--voice-bin", default=os.environ.get("VOICE_BIN", "voice"))
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
    parser.add_argument("--audio-path", type=Path)
    parser.add_argument(
        "--provider",
        default="voice",
        help=(
            "Provider name to write in isolated mode, and expected result "
            "provider in --use-existing-config mode."
        ),
    )
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--generate-timeout", type=float, default=180.0)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--expect-word",
        action="append",
        default=None,
        help="word expected in the transcript; repeatable",
    )
    args = parser.parse_args()
    if args.expect_word is None:
        args.expect_word = list(DEFAULT_EXPECT_WORDS)
    return args


def main() -> int:
    args = parse_args()
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")

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
        temporary_home = args.hermes_home is None
        hermes_home = (
            Path(tempfile.mkdtemp(prefix="hermes-voice-command-stt."))
            if temporary_home
            else args.hermes_home.expanduser().resolve()
        )
        config_path = write_isolated_config(
            hermes_home,
            provider=args.provider,
            voice_bin=voice_bin,
            timeout=args.timeout,
            force=args.force or temporary_home,
        )
        mode = "isolated"

    audio_temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.audio_path is None:
        audio_temp_dir = tempfile.TemporaryDirectory(
            prefix="hermes-voice-command-stt-audio."
        )
        audio_path = Path(audio_temp_dir.name) / "input.wav"
        generated_audio = True
    else:
        audio_path = args.audio_path.expanduser().resolve()
        generated_audio = False

    try:
        if generated_audio:
            generate_audio(
                voice_bin=voice_bin,
                text=args.text,
                output_path=audio_path,
                voice=args.voice,
                speed=args.speed,
                timeout=args.generate_timeout,
            )
        elif not audio_path.is_file() or audio_path.stat().st_size == 0:
            raise SystemExit(f"--audio-path is not a non-empty file: {audio_path}")

        result = run_transcription_tool(hermes_home=hermes_home, audio_path=audio_path)
        validate_result(
            result,
            expected_provider=args.provider,
            expected_words=args.expect_word,
        )
        retained_audio = not generated_audio
        print(
            json.dumps(
                {
                    "success": True,
                    "mode": mode,
                    "hermes_home": str(hermes_home),
                    "config_path": str(config_path),
                    "retained": bool(args.keep_home or not temporary_home),
                    "audio_path": (
                        str(audio_path)
                        if retained_audio
                        else "<temporary; pass --audio-path to use a retained fixture>"
                    ),
                    "provider": result.get("provider"),
                    "transcript": transcript_text(result),
                    "expected_words": args.expect_word,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        if audio_temp_dir is not None:
            audio_temp_dir.cleanup()
        if temporary_home and not args.keep_home:
            shutil.rmtree(hermes_home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
