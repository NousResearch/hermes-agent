#!/usr/bin/env python3
"""Validate the voice WebRTC sidecar full-duplex preflight from Hermes.

This is a small Hermes-side wrapper around the optional voice repo smoke:

    examples/webrtc-sidecar/full_duplex_loopback_smoke.py

It keeps the heavy WebRTC dependencies in the voice smoke environment, captures
the machine-readable JSON output, and verifies the fields Hermes cares about
before an operator attempts a real WhatsApp Calling session.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_INBOUND_TEXT = "hello world"
DEFAULT_OUTBOUND_TEXT = "Hello from Hermes through the voice WebRTC sidecar."
DEFAULT_EXPECT_WORDS = ("hello", "world")
DEFAULT_TIMEOUT = 90.0
DEFAULT_MAX_QUEUED_TX_MS = 1_000
MAX_CHILD_ERROR_CHARS = 4000


def resolve_executable(value: str, *, label: str) -> str:
    if "/" in value:
        path = Path(value).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"{label} is not executable: {path}")
        # Preserve virtualenv entrypoint symlinks. Resolving
        # /tmp/venv/bin/python to the base interpreter bypasses pyvenv.cfg and
        # drops the sidecar dependencies installed in that venv.
        return os.path.abspath(os.path.expanduser(value))

    found = shutil.which(value)
    if not found:
        raise SystemExit(f"{label} not found on PATH: {value}")
    return found


def default_python_bin() -> str:
    configured = os.environ.get("VOICE_WEBRTC_PYTHON")
    if configured:
        return configured
    common_venv = Path("/tmp/voice-webrtc-venv/bin/python")
    if common_venv.is_file() and os.access(common_venv, os.X_OK):
        return str(common_venv)
    return sys.executable


def full_duplex_smoke_path(voice_repo: Path) -> Path:
    return (
        voice_repo
        / "examples"
        / "webrtc-sidecar"
        / "full_duplex_loopback_smoke.py"
    )


def resolve_voice_smoke(voice_repo: Path) -> Path:
    path = full_duplex_smoke_path(voice_repo.expanduser().resolve())
    if not path.is_file():
        raise SystemExit(
            "voice full-duplex sidecar smoke not found: "
            f"{path}. Pass --voice-repo pointing at rgbkrk/voice main."
        )
    return path


def build_smoke_command(
    *,
    python_bin: str,
    smoke_path: Path,
    voice_bin: str,
    inbound_text: str,
    outbound_text: str,
    voice: str,
    speed: str,
    timeout: float,
    expect_words: list[str],
    max_queued_tx_ms: int,
) -> list[str]:
    command = [
        python_bin,
        str(smoke_path),
        "--voice-bin",
        voice_bin,
        "--inbound-text",
        inbound_text,
        "--outbound-text",
        outbound_text,
        "--voice",
        voice,
        "--speed",
        speed,
        "--timeout",
        f"{timeout:g}",
        "--max-queued-tx-ms",
        str(max_queued_tx_ms),
    ]
    for word in expect_words:
        command.extend(["--expect-word", word])
    return command


def run_smoke_command(
    command: list[str],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def format_child_error(text: str, *, max_chars: int = MAX_CHILD_ERROR_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    omitted = len(stripped) - max_chars
    return f"... omitted {omitted} chars from child stderr ...\n{stripped[-max_chars:]}"


def parse_smoke_json(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    text = stdout.strip()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if text[index + end :].strip():
            continue
        if not isinstance(parsed, dict):
            raise ValueError("full-duplex smoke JSON root must be an object")
        return parsed
    raise ValueError("full-duplex smoke did not print a JSON object")


def queued_tx_ms(queued_tx_bytes: object, audio: dict[str, Any]) -> int:
    try:
        queued_bytes = int(queued_tx_bytes or 0)
        sample_rate = int(audio.get("sample_rate") or 0)
        channels = int(audio.get("channels") or 0)
        bytes_per_sample = int(audio.get("bytes_per_sample") or 2)
    except (TypeError, ValueError):
        return 0

    bytes_per_second = sample_rate * channels * bytes_per_sample
    if queued_bytes <= 0 or bytes_per_second <= 0:
        return 0
    return round(queued_bytes * 1_000 / bytes_per_second)


def validate_smoke_result(
    result: dict[str, Any],
    *,
    max_queued_tx_ms: int,
) -> None:
    failures: list[str] = []

    if result.get("success") is not True:
        failures.append("success must be true")

    audio = result.get("audio")
    if not isinstance(audio, dict):
        failures.append("audio must be an object")
    else:
        if int(audio.get("sample_rate") or 0) != 48_000:
            failures.append("audio.sample_rate must be 48000")
        if int(audio.get("channels") or 0) != 1:
            failures.append("audio.channels must be 1")
        if int(audio.get("frame_ms") or 0) != 20:
            failures.append("audio.frame_ms must be 20")
        if str(audio.get("encoding") or "") != "pcm_s16le":
            failures.append("audio.encoding must be pcm_s16le")
        queued_ms = result.get("queued_tx_ms")
        if queued_ms is None:
            queued_ms = queued_tx_ms(result.get("queued_tx_bytes"), audio)
        try:
            queued_ms_int = int(queued_ms)
        except (TypeError, ValueError):
            failures.append("queued_tx_ms must be an integer")
        else:
            if queued_ms_int > max_queued_tx_ms:
                failures.append(
                    "queued_tx_ms must be <= "
                    f"{max_queued_tx_ms} (got {queued_ms_int})"
                )

    transcript = str(result.get("transcript") or "").strip()
    if not transcript:
        failures.append("transcript must be non-empty")
    if int(result.get("outbound_webrtc_bytes") or 0) <= 0:
        failures.append("outbound_webrtc_bytes must be positive")
    if int(result.get("decoded_pcm_bytes") or 0) <= 0:
        failures.append("decoded_pcm_bytes must be positive")

    stt = result.get("stt")
    if not isinstance(stt, dict):
        failures.append("stt must be an object")
    elif not str(stt.get("text") or "").strip():
        failures.append("stt.text must be non-empty")

    if failures:
        raise SystemExit("validation failed:\n- " + "\n- ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voice-repo",
        type=Path,
        default=Path(os.environ.get("VOICE_REPO", ".")),
        help="Path to an rgbkrk/voice checkout containing examples/webrtc-sidecar",
    )
    parser.add_argument("--python-bin", default=default_python_bin())
    parser.add_argument("--voice-bin", default=os.environ.get("VOICE_BIN", "voice"))
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--inbound-text", default=DEFAULT_INBOUND_TEXT)
    parser.add_argument("--outbound-text", default=DEFAULT_OUTBOUND_TEXT)
    parser.add_argument(
        "--max-queued-tx-ms",
        type=int,
        default=DEFAULT_MAX_QUEUED_TX_MS,
        help=(
            "Maximum outbound sidecar queue depth allowed at the end of the "
            "full-duplex smoke."
        ),
    )
    parser.add_argument(
        "--expect-word",
        action="append",
        default=None,
        help="word expected in the inbound transcript; repeatable",
    )
    args = parser.parse_args()
    if args.max_queued_tx_ms < 0:
        parser.error("--max-queued-tx-ms must be non-negative")
    if args.expect_word is None:
        args.expect_word = list(DEFAULT_EXPECT_WORDS)
    return args


def main() -> int:
    args = parse_args()
    python_bin = resolve_executable(args.python_bin, label="sidecar Python")
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")
    smoke_path = resolve_voice_smoke(args.voice_repo)
    command = build_smoke_command(
        python_bin=python_bin,
        smoke_path=smoke_path,
        voice_bin=voice_bin,
        inbound_text=args.inbound_text,
        outbound_text=args.outbound_text,
        voice=args.voice,
        speed=args.speed,
        timeout=args.timeout,
        expect_words=args.expect_word,
        max_queued_tx_ms=args.max_queued_tx_ms,
    )

    completed = run_smoke_command(command, timeout=args.timeout + 5)
    if completed.returncode != 0:
        stderr = format_child_error(completed.stderr)
        raise SystemExit(
            f"full-duplex smoke exited with code {completed.returncode}"
            + (f": {stderr}" if stderr else "")
        )

    try:
        result = parse_smoke_json(completed.stdout)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    validate_smoke_result(result, max_queued_tx_ms=args.max_queued_tx_ms)

    print(
        json.dumps(
            {
                "success": True,
                "voice_repo": str(args.voice_repo.expanduser().resolve()),
                "voice_bin": voice_bin,
                "python_bin": python_bin,
                "smoke": result,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
