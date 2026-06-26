#!/usr/bin/env python3
"""Validate a voice-backed WhatsApp Calling TTS stream command.

This script renders the same command template Hermes uses for
``calling_sidecar_tts_stream_command``, runs it against a temporary input file,
and verifies that stdout is raw 48 kHz mono 20 ms ``pcm_s16le`` frames suitable
for the voice WebRTC sidecar.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any


DEFAULT_TEXT = "Hermes voice stream command smoke test."
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CHANNELS = 1
DEFAULT_FRAME_MS = 20
DEFAULT_BYTES_PER_SAMPLE = 2
DEFAULT_ENCODING = "pcm_s16le"
DEFAULT_MIN_PEAK = 384
DEFAULT_MIN_DURATION_MS = 200


@dataclass(frozen=True)
class AudioContract:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    frame_ms: int = DEFAULT_FRAME_MS
    encoding: str = DEFAULT_ENCODING
    bytes_per_sample: int = DEFAULT_BYTES_PER_SAMPLE
    raw_outbound_pcm_command: str = ""
    raw_inbound_pcm_command: str = ""
    completed_voice_note_command: str = ""
    streamed_voice_note_command: str = ""

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate * self.frame_ms // 1_000

    @property
    def frame_bytes(self) -> int:
        return self.samples_per_frame * self.channels * self.bytes_per_sample

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.bytes_per_sample

    def validate(self) -> None:
        failures: list[str] = []
        if self.sample_rate <= 0:
            failures.append("sample_rate must be positive")
        if self.channels <= 0:
            failures.append("channels must be positive")
        if self.frame_ms <= 0:
            failures.append("frame_ms must be positive")
        if self.bytes_per_sample <= 0:
            failures.append("bytes_per_sample must be positive")
        if self.encoding != DEFAULT_ENCODING:
            failures.append(f"encoding must be {DEFAULT_ENCODING}")
        if self.samples_per_frame <= 0:
            failures.append("samples_per_frame must be positive")
        if self.frame_bytes <= 0:
            failures.append("frame_bytes must be positive")
        if failures:
            raise SystemExit("invalid audio contract:\n- " + "\n- ".join(failures))

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frame_ms": self.frame_ms,
            "encoding": self.encoding,
            "bytes_per_sample": self.bytes_per_sample,
            "samples_per_frame": self.samples_per_frame,
            "frame_bytes": self.frame_bytes,
        }


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


def build_default_command_template(voice_bin: str) -> str:
    return (
        f"{shlex.quote(voice_bin)} stream --quiet "
        f"--sample-rate {{sample_rate}} --frame-ms {{frame_ms}} "
        f"--raw-output - --input-file {{input_path}} "
        f"--voice {{voice}} --speed {{speed}}"
    )


def load_voice_stream_contract(voice_bin: str, *, timeout: float = 5.0) -> dict[str, Any]:
    completed = subprocess.run(
        [voice_bin, "stream-contract"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    parsed = json.loads(completed.stdout)
    if not isinstance(parsed, dict):
        raise ValueError("voice stream-contract root must be an object")
    return parsed


def audio_contract_from_voice(
    voice_bin: str,
    *,
    fallback: AudioContract,
) -> AudioContract:
    try:
        contract = load_voice_stream_contract(voice_bin)
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise SystemExit(f"failed to load `voice stream-contract`: {exc}") from exc

    audio = contract.get("audio")
    if not isinstance(audio, dict):
        raise SystemExit("voice stream-contract audio section must be an object")

    surface_commands = validate_voice_surfaces(contract.get("voice_surfaces"), audio)
    return AudioContract(
        sample_rate=int(audio.get("sample_rate") or fallback.sample_rate),
        channels=int(audio.get("channels") or fallback.channels),
        frame_ms=int(audio.get("frame_ms") or fallback.frame_ms),
        encoding=str(audio.get("encoding") or fallback.encoding),
        bytes_per_sample=int(audio.get("bytes_per_sample") or fallback.bytes_per_sample),
        raw_outbound_pcm_command=surface_commands.get("raw_outbound_pcm", ""),
        raw_inbound_pcm_command=surface_commands.get("raw_inbound_pcm", ""),
        completed_voice_note_command=surface_commands.get("completed_voice_note", ""),
        streamed_voice_note_command=surface_commands.get("streamed_voice_note", ""),
    )


def validate_voice_surfaces(
    surfaces: object,
    audio: dict[str, object],
) -> dict[str, str]:
    if surfaces is None:
        raise SystemExit("voice stream-contract is missing voice_surfaces")
    if not isinstance(surfaces, dict):
        raise SystemExit("voice stream-contract voice_surfaces section must be an object")

    frame_bytes = int(audio.get("frame_bytes") or 0)
    encoding = str(audio.get("encoding") or "")
    commands: dict[str, str] = {}

    completed = surface_object(surfaces, "completed_voice_note")
    if completed.get("output") != "audio/ogg; codecs=opus":
        raise SystemExit("completed_voice_note output must be audio/ogg; codecs=opus")
    if completed.get("transport") != "completed_file":
        raise SystemExit("completed_voice_note transport must be completed_file")
    completed_command = surface_command(completed, "completed_voice_note")
    require_command_parts(
        completed_command,
        "completed_voice_note",
        ["voice say", "--format ogg-opus", "--output"],
    )
    commands["completed_voice_note"] = completed_command

    streamed = surface_object(surfaces, "streamed_voice_note")
    if streamed.get("output") != "audio/ogg; codecs=opus":
        raise SystemExit("streamed_voice_note output must be audio/ogg; codecs=opus")
    if streamed.get("transport") != "daemon_stream_encoded_file":
        raise SystemExit("streamed_voice_note transport must be daemon_stream_encoded_file")
    streamed_command = surface_command(streamed, "streamed_voice_note")
    require_command_parts(
        streamed_command,
        "streamed_voice_note",
        ["voice stream", "--output", "--format ogg-opus"],
    )
    commands["streamed_voice_note"] = streamed_command

    outbound = surface_object(surfaces, "raw_outbound_pcm")
    if outbound.get("output") != encoding:
        raise SystemExit("raw_outbound_pcm output must match audio.encoding")
    if outbound.get("transport") != "stdout_pcm_frames":
        raise SystemExit("raw_outbound_pcm transport must be stdout_pcm_frames")
    if int(outbound.get("frame_bytes") or 0) != frame_bytes:
        raise SystemExit("raw_outbound_pcm frame_bytes must match audio.frame_bytes")
    outbound_command = surface_command(outbound, "raw_outbound_pcm")
    require_command_parts(
        outbound_command,
        "raw_outbound_pcm",
        ["voice stream", "--raw-output", "--sample-rate", "--frame-ms"],
    )
    commands["raw_outbound_pcm"] = outbound_command

    inbound = surface_object(surfaces, "raw_inbound_pcm")
    if inbound.get("input") != encoding:
        raise SystemExit("raw_inbound_pcm input must match audio.encoding")
    if inbound.get("transport") != "stdin_pcm_frames":
        raise SystemExit("raw_inbound_pcm transport must be stdin_pcm_frames")
    if int(inbound.get("frame_bytes") or 0) != frame_bytes:
        raise SystemExit("raw_inbound_pcm frame_bytes must match audio.frame_bytes")
    inbound_command = surface_command(inbound, "raw_inbound_pcm")
    require_command_parts(
        inbound_command,
        "raw_inbound_pcm",
        ["voice stream-transcribe", "--raw-input", "--sample-rate", "--frame-ms"],
    )
    commands["raw_inbound_pcm"] = inbound_command

    return commands


def surface_object(surfaces: dict[str, object], name: str) -> dict[str, object]:
    surface = surfaces.get(name)
    if not isinstance(surface, dict):
        raise SystemExit(f"voice stream-contract voice_surfaces.{name} must be an object")
    return surface


def surface_command(surface: dict[str, object], name: str) -> str:
    command = str(surface.get("command") or "")
    if not command:
        raise SystemExit(
            f"voice stream-contract voice_surfaces.{name}.command must be non-empty"
        )
    return command


def require_command_parts(command: str, name: str, parts: list[str]) -> None:
    missing = [part for part in parts if part not in command]
    if missing:
        raise SystemExit(
            f"voice stream-contract voice_surfaces.{name}.command is missing: "
            + ", ".join(missing)
        )


def render_stream_command(
    command_template: str,
    *,
    input_path: Path,
    text: str,
    contract: AudioContract,
    voice: str,
    speed: str,
) -> str:
    sys.path.insert(0, str(repo_root()))
    from tools.tts_tool import _render_command_tts_template

    return _render_command_tts_template(
        command_template,
        {
            "input_path": str(input_path),
            "text_path": str(input_path),
            "text": text,
            "sample_rate": str(contract.sample_rate),
            "channels": str(contract.channels),
            "frame_ms": str(contract.frame_ms),
            "encoding": contract.encoding,
            "voice": voice,
            "speed": speed,
        },
    )


def max_abs_pcm_s16le(pcm: bytes) -> int:
    if len(pcm) % DEFAULT_BYTES_PER_SAMPLE:
        raise ValueError("pcm_s16le payload must contain whole samples")
    if not pcm:
        return 0
    return max(abs(sample[0]) for sample in struct.iter_unpack("<h", pcm))


def validate_pcm(
    pcm: bytes,
    *,
    contract: AudioContract,
    min_peak: int = DEFAULT_MIN_PEAK,
    min_duration_ms: int = DEFAULT_MIN_DURATION_MS,
) -> dict[str, Any]:
    contract.validate()
    failures: list[str] = []

    if not pcm:
        failures.append("stream command produced no PCM")
    if len(pcm) % contract.bytes_per_sample:
        failures.append("PCM byte length is not aligned to whole s16le samples")
    if len(pcm) % contract.frame_bytes:
        failures.append(
            f"PCM byte length {len(pcm)} is not aligned to {contract.frame_bytes}-byte frames"
        )

    duration_ms = (
        len(pcm) * 1_000 // contract.bytes_per_second
        if contract.bytes_per_second > 0
        else 0
    )
    if duration_ms < min_duration_ms:
        failures.append(
            f"PCM duration {duration_ms}ms is shorter than minimum {min_duration_ms}ms"
        )

    peak = 0
    if len(pcm) % contract.bytes_per_sample == 0:
        peak = max_abs_pcm_s16le(pcm)
        if peak < min_peak:
            failures.append(f"PCM peak {peak} is below minimum {min_peak}")

    if failures:
        raise SystemExit("validation failed:\n- " + "\n- ".join(failures))

    return {
        "bytes": len(pcm),
        "frames": len(pcm) // contract.frame_bytes,
        "duration_ms": duration_ms,
        "peak": peak,
    }


def run_stream_command(command: str, *, timeout: float) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def build_streamed_ogg_command(
    *,
    voice_bin: str,
    input_path: Path,
    output_path: Path,
    contract: AudioContract,
    voice: str,
    speed: str,
) -> list[str]:
    return [
        voice_bin,
        "stream",
        "--quiet",
        "--sample-rate",
        str(contract.sample_rate),
        "--frame-ms",
        str(contract.frame_ms),
        "--output",
        str(output_path),
        "--format",
        "ogg-opus",
        "--input-file",
        str(input_path),
        "--voice",
        voice,
        "--speed",
        speed,
    ]


def run_streamed_ogg_command(
    command: list[str],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


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


def validate_streamed_ogg(path: Path, *, probe: dict[str, str]) -> dict[str, Any]:
    failures: list[str] = []

    if path.suffix.lower() not in {".ogg", ".opus"}:
        failures.append(f"expected .ogg or .opus output, got {path}")
    if not path.is_file() or path.stat().st_size == 0:
        failures.append(f"expected non-empty streamed Ogg/Opus output at {path}")
    if probe.get("codec_name") != "opus":
        failures.append(f"expected codec_name=opus, got {probe.get('codec_name')!r}")
    if probe.get("sample_rate") != "48000":
        failures.append(f"expected sample_rate=48000, got {probe.get('sample_rate')!r}")
    if probe.get("channels") != "1":
        failures.append(f"expected channels=1, got {probe.get('channels')!r}")

    if failures:
        raise SystemExit("streamed Ogg/Opus validation failed:\n- " + "\n- ".join(failures))

    return {
        "bytes": path.stat().st_size,
        "codec_name": probe.get("codec_name"),
        "sample_rate": probe.get("sample_rate"),
        "channels": probe.get("channels"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Hermes WhatsApp Calling TTS stream command and verify raw "
            "pcm_s16le frame output."
        )
    )
    parser.add_argument("--voice-bin", default=os.environ.get("VOICE_BIN", "voice"))
    parser.add_argument("--ffprobe-bin", default=os.environ.get("FFPROBE_BIN", "ffprobe"))
    parser.add_argument("--command-template")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--min-peak", type=int, default=DEFAULT_MIN_PEAK)
    parser.add_argument("--min-duration-ms", type=int, default=DEFAULT_MIN_DURATION_MS)
    parser.add_argument("--keep-input", action="store_true")
    parser.add_argument(
        "--skip-streamed-ogg",
        action="store_true",
        help="validate the advertised streamed_voice_note surface without running it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")
    ffprobe_bin = (
        None
        if args.skip_streamed_ogg
        else resolve_executable(args.ffprobe_bin, label="ffprobe")
    )
    fallback_contract = AudioContract(
        sample_rate=args.sample_rate,
        channels=args.channels,
        frame_ms=args.frame_ms,
    )
    contract = audio_contract_from_voice(voice_bin, fallback=fallback_contract)
    contract.validate()
    command_template = args.command_template or build_default_command_template(voice_bin)

    tmpdir = Path(tempfile.mkdtemp(prefix="hermes-voice-stream-tts."))
    try:
        input_path = tmpdir / "input.txt"
        input_path.write_text(args.text, encoding="utf-8")
        command = render_stream_command(
            command_template,
            input_path=input_path,
            text=args.text,
            contract=contract,
            voice=args.voice,
            speed=args.speed,
        )
        completed = run_stream_command(command, timeout=args.timeout)
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise SystemExit(
                f"stream command exited with code {completed.returncode}"
                + (f": {detail[:1000]}" if detail else "")
            )

        stats = validate_pcm(
            completed.stdout,
            contract=contract,
            min_peak=args.min_peak,
            min_duration_ms=args.min_duration_ms,
        )

        streamed_ogg: dict[str, Any]
        if args.skip_streamed_ogg:
            streamed_ogg = {
                "success": True,
                "skipped": True,
                "reason": "--skip-streamed-ogg was provided",
            }
        else:
            ogg_path = tmpdir / "streamed.ogg"
            ogg_command = build_streamed_ogg_command(
                voice_bin=voice_bin,
                input_path=input_path,
                output_path=ogg_path,
                contract=contract,
                voice=args.voice,
                speed=args.speed,
            )
            ogg_completed = run_streamed_ogg_command(ogg_command, timeout=args.timeout)
            if ogg_completed.returncode != 0:
                detail = ogg_completed.stderr.strip() or ogg_completed.stdout.strip()
                raise SystemExit(
                    f"streamed Ogg/Opus command exited with code {ogg_completed.returncode}"
                    + (f": {detail[:1000]}" if detail else "")
                )
            probe = probe_audio(ogg_path, ffprobe_bin=ffprobe_bin or "ffprobe")
            ogg_stats = validate_streamed_ogg(ogg_path, probe=probe)
            streamed_ogg = {
                "success": True,
                "path": str(ogg_path) if args.keep_input else "<temporary>",
                "command": " ".join(shlex.quote(part) for part in ogg_command),
                **ogg_stats,
            }

        retained = bool(args.keep_input)
        print(
            json.dumps(
                {
                    "success": True,
                    "input_path": str(input_path) if retained else "<temporary>",
                    "retained": retained,
                    "command_template": command_template,
                    "audio": contract.as_dict(),
                    "voice_surfaces": {
                        "raw_outbound_pcm_command": contract.raw_outbound_pcm_command,
                        "raw_inbound_pcm_command": contract.raw_inbound_pcm_command,
                        "completed_voice_note_command": contract.completed_voice_note_command,
                        "streamed_voice_note_command": contract.streamed_voice_note_command,
                    },
                    "pcm": stats,
                    "streamed_ogg": streamed_ogg,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        if not args.keep_input:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
