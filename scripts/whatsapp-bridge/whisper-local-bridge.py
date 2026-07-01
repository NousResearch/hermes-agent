#!/usr/bin/env python3
"""Local whisper.cpp transcription wrapper with OpenAI Whisper API fallback.

Designed to be plugged into Hermes' STT pipeline via:

    stt.provider: local_command
    HERMES_LOCAL_STT_COMMAND="python scripts/whatsapp-bridge/whisper-local-bridge.py {input} {output}"

Behavior:
  1. Run whisper-cli with a configured local model against the input audio.
  2. On any failure (non-zero exit, empty transcript, missing binary/model)
     fall back to OpenAI Whisper API (whisper-1).
  3. Write final transcript to {output} (the .txt path Hermes expects).
  4. Append one structured line per call to the Hermes logs directory:
       ts=... model=local|openai latency_ms=... bytes=... ok=true|false err=...

Exit codes:
  0, transcript successfully written to {output}
  1, both local and OpenAI paths failed (Hermes will surface the error)

This script intentionally has *no* third-party deps beyond the OpenAI SDK
(only loaded if local fails) so it stays cheap to import on every voice note.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WHISPER_CLI = os.environ.get("WHISPER_CLI_BIN") or shutil.which("whisper-cli") or "whisper-cli"
WHISPER_MODEL = os.environ.get(
    "WHISPER_LOCAL_MODEL",
    str(
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "whisper.cpp"
        / "ggml-medium.en.bin"
    ),
)
WHISPER_THREADS = os.environ.get("WHISPER_THREADS", "8")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "en")
WHISPER_NO_GPU = os.environ.get("WHISPER_NO_GPU", "1").lower() not in {"0", "false", "no"}
LOG_PATH = os.environ.get(
    "WHISPER_BRIDGE_LOG",
    str(
        Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        / "logs"
        / "whisper-bridge.log"
    ),
)


def _log(record: dict) -> None:
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    try:
        log_path = Path(LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        elif log_path.is_symlink():
            return
        fd = os.open(log_path, flags, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must never break transcription


def _try_local(input_path: str, output_path: str) -> tuple[bool, str, str]:
    """Return (ok, transcript_text, error_str)."""
    if not Path(WHISPER_CLI).exists():
        return False, "", f"whisper-cli not found at {WHISPER_CLI}"
    if not Path(WHISPER_MODEL).exists():
        return False, "", f"model not found at {WHISPER_MODEL}"

    source_input = str(Path(input_path).resolve())
    working_input = source_input
    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    if Path(input_path).suffix.lower() != ".wav":
        tmp_dir = tempfile.TemporaryDirectory(prefix="whisper-local-bridge-")
        working_input = str(Path(tmp_dir.name) / "input.wav")
        convert_cmd = [
            "ffmpeg",
            "-y",
            "-i", source_input,
            "-ar", "16000",
            "-ac", "1",
            working_input,
        ]
        try:
            conv = subprocess.run(
                convert_cmd, capture_output=True, text=True, timeout=60, check=False,
            )
        except subprocess.TimeoutExpired:
            if tmp_dir:
                tmp_dir.cleanup()
            return False, "", "ffmpeg conversion timeout (>60s)"
        except Exception as exc:  # noqa: BLE001
            if tmp_dir:
                tmp_dir.cleanup()
            return False, "", f"ffmpeg conversion launch error: {exc}"
        if conv.returncode != 0:
            if tmp_dir:
                tmp_dir.cleanup()
            return False, "", f"ffmpeg conversion rc={conv.returncode}: {conv.stderr.strip()[:300]}"

    output_txt = Path(output_path)
    output_base = output_txt.with_suffix("") if output_txt.suffix else output_txt
    cmd = [
        WHISPER_CLI,
        "-m", WHISPER_MODEL,
        "-f", working_input,
        "-l", WHISPER_LANG,
        "-t", WHISPER_THREADS,
        "-otxt",
        "-of", str(output_base),
        "-nt",  # no timestamps
    ]
    if WHISPER_NO_GPU:
        cmd.append("-ng")
    try:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "", "whisper-cli timeout (>120s)"
        except Exception as exc:  # noqa: BLE001
            return False, "", f"whisper-cli launch error: {exc}"

        if proc.returncode != 0:
            return False, "", f"whisper-cli rc={proc.returncode}: {proc.stderr.strip()[:300]}"

        txt_path = output_base.with_suffix(".txt")
        if not txt_path.exists():
            return False, "", f"whisper-cli produced no .txt output at {txt_path}"

        text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return False, "", "whisper-cli produced empty transcript"
        return True, text, ""
    finally:
        if tmp_dir:
            tmp_dir.cleanup()


def _try_openai(input_path: str) -> tuple[bool, str, str]:
    api_key = (
        os.environ.get("VOICE_TOOLS_OPENAI_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        return False, "", "no OPENAI_API_KEY/VOICE_TOOLS_OPENAI_KEY in env"
    try:
        from openai import OpenAI  # local import keeps cold-path cheap
    except Exception as exc:  # noqa: BLE001
        return False, "", f"openai SDK unavailable: {exc}"
    try:
        client = OpenAI(api_key=api_key, timeout=30, max_retries=1)
        with open(input_path, "rb") as fh:
            resp = client.audio.transcriptions.create(
                model=os.environ.get("STT_OPENAI_MODEL", "whisper-1"),
                file=fh,
                response_format="text",
            )
        text = resp if isinstance(resp, str) else getattr(resp, "text", "")
        text = (text or "").strip()
        if not text:
            return False, "", "OpenAI returned empty transcript"
        return True, text, ""
    except Exception as exc:  # noqa: BLE001
        return False, "", f"openai error: {exc}"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: whisper-local-bridge.py INPUT OUTPUT", file=sys.stderr)
        return 2
    input_path, output_path = argv[1], argv[2]
    try:
        nbytes = Path(input_path).stat().st_size
    except OSError:
        nbytes = -1

    t0 = time.monotonic()
    ok, text, err = _try_local(input_path, output_path)
    latency_ms = int((time.monotonic() - t0) * 1000)
    _log({
        "model": "local",
        "engine": "whisper-cli/medium.en",
        "latency_ms": latency_ms,
        "bytes": nbytes,
        "ok": ok,
        "err": err,
        "input": input_path,
    })

    if not ok:
        t0 = time.monotonic()
        ok, text, err2 = _try_openai(input_path)
        latency_ms = int((time.monotonic() - t0) * 1000)
        _log({
            "model": "openai",
            "engine": "whisper-1",
            "latency_ms": latency_ms,
            "bytes": nbytes,
            "ok": ok,
            "err": err2,
            "input": input_path,
            "fallback_reason": err,
        })
        if not ok:
            sys.stderr.write(f"local+openai both failed: local={err} openai={err2}\n")
            return 1

    Path(output_path).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
