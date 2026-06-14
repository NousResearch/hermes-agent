#!/usr/bin/env python3
"""Irodori-TTS client for an OpenAI-compatible /v1/audio/speech endpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.local_secretary.write_action_gate import check_write_action

DEFAULT_BASE = "http://127.0.0.1:8088"
SENTENCE_SPLIT = re.compile(r"(?<=[\u3002\uff01\uff1f!?\.])\s*")
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def chunk_sentences(text: str, max_chars: int = 220) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]
    if not parts:
        parts = [text]
    chunks: list[str] = []
    buf = ""
    for part in parts:
        candidate = f"{buf} {part}".strip() if buf else part
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        if len(part) <= max_chars:
            buf = part
        else:
            for i in range(0, len(part), max_chars):
                chunks.append(part[i : i + max_chars])
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _health_ok(base_url: str, timeout: float = 5.0) -> bool:
    url = f"{base_url.rstrip('/')}/health"
    try:
        with urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("status") == "ok"
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return False


def _safe_output_roots() -> list[Path]:
    roots: list[Path] = []
    for value in (
        os.getenv("IRODORI_TTS_OUTPUT_DIR"),
        os.getenv("HERMES_LOCAL_SECRETARY_OUTPUT_DIR"),
    ):
        if value:
            roots.append(Path(value).expanduser())
    roots.append(Path.home() / ".hermes" / "audio")

    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(root.resolve())
        except OSError:
            resolved.append(root.absolute())
    return resolved


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def _output_path_is_safe(path: Path) -> bool:
    return any(_path_is_under(path, root) for root in _safe_output_roots())


def _flag_enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_loopback_base_url(value: str) -> bool:
    text = value.strip()
    if "://" not in text:
        text = f"http://{text}"
    try:
        parsed = urlparse(text)
    except ValueError:
        return False
    return (parsed.hostname or "").strip().lower() in _LOOPBACK_HOSTS


def _synthesize_chunk(
    *,
    base_url: str,
    text: str,
    voice: str,
    response_format: str,
    speed: float,
    seed: int | None,
) -> bytes:
    endpoint = f"{base_url.rstrip('/')}/v1/audio/speech"
    body: dict[str, Any] = {
        "model": "irodori-tts",
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed,
    }
    if seed is not None:
        body["seed"] = seed
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "audio/*"}
    api_key = os.getenv("IRODORI_API_KEY")
    if api_key and (
        _is_loopback_base_url(base_url)
        or _flag_enabled(os.getenv("IRODORI_TTS_ALLOW_REMOTE_API_KEY"))
    ):
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(endpoint, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=300) as resp:
        return resp.read()


def _concat_wav(parts: list[bytes]) -> bytes:
    if len(parts) == 1:
        return parts[0]
    import io

    frames: list[bytes] = []
    params = None
    for blob in parts:
        with wave.open(io.BytesIO(blob), "rb") as src:
            if params is None:
                params = src.getparams()
            elif src.getparams()[:3] != params[:3]:
                raise ValueError("incompatible wav chunk parameters")
            frames.append(src.readframes(src.getnframes()))
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        assert params is not None
        dst.setparams(params)
        for frame in frames:
            dst.writeframes(frame)
    return out.getvalue()


def synthesize_speech(
    text: str,
    *,
    voice: str = "none",
    response_format: str = "wav",
    speed: float = 1.0,
    output_path: Path,
    seed: int | None = None,
    base_url: str = DEFAULT_BASE,
    dry_run: bool = False,
    autoplay: bool = False,
    confirmed: bool = False,
) -> str:
    gate = check_write_action("tts_generate")
    if not gate.ok:
        return json.dumps(gate.to_json())

    if not _output_path_is_safe(output_path):
        gate = check_write_action(
            "write",
            confirmed=confirmed,
            detail=f"tts output_path={output_path}",
        )
        if not gate.ok:
            return json.dumps(gate.to_json())

    chunks = chunk_sentences(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        meta = {
            "success": True,
            "action": "tts_generate",
            "dry_run": True,
            "output_path": str(output_path),
            "chunks": len(chunks or [text]),
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            "generated_at": _utc_now(),
            "autoplay": autoplay,
        }
        output_path.write_bytes(b"RIFF")
        return json.dumps(meta, ensure_ascii=False)

    if not _health_ok(base_url):
        return json.dumps(
            {
                "success": False,
                "action": "tts_generate",
                "error": f"Irodori-TTS health check failed for {base_url}",
            }
        )

    audio_parts: list[bytes] = []
    for chunk in chunks or [text]:
        audio_parts.append(
            _synthesize_chunk(
                base_url=base_url,
                text=chunk,
                voice=voice,
                response_format=response_format,
                speed=speed,
                seed=seed,
            )
        )

    if response_format == "wav":
        blob = _concat_wav(audio_parts)
    else:
        blob = audio_parts[0] if audio_parts else b""

    output_path.write_bytes(blob)
    meta = {
        "success": True,
        "action": "tts_generate",
        "output_path": str(output_path),
        "bytes": len(blob),
        "chunks": len(chunks or [text]),
        "voice": voice,
        "response_format": response_format,
        "speed": speed,
        "seed": seed,
        "generated_at": _utc_now(),
        "autoplay": autoplay,
    }
    sidecar = output_path.with_suffix(output_path.suffix + ".meta.json")
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.dumps(meta, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Irodori-TTS synthesis helper")
    parser.add_argument("--text", default="")
    parser.add_argument("--text-file", default="")
    parser.add_argument("--voice", default=os.getenv("IRODORI_TTS_DEFAULT_VOICE", "none"))
    parser.add_argument("--response-format", default="wav")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-url", default=os.getenv("IRODORI_TTS_BASE_URL", DEFAULT_BASE))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--confirmed", action="store_true")
    args = parser.parse_args()

    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    if not text.strip():
        print(json.dumps({"success": False, "error": "empty input text"}))
        return 1

    print(
        synthesize_speech(
            text,
            voice=args.voice,
            response_format=args.response_format,
            speed=args.speed,
            output_path=Path(args.output),
            seed=args.seed,
            base_url=args.base_url,
            dry_run=args.dry_run,
            autoplay=args.autoplay,
            confirmed=args.confirmed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
