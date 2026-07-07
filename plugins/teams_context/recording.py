"""Meeting recording and transcript ingestion for TeamContext."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from plugins.teams_context.store import TeamsContextStore


class RecordingIngestError(RuntimeError):
    pass


AUTHENTICATED_HOST_MARKERS = (
    "teams.microsoft.com",
    "sharepoint.com",
    "onedrive.live.com",
    "1drv.ms",
)


def ingest_recording(
    path_or_url: str,
    *,
    meeting_label: str,
    store: TeamsContextStore,
    transcript_path: str | None = None,
    artifact_cache: str | Path | None = None,
) -> dict[str, Any]:
    if not meeting_label.strip():
        raise RecordingIngestError("--meeting-label is required")
    cache_dir = Path(artifact_cache).expanduser() if artifact_cache else get_hermes_home() / "cache" / "teams_context"
    cache_dir.mkdir(parents=True, exist_ok=True)
    recording_path = resolve_recording_artifact(path_or_url, cache_dir=cache_dir)
    transcript_source = "transcript"
    if transcript_path:
        transcript_text = parse_vtt_file(Path(transcript_path).expanduser())
        source_type = "transcript"
    else:
        transcript_source = "stt"
        transcript_text = transcribe_recording_file(recording_path, cache_dir=cache_dir)
        source_type = "recording"
    source_id = stable_source_id(meeting_label, recording_path)
    chunks = chunk_text(transcript_text)
    metadata = {
        "meeting_label": meeting_label,
        "recording_path": str(recording_path),
        "transcript_path": str(Path(transcript_path).expanduser()) if transcript_path else None,
        "transcript_source": transcript_source,
    }
    for index, chunk in enumerate(chunks):
        store.upsert_kb_chunk(
            source_id=source_id,
            item_id=f"{source_id}:chunk:{index}",
            source_type=source_type,
            source_label=meeting_label,
            chunk_index=index,
            text=chunk,
            meeting_id=source_id,
            metadata=metadata,
        )
    return {
        "meeting_label": meeting_label,
        "source_id": source_id,
        "source_type": source_type,
        "chunks": len(chunks),
        "recording_path": str(recording_path),
        "transcript_source": transcript_source,
    }


def resolve_recording_artifact(path_or_url: str, *, cache_dir: Path) -> Path:
    raw = str(path_or_url or "").strip()
    if not raw:
        raise RecordingIngestError("recording path or URL is required")
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        return download_recording_url(raw, cache_dir=cache_dir)
    if parsed.scheme:
        raise RecordingIngestError(f"Unsupported recording URL scheme: {parsed.scheme}")
    path = Path(raw).expanduser()
    if not path.exists() or not path.is_file():
        raise RecordingIngestError(f"Recording file not found: {path}")
    return path


def download_recording_url(url: str, *, cache_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if any(marker in host for marker in AUTHENTICATED_HOST_MARKERS):
        raise RecordingIngestError(
            "Authenticated Teams, SharePoint, or OneDrive recording links are not supported. "
            "Download the recording locally first, then run ingest-recording on the local file."
        )
    filename = Path(urllib.parse.unquote(parsed.path)).name or "recording.bin"
    suffix = Path(filename).suffix or ".bin"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    destination = cache_dir / f"download-{digest}{suffix}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            status = getattr(response, "status", 200)
            if int(status) >= 400:
                raise RecordingIngestError(f"Recording download failed with HTTP {status}")
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except RecordingIngestError:
        raise
    except Exception as exc:
        raise RecordingIngestError(f"Recording download failed: {exc}") from exc
    return destination


def parse_vtt_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise RecordingIngestError(f"Transcript file not found: {path}")
    return parse_vtt_text(path.read_text(encoding="utf-8"))


def parse_vtt_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.upper() == "WEBVTT":
            continue
        if "-->" in line:
            continue
        if line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"^\s*[-\w ]+:\s+", lambda match: match.group(0), line)
        if line:
            lines.append(line)
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        raise RecordingIngestError("Transcript file did not contain readable VTT cues")
    return cleaned


def transcribe_recording_file(recording_path: Path, *, cache_dir: Path) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RecordingIngestError(
            "ffmpeg is required to extract recording audio. Install ffmpeg or pass --transcript."
        )
    audio_path = cache_dir / f"{recording_path.stem}-{hashlib.sha256(str(recording_path).encode()).hexdigest()[:8]}.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(recording_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode("utf-8", errors="replace")[-600:]
        raise RecordingIngestError(f"ffmpeg failed to extract audio: {error}") from exc

    try:
        from tools.transcription_tools import transcribe_audio
    except Exception as exc:
        raise RecordingIngestError(
            "STT dependencies are unavailable. Configure Hermes STT or pass --transcript."
        ) from exc
    result = transcribe_audio(str(audio_path))
    if not result.get("success"):
        raise RecordingIngestError(
            f"STT transcription failed: {result.get('error') or 'unknown error'}. "
            "Configure Hermes STT or pass --transcript."
        )
    transcript = str(result.get("transcript") or "").strip()
    if not transcript:
        raise RecordingIngestError("STT transcription returned an empty transcript")
    return transcript


def chunk_text(text: str, *, max_chars: int = 1800) -> list[str]:
    words = re.split(r"(\s+)", str(text or "").strip())
    chunks: list[str] = []
    current = ""
    for token in words:
        if len(current) + len(token) > max_chars and current.strip():
            chunks.append(current.strip())
            current = token
        else:
            current += token
    if current.strip():
        chunks.append(current.strip())
    return chunks or []


def stable_source_id(meeting_label: str, recording_path: Path) -> str:
    material = f"{meeting_label}\0{recording_path.name}\0{recording_path.stat().st_size if recording_path.exists() else 0}"
    return "meeting:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
