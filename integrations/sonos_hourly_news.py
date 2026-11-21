"""Hourly live-news playback for the Kitchen Speaker.

The scheduler resolves an official YouTube live page at run time, then starts a
short-lived local MP3 proxy. Sonos receives a normal HTTP audio stream instead
of a brittle YouTube URL or an unsupported HLS/DASH container.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import re
from typing import Any
from urllib.parse import urlparse
import http.server
import importlib
from hermes_constants import get_hermes_home

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from integrations.file_locks import atomic_write_text, file_lock

STATE_DIR: Path | None = None
CONFIG_PATH: Path | None = None
PID_PATH: Path | None = None
LOG_PATH: Path | None = None
STREAM_STATE_PATH: Path | None = None
RESUME_STATE_PATH: Path | None = None
SONOS_BRAIN: Path | None = None
SCRIPTS_DIR: Path | None = None
SCHEDULE_DIR: Path | None = None
WRAPPER_PATH: Path | None = None
SCHEDULE_NAME = "Sonos hourly news"


def _state_dir() -> Path:
    return Path(STATE_DIR) if STATE_DIR is not None else get_hermes_home() / "sonos-brain-state"


def _state_path(override: Path | None, name: str) -> Path:
    return Path(override) if override is not None else _state_dir() / name


def _config_path() -> Path:
    return _state_path(CONFIG_PATH, "hourly-news.json")


def _pid_path() -> Path:
    return _state_path(PID_PATH, "hourly-news.pid")


def _log_path() -> Path:
    return _state_path(LOG_PATH, "hourly-news.jsonl")


def _stream_state_path() -> Path:
    return _state_path(STREAM_STATE_PATH, "hourly-news-stream.json")


def _resume_state_path() -> Path:
    return _state_path(RESUME_STATE_PATH, "hourly-news-resume.json")


def _sonos_brain() -> Path:
    return Path(SONOS_BRAIN) if SONOS_BRAIN is not None else get_hermes_home() / "sonos-brain" / "sonos_brain.py"


def _scripts_dir() -> Path:
    return Path(SCRIPTS_DIR) if SCRIPTS_DIR is not None else get_hermes_home() / "scripts"


def _schedule_dir() -> Path:
    return Path(SCHEDULE_DIR) if SCHEDULE_DIR is not None else _scripts_dir() / "sonos-hourly-news"


def _wrapper_path() -> Path:
    return Path(WRAPPER_PATH) if WRAPPER_PATH is not None else _schedule_dir() / "hourly-news.py"

STATIONS: dict[str, dict[str, str]] = {
    "france24": {
        "label": "France 24 English",
        "url": "https://www.youtube.com/channel/UCQfwfsi5VrQ8yKZ-UWmAEFg/live",
    },
    "dw": {
        "label": "DW News",
        "url": "https://www.youtube.com/watch?v=LuKwFajn37U",
    },
    "bbc": {
        "label": "BBC News (when live)",
        "url": "https://www.youtube.com/@BBCNews/live",
    },
    "ua_news_1": {
        "label": "UA: Перший / Суспільне Новини",
        "url": "https://www.youtube.com/@SuspilneNews/live",
    },
}
ALLOWED_MODES = {"interrupt", "idle"}
ALLOWED_DURATIONS = {5, 10, 15, 30}
ALLOWED_POST_NEWS_ACTIONS = {"radio", "resume", "idle"}


def _redact_uri(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _log_event(event: str, **fields: Any) -> None:
    """Append best-effort local telemetry without leaking stream query tokens."""
    payload = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    try:
        _state_dir().mkdir(parents=True, exist_ok=True)
        _state_dir().chmod(0o700)
        with _log_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _log_path().chmod(0o600)
    except OSError:
        return


def _write_stream_state(**fields: Any) -> None:
    try:
        _state_dir().mkdir(parents=True, exist_ok=True)
        temporary = _stream_state_path().with_suffix(".tmp")
        temporary.write_text(json.dumps(fields, sort_keys=True) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(_stream_state_path())
        _stream_state_path().chmod(0o600)
    except OSError:
        return


def _read_stream_state() -> dict[str, Any]:
    try:
        value = json.loads(_stream_state_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_resume_state(state: dict[str, Any]) -> None:
    _state_dir().mkdir(parents=True, exist_ok=True)
    _state_dir().chmod(0o700)
    lock_path = _resume_state_path().with_suffix(".lock")
    lock_path.touch(mode=0o600, exist_ok=True)
    lock_path.chmod(0o600)
    with file_lock(lock_path):
        atomic_write_text(_resume_state_path(), json.dumps(state, sort_keys=True) + "\n", mode=0o600)
    _resume_state_path().chmod(0o600)


def _clear_resume_state(expected_news_uri: str | None = None) -> None:
    lock_path = _resume_state_path().with_suffix(".lock")
    if not _resume_state_path().exists():
        return
    lock_path.touch(mode=0o600, exist_ok=True)
    lock_path.chmod(0o600)
    with file_lock(lock_path):
        try:
            state = json.loads(_resume_state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if state.get("news_uri", "") != (expected_news_uri or ""):
            return
        _resume_state_path().unlink(missing_ok=True)


def _expected_source(client: Any, station: str) -> tuple[str, tuple[str, str]] | None:
    try:
        transport = client.transport_state()
        uri = str(client.media_info().get("current_uri", ""))
        return transport, (transport, uri)
    except Exception as exc:  # health: allow BLE001 -- SoCo exposes transport-specific failures; skip safely before stream resolution
        _log_event("playback_skipped", reason="speaker_unreachable", station=station, error=_safe_error(exc))
        return None


def _resolve_and_preflight(station: str) -> str:
    try:
        resolved_url = _resolve_live_url(station)
        metadata = _resolve_live_metadata(station)
        _log_event("source_resolved", station=station, **metadata)
    except Exception as exc:  # health: allow BLE001 -- stream resolvers expose provider-specific errors; record a sanitized failure before retrying
        _log_event("playback_failed", stage="source_resolution", station=station, error=_safe_error(exc))
        raise
    try:
        _preflight_audio(resolved_url)
    except Exception as exc:  # health: allow BLE001 -- FFmpeg implementations raise platform-specific failures; no Sonos handover has occurred
        _log_event("playback_failed", stage="audio_preflight", station=station, error=_safe_error(exc))
        raise
    return resolved_url


def _source_changed(client: Any, expected_source: tuple[str, str], station: str) -> bool:
    current_source = (client.transport_state(), str(client.media_info().get("current_uri", "")))
    if _playback_blocked() or current_source != expected_source:
        _log_event("playback_skipped", reason="source_or_policy_changed", station=station)
        return True
    return False


def _preflight_playback_gate(settings: dict[str, Any], station: str) -> dict[str, Any] | None:
    if not settings.get("enabled", True):
        _log_event("playback_skipped", reason="disabled", station=station)
        return {"success": True, "started": False, "skipped": "disabled"}
    if _playback_blocked():
        _log_event("playback_skipped", reason="policy_blocked", station=station)
        return {"success": True, "started": False, "skipped": "policy_blocked"}
    return None


def _busy_playback(only_if_idle: bool, mode: str, transport: str) -> bool:
    return (only_if_idle or mode == "idle") and transport != "STOPPED"


def _proxy_ready_port(process: subprocess.Popen[bytes], station: str) -> int:
    ready_line = ""
    startup_deadline = time.monotonic() + 10.0
    while time.monotonic() < startup_deadline and process.poll() is None:
        if process.stdout is None:
            break
        readable, _, _ = select.select([process.stdout], [], [], 0.25)
        if readable:
            ready_line = process.stdout.readline().decode("utf-8", errors="replace").strip()
            break
    if not ready_line.startswith("READY "):
        if process.poll() is None:
            raise RuntimeError("hourly-news proxy did not become ready within 10 seconds")
        detail = process.stderr.read().decode("utf-8", errors="replace").strip() if process.stderr else ""
        raise RuntimeError(f"hourly-news proxy exited before ready: {detail[:300] or 'no error output'}")
    port = int(ready_line.split()[1])
    _log_event("proxy_handoff_ready", station=station, port=port)
    return port


def _wait_for_news_bytes(initial_bytes: int) -> dict[str, Any]:
    verification_deadline = time.monotonic() + 8.0
    stream_state = _read_stream_state()
    while time.monotonic() < verification_deadline:
        stream_state = _read_stream_state()
        received = (int(stream_state.get("request_count", 0)) > 0
                    and int(stream_state.get("bytes_sent", 0)) > initial_bytes)
        if received:
            break
        time.sleep(0.5)
    if int(stream_state.get("bytes_sent", 0)) <= initial_bytes:
        raise RuntimeError("Sonos did not receive hourly-news audio bytes")
    stability_deadline = time.monotonic() + 5.0
    while time.monotonic() < stability_deadline:
        stream_state = _read_stream_state()
        stream_stopped = (int(stream_state.get("bytes_sent", 0)) <= initial_bytes
                          and int(stream_state.get("active_requests", 0)) == 0)
        if stream_stopped:
            break
        time.sleep(0.5)
    return stream_state


def _verify_news_playback(client: Any, uri: str, initial_bytes: int, stream_state: dict[str, Any]) -> dict[str, Any]:
    transport = client.transport_state()
    diagnostics = _speaker_diagnostics(client)
    request_count = int(stream_state.get("request_count", 0))
    bytes_sent = int(stream_state.get("bytes_sent", 0))
    if transport != "PLAYING":
        raise RuntimeError(f"Sonos did not start hourly news (state: {transport})")
    if client.media_info().get("current_uri", "") != uri:
        raise RuntimeError("Sonos source changed during news verification")
    if request_count == 0:
        raise RuntimeError("Sonos did not request the hourly-news proxy")
    if bytes_sent <= initial_bytes:
        raise RuntimeError("hourly-news audio stopped before stability was confirmed")
    return diagnostics


def _speaker_operation_lock():
    root = str(_sonos_brain().parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    sonos_brain = importlib.import_module("sonos_brain")
    return sonos_brain._exclusive_operation_lock(sonos_brain.SPEAKER_OPERATION_LOCK)


def _capture_resume_snapshot(client: Any, transport: str) -> dict[str, Any]:
    if transport != "PLAYING":
        return {}
    try:
        media = client.media_info()
        position = client.position_info()
        source_uri = str(media.get("current_uri") or position.get("track_uri") or "")
        rel_time = str(position.get("rel_time") or "")
        if not source_uri or not urlparse(source_uri).scheme or not rel_time:
            return {}
        parsed = urlparse(source_uri)
        active_stream = _read_stream_state()
        if parsed.path == "/news.mp3" and parsed.port == active_stream.get("port"):
            return {}
        return {
            "transport": "PLAYING",
            "source_uri": source_uri,
            "metadata": str(position.get("track_metadata") or "")[:65536],
            "position": rel_time,
            "duration": str(position.get("track_duration") or ""),
        }
    except Exception:  # health: allow BLE001 -- vendor SOAP failures have transport-specific types; without a valid snapshot, takeover remains guarded
        return {}


def _restore_interrupted_source(client: Any, news_uri: str, snapshot: dict[str, Any]) -> bool:
    if not snapshot or snapshot.get("transport") != "PLAYING":
        return False
    source_uri = str(snapshot.get("source_uri") or "")
    if not source_uri or not urlparse(source_uri).scheme:
        return False
    try:
        with _speaker_operation_lock():
            if _playback_blocked():
                return False
            if client.media_info().get("current_uri", "") != news_uri or client.transport_state() != "PLAYING":
                return False
            if _playback_blocked():
                return False
            if client.media_info().get("current_uri", "") != news_uri or client.transport_state() != "PLAYING":
                return False
            client.set_uri(source_uri, str(snapshot.get("metadata") or "")[:65536])
            if _playback_blocked() or client.media_info().get("current_uri", "") != source_uri:
                return False
            position = str(snapshot.get("position") or "")
            duration = str(snapshot.get("duration") or "")
            if position and duration not in {"", "0:00:00"} and position != "0:00:00":
                try:
                    client.seek_rel_time(position)
                except Exception as exc:  # health: allow BLE001 -- a failed seek must not prevent safe source restoration
                    _log_event("source_position_restore_skipped", error=_safe_error(exc))
            if _playback_blocked() or client.media_info().get("current_uri", "") != source_uri:
                return False
            client.play()
        if not _wait_for_transport(client, "PLAYING"):
            return False
        if client.media_info().get("current_uri", "") != source_uri:
            return False
        _log_event("source_restored")
        return True
    except Exception as exc:  # health: allow BLE001 -- SoCo and the worker expose vendor-specific playback errors; redact before recording and fail closed
        _log_event("source_restore_failed", error=_safe_error(exc))
        return False


def _safe_error(exc: BaseException) -> str:
    message = str(exc).replace("\n", " ").strip()
    return re.sub(r"https?://\S+", "<url>", message)[:500]


def _ffmpeg_error_excerpt(stderr: Any, limit_bytes: int = 4096) -> str:
    """Read a bounded tail of FFmpeg stderr and redact signed URLs before logging."""
    if stderr is None:
        return ""
    try:
        stderr.seek(0, os.SEEK_END)
        size = stderr.tell()
        stderr.seek(max(0, size - limit_bytes))
        detail = stderr.read(limit_bytes)
    except (OSError, ValueError):
        return ""
    text = detail.decode("utf-8", errors="replace") if isinstance(detail, bytes) else str(detail)
    return _safe_error(RuntimeError(text))


def _ffmpeg_ca_file() -> str | None:
    candidates = [os.environ.get("SSL_CERT_FILE"), ssl.get_default_verify_paths().cafile]
    try:
        import certifi
        candidates.append(certifi.where())
    except ImportError:
        pass
    return next((value for value in candidates if value and Path(value).is_file()), None)


def _ffmpeg_environment() -> dict[str, str]:
    # HLS segment requests need OpenSSL's trust setting as well as the input's -ca_file.
    ca_file = _ffmpeg_ca_file()
    return {**os.environ, **({"SSL_CERT_FILE": ca_file} if ca_file else {})}


def _ffmpeg_command(resolved_url: str) -> list[str]:
    command = [shutil.which("ffmpeg") or "ffmpeg", "-nostdin", "-v", "error", "-tls_verify", "1"]
    ca_file = _ffmpeg_ca_file()
    if ca_file:
        command.extend(["-ca_file", ca_file])
    command.extend([
        "-re", "-i", resolved_url,
        "-vn", "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame",
        "-b:a", "128k", "-f", "mp3", "pipe:1",
    ])
    return command


def _preflight_audio(resolved_url: str) -> None:
    command = _ffmpeg_command(resolved_url)
    env = _ffmpeg_environment()
    _log_event("audio_preflight_started", python=sys.executable, ffmpeg=command[0], ca_file=env.get("SSL_CERT_FILE", ""))
    try:
        result = subprocess.run(
            [*command[:-1], "-t", "2", command[-1]],
            env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=20, check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("hourly-news audio preflight timed out after 20 seconds") from None
    if result.returncode != 0 or len(result.stdout) < 16384:
        detail = _safe_error(RuntimeError(result.stderr.decode("utf-8", errors="replace")))
        raise RuntimeError(f"hourly-news audio preflight failed (FFmpeg {result.returncode}): {detail}")
    _log_event("audio_preflight_verified", audio_bytes=len(result.stdout))


def _speaker_diagnostics(client: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"proxy": _read_stream_state()}
    try:
        result["transport"] = client.transport_state()
    except Exception as exc:  # health: allow BLE001 -- diagnostics are independent best-effort SoCo calls with vendor-specific transport errors
        result["transport_error"] = _safe_error(exc)
    try:
        media = client.media_info()
        result["media"] = {
            "current_uri": _redact_uri(str(media.get("current_uri", ""))),
            "track_count": media.get("track_count", ""),
            "play_medium": media.get("play_medium", ""),
        }
    except Exception as exc:  # health: allow BLE001 -- diagnostics are independent best-effort SoCo calls with vendor-specific transport errors
        result["media_error"] = _safe_error(exc)
    try:
        position = client.position_info()
        result["position"] = {
            "track_uri": _redact_uri(str(position.get("track_uri", ""))),
            "rel_time": position.get("rel_time", ""),
            "track_duration": position.get("track_duration", ""),
        }
    except Exception as exc:  # health: allow BLE001 -- diagnostics are independent best-effort SoCo calls with vendor-specific transport errors
        result["position_error"] = _safe_error(exc)
    return result


def default_settings() -> dict[str, Any]:
    return {
        "enabled": True,
        "station": "france24",
        "mode": "interrupt",
        "duration_minutes": 15,
        "after_news_action": "radio",
        "after_news_station_id": "",
        "job_id": "",
    }


def update_settings(
    settings: dict[str, Any],
    *,
    enabled: bool | None = None,
    station: str | None = None,
    mode: str | None = None,
    duration_minutes: int | None = None,
    after_news_action: str | None = None,
    after_news_station_id: str | None = None,
) -> dict[str, Any]:
    defaults = default_settings()
    result = {**defaults, **settings}
    if enabled is not None:
        result["enabled"] = bool(enabled)
    if station is not None:
        if station not in STATIONS:
            raise ValueError("unknown news station")
        result["station"] = station
    if mode is not None:
        if mode not in ALLOWED_MODES:
            raise ValueError("mode must be interrupt or idle")
        result["mode"] = mode
    if duration_minutes is not None:
        value = int(duration_minutes)
        if value not in ALLOWED_DURATIONS:
            raise ValueError("duration must be 5, 10, 15, or 30 minutes")
        result["duration_minutes"] = value
    if after_news_action is not None:
        if after_news_action not in ALLOWED_POST_NEWS_ACTIONS:
            raise ValueError("after-news action must be radio, resume, or idle")
        result["after_news_action"] = after_news_action
    elif result.get("after_news_action") not in ALLOWED_POST_NEWS_ACTIONS:
        result["after_news_action"] = defaults["after_news_action"]
    if after_news_station_id is not None:
        if not isinstance(after_news_station_id, str) or len(after_news_station_id) > 128:
            raise ValueError("after-news station ID must be a string of at most 128 characters")
        result["after_news_station_id"] = after_news_station_id
    elif not isinstance(result.get("after_news_station_id"), str) or len(result["after_news_station_id"]) > 128:
        result["after_news_station_id"] = defaults["after_news_station_id"]
    return result


def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return dict(fallback)
    return value if isinstance(value, dict) else dict(fallback)


def load_settings() -> dict[str, Any]:
    return update_settings(_read_json(_config_path(), default_settings()))


def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = update_settings(settings)
    _state_dir().mkdir(parents=True, exist_ok=True)
    _state_dir().chmod(0o700)
    temporary = _config_path().with_suffix(".tmp")
    temporary.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(_config_path())
    _config_path().chmod(0o600)
    return normalized


def stream_content_type(path: Path) -> str:
    return "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/wav"


def wrapper_text() -> str:
    return f'''"""Generated by Hermes for the hourly-news live trigger."""
import subprocess
import sys

SCRIPT = {str(Path(__file__).resolve())!r}
result = subprocess.run([sys.executable, SCRIPT, "trigger"], capture_output=True, text=True, check=False)
if result.stdout:
    print(result.stdout.rstrip())
if result.returncode != 0:
    print((result.stderr or "hourly news failed")[:500])
    raise SystemExit(result.returncode)
'''


def _cron_call(action: str, **kwargs: Any) -> dict[str, Any]:
    from tools.cronjob_tools import cronjob

    raw = cronjob(action=action, **kwargs)
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"success": False, "error": "scheduler returned invalid JSON"}
    return value if isinstance(value, dict) else {"success": False, "error": "scheduler returned invalid data"}


def _job_id(result: dict[str, Any]) -> str:
    job = result.get("job")
    if isinstance(job, dict):
        return str(job.get("job_id") or job.get("id") or "")
    return str(result.get("job_id") or "")


def ensure_schedule(settings: dict[str, Any]) -> dict[str, Any]:
    current = str(settings.get("job_id") or "")
    if current:
        return settings
    _schedule_dir().mkdir(parents=True, exist_ok=True)
    _schedule_dir().chmod(0o700)
    _wrapper_path().write_text(wrapper_text(), encoding="utf-8")
    _wrapper_path().chmod(0o700)
    result = _cron_call(
        "create",
        schedule="0 * * * *",
        name=SCHEDULE_NAME,
        prompt=None,
        script=str(_wrapper_path().relative_to(_scripts_dir())),
        no_agent=True,
        deliver="local",
    )
    if not result.get("success"):
        raise RuntimeError(str(result.get("error") or "could not create hourly-news schedule"))
    job_id = _job_id(result)
    if not job_id:
        raise RuntimeError("scheduler created hourly-news job without an ID")
    return save_settings({**settings, "job_id": job_id})


def _resolve_live_url(station: str) -> str:
    if station not in STATIONS:
        raise ValueError("unknown news station")
    yt_dlp = shutil.which("yt-dlp")
    if yt_dlp is None:
        raise FileNotFoundError("yt-dlp is required for hourly live news")
    result = subprocess.run(
        [yt_dlp, "--no-warnings", "--no-playlist", "-f", "bestaudio/best", "-g", STATIONS[station]["url"]],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "live source unavailable").strip()
        raise RuntimeError(f"{STATIONS[station]['label']} stream unavailable: {detail[:300]}")
    resolved = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    parsed = urlparse(resolved)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("yt-dlp returned an invalid live media URL")
    return resolved


def _resolve_live_metadata(station: str) -> dict[str, str]:
    if station not in STATIONS:
        raise ValueError("unknown news station")
    yt_dlp = shutil.which("yt-dlp")
    if yt_dlp is None:
        raise FileNotFoundError("yt-dlp is required for hourly live news")
    result = subprocess.run(
        [
            yt_dlp,
            "--no-warnings",
            "--no-playlist",
            "--skip-download",
            "--print",
            "%(id)s\\t%(title)s\\t%(channel)s\\t%(live_status)s",
            STATIONS[station]["url"],
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-500:] or "could not inspect live source")
    fields = result.stdout.strip().splitlines()[-1].split("\\t", 3)
    if len(fields) != 4:
        raise RuntimeError("yt-dlp returned incomplete live metadata")
    video_id, title, channel, live_status = (field.strip() for field in fields)
    if live_status != "is_live":
        raise RuntimeError(f"selected source is not live (status: {live_status or 'unknown'})")
    return {"video_id": video_id, "title": title, "channel": channel, "live_status": live_status}


class _NewsStreamServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], resolved_url: str, duration: float):
        super().__init__(address, _NewsStreamHandler)
        self.resolved_url = resolved_url
        self.duration = duration
        self.ffmpeg_processes: set[subprocess.Popen[bytes]] = set()
        self.ffmpeg_lock = threading.Lock()
        self.request_count = 0
        self.bytes_sent = 0
        self.active_requests = 0
        self.stream_failed = False
        self.expired = threading.Event()
        self.request_lock = threading.Lock()


class _NewsStreamHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, format: str, *_args: object) -> None:
        return

    def do_HEAD(self) -> None:
        if self.path != "/news.mp3":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/news.mp3":
            self.send_error(404)
            return
        server = self.server
        if not isinstance(server, _NewsStreamServer):
            self.send_error(500)
            return
        ffmpeg: subprocess.Popen[bytes] | None = None
        ffmpeg_stderr: Any = None
        try:
            with server.request_lock:
                server.request_count += 1
                server.active_requests += 1
            _write_stream_state(
                status="streaming",
                request_count=server.request_count,
                bytes_sent=server.bytes_sent,
                active_requests=server.active_requests,
            )
            _log_event("proxy_requested", request_count=server.request_count)
            ffmpeg_stderr = tempfile.TemporaryFile()
            ffmpeg = subprocess.Popen(
                _ffmpeg_command(server.resolved_url),
                env=_ffmpeg_environment(),
                stdout=subprocess.PIPE,
                stderr=ffmpeg_stderr,
                stdin=subprocess.DEVNULL,
            )
            with server.ffmpeg_lock:
                server.ffmpeg_processes.add(ffmpeg)
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            deadline = time.monotonic() + server.duration
            assert ffmpeg.stdout is not None
            while time.monotonic() < deadline:
                chunk = ffmpeg.stdout.read(16384)
                if not chunk:
                    if time.monotonic() < deadline and not server.expired.is_set():
                        server.stream_failed = True
                        _log_event(
                            "proxy_stream_failed",
                            reason="upstream_ended_early",
                            ffmpeg_exit_code=ffmpeg.poll(),
                            error=_ffmpeg_error_excerpt(ffmpeg_stderr),
                        )
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                server.bytes_sent += len(chunk)
                _write_stream_state(
                    status="streaming",
                    request_count=server.request_count,
                    bytes_sent=server.bytes_sent,
                    active_requests=server.active_requests,
                )
        except (BrokenPipeError, ConnectionResetError, OSError):
            _write_stream_state(
                status="client_disconnected",
                request_count=server.request_count,
                bytes_sent=server.bytes_sent,
                active_requests=server.active_requests,
            )
            _log_event("proxy_client_disconnected", request_count=server.request_count, bytes_sent=server.bytes_sent)
        finally:
            if ffmpeg is not None and ffmpeg.poll() is None:
                ffmpeg.terminate()
            if ffmpeg is not None:
                with server.ffmpeg_lock:
                    server.ffmpeg_processes.discard(ffmpeg)
                try:
                    ffmpeg.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    ffmpeg.kill()
            if ffmpeg_stderr is not None:
                ffmpeg_stderr.close()
            with server.request_lock:
                server.active_requests = max(0, server.active_requests - 1)
                active_requests = server.active_requests
            _write_stream_state(
                status="streaming" if active_requests else "client_disconnected",
                request_count=server.request_count,
                bytes_sent=server.bytes_sent,
                active_requests=active_requests,
            )
            if server.stream_failed and not server.expired.is_set():
                server.shutdown()


def _handle_post_news_expiry(
    client: Any,
    news_uri: str,
    snapshot: dict[str, Any],
    action: str,
    station_id: str,
) -> bool:
    """Apply a post-news action only while the expected news stream is playing."""
    try:
            if action == "resume":
                restored = _restore_interrupted_source(client, news_uri, snapshot)
                _log_event("post_news_resumed" if restored else "post_news_skipped", reason="resume" if restored else "resume_unavailable")
                return restored
            with _speaker_operation_lock():
                if _playback_blocked():
                    _log_event("post_news_skipped", reason="policy_blocked", action=action)
                    return False
                media = client.media_info()
                if media.get("current_uri", "") != news_uri or client.transport_state() != "PLAYING":
                    _log_event("post_news_skipped", reason="news_no_longer_current", action=action)
                    return False
                if action == "idle":
                    client.stop()
                    client.clear_media()
                    _log_event("post_news_idle")
                    return False
                if action != "radio":
                    _log_event("post_news_skipped", reason="invalid_action")
                    return False
                station = _get_managed_radio_station(station_id)
                if not station:
                    _log_event("post_news_radio_unavailable", reason="station_not_found", station_id=station_id)
                    return False
                if _playback_blocked():
                    _log_event("post_news_skipped", reason="policy_blocked", action="radio")
                    return False
                media = client.media_info()
                if media.get("current_uri", "") != news_uri or client.transport_state() != "PLAYING":
                    _log_event("post_news_skipped", reason="news_no_longer_current", action="radio")
                    return False
                client.set_uri(station["url"])
                client.play()
                _log_event("post_news_radio_started", station_id=station_id, station_name=station.get("name", ""))
                return True
    except Exception as exc:  # health: allow BLE001 -- post-news device actions are best-effort; errors are URL-redacted before telemetry
        _log_event("post_news_action_failed", action=action, error=_safe_error(exc))
        return False


def _get_managed_radio_station(station_id: str) -> dict[str, str] | None:
    if not station_id:
        return None
    from integrations import sonos_radio_stations

    return sonos_radio_stations.get_station(station_id)


def _serve(resolved_file: Path, duration_minutes: int) -> int:
    resolved_url = resolved_file.read_text(encoding="utf-8").strip()
    resolved_file.unlink(missing_ok=True)
    server = _NewsStreamServer(("0.0.0.0", 0), resolved_url, duration_minutes * 60.0)
    port = int(server.server_address[1])
    state = {"pid": os.getpid(), "port": port, "duration_minutes": duration_minutes}
    _state_dir().mkdir(parents=True, exist_ok=True)
    _pid_path().write_text(json.dumps(state) + "\n", encoding="utf-8")
    _pid_path().chmod(0o600)
    _write_stream_state(status="ready", port=port, request_count=0, bytes_sent=0, active_requests=0)
    _log_event("proxy_ready", port=port, duration_minutes=duration_minutes)
    print(f"READY {port}", flush=True)
    expired = threading.Event()

    def expire_proxy() -> None:
        expired.set()
        server.expired.set()
        server.shutdown()

    timer = threading.Timer(duration_minutes * 60.0, expire_proxy)
    timer.daemon = True
    timer.start()
    try:
        server.serve_forever()
    finally:
        timer.cancel()
        server.expired.set()
        with server.ffmpeg_lock:
            active_processes = list(server.ffmpeg_processes)
        for ffmpeg in active_processes:
            if ffmpeg.poll() is None:
                ffmpeg.terminate()
        server.server_close()
        _write_stream_state(
            status="stopped",
            port=port,
            request_count=server.request_count,
            bytes_sent=server.bytes_sent,
            active_requests=server.active_requests,
        )
        _log_event("proxy_stopped", port=port, request_count=server.request_count, bytes_sent=server.bytes_sent)
        _pid_path().unlink(missing_ok=True)
        if expired.is_set() or server.stream_failed:
            resume_state = _read_json(_resume_state_path(), {})
            snapshot = resume_state.get("snapshot", {})
            news_uri = str(resume_state.get("news_uri", ""))
            stream_state = _read_stream_state()
            stream_ok = (
                not server.stream_failed
                and int(stream_state.get("request_count", 0)) > 0
                and int(stream_state.get("bytes_sent", 0)) > 0
            )
            own_news = urlparse(news_uri).path == "/news.mp3" and urlparse(news_uri).port == port
            if server.stream_failed and isinstance(snapshot, dict) and own_news:
                _restore_interrupted_source(_sonos_client(), news_uri, snapshot)
            elif stream_ok and isinstance(snapshot, dict) and own_news:
                settings = load_settings()
                try:
                    _handle_post_news_expiry(
                        _sonos_client(),
                        news_uri,
                        snapshot,
                        str(settings.get("after_news_action", "radio")),
                        str(settings.get("after_news_station_id", "")),
                    )
                except Exception as exc:  # health: allow BLE001 -- the proxy thread boundary records sanitized device-action errors and still reaps its child
                    _log_event("post_news_action_failed", error=_safe_error(exc))
            else:
                _log_event("post_news_skipped", reason="news_stream_not_verified")
            if own_news:
                _clear_resume_state(news_uri)
    return 0


def _sonos_client() -> Any:
    root = str(_sonos_brain().parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from sonos_brain import SonosClient

    return SonosClient()


def _playback_blocked() -> bool:
    root = str(_sonos_brain().parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from sonos_brain import playback_blocked

    return playback_blocked()


def _stop_existing_news() -> None:
    try:
        state = json.loads(_pid_path().read_text(encoding="utf-8"))
        pid = int(state.get("pid", 0))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, TypeError):
        pid = 0
    if pid > 0:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            pass
    _pid_path().unlink(missing_ok=True)


def _wait_for_transport(client: Any, expected: str, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        state = client.transport_state()
        if state == expected:
            return True
        if expected == "PLAYING" and state in {"STOPPED", "NO_MEDIA_PRESENT"}:
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def start_sonos_playback(
    client: Any, uri: str, *, interrupt: bool, expected_source: tuple[str, str] | None = None,
) -> bool:
    if _playback_blocked():
        return False
    with _speaker_operation_lock():
        if _playback_blocked():
            return False
        if expected_source is not None and (
            client.transport_state(), str(client.media_info().get("current_uri", ""))
        ) != expected_source:
            return False
        if not interrupt and client.transport_state() != "STOPPED":
            return False
        if interrupt:
            from sonos_brain import _stop_fm_program
            _stop_fm_program()
        if _playback_blocked() or (expected_source is not None and (
            client.transport_state(), str(client.media_info().get("current_uri", ""))
        ) != expected_source):
            return False
        client.set_uri(uri)
        if _playback_blocked() or client.media_info().get("current_uri", "") != uri:
            return False
        client.play()
    if not _wait_for_transport(client, "PLAYING"):
        raise RuntimeError("Sonos did not reach PLAYING")
    return True


def trigger_news(*, only_if_idle: bool = False) -> dict[str, Any]:
    settings = load_settings()
    station = str(settings.get("station", "france24"))
    mode = str(settings.get("mode", "interrupt"))
    duration = int(settings.get("duration_minutes", 10))
    _log_event("trigger_started", station=station, mode=mode, duration_minutes=duration)
    skip_result = _preflight_playback_gate(settings, station)
    if skip_result is not None:
        return skip_result
    client = _sonos_client()
    initial_source = _expected_source(client, station)
    if initial_source is None:
        return {"success": True, "started": False, "skipped": "speaker_unreachable"}
    transport, expected_source = initial_source
    if _busy_playback(only_if_idle, mode, transport):
        _log_event("playback_skipped", reason="busy", station=station)
        return {"success": True, "started": False, "skipped": "busy"}
    resolved_url = _resolve_and_preflight(station)
    if _source_changed(client, expected_source, station):
        return {"success": True, "started": False, "skipped": "source_or_policy_changed"}
    interrupt = mode == "interrupt" and not only_if_idle
    snapshot = _capture_resume_snapshot(client, transport) if interrupt else {}
    _stop_existing_news()
    _write_resume_state({"snapshot": snapshot, "news_uri": ""})
    _state_dir().mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", prefix="hourly-news-", suffix=".url", dir=_state_dir(), delete=False, encoding="utf-8") as handle:
        resolved_file = Path(handle.name)
        handle.write(resolved_url)
    resolved_file.chmod(0o600)
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "serve", str(resolved_file), str(duration)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    uri = ""
    port = 0
    try:
        port = _proxy_ready_port(process, station)
        host = _local_lan_ip()
        uri = f"http://{host}:{port}/news.mp3"
        _write_resume_state({"snapshot": snapshot, "news_uri": uri})
        initial_bytes = int(_read_stream_state().get("bytes_sent", 0))
        started = start_sonos_playback(client, uri, interrupt=interrupt, expected_source=expected_source)
        if not started:
            _log_event("playback_skipped", reason="source_or_policy_changed", station=station)
            process.terminate()
            resolved_file.unlink(missing_ok=True)
            _stream_state_path().unlink(missing_ok=True)
            _clear_resume_state(uri)
            return {"success": True, "started": False, "skipped": "source_or_policy_changed"}
        _log_event("sonos_play_requested", station=station, mode=mode, port=port)
        stream_state = _wait_for_news_bytes(initial_bytes)
        diagnostics = _verify_news_playback(client, uri, initial_bytes, stream_state)
        _log_event(
            "playback_verified",
            station=station,
            mode=mode,
            duration_minutes=duration,
            diagnostics=diagnostics,
        )
        return {"success": True, "started": True, "station": station, "duration_minutes": duration, "mode": mode}
    except Exception as exc:  # health: allow BLE001 -- vendor Sonos failures are redacted in telemetry and guarded recovery must still run
        diagnostics = _speaker_diagnostics(client)
        _log_event(
            "playback_not_confirmed",
            station=station,
            mode=mode,
            error=_safe_error(exc),
            diagnostics=diagnostics,
        )
        if uri:
            _restore_interrupted_source(client, uri, snapshot)
        process.terminate()
        resolved_file.unlink(missing_ok=True)
        _stream_state_path().unlink(missing_ok=True)
        _clear_resume_state(uri or None)
        raise


def _local_lan_ip(target_ip: str = "192.168.178.254") -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((target_ip, 1400))
        return str(sock.getsockname()[0])
    finally:
        sock.close()


def settings_operation(action: str, **kwargs: Any) -> dict[str, Any]:
    settings = load_settings()
    if action == "status":
        try:
            from integrations import sonos_radio_stations
            managed_stations = [
                {"id": station["id"], "name": station["name"]}
                for station in sonos_radio_stations.list_stations()
            ]
        except Exception as exc:  # health: allow BLE001 -- the optional station store can raise filesystem or database-specific errors; settings remain readable
            managed_stations = []
            _log_event("radio_station_list_failed", error=_safe_error(exc))
        return {
            "success": True,
            "settings": settings,
            "stations": {key: value["label"] for key, value in STATIONS.items()},
            "managed_radio_stations": managed_stations,
        }
    if action == "kill_automations":
        # Disable at the source before pausing cron: even a scheduler failure
        # cannot make a racing trigger start news or later play radio.
        settings = update_settings(settings, enabled=False, after_news_action="idle")
        saved_settings = save_settings(settings)
        if settings.get("job_id"):
            scheduler_result = _cron_call(
                "pause",
                job_id=str(settings["job_id"]),
                reason="Hourly news and post-news radio paused from Sonos settings",
            )
            if scheduler_result.get("success") is not True:
                return {"success": False, "settings": saved_settings, "error": "Could not pause scheduled news"}
        return {"success": True, "settings": saved_settings}
    if action == "set_enabled":
        settings = update_settings(settings, enabled=bool(kwargs["enabled"]))
        settings = save_settings(settings)
        scheduler_ok = True
        try:
            if settings["enabled"]:
                settings = ensure_schedule(settings)
                if settings.get("job_id"):
                    scheduler_ok = _cron_call("resume", job_id=str(settings["job_id"])).get("success") is True
            elif settings.get("job_id"):
                scheduler_ok = _cron_call(
                    "pause", job_id=str(settings["job_id"]), reason="Hourly news disabled from Sonos settings"
                ).get("success") is True
        except Exception:  # health: allow BLE001 -- cron backend failures are reduced to an explicit unsuccessful scheduler status
            scheduler_ok = False
        return {
            "success": scheduler_ok,
            "settings": settings,
            "scheduler": {"success": scheduler_ok},
            **({"error": "Could not update scheduled news"} if not scheduler_ok else {}),
        }
    if action == "set_station":
        return {"success": True, "settings": save_settings(update_settings(settings, station=str(kwargs["station"]))) }
    if action == "set_mode":
        return {"success": True, "settings": save_settings(update_settings(settings, mode=str(kwargs["mode"]))) }
    if action == "set_duration":
        return {"success": True, "settings": save_settings(update_settings(settings, duration_minutes=int(kwargs["duration_minutes"]))) }
    if action == "set_after_news_action":
        return {"success": True, "settings": save_settings(update_settings(settings, after_news_action=str(kwargs["after_news_action"]))) }
    if action == "set_after_news_station":
        return {"success": True, "settings": save_settings(update_settings(settings, after_news_station_id=str(kwargs["station_id"]))) }
    if action == "trigger":
        return trigger_news()
    if action == "trigger_if_idle":
        return trigger_news(only_if_idle=True)
    if action == "stop":
        _stop_existing_news()
        client = _sonos_client()
        client.stop()
        client.clear_media()
        return {"success": True, "stopped": True}
    raise ValueError("unsupported hourly-news operation")


def _duration_action(args: argparse.Namespace) -> dict[str, Any]:
    return settings_operation("set_duration", duration_minutes=int(args.value or "0"))


_ACTION_HANDLERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "trigger": lambda _: trigger_news(),
    "status": lambda _: settings_operation("status"),
    "enable": lambda _: settings_operation("set_enabled", enabled=True),
    "disable": lambda _: settings_operation("set_enabled", enabled=False),
    "station": lambda args: settings_operation("set_station", station=args.value or ""),
    "mode": lambda args: settings_operation("set_mode", mode=args.value or ""),
    "duration": _duration_action,
    "stop": lambda _: settings_operation("stop"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("serve", "trigger", "status", "enable", "disable", "station", "mode", "duration", "stop"))
    parser.add_argument("value", nargs="?")
    parser.add_argument("serve_duration", nargs="?")
    args = parser.parse_args()
    if args.action == "serve":
        if not args.value or not args.serve_duration:
            parser.error("serve requires a resolved URL file and duration")
        duration = int(args.serve_duration)
        return _serve(Path(args.value), duration)
    try:
        result = _ACTION_HANDLERS[args.action](args)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("success") else 2
    except Exception as exc:  # health: allow BLE001 -- CLI boundary converts integration errors into a nonzero exit; output is a redacted exception message
        print(f"hourly-news: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
