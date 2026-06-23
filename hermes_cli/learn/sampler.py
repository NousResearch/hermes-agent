"""Metadata-only Learn sampler.

The sampler records foreground-app metadata after Learn is explicitly running.
It does not capture keystrokes, clipboard content, screenshots, document
bodies, cookies, browser profiles, or full URLs.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from utils import atomic_replace

from . import state

Collector = Callable[[], Optional[Dict[str, Any]]]

_URL_RE = re.compile(r"\bhttps?://[^\s<>)\]]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")
_BARE_DOMAIN_RE = re.compile(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)+)\b", re.IGNORECASE)
_SENSITIVE_TITLE_RE = re.compile(r"\b(password|passcode|mfa|otp|2fa|credit card|ssn|cookie|token)\b", re.IGNORECASE)

_CATEGORY_BY_PROCESS = {
    "browser": {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "arc.exe"},
    "communication": {
        "discord.exe",
        "outlook.exe",
        "slack.exe",
        "teams.exe",
        "telegram.exe",
        "whatsapp.exe",
        "zoom.exe",
    },
    "development": {"code.exe", "cursor.exe", "pycharm64.exe", "webstorm64.exe", "devenv.exe"},
    "terminal": {"cmd.exe", "powershell.exe", "pwsh.exe", "windowsterminal.exe", "wt.exe"},
    "documents": {"excel.exe", "notepad.exe", "onenote.exe", "powerpnt.exe", "winword.exe"},
}


def _learn_dir(home: Path) -> Path:
    return home.resolve() / "learn"


def _events_path(home: Path) -> Path:
    return _learn_dir(home) / "events.jsonl"


def _secure_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass
    return _hermes_now()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _infer_duration_seconds(record: Dict[str, Any], previous_events: list[Dict[str, Any]]) -> int:
    if not previous_events:
        return 0
    last_timestamp = _parse_timestamp(previous_events[-1].get("timestamp"))
    current_timestamp = _parse_timestamp(record.get("timestamp"))
    delta = int((current_timestamp - last_timestamp).total_seconds())
    if delta <= 0:
        return 0
    return min(delta, 3600)


def _process_name(value: Any) -> str:
    text = str(value or "unknown").strip() or "unknown"
    return text.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _extract_domain(title: str) -> Optional[str]:
    match = _URL_RE.search(title)
    if match:
        host = urlparse(match.group(0)).hostname
        if host:
            return host.lower().removeprefix("www.")

    for candidate in _BARE_DOMAIN_RE.findall(title):
        lowered = candidate.lower().strip(".")
        if "." in lowered and not lowered.endswith(".exe"):
            return lowered.removeprefix("www.")
    return None


def _redact_title(title: str) -> str:
    title = str(title or "").strip()
    if not title:
        return ""
    without_urls = _URL_RE.sub("[url]", title)
    without_emails = _EMAIL_RE.sub("[email]", without_urls)
    without_numbers = _LONG_NUMBER_RE.sub("[number]", without_emails)
    if _SENSITIVE_TITLE_RE.search(without_numbers):
        return "[sensitive window]"
    return " ".join(without_numbers.split())[:160]


def _category_for(process_name: str, domain: Optional[str]) -> str:
    for category, processes in _CATEGORY_BY_PROCESS.items():
        if process_name in processes:
            return category
    if domain:
        return "browser"
    return "other"


def _matches_filter(record: Dict[str, Any], item: str) -> bool:
    target = item.strip().lower()
    if not target:
        return False
    haystack = [
        str(record.get("process_name") or "").lower(),
        str(record.get("domain") or "").lower(),
        str(record.get("category") or "").lower(),
        str(record.get("window_title") or "").lower(),
    ]
    return any(target in value for value in haystack)


def _passes_filters(record: Dict[str, Any], current_state: Dict[str, Any]) -> bool:
    denylist = current_state.get("denylist") if isinstance(current_state.get("denylist"), list) else []
    allowlist = current_state.get("allowlist") if isinstance(current_state.get("allowlist"), list) else []
    if any(_matches_filter(record, str(item)) for item in denylist):
        return False
    if allowlist and not any(_matches_filter(record, str(item)) for item in allowlist):
        return False
    return True


def normalize_sample(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return the persisted metadata shape with sensitive fields redacted."""
    process = _process_name(raw.get("process_name"))
    raw_title = str(raw.get("window_title") or "")
    domain = _extract_domain(raw_title)
    timestamp = _parse_timestamp(raw.get("timestamp"))
    category = str(raw.get("category") or _category_for(process, domain)).strip().lower() or "other"
    return {
        "timestamp": timestamp.isoformat(),
        "process_name": process,
        "window_title": _redact_title(raw_title),
        "domain": domain,
        "category": category,
        "idle": bool(raw.get("idle", False)),
        "idle_seconds": _safe_int(raw.get("idle_seconds"), 0),
        "duration_seconds": _safe_int(raw.get("duration_seconds"), 0),
    }


def _load_events(home: Path) -> list[Dict[str, Any]]:
    path = _events_path(home)
    if not path.exists():
        return []
    events: list[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    events.append(item)
    except (OSError, json.JSONDecodeError):
        return []
    return events


def _write_events(home: Path, events: list[Dict[str, Any]]) -> None:
    learn_dir = _learn_dir(home)
    learn_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(learn_dir), suffix=".tmp", prefix=".learn_events_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for event in events:
                json.dump(event, f, sort_keys=True)
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, _events_path(home))
        _secure_file(_events_path(home))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def prune_events(*, home: Path | None = None, retention_days: int = 14) -> int:
    resolved_home = (home or get_hermes_home()).resolve()
    cutoff = _hermes_now() - timedelta(days=max(1, int(retention_days)))
    events = _load_events(resolved_home)
    kept = [event for event in events if _parse_timestamp(event.get("timestamp")) >= cutoff]
    if len(kept) != len(events):
        _write_events(resolved_home, kept)
    return len(events) - len(kept)


def append_event(record: Dict[str, Any], *, home: Path | None = None) -> Dict[str, Any]:
    resolved_home = (home or get_hermes_home()).resolve()
    events = _load_events(resolved_home)
    events.append(record)
    _write_events(resolved_home, events)
    return record


def collect_windows_foreground_sample() -> Optional[Dict[str, Any]]:
    """Collect foreground-window metadata on Windows; return None elsewhere."""
    if os.name != "nt":
        return None
    try:
        import psutil  # type: ignore

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process_name = psutil.Process(int(pid.value)).name()

        class LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        last_input = LastInputInfo()
        last_input.cbSize = ctypes.sizeof(last_input)
        idle_seconds = 0
        if user32.GetLastInputInfo(ctypes.byref(last_input)):
            tick_count = ctypes.windll.kernel32.GetTickCount()
            idle_seconds = max(0, int((tick_count - last_input.dwTime) / 1000))

        return {
            "process_name": process_name,
            "window_title": buffer.value,
            "timestamp": _hermes_now().isoformat(),
            "idle": idle_seconds >= 300,
            "idle_seconds": idle_seconds,
            "duration_seconds": 0,
        }
    except Exception:
        return None


def sample_once(*, collector: Collector | None = None, home: Path | None = None) -> Optional[Dict[str, Any]]:
    """Collect and persist one metadata sample if Learn is running."""
    resolved_home = (home or get_hermes_home()).resolve()
    current_state = state._load_state(resolved_home)
    if current_state.get("mode") == "off" or current_state.get("state") != "running":
        return None

    raw = (collector or collect_windows_foreground_sample)()
    if not raw:
        prune_events(home=resolved_home, retention_days=int(current_state.get("retention_days", 14)))
        return None

    previous_events = _load_events(resolved_home)
    record = normalize_sample(raw)
    if record["duration_seconds"] == 0:
        record["duration_seconds"] = _infer_duration_seconds(record, previous_events)
    if not _passes_filters(record, current_state):
        prune_events(home=resolved_home, retention_days=int(current_state.get("retention_days", 14)))
        return None

    append_event(record, home=resolved_home)
    prune_events(home=resolved_home, retention_days=int(current_state.get("retention_days", 14)))
    return record
