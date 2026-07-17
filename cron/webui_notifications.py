"""Profile-local WebUI/Hermex notification store for cron deliveries.

The scheduler appends one JSON object per visible notification to
``$HERMES_HOME/cron/notifications.jsonl``.  WebUI/Hermex reads this inbox so
cron reports can land in the mobile app without depending on Telegram.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - Windows fallback is exercised by behaviour, not import.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home

_SCHEMA_VERSION = 1
_MAX_LIMIT = 200
_MAX_RECORDS = 2000
_MAX_BODY_CHARS = 20_000
_MAX_MEDIA_ITEMS = 50


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _profile_name_for_home(home: Path) -> str:
    home = Path(home).expanduser()
    if home.parent.name == "profiles" and home.name:
        return home.name
    return "default"


def notifications_file(home: str | Path | None = None) -> Path:
    """Return the profile-local cron notification JSONL file path."""
    base = Path(home).expanduser() if home is not None else get_hermes_home()
    return base / "cron" / "notifications.jsonl"


def _lock_file_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


@contextmanager
def _store_lock(path: Path):
    lock_path = _lock_file_for(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_fh:
        try:
            os.chmod(lock_path, 0o600)
        except OSError:
            pass
        if fcntl is not None:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def ensure_store(home: str | Path | None = None) -> Path:
    """Create the notification store with owner-only permissions where possible."""
    path = notifications_file(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    with open(path, "a", encoding="utf-8"):
        pass
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def _normalise_media(media: Iterable[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in media or []:
        if isinstance(item, dict):
            media_path = item.get("path") or item.get("file") or item.get("media_path")
            if not media_path:
                continue
            entry: dict[str, Any] = {"path": str(media_path)}
            if "is_voice" in item:
                entry["is_voice"] = bool(item.get("is_voice"))
            if "type" in item:
                entry["type"] = str(item.get("type"))
            out.append(entry)
            if len(out) >= _MAX_MEDIA_ITEMS:
                break
            continue
        if isinstance(item, (list, tuple)) and item:
            entry: dict[str, Any] = {"path": str(item[0])}
            if len(item) > 1:
                entry["is_voice"] = bool(item[1])
            out.append(entry)
            if len(out) >= _MAX_MEDIA_ITEMS:
                break
            continue
        if item:
            out.append({"path": str(item)})
            if len(out) >= _MAX_MEDIA_ITEMS:
                break
    return out


def _normalise_record(record: dict[str, Any], home: Path) -> dict[str, Any]:
    now = _utc_now_iso()
    normalised = dict(record)
    normalised.setdefault("id", f"notif_{uuid.uuid4().hex}")
    normalised.setdefault("schema_version", _SCHEMA_VERSION)
    normalised.setdefault("source", "cron")
    normalised.setdefault("profile", _profile_name_for_home(home))
    normalised.setdefault("severity", "info")
    normalised.setdefault("status", "ok")
    normalised.setdefault("body_format", "markdown")
    normalised.setdefault("created_at", now)
    normalised.setdefault("read_at", None)
    body = str(normalised.get("body") or "")
    if len(body) > _MAX_BODY_CHARS:
        normalised["body"] = body[:_MAX_BODY_CHARS] + "\n\n[notification body truncated]"
        normalised["body_truncated"] = True
    else:
        normalised["body"] = body
        normalised.setdefault("body_truncated", False)
    normalised["media"] = _normalise_media(normalised.get("media"))
    return normalised


def _trim_store_locked(path: Path) -> None:
    """Keep only the newest bounded raw records while the store lock is held."""
    newest: deque[str] = deque(maxlen=_MAX_RECORDS + 1)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        newest.extend(fh)
    if len(newest) <= _MAX_RECORDS:
        return

    fd, tmp_name = tempfile.mkstemp(
        prefix="notifications.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            for raw in list(newest)[-_MAX_RECORDS:]:
                tmp.write(raw if raw.endswith("\n") else raw + "\n")
            tmp.flush()
            try:
                os.fsync(tmp.fileno())
            except OSError:
                pass
        os.replace(tmp_name, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def append_notification(record: dict[str, Any], home: str | Path | None = None) -> dict[str, Any]:
    """Append and return one normalized notification record.

    The function stores metadata only for media paths; it never reads attachment
    file contents.
    """
    path = ensure_store(home)
    base = Path(home).expanduser() if home is not None else get_hermes_home()
    normalised = _normalise_record(record, base)
    line = json.dumps(normalised, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _store_lock(path):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        _trim_store_locked(path)
    return normalised


def _iter_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def list_notifications(
    home: str | Path | None = None,
    *,
    limit: int = 50,
    unread_only: bool = False,
) -> list[dict[str, Any]]:
    """List notifications newest-first, skipping malformed JSONL lines."""
    path = notifications_file(home)
    try:
        cap = int(limit)
    except (TypeError, ValueError):
        cap = 50
    cap = max(1, min(cap, _MAX_LIMIT))
    with _store_lock(path):
        records = _iter_records(path)
    if unread_only:
        records = [r for r in records if not r.get("read_at")]
    # Newer appended rows win ties from legacy second-resolution timestamps.
    records.reverse()
    records.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return records[:cap]


def unread_count(home: str | Path | None = None) -> int:
    return len(list_notifications(home, limit=_MAX_LIMIT, unread_only=True))


def mark_read(notification_id: str, home: str | Path | None = None) -> dict[str, Any] | None:
    """Mark one notification read and return the updated record, if found."""
    if not notification_id:
        return None
    path = ensure_store(home)
    updated: dict[str, Any] | None = None
    now = _utc_now_iso()
    with _store_lock(path):
        rows: list[tuple[str | None, dict[str, Any] | None]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    rows.append((raw, None))
                    continue
                if isinstance(parsed, dict):
                    if str(parsed.get("id") or "") == str(notification_id):
                        parsed["read_at"] = parsed.get("read_at") or now
                        updated = dict(parsed)
                    rows.append((None, parsed))
                else:
                    rows.append((raw, None))

        fd, tmp_name = tempfile.mkstemp(prefix="notifications.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                for raw, parsed in rows:
                    if parsed is not None:
                        tmp.write(json.dumps(parsed, ensure_ascii=False, separators=(",", ":")) + "\n")
                    elif raw:
                        tmp.write(raw if raw.endswith("\n") else raw + "\n")
                tmp.flush()
                try:
                    os.fsync(tmp.fileno())
                except OSError:
                    pass
            os.replace(tmp_name, path)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
    return updated
