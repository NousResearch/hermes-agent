"""Active Hermes session presence records.

This module intentionally knows nothing about MeshBoard, Tailscale, tmux, or
any one desktop shell.  It gives Hermes clients a small generic place to
publish "this session is live here" metadata that other local or synced
clients can discover and build adapters around.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1
DEFAULT_TTL_SECONDS = 90.0


def _presence_root(hermes_home: Path | str | None = None) -> Path:
    configured = os.environ.get("HERMES_SESSION_PRESENCE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()

    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    return home / "session-presence" / "active"


def _safe_id(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in value.strip())
    return cleaned.strip("._-") or "unknown"


def default_instance_id() -> str:
    """Return a stable-ish id for the current Hermes process."""
    env_value = os.environ.get("HERMES_SESSION_PRESENCE_INSTANCE", "").strip()
    if env_value:
        return _safe_id(env_value)
    return _safe_id(f"{socket.gethostname()}-{os.getpid()}")


def write_session_presence(
    *,
    session_id: str,
    session_key: str | None = None,
    status: str = "idle",
    title: str = "",
    model: str = "",
    cwd: str = "",
    source: str = "unknown",
    client: str = "",
    profile: str = "",
    endpoint: str = "",
    metadata: dict[str, Any] | None = None,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    hermes_home: Path | str | None = None,
    instance_id: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Publish or refresh an active-session presence record.

    The record is deliberately small and secret-free.  ``endpoint`` is optional
    and should contain only a user-configured local/private attach hint.
    """
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")

    ts = float(time.time() if now is None else now)
    ttl = max(1.0, float(ttl_seconds or DEFAULT_TTL_SECONDS))
    instance = _safe_id(instance_id or default_instance_id())
    root = _presence_root(hermes_home)
    root.mkdir(parents=True, exist_ok=True)

    record = {
        "version": SCHEMA_VERSION,
        "session_id": sid,
        "session_key": str(session_key or sid),
        "status": str(status or "idle"),
        "title": str(title or ""),
        "model": str(model or ""),
        "cwd": str(cwd or ""),
        "source": str(source or "unknown"),
        "client": str(client or ""),
        "profile": str(profile or ""),
        "endpoint": str(endpoint or ""),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "instance_id": instance,
        "updated_at": ts,
        "expires_at": ts + ttl,
        "metadata": metadata or {},
    }
    path = root / f"{instance}.{_safe_id(sid)}.json"
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
    # Invalidate cache for this root
    _presence_cache.clear()
    return record


# In-memory cache for list_session_presence: keyed by (hermes_home, include_expired).
# Stores (mtime_upper_bound, result) — skip re-read if all files are <= mtime_upper_bound.
_presence_cache: dict[tuple[str, bool], tuple[float, list[dict[str, Any]]]] = {}
_PRESENCE_CACHE_MAX_AGE = 3.0  # seconds — force re-read after this even if mtime matches


def _read_session_presence(
    *,
    root: Path,
    ts: float,
    include_expired: bool = False,
) -> list[dict[str, Any]]:
    """Core logic: glob + parse + dedup presence JSON files."""
    records_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in root.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not include_expired and float(record.get("expires_at") or 0) < ts:
            continue
        session_id = str(record.get("session_id") or "")
        endpoint = str(record.get("endpoint") or "").strip()
        stable_target = endpoint or session_id
        key = (
            str(record.get("profile") or ""),
            str(record.get("client") or record.get("source") or ""),
            stable_target,
        )
        if not stable_target:
            continue
        previous = records_by_key.get(key)
        if previous is None or float(record.get("updated_at") or 0) > float(
            previous.get("updated_at") or 0
        ):
            records_by_key[key] = record
    records = list(records_by_key.values())
    records.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
    return records


def list_session_presence(
    *,
    hermes_home: Path | str | None = None,
    now: float | None = None,
    include_expired: bool = False,
) -> list[dict[str, Any]]:
    """Read active-session presence records, newest first.

    Uses an in-memory cache keyed by mtime to avoid re-parsing JSON files
    when nothing on disk has changed. Cache expires after _PRESENCE_CACHE_MAX_AGE
    seconds or when any file's mtime is newer than the cached snapshot.
    """
    ts = float(time.time() if now is None else now)
    root = _presence_root(hermes_home)
    if not root.exists():
        return []

    # Cache key: normalized root path + include_expired flag
    cache_key = (str(root.resolve()), include_expired)
    now_real = time.time()

    # Check cache
    cached = _presence_cache.get(cache_key)
    if cached is not None:
        cached_mtime, cached_result = cached
        # Expire by age
        if now_real - cached_mtime > _PRESENCE_CACHE_MAX_AGE:
            _presence_cache.pop(cache_key, None)
        else:
            # Expire by mtime: check if any file is newer than cached snapshot
            try:
                newest_mtime = max(
                    (p.stat().st_mtime for p in root.glob("*.json")),
                    default=0.0,
                )
                if newest_mtime <= cached_mtime:
                    return cached_result
            except OSError:
                pass

    # Re-read from disk
    records = _read_session_presence(root=root, ts=ts, include_expired=include_expired)

    # Record the mtime upper bound for cache validity
    try:
        newest_mtime = max(
            (p.stat().st_mtime for p in root.glob("*.json")),
            default=0.0,
        )
    except OSError:
        newest_mtime = 0.0
    _presence_cache[cache_key] = (newest_mtime, records)

    # Evict stale entries if cache grows (shouldn't normally happen, but defensive)
    if len(_presence_cache) > 32:
        oldest_key = min(_presence_cache, key=lambda k: _presence_cache[k][0])
        del _presence_cache[oldest_key]

    return records


def clear_session_presence(
    *,
    session_id: str | None = None,
    instance_id: str | None = None,
    hermes_home: Path | str | None = None,
) -> int:
    """Remove matching presence records and return the number removed."""
    root = _presence_root(hermes_home)
    if not root.exists():
        return 0

    safe_session = _safe_id(session_id) if session_id else None
    safe_instance = _safe_id(instance_id) if instance_id else None
    removed = 0
    for path in root.glob("*.json"):
        if safe_instance and not path.name.startswith(f"{safe_instance}."):
            continue
        if safe_session and not path.name.endswith(f".{safe_session}.json"):
            continue
        try:
            path.unlink()
            removed += 1
            # Invalidate cache since we modified the directory
            _presence_cache.clear()
        except OSError:
            continue
    return removed
