"""Profile-scoped durable storage for Telegram location telemetry."""

from __future__ import annotations

import json
import math
import os
import threading
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_cli.profiles import normalize_profile_name, validate_profile_name
from hermes_constants import (
    assert_named_profile_home_live,
    get_default_hermes_root,
)
from utils import atomic_json_write


_LOCATION_LOCKS_GUARD = threading.Lock()
_LOCATION_WRITE_LOCKS: dict[str, threading.Lock] = {}
_SNAPSHOT_KEYS = frozenset(
    {
        "accuracy_m",
        "chat_id",
        "heading",
        "is_live",
        "latitude",
        "live_period",
        "longitude",
        "message_id",
        "message_thread_id",
        "profile",
        "source",
        "source_timestamp",
        "speed_mps",
        "update_id",
        "updated_at",
        "user_id",
    }
)


def _location_thread_lock(location_path: Path) -> threading.Lock:
    key = str(location_path.resolve(strict=False))
    with _LOCATION_LOCKS_GUARD:
        return _LOCATION_WRITE_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _location_file_lock(location_path: Path) -> Iterator[None]:
    """Serialize read/compare/write across gateway processes."""
    location_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = location_path.with_name(f".{location_path.name}.lock")
    raw_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    handle = os.fdopen(raw_fd, "r+b", buffering=0)
    acquired = False
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import msvcrt

            if lock_path.stat().st_size == 0:
                handle.write(b"\0")
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                with suppress(OSError):
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                with suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _profile_home(profile: Optional[str]) -> tuple[str, Path]:
    canonical = normalize_profile_name(profile or "default")
    validate_profile_name(canonical)
    root = get_default_hermes_root()
    if canonical == "default":
        return canonical, root
    home = root / "profiles" / canonical
    assert_named_profile_home_live(home)
    return canonical, home


def _aware_timestamp(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat()


def _number(
    value: Any,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    if minimum is not None and parsed < minimum:
        return None
    if maximum is not None and parsed > maximum:
        return None
    return parsed


def _integer(value: Any, *, allow_none: bool = False, nonzero: bool = False) -> Optional[int]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Telegram identifiers must be integers")
    if nonzero and value == 0:
        raise ValueError("Telegram identifiers must be non-zero")
    if not nonzero and value < 0:
        raise ValueError("Telegram sequence identifiers must be non-negative")
    return value


def _optional_number(
    value: Any,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    if value is None:
        return None
    parsed = _number(value, minimum=minimum, maximum=maximum)
    if parsed is None:
        raise ValueError("Invalid optional location measurement")
    return parsed


def _validated_snapshot(payload: Any, canonical_profile: str) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != _SNAPSHOT_KEYS:
        return None
    if payload.get("source") != "telegram" or payload.get("profile") != canonical_profile:
        return None
    source_timestamp = _aware_timestamp(payload.get("source_timestamp"))
    updated_at = _aware_timestamp(payload.get("updated_at"))
    latitude = _number(payload.get("latitude"), minimum=-90.0, maximum=90.0)
    longitude = _number(payload.get("longitude"), minimum=-180.0, maximum=180.0)
    if source_timestamp is None or updated_at is None or latitude is None or longitude is None:
        return None
    if not isinstance(payload.get("is_live"), bool):
        return None
    try:
        chat_id = _integer(payload.get("chat_id"), nonzero=True)
        user_id = _integer(payload.get("user_id"), nonzero=True)
        message_id = _integer(payload.get("message_id"))
        thread_id = _integer(payload.get("message_thread_id"), allow_none=True)
        update_id = _integer(payload.get("update_id"))
        live_period = _integer(payload.get("live_period"), allow_none=True)
        accuracy = _optional_number(payload.get("accuracy_m"), minimum=0.0)
        heading = _optional_number(payload.get("heading"), minimum=0.0, maximum=360.0)
        speed = _optional_number(payload.get("speed_mps"), minimum=0.0)
    except ValueError:
        return None
    if payload["is_live"] is not bool(live_period is not None and live_period > 0):
        return None
    return {
        "accuracy_m": accuracy,
        "chat_id": chat_id,
        "heading": heading,
        "is_live": payload["is_live"],
        "latitude": latitude,
        "live_period": live_period,
        "longitude": longitude,
        "message_id": message_id,
        "message_thread_id": thread_id,
        "profile": canonical_profile,
        "source": "telegram",
        "source_timestamp": source_timestamp,
        "speed_mps": speed,
        "update_id": update_id,
        "updated_at": updated_at,
        "user_id": user_id,
    }


def _source_order(payload: Any) -> Optional[tuple[datetime, int]]:
    if not isinstance(payload, dict):
        return None
    raw_timestamp = _aware_timestamp(payload.get("source_timestamp"))
    update_id = payload.get("update_id")
    if raw_timestamp is None or isinstance(update_id, bool) or not isinstance(update_id, int) or update_id < 0:
        return None
    return datetime.fromisoformat(raw_timestamp), update_id


def persist_location_snapshot(payload: dict[str, Any], *, profile: Optional[str]) -> bool:
    """Atomically advance one profile's latest location and reject unsafe state."""
    try:
        canonical_profile, home = _profile_home(profile)
    except (OSError, TypeError, ValueError):
        return False
    incoming = _validated_snapshot(payload, canonical_profile)
    incoming_order = _source_order(incoming)
    if incoming is None or incoming_order is None:
        return False
    location_path = home / "location" / "latest.json"
    try:
        with _location_thread_lock(location_path):
            with _location_file_lock(location_path):
                try:
                    existing = json.loads(location_path.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    existing = None
                except (json.JSONDecodeError, OSError, TypeError):
                    return False
                if existing is not None and not isinstance(existing, dict):
                    return False
                if isinstance(existing, dict):
                    validated_existing = _validated_snapshot(existing, canonical_profile)
                    existing_order = _source_order(validated_existing)
                    if existing_order is None:
                        return False
                    incoming_timestamp, incoming_update_id = incoming_order
                    existing_timestamp, existing_update_id = existing_order
                    if incoming_update_id <= existing_update_id or incoming_timestamp < existing_timestamp:
                        return False
                atomic_json_write(location_path, incoming, indent=2, mode=0o600, sort_keys=True)
                return True
    except (OSError, TimeoutError):
        return False
