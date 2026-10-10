"""Durable bounded gateway lifecycle facts for external health checks.

The writer is optional and activates only when
``HERMES_GATEWAY_CHURN_PATH`` names an absolute path. Records contain process
identity metadata only; policy such as churn windows and thresholds belongs to
the consumer.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


GATEWAY_CHURN_HOOK_VERSION = 1
GATEWAY_CHURN_PATH_ENV = "HERMES_GATEWAY_CHURN_PATH"
GATEWAY_CHURN_MAX_RECORDS = 256
_GATEWAY_CHURN_READ_BYTES = 1024 * 1024
_EVENT_TYPES = frozenset({"start", "replace"})
_IS_WINDOWS = sys.platform == "win32"
_WINDOWS_LOCK_OFFSET = 1024 * 1024


def configure_gateway_churn_from_config(config: object) -> Path | None:
    """Bridge an explicit gateway.churn_path setting to the writer."""

    if not isinstance(config, dict):
        return None
    gateway_config = config.get("gateway")
    if not isinstance(gateway_config, dict) or "churn_path" not in gateway_config:
        return None

    raw = gateway_config["churn_path"]
    if isinstance(raw, str) and raw.strip():
        try:
            path = Path(raw.strip()).expanduser()
        except (OSError, RuntimeError):
            path = None
        if path is not None and path.is_absolute():
            os.environ[GATEWAY_CHURN_PATH_ENV] = str(path)
            return path

    logging.getLogger(__name__).warning(
        "Ignoring gateway.churn_path: expected a nonempty absolute path"
    )
    return None


def gateway_churn_record_path() -> Path | None:
    """Return the configured absolute churn-record path, if enabled."""

    raw = os.environ.get(GATEWAY_CHURN_PATH_ENV, "").strip()
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
    except (OSError, RuntimeError):
        return None
    return path if path.is_absolute() else None


def gateway_churn_hook_info() -> dict[str, object]:
    """Return a side-effect-free feature-probe result for deployment checks."""

    path = gateway_churn_record_path()
    return {
        "version": GATEWAY_CHURN_HOOK_VERSION,
        "enabled": path is not None,
        "path": str(path) if path is not None else None,
        "max_records": GATEWAY_CHURN_MAX_RECORDS,
    }


def _validated_pid(name: str, value: int | None, *, required: bool) -> int | None:
    if value is None and not required:
        return None
    if type(value) is not int or value <= 0:
        suffix = " or None" if not required else ""
        raise ValueError(f"{name} must be a positive integer{suffix}")
    return value


def _validated_timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("timestamp must be a timezone-aware ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("timestamp must be a timezone-aware ISO 8601 string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be a timezone-aware ISO 8601 string")
    return value


def _acquire_file_lock(handle) -> None:
    if _IS_WINDOWS:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write("\n")
            handle.flush()
        handle.seek(_WINDOWS_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_file_lock(handle) -> None:
    try:
        if _IS_WINDOWS:
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextmanager
def _record_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            lock_path.chmod(0o600)
        except OSError:
            pass
        _acquire_file_lock(handle)
        yield
    finally:
        _release_file_lock(handle)
        handle.close()


def _read_tail(path: Path) -> list[bytes]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            start = max(0, size - _GATEWAY_CHURN_READ_BYTES)
            starts_mid_line = False
            if start > 0:
                handle.seek(start - 1)
                starts_mid_line = handle.read(1) not in (b"\n", b"\r")
            handle.seek(start)
            data = handle.read(_GATEWAY_CHURN_READ_BYTES + 1)
    except FileNotFoundError:
        return []

    lines = data.splitlines(keepends=True)
    if starts_mid_line and lines:
        lines = lines[1:]
    complete = [line for line in lines if line.endswith((b"\n", b"\r"))]
    return complete[-(GATEWAY_CHURN_MAX_RECORDS - 1) :]


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_append_capped(path: Path, encoded_record: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _record_lock(path):
        payload = b"".join([*_read_tail(path), encoded_record])
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        fd = -1
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
            _fsync_directory(path.parent)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def append_gateway_churn_event(
    event_type: str,
    *,
    pid_old: int | None,
    pid_new: int,
    timestamp: str | None = None,
) -> bool:
    """Append one lifecycle fact while preserving a bounded atomic window.

    Returns ``False`` when the hook is disabled or the record cannot be written.
    Invalid caller-supplied facts raise ``ValueError`` instead of serializing an
    ambiguous record.
    """

    if event_type not in _EVENT_TYPES:
        raise ValueError(f"event_type must be one of {sorted(_EVENT_TYPES)}")
    old_pid = _validated_pid("pid_old", pid_old, required=False)
    new_pid = _validated_pid("pid_new", pid_new, required=True)
    observed_at = _validated_timestamp(timestamp)
    path = gateway_churn_record_path()
    if path is None:
        return False

    record = {
        "event_type": event_type,
        "timestamp": observed_at,
        "pid_old": old_pid,
        "pid_new": new_pid,
    }
    encoded = (
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    try:
        _atomic_append_capped(path, encoded)
    except OSError:
        return False
    return True
