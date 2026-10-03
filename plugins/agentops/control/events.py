"""Schema-v1 event validation and crash-safe local spool handling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from plugins.agentops.control.models import AppendResult, EventEnvelope, SpoolReplayResult


class EventValidationError(ValueError):
    """Raised without echoing untrusted event content."""


class SpoolCapacityError(RuntimeError):
    """Raised when the configured local spool budget has been reached."""


class SpoolQuarantineError(RuntimeError):
    """An untrusted spool file could not be durably redacted or removed."""


class EventStore(Protocol):
    def append_event(self, event: EventEnvelope) -> AppendResult: ...


_SECRET_KEY = re.compile(r"(?:api[_-]?key|token|cookie|password|secret|authorization|credential)", re.I)
_SECRET_VALUE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{8,}\b|\bBearer\s+[A-Za-z0-9._-]{8,}\b|\bgh[pousr]_[A-Za-z0-9]{8,}\b)",
    re.I,
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically after recursively rejecting unsafe values."""
    safe_value = _validate_json(value)
    return json.dumps(safe_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_json(value: Any) -> Any:
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, str):
        validate_string_value(value, required=False)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EventValidationError("event validation failed")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise EventValidationError("event validation failed")
            try:
                validate_string_value(key, required=True)
            except ValueError as exc:
                raise EventValidationError("event validation failed")
            output[key] = _validate_json(child)
        return output
    if isinstance(value, (list, tuple)):
        return [_validate_json(child) for child in value]
    raise EventValidationError("event validation failed")


def contains_secret(value: Any) -> bool:
    try:
        encoded = canonical_json(value)
    except EventValidationError:
        return True
    return bool(_SECRET_VALUE.search(encoded))


def validate_string_value(value: Any, *, required: bool) -> None:
    """Reject secret-looking and non-text values without reflecting them."""
    if not isinstance(value, str) or (required and not value.strip()):
        raise EventValidationError("event validation failed")
    if _SECRET_KEY.search(value) or _SECRET_VALUE.search(value):
        raise EventValidationError("event validation failed")


def contains_secret_blob(value: bytes) -> bool:
    try:
        decoded = value.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return bool(_SECRET_VALUE.search(decoded) or _SECRET_KEY.search(decoded))


def validate_event_fields(
    *,
    schema_version: Any,
    event_id: Any,
    event_type: Any,
    occurred_at: Any,
    producer: Any,
    target_id: Any,
    correlation_id: Any,
    payload: Any,
    redaction_version: Any,
) -> None:
    if schema_version != 1 or not isinstance(redaction_version, int) or redaction_version < 1:
        raise EventValidationError("event validation failed")
    for value in (event_id, event_type, producer, target_id):
        try:
            validate_string_value(value, required=True)
        except ValueError as exc:
            raise EventValidationError("event validation failed") from exc
        if not _SAFE_ID.fullmatch(value):
            raise EventValidationError("event validation failed")
    if correlation_id is not None:
        try:
            validate_string_value(correlation_id, required=True)
        except ValueError as exc:
            raise EventValidationError("event validation failed") from exc
        if not _SAFE_ID.fullmatch(correlation_id):
            raise EventValidationError("event validation failed")
    if not isinstance(occurred_at, datetime) or occurred_at.tzinfo is None:
        raise EventValidationError("event validation failed")
    _validate_json(payload)


class EventSpool:
    """Bounded event spool that is independent of any target or Gateway."""

    def __init__(self, root: Path, *, max_bytes: int = 256 * 1024 * 1024):
        self.root = Path(root)
        self.quarantine_dir = self.root / "quarantine"
        self.max_bytes = max_bytes

    def _ensure_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        os.chmod(self.quarantine_dir, 0o700)

    def _cleanup_orphan_temps(self) -> int:
        """Remove temp artifacts without reading their potentially unsafe bytes."""
        self._ensure_directories()
        failed = 0
        for directory in (self.root, self.quarantine_dir):
            changed = False
            for temporary in directory.glob(".*.tmp"):
                try:
                    metadata = temporary.lstat()
                    if stat.S_ISDIR(metadata.st_mode):
                        raise SpoolQuarantineError("orphan temp directory")
                    temporary.unlink()
                    changed = True
                except (OSError, SpoolQuarantineError):
                    failed += 1
            if changed:
                try:
                    self._fsync_parent(directory)
                except OSError:
                    failed += 1
        return failed

    def pending_paths(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(path for path in self.root.glob("*.json") if path.is_file())

    def depth(self) -> int:
        return len(self.pending_paths())

    def _size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.pending_paths())

    def quarantine_size_bytes(self) -> int:
        if not self.quarantine_dir.exists():
            return 0
        return sum(path.stat().st_size for path in self.quarantine_dir.glob("*.json") if path.is_file())

    def total_size_bytes(self) -> int:
        return self._size_bytes() + self.quarantine_size_bytes()

    def healthy(self) -> bool:
        return self.total_size_bytes() <= self.max_bytes and self.quarantine_size_bytes() <= self.max_bytes // 4

    @staticmethod
    def _fsync_parent(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def write(self, event: EventEnvelope) -> Path:
        payload = canonical_json(event.to_dict()).encode("utf-8")
        self._ensure_directories()
        destination = self.root / f"{event.event_id}.json"
        if destination.exists():
            if destination.read_bytes() == payload:
                return destination
            raise EventValidationError("event validation failed")
        if self.total_size_bytes() + len(payload) > self.max_bytes:
            raise SpoolCapacityError("event spool capacity reached")
        temporary = self.root / f".event-{event.event_id}-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self._fsync_parent(self.root)
            return destination
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
                self._fsync_parent(self.root)

    def _remove_untrusted_source(self, path: Path) -> bool:
        try:
            path.unlink()
            self._fsync_parent(self.root)
            return True
        except OSError:
            return False

    def _quarantine(self, path: Path, raw: bytes, reason: str) -> str:
        """Persist redacted metadata or report a fatal, non-silent failure.

        The source is untrusted.  If metadata persistence fails, it is still
        removed where possible; the caller receives ``failed`` either way so a
        daemon cannot claim healthy replay after a redaction-path error.
        """
        self._ensure_directories()
        destination = self.quarantine_dir / path.name
        content = canonical_json(
            {
                "reason": reason,
                "content_hash": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "redacted": True,
            }
        ).encode("utf-8")
        if self.quarantine_size_bytes() + len(content) > self.max_bytes // 4:
            return "dropped" if self._remove_untrusted_source(path) else "failed"
        temporary = self.quarantine_dir / f".quarantine-{path.stem}-{uuid.uuid4().hex}.tmp"
        outcome = "failed"
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self._fsync_parent(self.quarantine_dir)
            if not self._remove_untrusted_source(path):
                outcome = "failed"
            else:
                outcome = "quarantined"
        except OSError:
            self._remove_untrusted_source(path)
            outcome = "failed"
        finally:
            try:
                if temporary.exists() or temporary.is_symlink():
                    temporary.unlink()
                    self._fsync_parent(self.quarantine_dir)
            except OSError:
                # A leftover temp is itself a fatal replay condition; the next
                # start will retry deletion without reading it.
                outcome = "failed"
        return outcome

    def replay(self, store: EventStore) -> SpoolReplayResult:
        appended = duplicates = quarantined = dropped = failed = 0
        failed += self._cleanup_orphan_temps()
        for path in self.pending_paths():
            raw: bytes = b""
            try:
                raw = path.read_bytes()
                decoded = raw.decode("utf-8")
                event = EventEnvelope.from_dict(json.loads(decoded))
                result = store.append_event(event)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, EventValidationError, TypeError, AttributeError, RuntimeError):
                try:
                    outcome = self._quarantine(path, raw, "event_invalid")
                    if outcome == "quarantined":
                        quarantined += 1
                    elif outcome == "dropped":
                        dropped += 1
                    else:
                        failed += 1
                except OSError:
                    failed += 1
                continue
            path.unlink(missing_ok=True)
            self._fsync_parent(self.root)
            if result.inserted:
                appended += 1
            else:
                duplicates += 1
        return SpoolReplayResult(
            appended=appended,
            duplicates=duplicates,
            quarantined=quarantined,
            dropped=dropped,
            failed=failed,
        )
