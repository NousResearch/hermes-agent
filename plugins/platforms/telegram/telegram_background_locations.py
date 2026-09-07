"""Private state and per-turn context for Telegram background live locations.

The adapter owns Telegram intake; this mixin owns the sensitive state lifecycle.
State is shared by path so a replacement adapter cannot race an older timed-out
writer during reconnect. Writes are serialized on one daemon worker per state
file and coalesced to the newest complete snapshot.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_MAX_BACKGROUND_LOCATION_SUBJECTS = 512

_LOCATION_STATE_FILE_LOCKS_GUARD = threading.Lock()
_LOCATION_STATE_FILE_LOCKS: dict[str, threading.RLock] = {}
_COORDINATE_FREE_LOCATION_SOURCES = frozenset(
    {
        "live_location_stop",
        "live_location_expired",
        "live_location_restart",
        "live_location_persist_failed",
    }
)
_IRREVOCABLE_LOCATION_SOURCES = frozenset(
    {"live_location_stop", "live_location_expired"}
)
_MISSING = object()


def _new_location_writer_epoch() -> str:
    """Opaque ownership generation for coordinate-bearing state writes."""
    return secrets.token_hex(16)


def _location_writer_control_path(root: Path, bot_scope: str) -> Path:
    """Bot-wide ownership fence shared by every profile-local state file."""
    return (
        root
        / "state"
        / "telegram_background_locations"
        / f".{bot_scope}.writer.json"
    )


def _location_state_lock_path(path: Path, owner_home: Optional[Path]) -> Path:
    """Return a lock path that survives named-profile deletion/recreation."""
    digest = hashlib.sha256(str(path.absolute()).encode("utf-8")).hexdigest()[:24]
    if owner_home is not None:
        from hermes_constants import profile_incarnation_path

        lock_dir = profile_incarnation_path(owner_home).parent
    else:
        lock_dir = path.parent
    return lock_dir / f".telegram-background-location-{digest}.lock"


@contextlib.contextmanager
def _background_location_state_file_lock(
    path: Path, owner_home: Optional[Path]
):
    """Serialize one sensitive state file across threads and processes."""
    lock_path = _location_state_lock_path(path, owner_home)
    key = os.path.normcase(str(lock_path.resolve(strict=False)))
    with _LOCATION_STATE_FILE_LOCKS_GUARD:
        thread_lock = _LOCATION_STATE_FILE_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            with contextlib.suppress(OSError):
                os.chmod(lock_path, 0o600)
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _location_record_identity(record: Any) -> Optional[tuple[str, str, str]]:
    if not isinstance(record, dict):
        return None
    identity = tuple(
        str(record.get(key, "") or "")
        for key in ("chat_id", "user_id", "message_id")
    )
    return identity if all(identity) else None


def _location_record_timestamp(record: Any) -> Optional[datetime]:
    if not isinstance(record, dict):
        return None
    for key in ("telegram_timestamp", "recorded_at"):
        value = record.get(key)
        if not isinstance(value, str):
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _location_record_update_id(record: Any) -> Optional[int]:
    if not isinstance(record, dict):
        return None
    value = record.get("update_id")
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _coordinate_free_record(record: Dict[str, Any], source: str) -> Dict[str, Any]:
    marker: Dict[str, Any] = {
        "source": source,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    for key in (
        "subject_key",
        "chat_type",
        "chat_id",
        "user_id",
        "thread_id",
        "message_id",
        "telegram_timestamp",
        "update_id",
    ):
        value = record.get(key)
        if value is not None and str(value):
            marker[key] = str(value)
    return marker


def _safer_or_newer_location_record(
    current: Dict[str, Any], desired: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve concurrent mutations without reviving a terminal lifecycle."""
    current_source = str(current.get("source", ""))
    desired_source = str(desired.get("source", ""))
    same_lifecycle = (
        _location_record_identity(current) is not None
        and _location_record_identity(current) == _location_record_identity(desired)
    )
    if same_lifecycle:
        if (
            current_source in _IRREVOCABLE_LOCATION_SOURCES
            and desired_source == "live_location"
        ):
            return current
        if (
            desired_source in _IRREVOCABLE_LOCATION_SOURCES
            and current_source == "live_location"
        ):
            return desired

    current_timestamp = _location_record_timestamp(current)
    desired_timestamp = _location_record_timestamp(desired)
    if current_timestamp is not None and desired_timestamp is not None:
        if current_timestamp != desired_timestamp:
            return desired if desired_timestamp > current_timestamp else current
        current_update_id = _location_record_update_id(current)
        desired_update_id = _location_record_update_id(desired)
        if (
            current_update_id is not None
            and desired_update_id is not None
            and current_update_id != desired_update_id
        ):
            return desired if desired_update_id > current_update_id else current

    # Equal or unorderable records fail closed: coordinate-free state wins.
    current_safe = current_source in _COORDINATE_FREE_LOCATION_SOURCES
    desired_safe = desired_source in _COORDINATE_FREE_LOCATION_SOURCES
    if current_safe != desired_safe:
        return current if current_safe else desired
    # The on-disk value is already durable; retain it when order cannot be proven.
    return current


def _merge_location_snapshots(
    current: Dict[str, dict],
    baseline: Dict[str, dict],
    desired: Dict[str, dict],
) -> Dict[str, dict]:
    """Rebase a whole-file snapshot over changes made by another process."""
    merged: Dict[str, dict] = {}
    for key in set(current) | set(baseline) | set(desired):
        old = baseline.get(key, _MISSING)
        disk = current.get(key, _MISSING)
        wanted = desired.get(key, _MISSING)

        if disk == old:
            chosen = wanted
            # Persist a revocation marker for an active record removed by this
            # snapshot. A writer that staged before the removal can then never
            # resurrect its coordinates after acquiring the file lock later.
            if wanted is _MISSING and isinstance(old, dict) and old.get(
                "source"
            ) == "live_location":
                chosen = _coordinate_free_record(old, "live_location_restart")
        elif wanted == old or disk == wanted:
            chosen = disk
        elif disk is _MISSING:
            # A concurrent removal is safer than an older active rewrite.
            if (
                isinstance(old, dict)
                and isinstance(wanted, dict)
                and old.get("source") == "live_location"
                and wanted.get("source") == "live_location"
                and _location_record_identity(old)
                == _location_record_identity(wanted)
            ):
                chosen = _coordinate_free_record(
                    wanted, "live_location_restart"
                )
            else:
                chosen = wanted
        elif wanted is _MISSING:
            chosen = disk
        elif isinstance(disk, dict) and isinstance(wanted, dict):
            chosen = _safer_or_newer_location_record(disk, wanted)
        else:
            chosen = disk

        if isinstance(chosen, dict):
            merged[str(key)] = dict(chosen)

    # A lifecycle can move between subject keys during profile/topic routing.
    # Collapse duplicates after the per-key merge so a terminal marker in any
    # key defeats an older coordinate-bearing copy of that same live share.
    winner_by_lifecycle: dict[tuple[str, str, str], tuple[str, Dict[str, Any]]] = {}
    for key, record in list(merged.items()):
        identity = _location_record_identity(record)
        if identity is None:
            continue
        previous = winner_by_lifecycle.get(identity)
        if previous is None:
            winner_by_lifecycle[identity] = (key, record)
            continue
        previous_key, previous_record = previous
        winner = _safer_or_newer_location_record(previous_record, record)
        if winner is previous_record:
            merged.pop(key, None)
        else:
            merged.pop(previous_key, None)
            winner_by_lifecycle[identity] = (key, record)
    return merged


def _read_location_snapshot_unlocked(
    path: Path,
    owner_incarnation: Optional[tuple[str, int, int, int]],
) -> Dict[str, dict]:
    try:
        with path.open("rb") as state_file:
            raw_payload = state_file.read(2 * 1024 * 1024 + 1)
        if len(raw_payload) > 2 * 1024 * 1024:
            return {}
        payload = json.loads(raw_payload)
        if not isinstance(payload, dict) or payload.get("version") != 2:
            return {}
        if owner_incarnation is not None and payload.get(
            "owner_incarnation"
        ) != list(owner_incarnation):
            return {}
        locations = payload.get("locations")
        if not isinstance(locations, dict):
            return {}
        return {
            str(key): dict(value)
            for key, value in locations.items()
            if isinstance(value, dict)
        }
    except (FileNotFoundError, OSError, ValueError, TypeError, RecursionError):
        return {}


def _read_location_writer_epoch_unlocked(path: Path) -> Optional[str]:
    """Read an epoch while the corresponding file lock is held."""
    try:
        with path.open("rb") as state_file:
            raw_payload = state_file.read(2 * 1024 * 1024 + 1)
        if len(raw_payload) > 2 * 1024 * 1024:
            return None
        payload = json.loads(raw_payload)
        value = payload.get("writer_epoch") if isinstance(payload, dict) else None
        if isinstance(value, str) and 0 < len(value) <= 128:
            return value
    except (FileNotFoundError, OSError, ValueError, TypeError, RecursionError):
        pass
    return None


def _write_location_writer_epoch_unlocked(path: Path, epoch: str) -> None:
    """Durably replace the bot-wide writer epoch while its lock is held."""
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)
    atomic_json_write(
        path,
        {"version": 1, "writer_epoch": epoch},
        mode=0o600,
        sort_keys=True,
        create_parent=False,
    )


def _claim_location_writer_epoch(path: Path) -> tuple[str, Optional[str]]:
    """Synchronously claim a fresh bot-wide generation.

    This write is intentionally not queued, coalesced, or timeout-bounded.  A
    reconnect must not start polling until its ownership fence is durable, and
    a stop must not return while an older process can still commit coordinates.
    """
    epoch = _new_location_writer_epoch()
    with _background_location_state_file_lock(path, None):
        previous_epoch = _read_location_writer_epoch_unlocked(path)
        _write_location_writer_epoch_unlocked(path, epoch)
    return epoch, previous_epoch


def _get_or_create_location_writer_epoch(
    path: Path, proposed_epoch: str
) -> Optional[str]:
    """Read the current bot epoch, creating it only when truly absent.

    An existing but malformed control file fails closed instead of letting an
    arbitrary process become the coordinate writer again.
    """
    with _background_location_state_file_lock(path, None):
        epoch = _read_location_writer_epoch_unlocked(path)
        if epoch is not None:
            return epoch
        try:
            exists = path.exists()
        except OSError:
            return None
        if exists:
            return None
        _write_location_writer_epoch_unlocked(path, proposed_epoch)
        return proposed_epoch


def _read_location_writer_epoch(path: Path) -> Optional[str]:
    """Read the bot-wide epoch behind its cross-process lock."""
    with _background_location_state_file_lock(path, None):
        return _read_location_writer_epoch_unlocked(path)


def _directory_incarnation(path: Path) -> Optional[tuple[str, int, int, int]]:
    """Stable identity for one existing directory incarnation.

    Named profile paths may be deleted and recreated while a multiplex gateway
    remains alive. Path equality alone must not make the new profile inherit
    the old profile's sensitive in-memory state.
    """
    try:
        from hermes_constants import ensure_named_profile_incarnation

        token = ensure_named_profile_incarnation(path)
        stat_result = path.stat()
        if token is None or not path.is_dir():
            return None
    except OSError:
        return None
    birth_ns = getattr(stat_result, "st_birthtime_ns", None)
    if birth_ns is None:
        birth = getattr(stat_result, "st_birthtime", None)
        birth_ns = int(float(birth) * 1_000_000_000) if birth is not None else 0
    return token, int(stat_result.st_dev), int(stat_result.st_ino), int(birth_ns)


def _current_directory_incarnation(
    path: Path,
) -> Optional[tuple[str, int, int, int]]:
    """Read (without creating) the current identity of an existing profile."""
    try:
        from hermes_constants import (
            named_profile_is_deleted,
            read_named_profile_incarnation,
        )

        if named_profile_is_deleted(path):
            return None
        token = read_named_profile_incarnation(path)
        stat_result = path.stat()
        if token is None or not path.is_dir():
            return None
    except OSError:
        return None
    birth_ns = getattr(stat_result, "st_birthtime_ns", None)
    if birth_ns is None:
        birth = getattr(stat_result, "st_birthtime", None)
        birth_ns = int(float(birth) * 1_000_000_000) if birth is not None else 0
    return token, int(stat_result.st_dev), int(stat_result.st_ino), int(birth_ns)


def _supports_directory_fd_snapshot_write() -> bool:
    return os.name == "posix" and hasattr(os, "O_DIRECTORY")


def _write_snapshot_if_current_unlocked(
    path: Path,
    records: Dict[str, dict],
    is_current: Any,
    owner_incarnation: Optional[tuple[str, int, int, int]],
    writer_epoch: Optional[str] = None,
) -> None:
    """Atomically write without ever recreating a deleted named profile."""
    from hermes_constants import mkdir_under_hermes_home

    if not is_current():
        raise FileNotFoundError(f"background location state owner changed: {path}")
    mkdir_under_hermes_home(path.parent)
    if not is_current():
        raise FileNotFoundError(f"background location state owner changed: {path}")
    payload: Dict[str, Any] = {"version": 2, "locations": records}
    if owner_incarnation is not None:
        payload["owner_incarnation"] = list(owner_incarnation)
    if writer_epoch:
        payload["writer_epoch"] = writer_epoch

    if _supports_directory_fd_snapshot_write():
        # Pin the already-validated directory itself. If another process now
        # deletes and recreates the same profile path, dir-fd-relative rename
        # still targets the old directory (or fails); it can never land in the
        # new profile incarnation.
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        temporary_name = (
            f".{path.stem}.{os.getpid()}.{threading.get_ident()}."
            f"{secrets.token_hex(8)}.tmp"
        )
        temporary_fd: Optional[int] = None
        try:
            if not is_current():
                raise FileNotFoundError(
                    f"background location state owner changed: {path}"
                )
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
                temporary_fd = None
                json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.rename(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
            return
        finally:
            if temporary_fd is not None:
                os.close(temporary_fd)
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass
            os.close(directory_fd)

    # On platforms without dir-fd rename, the caller's profile-lifecycle lock
    # closes the final validation-to-replace interval. The embedded owner token
    # is still validated on every subsequent load.
    atomic_json_write(
        path,
        payload,
        mode=0o600,
        sort_keys=True,
        create_parent=False,
    )


def _write_snapshot_if_current(
    path: Path,
    records: Dict[str, dict],
    is_current: Any,
    owner_incarnation: Optional[tuple[str, int, int, int]],
    owner_home: Optional[Path] = None,
    baseline_records: Optional[Dict[str, dict]] = None,
    expected_writer_epoch: Optional[str] = None,
    writer_control_path: Optional[Path] = None,
) -> tuple[Dict[str, dict], Optional[str]]:
    """Merge and write one snapshot behind cross-process lifecycle fences.

    POSIX additionally pins the directory with a file descriptor. Windows has
    no dir-fd-relative replace, so the cross-process incarnation lock prevents
    delete/recreate from crossing the last validation-to-replace interval.
    """
    control_lock = (
        _background_location_state_file_lock(writer_control_path, None)
        if writer_control_path is not None
        else contextlib.nullcontext()
    )
    # Global order is always control lock -> profile state lock.  A stop or
    # reconnect can therefore rotate one bot-wide fence between any two state
    # commits without deadlocking profile-local writers.
    with control_lock:
        current_writer_epoch: Optional[str] = None
        if writer_control_path is not None:
            current_writer_epoch = _read_location_writer_epoch_unlocked(
                writer_control_path
            )
            if current_writer_epoch is None:
                try:
                    control_exists = writer_control_path.exists()
                except OSError:
                    control_exists = True
                if not control_exists and expected_writer_epoch is not None:
                    _write_location_writer_epoch_unlocked(
                        writer_control_path, expected_writer_epoch
                    )
                    current_writer_epoch = expected_writer_epoch

        with _background_location_state_file_lock(path, owner_home):
            current_records = _read_location_snapshot_unlocked(
                path, owner_incarnation
            )
            current_snapshot_epoch = _read_location_writer_epoch_unlocked(path)
            desired_records = {
                str(key): dict(value)
                for key, value in records.items()
                if isinstance(value, dict)
            }
            stale_writer = (
                writer_control_path is not None
                and expected_writer_epoch is not None
                and current_writer_epoch != expected_writer_epoch
            )
            if stale_writer:
                # A stop or replacement polling owner has invalidated this
                # process. Safety mutations may still merge, but coordinates
                # staged under the old epoch can never cross the durable fence.
                desired_records = {
                    key: record
                    for key, record in desired_records.items()
                    if record.get("source") != "live_location"
                }
                if current_snapshot_epoch != current_writer_epoch:
                    # Do not let a stale writer re-sign an old on-disk active
                    # record merely because it appeared on the ``current``
                    # side of the three-way merge.
                    current_records = {
                        key: (
                            _coordinate_free_record(
                                record, "live_location_restart"
                            )
                            if record.get("source") == "live_location"
                            else record
                        )
                        for key, record in current_records.items()
                    }
            baseline = (
                current_records
                if baseline_records is None
                else {
                    str(key): dict(value)
                    for key, value in baseline_records.items()
                    if isinstance(value, dict)
                }
            )
            committed_records = _merge_location_snapshots(
                current_records,
                baseline,
                desired_records,
            )
            if len(committed_records) > _MAX_BACKGROUND_LOCATION_SUBJECTS:
                def _premerge_recorded_at(item: tuple[str, dict]) -> str:
                    key, committed = item
                    candidates = (
                        desired_records.get(key), current_records.get(key)
                    )
                    original_values = [
                        str(candidate.get("recorded_at", ""))
                        for candidate in candidates
                        if isinstance(candidate, dict)
                    ]
                    if original_values:
                        return max(original_values)
                    return max(
                        str(committed.get("recorded_at", "")), ""
                    )

                committed_records = dict(
                    sorted(
                        committed_records.items(),
                        key=_premerge_recorded_at,
                        reverse=True,
                    )[:_MAX_BACKGROUND_LOCATION_SUBJECTS]
                )
            committed_writer_epoch = current_writer_epoch or expected_writer_epoch
            lock = contextlib.nullcontext()
            if owner_incarnation is not None and owner_home is not None:
                from hermes_constants import named_profile_incarnation_lock

                lock = named_profile_incarnation_lock(owner_home)
            with lock:
                _write_snapshot_if_current_unlocked(
                    path,
                    committed_records,
                    is_current,
                    owner_incarnation,
                    committed_writer_epoch,
                )
            return committed_records, committed_writer_epoch


@dataclass
class _SnapshotWrite:
    records: Dict[str, dict]
    baseline_records: Dict[str, dict]
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    expected_writer_epoch: Optional[str]
    attempts: int = 0


@dataclass(frozen=True)
class _SnapshotWriteResult:
    succeeded: bool
    records: Optional[Dict[str, dict]] = None
    writer_epoch: Optional[str] = None


class _LatestSnapshotWriter:
    """Serialize whole-file replacements and retain only the newest queued one."""

    _MAX_WRITE_ATTEMPTS = 3
    _RETRY_BASE_DELAY_SECONDS = 0.05

    def __init__(
        self,
        path: Path,
        is_current: Any,
        owner_incarnation: Optional[tuple[str, int, int, int]],
        owner_home: Optional[Path] = None,
        writer_control_path: Optional[Path] = None,
    ) -> None:
        self.path = path
        self._is_current = is_current
        self._owner_incarnation = owner_incarnation
        self._owner_home = owner_home
        self._writer_control_path = writer_control_path
        self.io_lock = threading.Lock()
        self._lock = threading.Lock()
        self._pending: Optional[_SnapshotWrite] = None
        self._thread: Optional[threading.Thread] = None
        self._inflight = False
        self._latest_records: Optional[Dict[str, dict]] = None
        self._closed = False

    @staticmethod
    def _copy_records(records: Dict[str, dict]) -> Dict[str, dict]:
        return {str(key): dict(value) for key, value in records.items()}

    @staticmethod
    def _resolve(request: _SnapshotWrite, result: _SnapshotWriteResult) -> None:
        def finish() -> None:
            if not request.future.done():
                request.future.set_result(result)

        try:
            request.loop.call_soon_threadsafe(finish)
        except RuntimeError:
            # The event loop may already be closed during gateway shutdown.
            pass

    @property
    def thread(self) -> Optional[threading.Thread]:
        with self._lock:
            return self._thread

    def outstanding_snapshot(self) -> Optional[Dict[str, dict]]:
        """Return the newest staged snapshot while disk is known to lag it."""
        with self._lock:
            if self._pending is None and not self._inflight:
                return None
            if self._latest_records is None:
                return None
            return self._copy_records(self._latest_records)

    def submit(
        self,
        records: Dict[str, dict],
        baseline_records: Dict[str, dict],
        loop: asyncio.AbstractEventLoop,
        expected_writer_epoch: Optional[str],
    ) -> tuple[asyncio.Future, Optional[threading.Thread]]:
        snapshot = self._copy_records(records)
        baseline = self._copy_records(baseline_records)
        request = _SnapshotWrite(
            snapshot,
            baseline,
            loop,
            loop.create_future(),
            expected_writer_epoch,
        )
        replaced: Optional[_SnapshotWrite]
        with self._lock:
            if self._closed or not self._is_current():
                self._resolve(request, _SnapshotWriteResult(False))
                return request.future, self._thread
            replaced = self._pending
            self._pending = request
            self._latest_records = snapshot
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._drain,
                    name="telegram-background-location-writer",
                    daemon=True,
                )
                self._thread.start()
            worker = self._thread
        if replaced is not None:
            self._resolve(replaced, _SnapshotWriteResult(False))
        return request.future, worker

    def close(self) -> Optional[threading.Thread]:
        """Reject queued writes and return any in-flight worker for joining."""
        with self._lock:
            self._closed = True
            pending = self._pending
            self._pending = None
            self._latest_records = None
            worker = self._thread
        if pending is not None:
            self._resolve(pending, _SnapshotWriteResult(False))
        return worker

    def _drain(self) -> None:
        while True:
            with self._lock:
                request = self._pending
                self._pending = None
                if request is None:
                    self._thread = None
                    self._inflight = False
                    return
                self._inflight = True

            request.attempts += 1
            try:
                with self.io_lock:
                    committed = _write_snapshot_if_current(
                        self.path,
                        request.records,
                        self._is_current,
                        self._owner_incarnation,
                        self._owner_home,
                        request.baseline_records,
                        request.expected_writer_epoch,
                        self._writer_control_path,
                    )
                # Tests and third-party subclasses may wrap the legacy helper.
                # Preserve compatibility while production returns both the
                # merged records and the epoch physically stored beside them.
                if isinstance(committed, tuple) and len(committed) == 2:
                    committed_records, committed_epoch = committed
                elif isinstance(committed, dict):
                    committed_records = committed
                    committed_epoch = request.expected_writer_epoch
                else:
                    committed_records = request.records
                    committed_epoch = request.expected_writer_epoch
                result = _SnapshotWriteResult(
                    True,
                    self._copy_records(committed_records),
                    committed_epoch,
                )
            except OSError:
                logger.warning(
                    "[Telegram] Could not persist background location state at %s "
                    "(attempt %d/%d)",
                    self.path,
                    request.attempts,
                    self._MAX_WRITE_ATTEMPTS,
                    exc_info=True,
                )
                result = _SnapshotWriteResult(False)
            except BaseException:
                logger.warning(
                    "[Telegram] Unexpected background location write failure",
                    exc_info=True,
                )
                result = _SnapshotWriteResult(False)
            finally:
                with self._lock:
                    self._inflight = False

            if not result.succeeded and request.attempts < self._MAX_WRITE_ATTEMPTS:
                # Do not retry an obsolete whole-file snapshot over a newer
                # pending update. A short bounded backoff handles transient
                # filesystem failures without stalling Telegram's event loop.
                time.sleep(
                    self._RETRY_BASE_DELAY_SECONDS * (2 ** (request.attempts - 1))
                )
                with self._lock:
                    if self._pending is None:
                        self._pending = request
                        continue
            self._resolve(request, result)


class _SharedLocationState:
    def __init__(
        self,
        path: Path,
        subject_prefix: Optional[str] = None,
        owner_home: Optional[Path] = None,
        writer_control_path: Optional[Path] = None,
    ) -> None:
        self.path = path
        self.subject_prefix = subject_prefix
        self.owner_home = owner_home
        self.writer_control_path = writer_control_path
        self.owner_incarnation = (
            _directory_incarnation(owner_home) if owner_home is not None else None
        )
        self.released = False
        self.records: Optional[Dict[str, dict]] = None
        self.records_writer_epoch: Optional[str] = None
        # Last snapshot proven to have come from (or been committed to) disk.
        # ``records`` may be newer while an async write is pending.
        self.persisted_records: Optional[Dict[str, dict]] = None
        # Preserve the original three-way-merge base across retries and
        # coalesced mutations; rebasing from the uncommitted desired snapshot
        # would make a retry incorrectly prefer stale disk coordinates.
        self.write_baseline_records: Optional[Dict[str, dict]] = None
        self.cached_at_monotonic: Optional[float] = None
        self.dirty = False
        self.generation = 0
        self.retry_not_before_monotonic = 0.0
        self.writer_epoch = _new_location_writer_epoch()
        self.expiry_timer: Optional[asyncio.TimerHandle] = None
        # Telegram polling intentionally drops queued updates on a cold process
        # start. Until the first connect invalidates persisted active records,
        # an offline stop could otherwise be missed forever.
        self.cold_start_pending = True
        self.mutation_lock = asyncio.Lock()
        self.writer = _LatestSnapshotWriter(
            path,
            self.is_current,
            self.owner_incarnation,
            owner_home,
            writer_control_path,
        )

    def is_current(self) -> bool:
        if self.released:
            return False
        if self.owner_home is None:
            return True
        try:
            from hermes_constants import named_profile_is_deleted

            if named_profile_is_deleted(self.owner_home):
                return False
        except OSError:
            return False
        return (
            self.owner_incarnation is not None
            and _current_directory_incarnation(self.owner_home)
            == self.owner_incarnation
        )

    def release(self) -> Optional[threading.Thread]:
        self.released = True
        self.records = {}
        self.records_writer_epoch = None
        self.persisted_records = {}
        self.write_baseline_records = None
        self.cached_at_monotonic = None
        self.dirty = False
        if self.expiry_timer is not None:
            self.expiry_timer.cancel()
            self.expiry_timer = None
        return self.writer.close()


_SHARED_STATE_LOCK = threading.Lock()
_SHARED_STATE_BY_PATH: Dict[str, _SharedLocationState] = {}
_STOP_FENCE_LOCK = threading.Lock()
_STOP_FENCES_BY_BOT: Dict[str, set[tuple[str, str, str]]] = {}


def _shared_location_state(
    path: Path,
    subject_prefix: Optional[str] = None,
    owner_home: Optional[Path] = None,
    writer_control_path: Optional[Path] = None,
) -> _SharedLocationState:
    key = str(path.absolute())
    retired: Optional[_SharedLocationState] = None
    with _SHARED_STATE_LOCK:
        state = _SHARED_STATE_BY_PATH.get(key)
        if state is not None and (
            state.owner_home != owner_home
            or state.writer_control_path != writer_control_path
            or not state.is_current()
        ):
            retired = state
            state = None
        if state is None:
            state = _SharedLocationState(
                path, subject_prefix, owner_home, writer_control_path
            )
            _SHARED_STATE_BY_PATH[key] = state
        elif subject_prefix is not None:
            state.subject_prefix = subject_prefix
    if retired is not None:
        retired.release()
    return state


def release_background_location_states_under(profile_dir: Path) -> int:
    """Retire sensitive state owned by ``profile_dir`` in this process."""
    target = profile_dir.absolute()
    retired: list[_SharedLocationState] = []
    with _SHARED_STATE_LOCK:
        for key, state in list(_SHARED_STATE_BY_PATH.items()):
            if state.owner_home is not None and state.owner_home.absolute() == target:
                _SHARED_STATE_BY_PATH.pop(key, None)
                retired.append(state)
    workers = [state.release() for state in retired]
    for worker in workers:
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=10.0)
    return len(retired)


class TelegramBackgroundLocationsMixin:
    """Background-location lifecycle mixed into :class:`TelegramAdapter`."""

    _BACKGROUND_LOCATION_CONTEXT_HEADER = "[Background Telegram location context]"
    _BACKGROUND_LOCATION_STATE_VERSION = 2
    _BACKGROUND_LOCATION_MAX_SUBJECTS = _MAX_BACKGROUND_LOCATION_SUBJECTS
    _BACKGROUND_LOCATION_MAX_STATE_BYTES = 2 * 1024 * 1024
    _BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD = 0x7FFFFFFF
    _BACKGROUND_LOCATION_WRITE_TIMEOUT_SECONDS = 10.0
    _BACKGROUND_LOCATION_STATE_CACHE_TTL_SECONDS = 30.0

    def _init_background_locations(self, config: Any) -> None:
        raw_opt_in = (
            self.config.extra.get("background_locations")
            if getattr(self.config, "extra", None)
            else None
        )
        if isinstance(raw_opt_in, bool):
            configured = raw_opt_in
        elif isinstance(raw_opt_in, str):
            normalized = raw_opt_in.strip().lower()
            configured = normalized in {"true", "1", "yes", "on"}
            if normalized not in {
                "true", "1", "yes", "on", "false", "0", "no", "off", ""
            }:
                configured = False
                logger.warning(
                    "[Telegram] Ignoring invalid background_locations value; "
                    "explicit true/false is required"
                )
        else:
            # Exact coordinate retention is sensitive and must be an explicit
            # opt-in. In particular, bool([False]), bool({"enabled": False}),
            # and bool(2) must never silently enable it.
            configured = False
            if raw_opt_in is not None:
                logger.warning(
                    "[Telegram] Ignoring non-scalar background_locations value; "
                    "explicit true/false is required"
                )
        self._background_locations_configured = configured
        self._background_locations_enabled = self._background_locations_configured
        from hermes_constants import (
            get_default_hermes_root,
            get_hermes_home,
            named_profile_home,
        )

        self._background_location_bot_scope = (
            self._resolve_background_location_bot_scope(config.token)
        )
        self._background_location_default_root = Path(get_default_hermes_root())
        self._background_location_writer_control_path = _location_writer_control_path(
            self._background_location_default_root,
            self._background_location_bot_scope,
        )
        self._background_location_home = Path(get_hermes_home())
        initial_owner_home = named_profile_home(self._background_location_home)
        self._background_location_state_path = (
            self._background_location_home
            / "state"
            / "telegram_background_locations"
            / f"{self._background_location_bot_scope}.json"
        )
        self._background_location_state = _shared_location_state(
            self._background_location_state_path,
            f"bot:{self._background_location_bot_scope}:chat:",
            initial_owner_home,
            self._background_location_writer_control_path,
        )
        self._background_location_states = {
            str(
                self._background_location_state_path.absolute()
            ): self._background_location_state
        }
        self._background_location_profile_scan_complete = False

    @property
    def _background_location_records(self) -> Optional[Dict[str, dict]]:
        return self._background_location_state.records

    @_background_location_records.setter
    def _background_location_records(self, records: Optional[Dict[str, dict]]) -> None:
        self._background_location_state.records = records
        if records is not None:
            # Direct callers are publishing warm in-process state, not loading
            # an unverifiable snapshot from a prior process.
            state = self._background_location_state
            try:
                epoch = (
                    _get_or_create_location_writer_epoch(
                        state.writer_control_path, state.writer_epoch
                    )
                    if state.writer_control_path is not None
                    else state.writer_epoch
                )
            except OSError:
                epoch = None
            if epoch is not None:
                state.writer_epoch = epoch
            state.records_writer_epoch = epoch
            self._background_location_state.cold_start_pending = False

    @property
    def _background_location_records_cached_at_monotonic(self) -> Optional[float]:
        return self._background_location_state.cached_at_monotonic

    @_background_location_records_cached_at_monotonic.setter
    def _background_location_records_cached_at_monotonic(
        self, value: Optional[float]
    ) -> None:
        self._background_location_state.cached_at_monotonic = value

    @property
    def _background_location_write_lock(self) -> asyncio.Lock:
        return self._background_location_state.mutation_lock

    @property
    def _background_location_write_thread(self) -> Optional[threading.Thread]:
        return self._background_location_state.writer.thread

    def _background_location_state_for_source(
        self, source: Any
    ) -> Optional[_SharedLocationState]:
        """Resolve sensitive state to the routed profile's own HERMES_HOME."""
        if source is None or getattr(source, "profile_route_rejected", False) is True:
            return None
        profile = str(getattr(source, "profile", "") or "").strip()
        if not profile:
            profile = str(getattr(self, "_owner_profile", "") or "").strip()
        if not profile:
            self._background_location_state.subject_prefix = (
                f"bot:{self._background_location_bot_scope}:chat:"
            )
            return self._background_location_state
        try:
            from hermes_cli.profiles import (
                get_profile_dir,
                normalize_profile_name,
                validate_profile_name,
            )

            profile = normalize_profile_name(profile)
            validate_profile_name(profile)
            profile_home = Path(get_profile_dir(profile))
        except Exception:
            logger.warning(
                "[Telegram] Refusing background location state for invalid profile %r",
                profile,
                exc_info=True,
            )
            return None
        path = (
            profile_home
            / "state"
            / "telegram_background_locations"
            / f"{self._background_location_bot_scope}.json"
        )
        key = str(path.absolute())
        subject_prefix = f"bot:{self._background_location_bot_scope}:chat:"
        try:
            from hermes_constants import named_profile_home, named_profile_is_deleted

            owner_home = named_profile_home(profile_home)
            if (
                owner_home is not None
                and named_profile_is_deleted(owner_home)
            ) or not profile_home.is_dir():
                return None
        except OSError:
            return None
        # Always consult the shared registry: a CLI in another process may have
        # deleted and recreated this same profile path while this adapter stayed
        # alive. The incarnation check replaces, rather than reuses, old RAM.
        state = _shared_location_state(
            path,
            subject_prefix,
            owner_home,
            self._background_location_writer_control_path,
        )
        self._background_location_states[key] = state
        if key == str(self._background_location_state_path.absolute()):
            # Multiplex assigns ``_owner_profile`` after adapter construction.
            # The initial same-path state therefore lacked an incarnation and
            # may just have been retired by the registry; keep the canonical
            # pointer (used by reconnect invalidation) on the live replacement.
            self._background_location_state = state
        return state

    def _discover_background_location_profile_paths(self) -> list[Path]:
        """Filesystem half of profile-state discovery; always called off-loop."""
        root = self._background_location_default_root
        relative = (
            Path("state")
            / "telegram_background_locations"
            / f"{self._background_location_bot_scope}.json"
        )
        candidate_paths = [root / relative]
        profiles_root = root / "profiles"
        if profiles_root.is_dir():
            candidate_paths.extend(
                child / relative for child in profiles_root.iterdir() if child.is_dir()
            )
        return [path for path in candidate_paths if path.is_file()]

    async def _background_location_candidate_states(
        self, target: _SharedLocationState
    ) -> list[_SharedLocationState]:
        """Known/existing profile-local files that may own one Telegram lifecycle."""
        states = {
            key: state
            for key, state in self._background_location_states.items()
            if state.is_current()
        }
        root = self._background_location_default_root.absolute()
        # Replacement adapters must see staged/in-flight state even before its
        # profile-local file exists. The registry lookup itself performs no I/O.
        with _SHARED_STATE_LOCK:
            for key, state in _SHARED_STATE_BY_PATH.items():
                if (
                    state.path.name == f"{self._background_location_bot_scope}.json"
                    and state.path.absolute().is_relative_to(root)
                    and state.is_current()
                ):
                    states[key] = state
        if not self._background_location_profile_scan_complete:
            try:
                candidate_paths = await asyncio.to_thread(
                    self._discover_background_location_profile_paths
                )
            except OSError:
                logger.warning(
                    "[Telegram] Could not enumerate profile-local background location state",
                    exc_info=True,
                )
                candidate_paths = []
            else:
                self._background_location_profile_scan_complete = True
            for path in candidate_paths:
                key = str(path.absolute())
                if key not in states:
                    from hermes_constants import named_profile_home

                    subject_prefix = f"bot:{self._background_location_bot_scope}:chat:"
                    state = _shared_location_state(
                        path,
                        subject_prefix,
                        named_profile_home(path),
                        self._background_location_writer_control_path,
                    )
                    self._background_location_states[key] = state
                    states[key] = state
        target_key = str(target.path.absolute())
        # Never let a retired pre-owner state shadow the incarnation-aware
        # same-path state collected from the shared registry above.
        if target.is_current() or target_key not in states:
            states[target_key] = target
        return [states[key] for key in sorted(states)]

    def _publish_background_location_writer_epoch(self, epoch: str) -> None:
        """Make a durable bot-wide epoch current for every local profile state."""
        control_path = self._background_location_writer_control_path
        with _SHARED_STATE_LOCK:
            shared_states = [
                state
                for state in _SHARED_STATE_BY_PATH.values()
                if state.writer_control_path == control_path and state.is_current()
            ]
        for state in shared_states:
            state.writer_epoch = epoch
        for state in self._background_location_states.values():
            if state.is_current():
                state.writer_epoch = epoch

    async def _claim_background_location_writer_epoch(
        self,
    ) -> tuple[str, Optional[str]]:
        epoch, previous_epoch = await asyncio.to_thread(
            _claim_location_writer_epoch,
            self._background_location_writer_control_path,
        )
        self._publish_background_location_writer_epoch(epoch)
        return epoch, previous_epoch

    async def _sync_background_location_writer_epoch(self) -> Optional[str]:
        epoch = await asyncio.to_thread(
            _get_or_create_location_writer_epoch,
            self._background_location_writer_control_path,
            self._background_location_state.writer_epoch,
        )
        if epoch is not None:
            self._publish_background_location_writer_epoch(epoch)
        return epoch

    def _background_location_records_epoch_is_current(
        self, state: _SharedLocationState
    ) -> bool:
        """Verify that an in-memory snapshot belongs to the durable bot epoch."""
        if state.writer_control_path is None:
            return True
        try:
            durable_epoch = _read_location_writer_epoch(
                state.writer_control_path
            )
        except OSError:
            return False
        return (
            durable_epoch is not None
            and state.records_writer_epoch == durable_epoch
        )

    @staticmethod
    def _resolve_background_location_bot_scope(token: Any) -> str:
        """Return a non-secret stable identifier for the configured bot."""
        raw_token = str(token or "")
        bot_id, separator, _secret = raw_token.partition(":")
        if separator and bot_id.isdigit():
            return bot_id
        return f"token-{hashlib.sha256(raw_token.encode('utf-8')).hexdigest()[:16]}"

    def _background_location_source_for_message(self, message: Any) -> Optional[Any]:
        """Build and route the source that owns a sensitive location record."""
        # SessionSource does not yet carry Telegram's business-account
        # connection identity. Without that namespace, the same bot/chat/user
        # tuple could read coordinates retained for a different connected
        # business account. Fail closed until routing supports it end to end.
        if getattr(message, "business_connection_id", None) is not None:
            return None
        if getattr(getattr(message, "sender_chat", None), "id", None) is not None:
            return None
        source = self._source_from_message_for_auth(message)
        runner = getattr(self, "gateway_runner", None)
        resolver = getattr(runner, "_profile_name_for_source", None)
        if callable(resolver) and not getattr(source, "profile", None):
            try:
                source.profile = resolver(source)
            except Exception:
                # A rejected or broken profile route must never fall back to a
                # different profile's retained coordinates.
                logger.warning(
                    "[Telegram] Could not resolve the background location profile; "
                    "dropping the update",
                    exc_info=True,
                )
                return None
        if not getattr(source, "profile", None):
            owner_profile = getattr(self, "_owner_profile", None)
            if isinstance(owner_profile, str) and owner_profile.strip():
                source.profile = owner_profile.strip()
        return source

    def _background_location_subject_key(self, message: Any) -> Optional[str]:
        """Return a bot/profile/chat/sender key, failing closed for shared personas."""
        source = self._background_location_source_for_message(message)
        return (
            self._background_location_subject_key_from_source(source)
            if source is not None
            else None
        )

    def _background_location_subject_key_from_source(
        self, source: Any
    ) -> Optional[str]:
        user_id = getattr(source, "user_id", None)
        chat_id = getattr(source, "chat_id", None)
        if user_id is None or chat_id is None:
            return None
        prefix = f"bot:{self._background_location_bot_scope}"
        key = f"{prefix}:chat:{chat_id}:user:{user_id}"
        chat_type = str(getattr(source, "chat_type", "") or "").lower()
        thread_id = getattr(source, "thread_id", None)
        if chat_type not in {"private", "dm"} and thread_id is not None:
            key += f":thread:{thread_id}"
        return key

    @staticmethod
    def _background_location_lifecycle_identity(
        message: Any,
    ) -> Optional[tuple[str, str, str]]:
        """Stable Telegram live-share identity, independent of profile routing."""
        chat_id = str(getattr(getattr(message, "chat", None), "id", "") or "")
        user_id = str(getattr(getattr(message, "from_user", None), "id", "") or "")
        message_id = str(getattr(message, "message_id", "") or "")
        if not chat_id or not user_id or not message_id:
            return None
        return chat_id, user_id, message_id

    @staticmethod
    def _background_location_record_lifecycle_identity(
        record: Any,
    ) -> Optional[tuple[str, str, str]]:
        if not isinstance(record, dict):
            return None
        chat_id = str(record.get("chat_id", "") or "")
        user_id = str(record.get("user_id", "") or "")
        message_id = str(record.get("message_id", "") or "")
        if not chat_id or not user_id or not message_id:
            return None
        return chat_id, user_id, message_id

    def _add_background_location_stop_fence(
        self, identity: tuple[str, str, str]
    ) -> None:
        """Hide a lifecycle synchronously while its stop is being persisted."""
        scope = (
            f"{self._background_location_default_root.absolute()}\0"
            f"{self._background_location_bot_scope}"
        )
        with _STOP_FENCE_LOCK:
            _STOP_FENCES_BY_BOT.setdefault(scope, set()).add(identity)

    def _remove_background_location_stop_fence(
        self, identity: tuple[str, str, str]
    ) -> None:
        scope = (
            f"{self._background_location_default_root.absolute()}\0"
            f"{self._background_location_bot_scope}"
        )
        with _STOP_FENCE_LOCK:
            fences = _STOP_FENCES_BY_BOT.get(scope)
            if fences is None:
                return
            fences.discard(identity)
            if not fences:
                _STOP_FENCES_BY_BOT.pop(scope, None)

    def _background_location_lifecycle_is_stop_fenced(self, record: Any) -> bool:
        identity = self._background_location_record_lifecycle_identity(record)
        if identity is None:
            return False
        scope = (
            f"{self._background_location_default_root.absolute()}\0"
            f"{self._background_location_bot_scope}"
        )
        with _STOP_FENCE_LOCK:
            return identity in _STOP_FENCES_BY_BOT.get(scope, set())

    @staticmethod
    def _background_location_record_matches_lifecycle(
        record: Any, identity: Optional[tuple[str, str, str]]
    ) -> bool:
        if not isinstance(record, dict) or identity is None:
            return False
        chat_id, user_id, message_id = identity
        return (
            str(record.get("chat_id", "")) == chat_id
            and str(record.get("user_id", "")) == user_id
            and str(record.get("message_id", "")) == message_id
        )

    @staticmethod
    def _background_location_record_matches_subject_key(
        key: str, record: Any, expected_prefix: str
    ) -> bool:
        """Verify that an on-disk record belongs to the key that contains it.

        The JSON file is intentionally user-editable and therefore untrusted.
        Lookup by an event-derived key is safe only when the record's embedded
        sender/chat/topic identity agrees with that key. Older records did not
        include ``subject_key`` or ``chat_type``; their Telegram numeric IDs are
        still sufficient to reconstruct the only safe candidate key.
        """
        identity = _location_record_identity(record)
        if identity is None:
            return False
        chat_id, user_id, _message_id = identity
        stored_subject_key = record.get("subject_key")
        if stored_subject_key is not None and str(stored_subject_key) != key:
            return False

        if expected_prefix.endswith(":chat:"):
            base_key = f"{expected_prefix}{chat_id}:user:{user_id}"
        else:
            base_key = (
                f"{expected_prefix.rstrip(':')}:chat:{chat_id}:user:{user_id}"
            )
        raw_thread_id = record.get("thread_id")
        thread_id = (
            str(raw_thread_id)
            if raw_thread_id is not None and str(raw_thread_id)
            else None
        )
        chat_type = str(record.get("chat_type", "") or "").lower()
        if chat_type in {"private", "dm"}:
            expected_key = base_key
        elif thread_id is not None:
            expected_key = f"{base_key}:thread:{thread_id}"
        else:
            expected_key = base_key
        return key == expected_key

    def _background_location_cache_is_fresh(
        self, state: Optional[_SharedLocationState] = None
    ) -> bool:
        state = state or self._background_location_state
        if not state.is_current():
            return False
        cached = state.records
        cached_at = state.cached_at_monotonic
        return cached is not None and (
            state.dirty
            or cached_at is None
            or time.monotonic() - cached_at
            < self._BACKGROUND_LOCATION_STATE_CACHE_TTL_SECONDS
        )

    def _background_location_receive_is_degraded(self) -> bool:
        """Fail closed when polling cannot currently prove stop continuity."""
        checker = getattr(
            self, "_background_location_receive_path_degraded", None
        )
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return True
        return bool(getattr(self, "_send_path_degraded", False))

    @staticmethod
    def _background_location_timestamp(value: Any) -> Optional[str]:
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _parse_background_location_datetime(value: Any) -> Optional[datetime]:
        if not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _coerce_finite_float(
        value: Any,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> Optional[float]:
        import math

        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number):
            return None
        if minimum is not None and number < minimum:
            return None
        if maximum is not None and number > maximum:
            return None
        return number

    @staticmethod
    def _coerce_nonnegative_int(value: Any) -> Optional[int]:
        import math

        if isinstance(value, bool):
            return None
        if isinstance(value, timedelta):
            seconds = value.total_seconds()
            if not math.isfinite(seconds) or seconds < 0 or not seconds.is_integer():
                return None
            return int(seconds)
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if number >= 0 else None

    def _active_live_location_period(self, location: Any) -> Optional[int]:
        live_period = self._coerce_nonnegative_int(
            getattr(location, "live_period", None)
        )
        return (
            live_period
            if live_period is not None
            and 0 < live_period <= self._BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD
            else None
        )

    @classmethod
    def _background_location_candidate_is_newer(
        cls, existing: Any, candidate: Dict[str, Any]
    ) -> bool:
        if not isinstance(existing, dict):
            return True
        existing_timestamp = cls._parse_background_location_datetime(
            existing.get("telegram_timestamp")
        )
        candidate_timestamp = cls._parse_background_location_datetime(
            candidate.get("telegram_timestamp")
        )
        existing_update_id = cls._coerce_nonnegative_int(existing.get("update_id"))
        candidate_update_id = cls._coerce_nonnegative_int(candidate.get("update_id"))
        if existing_timestamp is not None and candidate_timestamp is not None:
            if candidate_timestamp != existing_timestamp:
                return candidate_timestamp > existing_timestamp
            return candidate_update_id is not None and (
                existing_update_id is None or candidate_update_id > existing_update_id
            )
        if existing_update_id is not None and candidate_update_id is not None:
            return candidate_update_id > existing_update_id
        return True

    def _coordinate_free_lifecycle_marker(
        self, record: Dict[str, Any], source: str
    ) -> Dict[str, Any]:
        marker: Dict[str, Any] = {
            "source": source,
            "recorded_at": str(
                record.get("recorded_at") or datetime.now(timezone.utc).isoformat()
            ),
        }
        for key in (
            "subject_key",
            "chat_type",
            "chat_id",
            "user_id",
            "thread_id",
            "message_id",
            "telegram_timestamp",
            "update_id",
        ):
            value = record.get(key)
            if value is not None and str(value):
                marker[key] = str(value)
        return marker

    def _prune_expired_background_locations(
        self, records: Dict[str, dict]
    ) -> tuple[Dict[str, dict], bool]:
        now = datetime.now(timezone.utc)
        cleaned: Dict[str, dict] = {}
        changed = False
        for key, record in records.items():
            if record.get("source") != "live_location":
                cleaned[key] = record
                continue
            live_period = self._coerce_nonnegative_int(record.get("live_period"))
            if live_period == self._BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD:
                cleaned[key] = record
                continue
            expires_at = self._parse_background_location_datetime(
                record.get("live_expires_at")
            )
            if live_period is None or expires_at is None or now >= expires_at:
                cleaned[key] = self._coordinate_free_lifecycle_marker(
                    record, "live_location_expired"
                )
                changed = True
            else:
                cleaned[key] = record
        return cleaned, changed

    def _schedule_background_location_expiry(
        self, state: _SharedLocationState, records: Dict[str, dict]
    ) -> None:
        """Arm one loop timer for the nearest finite active-share expiry."""
        if state.expiry_timer is not None:
            state.expiry_timer.cancel()
            state.expiry_timer = None
        expiries = [
            parsed
            for record in records.values()
            if isinstance(record, dict)
            and record.get("source") == "live_location"
            and self._coerce_nonnegative_int(record.get("live_period"))
            != self._BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD
            for parsed in [
                self._parse_background_location_datetime(record.get("live_expires_at"))
            ]
            if parsed is not None
        ]
        if not expiries:
            return
        delay = max(
            0.0,
            (min(expiries) - datetime.now(timezone.utc)).total_seconds(),
        )
        loop = asyncio.get_running_loop()

        def expire() -> None:
            state.expiry_timer = None
            loop.create_task(self._expire_background_location_state(state))

        state.expiry_timer = loop.call_later(delay, expire)

    async def _expire_background_location_state(
        self, state: _SharedLocationState
    ) -> None:
        """Replace elapsed finite coordinates with markers even without traffic."""
        result_future: Optional[asyncio.Future] = None
        async with state.mutation_lock:
            records = dict(state.records or {})
            records, changed = self._prune_expired_background_locations(records)
            if changed:
                result_future = self._stage_background_location_records(records, state)
            else:
                self._schedule_background_location_expiry(state, records)
        if result_future is not None:
            await self._await_background_location_write(
                result_future, operation="live-location expiry"
            )

    def _invalidate_cold_start_background_locations(
        self, records: Dict[str, dict]
    ) -> tuple[Dict[str, dict], bool]:
        """Remove coordinates whose stop may have arrived while Hermes was down.

        Telegram's cold polling startup deliberately discards queued updates.
        Consequently, persisted active state cannot prove that an indefinite
        share is still active after a process restart. Adapter replacements in
        the same process share ``_SharedLocationState`` and do not take this
        path.
        """
        cleaned: Dict[str, dict] = {}
        changed = False
        for key, record in records.items():
            if record.get("source") == "live_location":
                cleaned[key] = self._coordinate_free_lifecycle_marker(
                    record, "live_location_restart"
                )
                changed = True
            else:
                cleaned[key] = record
        return cleaned, changed

    def _load_background_location_records(
        self, state: Optional[_SharedLocationState] = None
    ) -> Dict[str, dict]:
        """Load bounded state; production handlers call this through ``to_thread``."""
        state = state or self._background_location_state
        if not state.is_current():
            state.records = {}
            state.persisted_records = {}
            state.write_baseline_records = None
            state.records_writer_epoch = None
            state.cached_at_monotonic = None
            state.dirty = False
            return {}
        control_epoch = (
            _read_location_writer_epoch(state.writer_control_path)
            if state.writer_control_path is not None
            else state.writer_epoch
        )
        if (
            self._background_location_cache_is_fresh(state)
            and state.records_writer_epoch == control_epoch
        ):
            return state.records or {}

        writer = state.writer
        staged = writer.outstanding_snapshot()
        if staged is not None and state.records_writer_epoch == control_epoch:
            state.records = staged
            state.cached_at_monotonic = time.monotonic()
            return staged

        records: Dict[str, dict] = {}
        disk_records: Dict[str, dict] = {}
        rewrite_pruned = False
        payload_epoch: Optional[str] = None
        try:
            with writer.io_lock:
                with state.path.open("rb") as state_file:
                    raw_payload = state_file.read(
                        self._BACKGROUND_LOCATION_MAX_STATE_BYTES + 1
                    )
                if len(raw_payload) > self._BACKGROUND_LOCATION_MAX_STATE_BYTES:
                    raise ValueError("background location state exceeds size limit")
                payload = json.loads(raw_payload)
                if not isinstance(payload, dict) or payload.get("version") != (
                    self._BACKGROUND_LOCATION_STATE_VERSION
                ):
                    raise ValueError("unsupported background location state version")
                raw_payload_epoch = payload.get("writer_epoch")
                if isinstance(raw_payload_epoch, str) and (
                    0 < len(raw_payload_epoch) <= 128
                ):
                    payload_epoch = raw_payload_epoch
                if state.owner_incarnation is not None and payload.get(
                    "owner_incarnation"
                ) != list(state.owner_incarnation):
                    raise ValueError(
                        "background location state belongs to another profile incarnation"
                    )
                raw_records = payload.get("locations", {})
                if isinstance(raw_records, dict):
                    expected_prefix = state.subject_prefix or (
                        f"bot:{self._background_location_bot_scope}:"
                    )
                    valid_records = []
                    for key, value in raw_records.items():
                        normalized_key = str(key)
                        if not (
                            normalized_key.startswith(expected_prefix)
                            and isinstance(value, dict)
                            and value.get("source")
                            in {
                                "live_location",
                                "live_location_stop",
                                "live_location_expired",
                                "live_location_restart",
                                "live_location_persist_failed",
                            }
                            and self._background_location_record_matches_subject_key(
                                normalized_key, value, expected_prefix
                            )
                        ):
                            rewrite_pruned = True
                            continue
                        valid_records.append((normalized_key, dict(value)))
                    valid_records.sort(
                        key=lambda item: str(item[1].get("recorded_at", "")),
                        reverse=True,
                    )
                    records = dict(
                        valid_records[: self._BACKGROUND_LOCATION_MAX_SUBJECTS]
                    )
                    disk_records = {
                        str(key): dict(value) for key, value in records.items()
                    }
                    records, expiry_pruned = self._prune_expired_background_locations(
                        records
                    )
                    rewrite_pruned = rewrite_pruned or expiry_pruned
                final_control_epoch = (
                    _read_location_writer_epoch(state.writer_control_path)
                    if state.writer_control_path is not None
                    else state.writer_epoch
                )
                epoch_is_current = (
                    state.writer_control_path is None
                    or (
                        control_epoch is not None
                        and control_epoch == final_control_epoch == payload_epoch
                    )
                )
                if not epoch_is_current:
                    # A stop/reconnect rotated the bot-wide fence before this
                    # snapshot was re-signed. Never surface its coordinates;
                    # a later refresh can load the replacement snapshot.
                    records, _ = self._invalidate_cold_start_background_locations(
                        records
                    )
                if rewrite_pruned and (
                    epoch_is_current
                    or all(
                        record.get("source") != "live_location"
                        for record in records.values()
                    )
                ):
                    committed = _write_snapshot_if_current(
                        state.path,
                        records,
                        state.is_current,
                        state.owner_incarnation,
                        state.owner_home,
                        expected_writer_epoch=state.writer_epoch,
                        writer_control_path=state.writer_control_path,
                    )
                    if isinstance(committed, tuple) and len(committed) == 2:
                        records, payload_epoch = committed
                        disk_records = {
                            str(key): dict(value)
                            for key, value in records.items()
                        }
        except FileNotFoundError:
            payload_epoch = control_epoch
        except (OSError, ValueError, TypeError, RecursionError):
            logger.warning(
                "[Telegram] Could not read background location state at %s; "
                "starting with an empty cache",
                state.path,
                exc_info=True,
            )
            records = {}
        state.records = records
        state.records_writer_epoch = payload_epoch
        state.persisted_records = {
            str(key): dict(value) for key, value in disk_records.items()
        }
        state.write_baseline_records = None
        state.cached_at_monotonic = time.monotonic()
        state.dirty = False
        return records

    async def _refresh_background_location_records(
        self, state: Optional[_SharedLocationState] = None
    ) -> Dict[str, dict]:
        """Refresh stale state off-loop, single-flight across replacement adapters."""
        state = state or self._background_location_state
        if not state.is_current():
            state.records = {}
            state.persisted_records = {}
            state.write_baseline_records = None
            state.records_writer_epoch = None
            state.cached_at_monotonic = None
            state.dirty = False
            return {}
        result_future: Optional[asyncio.Future] = None
        async with state.mutation_lock:
            if state.dirty and state.records is not None:
                records = state.records or {}
                if (
                    state.writer.outstanding_snapshot() is None
                    and time.monotonic() >= state.retry_not_before_monotonic
                ):
                    result_future = self._stage_background_location_records(
                        records, state
                    )
            elif self._background_location_cache_is_fresh(state):
                records = state.records or {}
            else:
                if state is self._background_location_state:
                    records = await asyncio.to_thread(
                        self._load_background_location_records
                    )
                else:
                    records = await asyncio.to_thread(
                        self._load_background_location_records, state
                    )
            if state.cold_start_pending:
                state.cold_start_pending = False
                records, changed = self._invalidate_cold_start_background_locations(
                    dict(records)
                )
                if changed:
                    result_future = self._stage_background_location_records(
                        records, state
                    )
            if result_future is None:
                self._schedule_background_location_expiry(state, records)
        if result_future is not None:
            await self._await_background_location_write(
                result_future, operation="cold-start invalidation"
            )
        return records

    async def _prepare_background_locations_for_connect(self) -> Dict[str, dict]:
        """Invalidate active coordinates whenever Telegram continuity is re-established.

        A stop can fall out of Telegram's retained update window during a long
        outage even when this Python process stays alive. Requiring a fresh
        live edit after every connect is the only fail-closed proof that the
        share is still active.
        """
        # ``connect()`` calls this only after taking the platform's exclusive
        # polling lock. Claim ownership synchronously and durably before any
        # profile invalidation; a timeout or coalesced background write here
        # could let an older process publish coordinates after polling starts.
        epoch, _previous_epoch = (
            await self._claim_background_location_writer_epoch()
        )
        states = await self._background_location_candidate_states(
            self._background_location_state
        )
        default_records: Dict[str, dict] = {}
        for state in states:
            async with state.mutation_lock:
                state.writer_epoch = epoch
                if self._background_location_cache_is_fresh(state):
                    records = state.records or {}
                else:
                    if state is self._background_location_state:
                        records = await asyncio.to_thread(
                            self._load_background_location_records
                        )
                    else:
                        records = await asyncio.to_thread(
                            self._load_background_location_records, state
                        )
                state.cold_start_pending = False
                records, _changed = self._invalidate_cold_start_background_locations(
                    dict(records)
                )
                if state is self._background_location_state:
                    default_records = records
                persisted = await asyncio.to_thread(
                    self._write_background_location_records,
                    records,
                    state,
                )
                if not persisted:
                    raise OSError(
                        "could not durably invalidate Telegram background "
                        f"locations at {state.path}"
                    )
                if state is self._background_location_state:
                    default_records = state.records or {}
        return default_records

    @staticmethod
    def _is_background_location_edited_update(update: Any) -> bool:
        return bool(
            getattr(update, "edited_message", None)
            or getattr(update, "edited_channel_post", None)
            or getattr(update, "edited_business_message", None)
        )

    @staticmethod
    def _is_background_location_business_update(update: Any, message: Any) -> bool:
        return bool(
            getattr(message, "business_connection_id", None) is not None
            or getattr(update, "business_message", None)
            or getattr(update, "edited_business_message", None)
        )

    def _is_background_live_location_update(self, update: Any, message: Any) -> bool:
        """Classify the live lifecycle without consulting fallible local state.

        Telegram only emits edited location messages for the live-location
        lifecycle. Treat an edited non-venue location without ``live_period``
        as a privacy-sensitive stop even if local state is missing or corrupt.
        """
        if getattr(message, "venue", None) is not None:
            return False
        location = getattr(message, "location", None)
        if location is None:
            return False
        if self._active_live_location_period(location) is not None:
            return True
        return self._is_background_location_edited_update(update)

    def _updated_background_location_records(
        self,
        update: Any,
        message: Any,
        current_records: Dict[str, dict],
        *,
        subject_key: Optional[str] = None,
        record_subject_key: Optional[str] = None,
        enforce_subject_cap: bool = True,
    ) -> tuple[bool, bool, Dict[str, dict]]:
        """Return ``(accepted, changed, records)`` for one lifecycle update."""
        if getattr(message, "venue", None) is not None:
            return False, False, current_records
        location = getattr(message, "location", None)
        if location is None:
            return False, False, current_records

        lifecycle_identity = self._background_location_lifecycle_identity(message)
        if lifecycle_identity is None:
            logger.warning(
                "[Telegram] Ignoring background location without a complete "
                "chat/user/message lifecycle identity"
            )
            return False, False, current_records
        records = {key: dict(value) for key, value in current_records.items()}
        live_period = self._active_live_location_period(location)
        edited = self._is_background_location_edited_update(update)
        telegram_timestamp = self._background_location_timestamp(
            getattr(message, "edit_date", None) or getattr(message, "date", None)
        )
        update_id = self._coerce_nonnegative_int(getattr(update, "update_id", None))

        if edited and live_period is None:
            message_id = str(getattr(message, "message_id", ""))
            matching_records = [
                (key, record)
                for key, record in records.items()
                if record.get("source") == "live_location"
                and self._background_location_record_matches_lifecycle(
                    record, lifecycle_identity
                )
            ]
            if not message_id:
                return True, False, records
            if not matching_records:
                # The bot-wide epoch was already rotated before any await. It
                # fences an unflushed writer without consuming this subject's
                # slot or overwriting a newer live share under the same key.
                return True, False, records
            changed = False
            for matching_key, matching_record in matching_records:
                marker = self._coordinate_free_lifecycle_marker(
                    matching_record, "live_location_stop"
                )
                marker["recorded_at"] = datetime.now(timezone.utc).isoformat()
                if telegram_timestamp:
                    marker["telegram_timestamp"] = telegram_timestamp
                if update_id is not None:
                    marker["update_id"] = str(update_id)
                if not self._background_location_candidate_is_newer(
                    matching_record, marker
                ):
                    continue
                records[matching_key] = marker
                changed = True
            return True, changed, records

        subject_key = subject_key or self._background_location_subject_key(message)
        if subject_key is None:
            return False, False, current_records
        existing_record = records.get(subject_key)
        if live_period is None:
            return False, False, current_records

        latitude = self._coerce_finite_float(
            getattr(location, "latitude", None), minimum=-90.0, maximum=90.0
        )
        longitude = self._coerce_finite_float(
            getattr(location, "longitude", None), minimum=-180.0, maximum=180.0
        )
        if latitude is None or longitude is None:
            logger.warning(
                "[Telegram] Ignoring background location with invalid coordinates"
            )
            return False, False, current_records

        now = datetime.now(timezone.utc)
        record: Dict[str, Any] = {
            # Multi-profile mutation temporarily namespaces dictionary keys
            # with an in-memory state id.  Persist the real lookup key, never
            # that aggregate bookkeeping key.
            "subject_key": record_subject_key or subject_key,
            "latitude": latitude,
            "longitude": longitude,
            "recorded_at": now.isoformat(),
            "source": "live_location",
            "is_edited_update": edited,
            "chat_id": str(getattr(getattr(message, "chat", None), "id", "")),
            "chat_type": str(
                getattr(getattr(message, "chat", None), "type", "") or ""
            ).lower(),
            "message_id": str(getattr(message, "message_id", "")),
            "live_period": live_period,
        }
        if update_id is not None:
            record["update_id"] = str(update_id)
        user_id = getattr(getattr(message, "from_user", None), "id", None)
        if user_id is not None:
            record["user_id"] = str(user_id)
        thread_id = self._effective_message_thread_id(message)
        if thread_id is not None:
            record["thread_id"] = str(thread_id)
        if telegram_timestamp:
            record["telegram_timestamp"] = telegram_timestamp
        for key in ("horizontal_accuracy", "heading", "proximity_alert_radius"):
            number = self._coerce_finite_float(
                getattr(location, key, None), minimum=0.0
            )
            if number is not None:
                record[key] = number

        live_started_at = (
            self._parse_background_location_datetime(
                self._background_location_timestamp(getattr(message, "date", None))
            )
            or now
        )
        record["live_started_at"] = live_started_at.isoformat()
        if live_period != self._BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD:
            record["live_expires_at"] = (
                live_started_at + timedelta(seconds=live_period)
            ).isoformat()

        lifecycle_records = [
            (key, candidate)
            for key, candidate in records.items()
            if self._background_location_record_matches_lifecycle(
                candidate, lifecycle_identity
            )
        ]
        if any(
            candidate.get("source")
            in {"live_location_stop", "live_location_expired"}
            for _key, candidate in lifecycle_records
        ):
            # A Telegram live-share message cannot resume after a terminal stop
            # or expiry. Reject a reordered late edit rather than resurrecting
            # coordinates for that lifecycle.
            return True, False, records
        comparison_records = [candidate for _key, candidate in lifecycle_records]
        if existing_record is not None and all(
            candidate is not existing_record for candidate in comparison_records
        ):
            comparison_records.append(existing_record)
        if any(
            not self._background_location_candidate_is_newer(candidate, record)
            for candidate in comparison_records
        ):
            logger.debug(
                "[Telegram] Ignoring stale or duplicate background location update"
            )
            return True, False, records
        if (
            live_period != self._BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD
            and datetime.now(timezone.utc)
            >= live_started_at + timedelta(seconds=live_period)
        ):
            record = self._coordinate_free_lifecycle_marker(
                record, "live_location_expired"
            )
        # A profile route may change while Telegram keeps editing the same live
        # message. Remove the prior profile's copy before publishing the new
        # owner so coordinates can never remain reachable in both islands.
        for matching_key, _candidate in lifecycle_records:
            if matching_key != subject_key:
                records.pop(matching_key, None)
        records[subject_key] = record
        if (
            enforce_subject_cap
            and len(records) > self._BACKGROUND_LOCATION_MAX_SUBJECTS
        ):
            records = dict(
                sorted(
                    records.items(),
                    key=lambda item: str(item[1].get("recorded_at", "")),
                    reverse=True,
                )[: self._BACKGROUND_LOCATION_MAX_SUBJECTS]
            )
        return True, True, records

    def _write_background_location_records(
        self,
        records: Dict[str, dict],
        state: Optional[_SharedLocationState] = None,
    ) -> bool:
        """Synchronously write state (used by direct/unit callers)."""
        state = state or self._background_location_state
        if not state.is_current():
            return False
        baseline_source = (
            state.write_baseline_records
            if state.write_baseline_records is not None
            else state.persisted_records or {}
        )
        baseline_records = {
            str(key): dict(value)
            for key, value in baseline_source.items()
            if isinstance(value, dict)
        }
        try:
            with state.writer.io_lock:
                committed = _write_snapshot_if_current(
                    state.path,
                    records,
                    state.is_current,
                    state.owner_incarnation,
                    state.owner_home,
                    baseline_records,
                    state.writer_epoch,
                    state.writer_control_path,
                )
        except OSError:
            logger.warning(
                "[Telegram] Could not persist background location state at %s",
                state.path,
                exc_info=True,
            )
            return False
        if isinstance(committed, tuple) and len(committed) == 2:
            committed_records, committed_epoch = committed
        elif isinstance(committed, dict):
            committed_records = committed
            committed_epoch = state.writer_epoch
        else:
            committed_records = records
            committed_epoch = state.writer_epoch
        state.records = committed_records
        state.records_writer_epoch = committed_epoch
        state.persisted_records = {
            str(key): dict(value) for key, value in committed_records.items()
        }
        state.write_baseline_records = None
        state.cached_at_monotonic = time.monotonic()
        state.dirty = False
        return True

    def _record_background_location(self, update: Any, message: Any) -> bool:
        """Synchronous compatibility path; async intake uses staged writes below."""
        subject_key = self._background_location_subject_key(message)
        if subject_key is None:
            return False
        source = self._background_location_source_for_message(message)
        if source is None:
            return False
        state = self._background_location_state_for_source(source)
        if state is None:
            return False
        accepted, changed, records = self._updated_background_location_records(
            update,
            message,
            dict(self._load_background_location_records(state)),
            subject_key=subject_key,
        )
        if not accepted:
            return False
        if not changed:
            return True
        return self._write_background_location_records(records, state)

    def _stage_background_location_records(
        self,
        records: Dict[str, dict],
        state: Optional[_SharedLocationState] = None,
    ) -> asyncio.Future:
        """Publish a safe RAM snapshot and enqueue its complete disk replacement."""
        state = state or self._background_location_state
        if not state.is_current():
            loop = asyncio.get_running_loop()
            result = loop.create_future()
            result.set_result(_SnapshotWriteResult(False))
            return result
        baseline_source = (
            state.write_baseline_records
            if state.write_baseline_records is not None
            else state.persisted_records or {}
        )
        baseline_records = {
            str(key): dict(value)
            for key, value in baseline_source.items()
            if isinstance(value, dict)
        }
        if state.write_baseline_records is None:
            state.write_baseline_records = {
                str(key): dict(value) for key, value in baseline_records.items()
            }
        state.records = records
        state.records_writer_epoch = state.writer_epoch
        state.cached_at_monotonic = time.monotonic()
        state.dirty = True
        self._schedule_background_location_expiry(state, records)
        state.generation += 1
        generation = state.generation
        loop = asyncio.get_running_loop()
        result_future, _worker = state.writer.submit(
            records,
            baseline_records,
            loop,
            state.writer_epoch,
        )

        def _mark_persisted(future: asyncio.Future) -> None:
            if generation != state.generation or future.cancelled():
                return
            try:
                write_result = future.result()
                succeeded = (
                    isinstance(write_result, _SnapshotWriteResult)
                    and write_result.succeeded
                )
            except Exception:
                write_result = None
                succeeded = False
            state.dirty = not succeeded
            if succeeded:
                if write_result.records is not None:
                    state.records = {
                        str(key): dict(value)
                        for key, value in write_result.records.items()
                    }
                    state.persisted_records = {
                        str(key): dict(value)
                        for key, value in write_result.records.items()
                    }
                    self._schedule_background_location_expiry(
                        state, state.records
                    )
                state.records_writer_epoch = write_result.writer_epoch
                state.write_baseline_records = None
                state.cached_at_monotonic = time.monotonic()
                state.retry_not_before_monotonic = 0.0
                return

            state.retry_not_before_monotonic = time.monotonic() + 1.0
            # A location that could not be durably committed must not remain
            # available to later model turns. Keep a coordinate-free snapshot
            # authoritative in RAM and retry that safe snapshot on subsequent
            # traffic; never reload older on-disk coordinates while dirty.
            current = state.records or {}
            safe_records = {
                key: (
                    self._coordinate_free_lifecycle_marker(
                        record, "live_location_persist_failed"
                    )
                    if record.get("source") == "live_location"
                    else record
                )
                for key, record in current.items()
            }
            if safe_records != current:
                state.records = safe_records
                state.cached_at_monotonic = time.monotonic()
                state.generation += 1

        result_future.add_done_callback(_mark_persisted)
        return result_future

    async def _await_background_location_write(
        self, result_future: asyncio.Future, *, operation: str = "state update"
    ) -> bool:
        try:
            write_result = await asyncio.wait_for(
                asyncio.shield(result_future),
                timeout=self._BACKGROUND_LOCATION_WRITE_TIMEOUT_SECONDS,
            )
            return (
                isinstance(write_result, _SnapshotWriteResult)
                and write_result.succeeded
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[Telegram] Background location %s exceeded %.1fs; "
                "the newest snapshot remains queued",
                operation,
                self._BACKGROUND_LOCATION_WRITE_TIMEOUT_SECONDS,
            )
            return False

    async def _persist_background_location(
        self,
        update: Any,
        message: Any,
        *,
        polling_admission: Optional[tuple[int, bool]] = None,
    ) -> bool:
        """Update every profile-local copy of one lifecycle, then persist off-loop."""
        if polling_admission is None:
            admitted_polling_generation = getattr(self, "_polling_generation", 0)
            # Once polling exists, every real Bot API location update is
            # stamped by the raw response observer. Missing provenance means
            # the bounded admission cache evicted it, instrumentation was
            # bypassed, or the update came from an unverified source: fail
            # closed instead of inferring freshness from the current barrier.
            admitted_after_recovery = (
                getattr(self, "_polling_generation_started_monotonic", None) is None
            )
        else:
            admitted_polling_generation, admitted_after_recovery = polling_admission
        is_stop = (
            self._is_background_location_edited_update(update)
            and self._active_live_location_period(getattr(message, "location", None))
            is None
        )
        lifecycle_identity = self._background_location_lifecycle_identity(message)
        stop_fenced = False
        if is_stop and lifecycle_identity is not None:
            # Register before the first await. A queued turn may reach its final
            # dispatch boundary while profile discovery or a stale disk read is
            # blocked; it must observe the stop immediately, not after I/O.
            self._add_background_location_stop_fence(lifecycle_identity)
            stop_fenced = True
            try:
                writer_epoch, previous_writer_epoch = (
                    await self._claim_background_location_writer_epoch()
                )
            except OSError:
                # Without a durable fence another process could still commit
                # coordinates after the stop. Disable reads locally and keep
                # the in-memory lifecycle fence until the process restarts.
                self._background_locations_enabled = False
                logger.error(
                    "[Telegram] Could not durably fence a stopped live location; "
                    "background location context has been disabled",
                    exc_info=True,
                )
                return False
        elif not is_stop:
            previous_writer_epoch = None
            try:
                writer_epoch = await self._sync_background_location_writer_epoch()
            except OSError:
                writer_epoch = None
            if writer_epoch is None:
                self._background_locations_enabled = False
                logger.error(
                    "[Telegram] Could not establish the background location "
                    "writer fence; background location context has been disabled"
                )
                return False
        else:
            # An edited location without a complete Telegram lifecycle identity
            # cannot safely revoke or create retained state.
            return False
        subject_key: Optional[str] = None
        target = self._background_location_state
        if not is_stop:
            subject_key = self._background_location_subject_key(message)
            if subject_key is None:
                return False
            source = self._background_location_source_for_message(message)
            if source is None:
                return False
            target = self._background_location_state_for_source(source)
            if target is None:
                return False
        states = await self._background_location_candidate_states(target)

        # First access to any profile after process start invalidates records
        # whose Telegram stop may have been discarded while Hermes was down.
        if not is_stop:
            for state in states:
                await self._refresh_background_location_records(state)

        acquired: list[_SharedLocationState] = []
        result_futures: list[asyncio.Future] = []
        accepted = False
        try:
            for state in states:
                await state.mutation_lock.acquire()
                acquired.append(state)
                state.writer_epoch = writer_epoch

            if not is_stop and (
                admitted_polling_generation
                != getattr(self, "_polling_generation", 0)
                or not admitted_after_recovery
                or self._background_location_receive_is_degraded()
                or getattr(self, "_polling_teardown_started", False)
                or getattr(self, "_teardown_started", False)
            ):
                # This handler belongs to a receive generation whose
                # continuity was lost while it awaited discovery/locks. A
                # missed stop is now possible; only an edit delivered by the
                # newly healthy generation may reactivate coordinates.
                return False

            current_by_state: dict[_SharedLocationState, Dict[str, dict]] = {}
            baseline_by_state: dict[_SharedLocationState, Dict[str, dict]] = {}
            cold_start_states: set[_SharedLocationState] = set()
            for state in states:
                if is_stop and (
                    state.records is None
                    or state.records_writer_epoch != previous_writer_epoch
                ):
                    if state is self._background_location_state:
                        loaded = await asyncio.to_thread(
                            self._load_background_location_records
                        )
                    else:
                        loaded = await asyncio.to_thread(
                            self._load_background_location_records, state
                        )
                else:
                    loaded = state.records or {}
                baseline = {key: dict(value) for key, value in loaded.items()}
                if is_stop and state.cold_start_pending:
                    state.cold_start_pending = False
                    cold_start_states.add(state)
                baseline_by_state[state] = baseline
                current_by_state[state] = baseline
            state_ids = {state: str(index) for index, state in enumerate(states)}
            owner_by_key: dict[str, _SharedLocationState] = {}
            persistent_key_by_key: dict[str, str] = {}
            aggregate: Dict[str, dict] = {}
            for state, state_records in current_by_state.items():
                for key, record in state_records.items():
                    aggregate_key = f"{state_ids[state]}\x1f{key}"
                    aggregate[aggregate_key] = record
                    owner_by_key[aggregate_key] = state
                    persistent_key_by_key[aggregate_key] = key

            target_key = (
                f"{state_ids[target]}\x1f{subject_key}"
                if subject_key is not None
                else None
            )

            accepted, changed, records = self._updated_background_location_records(
                update,
                message,
                aggregate,
                subject_key=target_key,
                record_subject_key=subject_key,
                enforce_subject_cap=False,
            )
            if not accepted:
                records = aggregate
            if (
                accepted
                and changed
                and target_key is not None
                and subject_key is not None
            ):
                owner_by_key[target_key] = target
                persistent_key_by_key[target_key] = subject_key
            desired_by_state: dict[_SharedLocationState, Dict[str, dict]] = {
                state: {} for state in states
            }
            if accepted:
                for key, record in records.items():
                    owner = owner_by_key.get(key)
                    if owner is not None:
                        desired_by_state[owner][persistent_key_by_key[key]] = record
            else:
                desired_by_state = current_by_state

            # Apply the actual stop first so its update ordering becomes the
            # durable terminal marker. Any other unverifiable cold-start
            # lifecycles in that profile are then reduced to restart markers.
            for state in cold_start_states:
                desired_by_state[state], _changed = (
                    self._invalidate_cold_start_background_locations(
                        dict(desired_by_state[state])
                    )
                )

            for state, desired in desired_by_state.items():
                if len(desired) > self._BACKGROUND_LOCATION_MAX_SUBJECTS:
                    desired_by_state[state] = dict(
                        sorted(
                            desired.items(),
                            key=lambda item: str(item[1].get("recorded_at", "")),
                            reverse=True,
                        )[: self._BACKGROUND_LOCATION_MAX_SUBJECTS]
                    )

            # Publish every coordinate removal before awaiting filesystem I/O.
            for state in states:
                desired = desired_by_state[state]
                stop_needs_resign = is_stop and (
                    bool(desired)
                    or bool(baseline_by_state[state])
                    or state.path.exists()
                )
                if stop_needs_resign or desired != baseline_by_state[state]:
                    result_futures.append(
                        self._stage_background_location_records(desired, state)
                    )
        finally:
            for state in reversed(acquired):
                state.mutation_lock.release()

        if not accepted:
            return False
        persisted = all([
            await self._await_background_location_write(future)
            for future in result_futures
        ])
        if (
            stop_fenced
            and persisted
            and self._background_location_profile_scan_complete
        ):
            assert lifecycle_identity is not None
            self._remove_background_location_stop_fence(lifecycle_identity)
        return persisted

    def _is_background_location_authorized(self, message: Any) -> bool:
        source = self._source_from_message_for_auth(message)
        return (
            self._is_sender_authorized(
                source.user_id,
                source.chat_type,
                source.chat_id,
                is_bot=source.is_bot,
                thread_id=source.thread_id,
            )
            is True
        )

    def _should_accept_background_location(self, message: Any) -> bool:
        if self._is_own_message(message):
            return False
        thread_id = self._effective_message_thread_id(message)
        if thread_id is not None:
            try:
                if int(thread_id) in self._telegram_ignored_threads():
                    return False
            except (TypeError, ValueError):
                return False
        if not self._is_group_chat(message):
            return True
        allowed_topics = self._telegram_allowed_topics()
        if allowed_topics:
            topic_id = (
                str(thread_id)
                if thread_id is not None
                else self._GENERAL_TOPIC_THREAD_ID
            )
            if topic_id not in allowed_topics:
                return False
        allowed_chats = self._telegram_allowed_chats()
        chat_id = str(getattr(getattr(message, "chat", None), "id", ""))
        return not allowed_chats or chat_id in allowed_chats

    def _build_background_location_context_for_subject(
        self, subject_key: Optional[str], records: Dict[str, dict]
    ) -> Optional[str]:
        if not getattr(self, "_background_locations_enabled", False) or not subject_key:
            return None
        record = records.get(subject_key)
        if not isinstance(record, dict) or record.get("source") != "live_location":
            return None
        if self._background_location_lifecycle_is_stop_fenced(record):
            return None
        latitude = self._coerce_finite_float(
            record.get("latitude"), minimum=-90.0, maximum=90.0
        )
        longitude = self._coerce_finite_float(
            record.get("longitude"), minimum=-180.0, maximum=180.0
        )
        if latitude is None or longitude is None:
            return None
        live_period = self._coerce_nonnegative_int(record.get("live_period"))
        if live_period is None:
            return None
        if live_period != self._BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD:
            expires_at = self._parse_background_location_datetime(
                record.get("live_expires_at")
            )
            if expires_at is None or datetime.now(timezone.utc) >= expires_at:
                return None
        recorded_at = self._parse_background_location_datetime(
            record.get("telegram_timestamp")
        ) or self._parse_background_location_datetime(record.get("recorded_at"))
        lines = [
            "[Background Telegram location context]",
            "Source: live_location",
            "This is the latest snapshot of an active live location share, not a fixed one-time pin.",
            "The recorded position may be stale; use it only when relevant to the user's explicit request.",
            "Recorded at (UTC): "
            + (recorded_at.isoformat() if recorded_at is not None else "unknown"),
            f"Latitude: {latitude}",
            f"Longitude: {longitude}",
        ]
        accuracy = self._coerce_finite_float(
            record.get("horizontal_accuracy"), minimum=0.0
        )
        if accuracy is not None:
            lines.append(f"Horizontal accuracy: {accuracy} metres")
        return "\n".join(lines)

    def _build_background_location_context(
        self,
        message: Any,
        records: Optional[Dict[str, dict]] = None,
    ) -> Optional[str]:
        subject_key = self._background_location_subject_key(message)
        if records is None:
            source = self._background_location_source_for_message(message)
            state = self._background_location_state_for_source(source)
            records = (
                self._load_background_location_records(state)
                if state is not None
                else {}
            )
        return self._build_background_location_context_for_subject(
            subject_key,
            records,
        )

    def _set_event_background_location_context(
        self, event: MessageEvent, location_context: Optional[str]
    ) -> None:
        """Replace only this mixin's suffix, preserving other volatile context."""
        base_attr = "_telegram_background_location_base_context"
        if not hasattr(event, base_attr):
            existing = getattr(event, "ephemeral_user_context", None)
            base = existing.strip() if isinstance(existing, str) else ""
            marker = f"\n\n{self._BACKGROUND_LOCATION_CONTEXT_HEADER}"
            if base.startswith(self._BACKGROUND_LOCATION_CONTEXT_HEADER):
                base = ""
            elif marker in base:
                base = base.split(marker, 1)[0].rstrip()
            setattr(event, base_attr, base or None)
        base = getattr(event, base_attr, None)
        if base and location_context:
            event.ephemeral_user_context = f"{base}\n\n{location_context}"
        else:
            event.ephemeral_user_context = base or location_context or None

    def _scrub_queued_background_location_context(
        self, subject_key: str, state_path: Optional[str] = None
    ) -> None:
        """Remove a stopped subject's coordinates from every in-memory queue we own."""
        candidates = []
        for mapping_name in ("_pending_text_batches", "_pending_messages"):
            mapping = getattr(self, mapping_name, None)
            if isinstance(mapping, dict):
                candidates.extend(mapping.values())
        candidates.extend(list(getattr(self, "_held_inbound_events", None) or []))
        debounce_store = getattr(self, "_text_debounce_store", lambda: {})()
        if isinstance(debounce_store, dict):
            candidates.extend(
                getattr(state, "event", None) for state in debounce_store.values()
            )
        runner = getattr(self, "gateway_runner", None)
        queued_by_session = getattr(runner, "_queued_events", None)
        if isinstance(queued_by_session, dict):
            for queued in queued_by_session.values():
                candidates.extend(list(queued or []))

        seen = set()
        for event in candidates:
            if event is None or id(event) in seen:
                continue
            seen.add(id(event))
            event_subject = getattr(
                event, "_telegram_background_location_subject_key", None
            ) or self._background_location_subject_key_from_source(
                getattr(event, "source", None)
            )
            event_state_path = getattr(
                event, "_telegram_background_location_state_path", None
            )
            if event_subject == subject_key and (
                state_path is None or str(event_state_path or "") == state_path
            ):
                self._set_event_background_location_context(event, None)

    def _background_location_stopped_subject_keys(
        self, message: Any
    ) -> list[tuple[str, str]]:
        identity = self._background_location_lifecycle_identity(message)
        return [
            (str(state.path.absolute()), key)
            for state in self._background_location_states.values()
            for key, record in (state.records or {}).items()
            if (
                record.get("source") == "live_location_stop"
                and self._background_location_record_matches_lifecycle(record, identity)
            )
        ]

    async def _attach_background_location_context(
        self, event: MessageEvent, message: Any
    ) -> MessageEvent:
        # Group-observation attribution may intentionally neutralize
        # ``event.source.user_id``. Sensitive identity still comes from the
        # authenticated raw Telegram message; its routed profile selects the
        # profile-local state file.
        source = self._background_location_source_for_message(message)
        subject_key = self._background_location_subject_key_from_source(source)
        state = self._background_location_state_for_source(source)
        event_state = self._background_location_state_for_source(
            getattr(event, "source", source)
        )
        if (
            state is None
            or event_state is None
            or state.path.absolute() != event_state.path.absolute()
        ):
            # Event construction pins the runtime profile before awaited media
            # work. If routing changed meanwhile, never attach coordinates from
            # the newly resolved profile to the already-pinned event.
            setattr(event, "_telegram_background_location_subject_key", None)
            self._set_event_background_location_context(event, None)
            return event
        records = (
            await self._refresh_background_location_records(state)
            if state is not None
            else {}
        )
        epoch_is_current = state is not None and await asyncio.to_thread(
            self._background_location_records_epoch_is_current, state
        )
        location_context = (
            self._build_background_location_context_for_subject(subject_key, records)
            if epoch_is_current
            else None
        )
        if location_context is not None and not await asyncio.to_thread(
            self._background_location_records_epoch_is_current, state
        ):
            location_context = None
        setattr(event, "_telegram_background_location_subject_key", subject_key)
        if state is not None:
            setattr(
                event,
                "_telegram_background_location_bot_scope",
                self._background_location_bot_scope,
            )
            setattr(
                event,
                "_telegram_background_location_state_path",
                str(state.path.absolute()),
            )
            setattr(
                event,
                "_telegram_background_location_state_incarnation",
                state.owner_incarnation,
            )
        self._set_event_background_location_context(event, location_context)
        return event

    async def _refresh_ephemeral_user_context_for_dispatch(
        self, event: MessageEvent
    ) -> None:
        """Resolve queued Telegram location context at the actual dispatch boundary."""
        if not getattr(self, "_background_locations_enabled", False):
            if hasattr(event, "_telegram_background_location_subject_key"):
                self._set_event_background_location_context(event, None)
            return
        # Only text/command intake explicitly opts an event into background
        # location context. Never infer eligibility from a generic Telegram
        # source: media and synthetic internal events may carry the same user
        # identity but are not fresh user-authored requests for this context.
        marker = "_telegram_background_location_subject_key"
        if getattr(event, "internal", False):
            self._set_event_background_location_context(event, None)
            return
        if not hasattr(event, marker):
            return
        if getattr(event, "_ephemeral_context_refresh_unsafe", False):
            self._set_event_background_location_context(event, None)
            return
        if getattr(
            event, "_telegram_background_location_bot_scope", None
        ) != self._background_location_bot_scope:
            # A queued event can outlive a token/config-driven adapter
            # replacement. Never let the current bot resolve the prior bot's
            # globally shared state path.
            self._set_event_background_location_context(event, None)
            return
        if (
            self._background_location_receive_is_degraded()
            or getattr(self, "_polling_teardown_started", False)
            or getattr(self, "_teardown_started", False)
        ):
            # A stop edit may be waiting behind the broken polling generation.
            # Detached and queued turns must not use its last known coordinates.
            self._set_event_background_location_context(event, None)
            return
        subject_key = getattr(event, marker, None)
        if not isinstance(subject_key, str) or not subject_key:
            self._set_event_background_location_context(event, None)
            return
        state_path = getattr(event, "_telegram_background_location_state_path", None)
        state = (
            self._background_location_states.get(str(state_path))
            if state_path
            else None
        )
        if state is None:
            with _SHARED_STATE_LOCK:
                state = (
                    _SHARED_STATE_BY_PATH.get(str(state_path)) if state_path else None
                )
        if state is None:
            state = self._background_location_state_for_source(event.source)
        event_incarnation = getattr(
            event, "_telegram_background_location_state_incarnation", None
        )
        if state is None or state.owner_incarnation != event_incarnation:
            # A queued event from a deleted profile incarnation must never
            # resolve through the same path into a newly-created profile's
            # coordinates.
            self._set_event_background_location_context(event, None)
            return
        records = await self._refresh_background_location_records(state)
        if (
            self._background_location_receive_is_degraded()
            or getattr(self, "_polling_teardown_started", False)
            or getattr(self, "_teardown_started", False)
        ):
            # Polling can fail while the cache refresh awaits disk. A stop may
            # already be queued behind that failed generation, so the health
            # decision must be revalidated at the actual attachment edge.
            self._set_event_background_location_context(event, None)
            return
        self._set_event_background_location_context(
            event,
            self._build_background_location_context_for_subject(subject_key, records),
        )

    def _resolve_ephemeral_user_context_for_dispatch_sync(
        self, event: MessageEvent
    ) -> Optional[str]:
        """Return current volatile context for one provider call.

        The async dispatch refresh warms the state. This final resolver runs in
        the model worker on every tool-loop iteration and observes synchronous
        stop fences, the cross-process writer fence, polling health, expiry,
        and the latest published in-memory snapshot.
        """
        base = getattr(event, "_telegram_background_location_base_context", None)
        if (
            not getattr(self, "_background_locations_enabled", False)
            or getattr(event, "internal", False)
            or getattr(event, "_ephemeral_context_refresh_unsafe", False)
            or getattr(
                event, "_telegram_background_location_bot_scope", None
            )
            != self._background_location_bot_scope
            or self._background_location_receive_is_degraded()
            or getattr(self, "_polling_teardown_started", False)
            or getattr(self, "_teardown_started", False)
        ):
            return base

        subject_key = getattr(
            event, "_telegram_background_location_subject_key", None
        )
        if not isinstance(subject_key, str) or not subject_key:
            return base
        state_path = getattr(
            event, "_telegram_background_location_state_path", None
        )
        state = (
            self._background_location_states.get(str(state_path))
            if state_path
            else None
        )
        if state is None:
            with _SHARED_STATE_LOCK:
                state = (
                    _SHARED_STATE_BY_PATH.get(str(state_path))
                    if state_path
                    else None
                )
        if (
            state is None
            or state.owner_incarnation
            != getattr(
                event, "_telegram_background_location_state_incarnation", None
            )
            or state.records is None
            or not self._background_location_cache_is_fresh(state)
            or not self._background_location_records_epoch_is_current(state)
        ):
            return base
        records = state.records
        # ``_background_location_cache_is_fresh`` may inspect a named-profile
        # incarnation on disk and release the GIL. Polling can be fenced (or the
        # profile recycled) during that check, so revalidate every revocation
        # condition at the actual provider-return edge.
        if (
            not getattr(self, "_background_locations_enabled", False)
            or self._background_location_receive_is_degraded()
            or getattr(self, "_polling_teardown_started", False)
            or getattr(self, "_teardown_started", False)
            or not state.is_current()
            or state.owner_incarnation
            != getattr(
                event, "_telegram_background_location_state_incarnation", None
            )
            or records is not state.records
        ):
            return base
        location_context = self._build_background_location_context_for_subject(
            subject_key, records
        )
        if (
            location_context is not None
            and not self._background_location_records_epoch_is_current(state)
        ):
            return base
        if base and location_context:
            return f"{base}\n\n{location_context}"
        return base or location_context or None
