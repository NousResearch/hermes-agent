from __future__ import annotations

import errno
import json
import logging
import os
import re
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from hashlib import blake2s
from pathlib import Path
from typing import Any, Mapping, Sequence

from hermes_constants import get_hermes_home
from hermes_state import (
    AppendMessagesBatchConflictError,
    CompressionSessionBusyError,
    CompressionSessionClosedError,
    SessionDBBatchMessage,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - Windows-only import path
    import fcntl  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Windows-only fallback
    fcntl = None

try:  # pragma: no cover - POSIX-only import path
    import msvcrt  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - POSIX-only fallback
    msvcrt = None


SPOOL_ROOT_NAME = "session_fallback_spool"
ACTIVE_SPOOL_NAME = "active.spool"
LOCK_FILE_NAME = "append.lock"
REPLAY_OWNER_LOCK_NAME = "replay-owner.lock"
QUARANTINE_DIR_NAME = "quarantine"
SEALED_DIR_NAME = "sealed"
ACKS_DIR_NAME = "acks"
BLOCKERS_DIR_NAME = "blockers"
HIGHWATER_FILE_NAME = "segment-sequence.highwater.json"
HEADER_MAGIC = b"HSPL"
HEADER_SIZE = 32
FRAME_VERSION = 0x01
RECORD_KIND_SESSION_PERSISTENCE_UNIT = 0x01
ROOT_MODE = 0o700
FILE_MODE = 0o600
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
MAX_FRAME_BYTES = MAX_PAYLOAD_BYTES + HEADER_SIZE
TOTAL_CAP_BYTES = 64 * 1024 * 1024
LOCK_TIMEOUT_SECONDS = 5.0
LOCK_RETRY_SECONDS = 0.02
STATUS_LOCK_TIMEOUT_SECONDS = 0.25
STATUS_LOCK_RETRY_SECONDS = 0.01
STATUS_SCAN_ARTIFACT_LIMIT = 4096
SEGMENT_SEQUENCE_WIDTH = 20
MAX_SEGMENT_SEQUENCE = 18446744073709551615
_REPLAY_RETRYABLE_ERRNOS = {
    errno.EAGAIN,
    getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
    getattr(errno, "EBUSY", errno.EAGAIN),
}
_LOCK_CONTENTION_ERRNOS = {
    errno.EACCES,
    errno.EAGAIN,
    getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
}
_CURRENT_QUARANTINE_DIR_FD: int | None = None
_REPLAY_COOLDOWNS: dict[str, dict[str, Any]] = {}
_REPLAY_LOG_STATE: dict[str, dict[str, Any]] = {}
_STATUS_REASON_ORDER = (
    "pending_backlog",
    "blocker",
    "retry_cooldown",
    "ack_pending",
    "capacity_constrained",
    "capacity_full",
    "disk_low",
    "inspection_error",
)


class SpoolTailStatus(str, Enum):
    CLEAN = "clean"
    INCOMPLETE_EOF = "incomplete_eof"
    BAD_MAGIC = "bad_magic"
    BAD_VERSION = "bad_version"
    BAD_RECORD_KIND = "bad_record_kind"
    NONZERO_RESERVED = "nonzero_reserved"
    OVERSIZED_LENGTH = "oversized_length"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"
    SCAN_LIMIT_EXCEEDED = "scan_limit_exceeded"


@dataclass(frozen=True)
class SessionSpoolBootstrap:
    session_id: str | None
    source: str | None
    started_at: float | None
    model: str | None
    model_config: Any
    system_prompt: str | None
    parent_session_id: str | None
    cwd: str | None
    profile_name: str | None
    user_id: str | None
    session_key: str | None
    chat_id: str | None
    chat_type: str | None
    thread_id: str | None


@dataclass(frozen=True)
class SessionSpoolRecord:
    bootstrap: SessionSpoolBootstrap
    persist_attempt_id: str
    persist_attempt_unit_index: int
    canonical_failure: Mapping[str, Any]
    batch_messages: tuple[SessionDBBatchMessage, ...]


@dataclass(frozen=True)
class SpoolFrameReceipt:
    path: str
    offset: int
    frame_length: int
    payload_length: int
    checksum_hex: str


@dataclass(frozen=True)
class SpoolUnitAppendResult:
    persistence_unit_id: str
    message_keys: tuple[str, ...]
    receipt: SpoolFrameReceipt


@dataclass(frozen=True)
class SpoolAppendAttemptResult:
    unit_results: tuple[SpoolUnitAppendResult, ...]


@dataclass(frozen=True)
class SpoolScanResult:
    valid_prefix_bytes: int
    frame_count: int
    tail_status: SpoolTailStatus
    tail_offset: int | None


@dataclass(frozen=True)
class SessionSpoolFrame:
    record: SessionSpoolRecord
    frame_offset: int
    frame_length: int
    payload_length: int
    checksum_hex: str


@dataclass(frozen=True)
class DecodedSegment:
    prefix_frames: tuple[SessionSpoolFrame, ...]
    valid_prefix_bytes: int
    tail_status: SpoolTailStatus
    tail_offset: int | None


class ReplayFrameState(str, Enum):
    REPLAYED = "replayed"
    DUPLICATE = "duplicate"
    RETRY_PENDING = "retry_pending"
    BLOCKED_INTEGRITY = "blocked_integrity"


class ReplayRunState(str, Enum):
    EMPTY = "empty"
    OWNER_BUSY = "owner_busy"
    REPLAYED = "replayed"
    PARTIALLY_REPLAYED = "partially_replayed"
    RETRY_PENDING = "retry_pending"
    BLOCKED_INTEGRITY = "blocked_integrity"
    CORRUPT_QUARANTINED = "corrupt_quarantined"
    NOT_DURABLE = "not_durable"


@dataclass(frozen=True)
class ReplayFrameResult:
    state: ReplayFrameState
    segment_sequence: int
    frame_offset: int
    frame_length: int
    checksum_hex: str
    retry_class: str | None = None
    error_class: str | None = None


@dataclass(frozen=True)
class ReplayRunResult:
    state: ReplayRunState
    trigger: str
    segment_count_seen: int = 0
    frames_decoded: int = 0
    frames_committed: int = 0
    frames_duplicated: int = 0
    frames_acked: int = 0
    bytes_decoded: int = 0
    bytes_acked: int = 0
    pending_bytes_after: int = 0
    pending_frames_after: int | None = None
    first_blocked_segment: int | None = None
    first_blocked_offset: int | None = None
    retry_class: str | None = None
    error_class: str | None = None
    ack_pending: bool = False
    cooldown_seconds: float = 0.0


@dataclass(frozen=True)
class SessionFallbackSpoolStatus:
    schema_version: int
    state: str
    reasons: tuple[str, ...]
    pending_units: int
    pending_frames: int
    pending_bytes: int
    oldest_pending_age_seconds: float | None
    retry_pending: bool
    retry_class: str | None
    cooldown_seconds: float
    ack_pending: bool
    blocker_present: bool
    blocker_sequence: int | None
    blocker_offset: int | None
    blocker_reason_class: str | None
    blocker_source_kind: str | None
    capacity_used_bytes: int
    capacity_cap_bytes: int
    capacity_remaining_bytes: int
    capacity_state: str
    disk_free_bytes: int | None
    disk_total_bytes: int | None
    disk_headroom_threshold_bytes: int
    disk_state: str
    inspection_error_class: str | None


@dataclass(frozen=True)
class _AnchoredRuntime:
    home_path: Path
    root_path: Path
    quarantine_path: Path
    active_path: Path
    home_fd: int
    root_fd: int
    lock_fd: int


@dataclass(frozen=True)
class _BlockerBackedPrefixReplayState:
    segment_sequence: int
    blocking_offset: int
    tail_status: SpoolTailStatus
    prefix_segment_name: str
    prefix_segment_path: Path
    valid_prefix_bytes: int
    acked_prefix_bytes: int


@dataclass(frozen=True)
class _DurableCapacityInventory:
    quarantine_bytes: int
    other_artifact_bytes: int


@dataclass(frozen=True)
class _StatusSegmentSnapshot:
    sequence: int
    name: str
    size_bytes: int
    frame_count: int
    mtime_ns: int
    decoded: DecodedSegment


@dataclass(frozen=True)
class _StatusAckDirectorySnapshot:
    winners_by_segment: Mapping[str, Mapping[str, Any]]
    orphan_tombstone_sequence: int | None = None
    orphan_tombstone_mtime_ns: int | None = None
    blocked_orphan_sequence: int | None = None


@dataclass(frozen=True)
class _StatusBlockerSnapshot:
    present: bool = False
    sequence: int | None = None
    offset: int | None = None
    reason_class: str | None = None
    source_kind: str | None = None
    mtime_ns: int | None = None
    acked_prefix_bytes: int = 0
    valid_prefix_bytes: int = 0
    prefix_segment_name: str | None = None
    zero_prefix: bool = False


@dataclass(frozen=True)
class _PendingBacklogSnapshot:
    pending_bytes: int
    pending_frames: int
    ack_pending: bool = False
    first_blocked_segment: int | None = None
    first_blocked_offset: int | None = None


class SessionFallbackSpoolError(RuntimeError):
    pass


class _StatusInspectionError(SessionFallbackSpoolError):
    def __init__(self, error_class: str):
        super().__init__(error_class)
        self.error_class = error_class


class SpoolPathSecurityError(SessionFallbackSpoolError):
    pass


class SpoolLockTimeoutError(SessionFallbackSpoolError):
    pass


class SpoolFrameTooLargeError(SessionFallbackSpoolError):
    def __init__(self, payload_bytes: int, frame_bytes: int):
        super().__init__(
            f"fallback spool frame too large: payload={payload_bytes} frame={frame_bytes}"
        )
        self.payload_bytes = payload_bytes
        self.frame_bytes = frame_bytes


class SpoolCapacityError(SessionFallbackSpoolError):
    def __init__(
        self,
        *,
        active_bytes: int,
        quarantine_bytes: int,
        other_artifact_bytes: int = 0,
        requested_bytes: int,
        cap_bytes: int,
    ):
        super().__init__(
            "fallback spool capacity exceeded: "
            f"active_bytes={active_bytes} quarantine_bytes={quarantine_bytes} "
            f"other_artifact_bytes={other_artifact_bytes} "
            f"requested_bytes={requested_bytes} cap_bytes={cap_bytes}"
        )
        self.active_bytes = active_bytes
        self.quarantine_bytes = quarantine_bytes
        self.other_artifact_bytes = other_artifact_bytes
        self.requested_bytes = requested_bytes
        self.cap_bytes = cap_bytes


class SpoolDurabilityError(SessionFallbackSpoolError):
    pass


class SpoolAppendAttemptPartialError(SessionFallbackSpoolError):
    def __init__(
        self,
        durable_results: Sequence[SpoolUnitAppendResult],
        cause: BaseException,
    ):
        super().__init__(f"fallback spool append partially durable: {cause}")
        self.durable_results = tuple(durable_results)
        self.cause = cause


class SpoolRetryableReplayError(SessionFallbackSpoolError):
    def __init__(self, retry_class: str, *, ack_pending: bool = False):
        super().__init__(retry_class)
        self.retry_class = retry_class
        self.ack_pending = ack_pending


class SpoolBlockedReplayError(SessionFallbackSpoolError):
    def __init__(self, error_class: str, *, frame_offset: int):
        super().__init__(error_class)
        self.error_class = error_class
        self.frame_offset = frame_offset


def _spool_root() -> Path:
    return get_hermes_home() / SPOOL_ROOT_NAME


def _active_spool_path() -> Path:
    return _spool_root() / ACTIVE_SPOOL_NAME


def _lock_path() -> Path:
    return _spool_root() / LOCK_FILE_NAME


def _quarantine_dir() -> Path:
    return _spool_root() / QUARANTINE_DIR_NAME


def _sealed_dir() -> Path:
    return _spool_root() / SEALED_DIR_NAME


def _acks_dir() -> Path:
    return _sealed_dir() / ACKS_DIR_NAME


def _blockers_dir() -> Path:
    return _sealed_dir() / BLOCKERS_DIR_NAME


def _owner_lock_path() -> Path:
    return _spool_root() / REPLAY_OWNER_LOCK_NAME


def _segment_sequence_highwater_path() -> Path:
    return _spool_root() / HIGHWATER_FILE_NAME


def _format_segment_sequence(sequence: int) -> str:
    return f"{sequence:0{SEGMENT_SEQUENCE_WIDTH}d}"


def _replay_root_key(runtime: _AnchoredRuntime) -> str:
    return str(runtime.root_path)


def _trigger_budget(trigger: str) -> tuple[int | None, int | None, float | None]:
    if trigger == "startup":
        return 64, 8 * 1024 * 1024, 2.0
    if trigger == "pre_persist":
        return 16, 2 * 1024 * 1024, 0.5
    return None, None, None


def _retry_delay_seconds(failures: int) -> float:
    return min(30.0, 0.5 * (2 ** max(failures - 1, 0)))


def _register_replay_cooldown(
    runtime: _AnchoredRuntime,
    *,
    retry_class: str,
    ack_pending: bool,
) -> float:
    key = _replay_root_key(runtime)
    now = time.monotonic()
    previous = _REPLAY_COOLDOWNS.get(key)
    failures = int(previous.get("failures", 0)) + 1 if previous else 1
    delay = _retry_delay_seconds(failures)
    _REPLAY_COOLDOWNS[key] = {
        "failures": failures,
        "next_eligible": now + delay,
        "retry_class": retry_class,
        "ack_pending": ack_pending,
    }
    return delay


def _clear_replay_cooldown(runtime: _AnchoredRuntime) -> None:
    _REPLAY_COOLDOWNS.pop(_replay_root_key(runtime), None)


def _cooldown_result(runtime: _AnchoredRuntime, *, trigger: str) -> ReplayRunResult | None:
    state = _REPLAY_COOLDOWNS.get(_replay_root_key(runtime))
    if not state:
        return None
    remaining = float(state["next_eligible"] - time.monotonic())
    if remaining <= 0:
        return None
    pending_snapshot = _capture_replay_terminal_backlog(
        runtime,
        ack_pending=bool(state.get("ack_pending", False)),
    )
    return ReplayRunResult(
        state=ReplayRunState.RETRY_PENDING,
        trigger=trigger,
        pending_bytes_after=pending_snapshot.pending_bytes,
        pending_frames_after=pending_snapshot.pending_frames,
        first_blocked_segment=pending_snapshot.first_blocked_segment,
        first_blocked_offset=pending_snapshot.first_blocked_offset,
        retry_class=str(state.get("retry_class") or "retry_pending"),
        ack_pending=pending_snapshot.ack_pending,
        cooldown_seconds=remaining,
    )


def _classify_nonretryable_replay_error(exc: BaseException) -> str:
    if isinstance(exc, OSError) and exc.errno is not None:
        errno_name = errno.errorcode.get(exc.errno)
        if errno_name:
            return f"errno_{errno_name.lower()}"
    return exc.__class__.__name__


def _pending_frames_for_log(result: ReplayRunResult) -> int:
    if result.pending_frames_after is not None:
        return int(result.pending_frames_after)
    return -1


def _pending_backlog_signature_state(result: ReplayRunResult) -> str | None:
    if result.state is not ReplayRunState.NOT_DURABLE:
        return None
    pending_frames = result.pending_frames_after
    if int(result.pending_bytes_after) < 0:
        return "pending_bytes_unknown"
    if pending_frames is not None and int(pending_frames) < 0:
        return "pending_frames_unknown"
    return None


def _log_replay_run_result(runtime: _AnchoredRuntime, result: ReplayRunResult) -> None:
    state = getattr(result.state, "value", result.state)
    state_text = str(state)
    signature = (
        state_text,
        result.retry_class,
        result.error_class,
        result.first_blocked_segment,
        result.first_blocked_offset,
        result.ack_pending,
        _pending_backlog_signature_state(result),
    )
    key = _replay_root_key(runtime)
    previous = _REPLAY_LOG_STATE.get(key)
    now = time.monotonic()
    if state_text in {
        ReplayRunState.RETRY_PENDING.value,
        ReplayRunState.BLOCKED_INTEGRITY.value,
        ReplayRunState.NOT_DURABLE.value,
    }:
        if (
            previous is not None
            and previous.get("signature") == signature
            and (now - float(previous.get("logged_at", 0.0))) < 300.0
        ):
            return
        pending_frames = _pending_frames_for_log(result)
        pending_bytes = int(result.pending_bytes_after)
        if state_text == ReplayRunState.RETRY_PENDING.value:
            logger.warning(
                "Fallback spool replay degraded state=%s trigger=%s retry_class=%s ack_pending=%s cooldown_seconds=%.3f pending_frames=%d pending_bytes=%d",
                state_text,
                result.trigger,
                result.retry_class,
                result.ack_pending,
                result.cooldown_seconds,
                pending_frames,
                pending_bytes,
            )
        elif state_text == ReplayRunState.BLOCKED_INTEGRITY.value:
            logger.error(
                "Fallback spool replay degraded state=%s trigger=%s error_class=%s first_blocked_segment=%s first_blocked_offset=%s pending_frames=%d pending_bytes=%d",
                state_text,
                result.trigger,
                result.error_class,
                result.first_blocked_segment,
                result.first_blocked_offset,
                pending_frames,
                pending_bytes,
            )
        else:
            logger.error(
                "Fallback spool replay degraded state=%s trigger=%s error_class=%s pending_frames=%d pending_bytes=%d",
                state_text,
                result.trigger,
                result.error_class,
                pending_frames,
                pending_bytes,
            )
        _REPLAY_LOG_STATE[key] = {"signature": signature, "logged_at": now}
        return
    if (
        previous is not None
        and state_text in {ReplayRunState.EMPTY.value, ReplayRunState.REPLAYED.value}
    ):
        logger.info(
            "Fallback spool replay recovered state=%s trigger=%s",
            state_text,
            result.trigger,
        )
        _REPLAY_LOG_STATE.pop(key, None)


def _log_and_return_replay_result(runtime: _AnchoredRuntime, result: ReplayRunResult) -> ReplayRunResult:
    _log_replay_run_result(runtime, result)
    return result


def _is_retryable_replay_os_error(exc: OSError) -> bool:
    return exc.errno in _REPLAY_RETRYABLE_ERRNOS


def _remaining_segment_bytes(ordered_segments: Sequence[tuple[int, str, Path]], start_index: int) -> int:
    total = 0
    for _sequence, _name, path in ordered_segments[start_index:]:
        try:
            total += int(path.stat().st_size)
        except OSError:
            continue
    return total


def _measure_active_pending_backlog(runtime: _AnchoredRuntime) -> tuple[int, int]:
    active_fd: int | None = None
    try:
        active_fd = _open_file_optional(
            runtime.root_fd,
            ACTIVE_SPOOL_NAME,
            full_path=runtime.active_path,
        )
        if active_fd is None:
            return 0, 0
        pending_bytes = max(0, int(os.fstat(active_fd).st_size))
        if pending_bytes == 0:
            return 0, 0
        try:
            return pending_bytes, _scan_fd(active_fd).frame_count
        except OSError:
            return pending_bytes, -1
    except (OSError, SpoolDurabilityError, SpoolPathSecurityError):
        return -1, -1
    finally:
        if active_fd is not None:
            _close_fd_quietly(active_fd)


def _measure_remaining_segment_backlog(
    runtime: _AnchoredRuntime,
    *,
    ordered_segments: Sequence[tuple[int, str, Path]],
    start_index: int,
    current_segment_name: str | None = None,
    current_segment_frame_count: int | None = None,
    current_segment_pending_bytes: int | None = None,
) -> tuple[int, int]:
    if start_index >= len(ordered_segments):
        return 0, 0
    sealed_fd: int | None = None
    pending_bytes = 0
    pending_frames = 0
    try:
        sealed_fd = _open_dir_optional(runtime.root_fd, SEALED_DIR_NAME, full_path=_sealed_dir())
        if sealed_fd is None:
            return -1, -1
        for index, (_sequence, segment_name, _segment_path) in enumerate(
            ordered_segments[start_index:], start=start_index
        ):
            segment_fd: int | None = None
            try:
                segment_fd = os.open(segment_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=sealed_fd)
                segment_stat = os.fstat(segment_fd)
                _status_assert_regular_fd_matches_entry(
                    parent_fd=sealed_fd,
                    name=segment_name,
                    entry_stat=segment_stat,
                    fd=segment_fd,
                    error_class="entry_replaced",
                    default_error_class="inspection_error",
                )
                if (
                    index == start_index
                    and current_segment_name is not None
                    and current_segment_frame_count is not None
                    and segment_name == current_segment_name
                ):
                    if current_segment_pending_bytes is not None:
                        pending_bytes += max(0, int(current_segment_pending_bytes))
                    else:
                        pending_bytes += max(0, int(segment_stat.st_size))
                    pending_frames += max(0, int(current_segment_frame_count))
                    continue
                pending_bytes += max(0, int(segment_stat.st_size))
                pending_frames += _scan_fd(segment_fd).frame_count
            except (
                _StatusInspectionError,
                OSError,
                SpoolDurabilityError,
                SpoolPathSecurityError,
            ):
                return -1, -1
            finally:
                if segment_fd is not None:
                    _close_fd_quietly(segment_fd)
        return pending_bytes, pending_frames
    except (OSError, SpoolDurabilityError, SpoolPathSecurityError):
        return -1, -1
    finally:
        if sealed_fd is not None:
            _close_fd_quietly(sealed_fd)


def _unknown_pending_backlog_snapshot(
    *,
    ack_pending: bool = False,
    first_blocked_segment: int | None = None,
    first_blocked_offset: int | None = None,
) -> _PendingBacklogSnapshot:
    return _PendingBacklogSnapshot(
        pending_bytes=-1,
        pending_frames=-1,
        ack_pending=ack_pending,
        first_blocked_segment=first_blocked_segment,
        first_blocked_offset=first_blocked_offset,
    )


def _snapshot_pending_backlog(runtime: _AnchoredRuntime) -> _PendingBacklogSnapshot:
    active_fd: int | None = None
    lock_held = False
    try:
        if not _try_acquire_status_lock(runtime.lock_fd):
            return _unknown_pending_backlog_snapshot()
        lock_held = True
        _status_count_protocol_artifacts(
            root_fd=runtime.root_fd,
            limit=STATUS_SCAN_ARTIFACT_LIMIT,
        )
        pending_bytes = 0
        pending_frames = 0
        active_fd = _open_file_optional(
            runtime.root_fd,
            ACTIVE_SPOOL_NAME,
            full_path=runtime.active_path,
        )
        if active_fd is not None:
            active_stat = os.fstat(active_fd)
            _status_assert_active_fd_matches_entry(
                root_fd=runtime.root_fd,
                entry_stat=active_stat,
                fd=active_fd,
            )
            pending_bytes = max(0, int(active_stat.st_size))
            if pending_bytes > 0:
                active_scan = _scan_fd(active_fd)
                _status_assert_active_fd_matches_entry(
                    root_fd=runtime.root_fd,
                    entry_stat=active_stat,
                    fd=active_fd,
                )
                pending_frames = active_scan.frame_count

        blocker = _status_load_blocker_snapshot(root_fd=runtime.root_fd)
        segments = _status_collect_segment_snapshots(root_fd=runtime.root_fd)
        ack_snapshot = _status_scan_ack_sidecars(
            root_fd=runtime.root_fd,
            segment_sizes={item.name: item.size_bytes for item in segments},
        )
        if ack_snapshot.blocked_orphan_sequence is not None:
            return _unknown_pending_backlog_snapshot(
                ack_pending=True,
                first_blocked_segment=ack_snapshot.blocked_orphan_sequence,
            )

        ack_pending = (
            blocker.acked_prefix_bytes > 0
            or ack_snapshot.orphan_tombstone_sequence is not None
        )
        for item in segments:
            acked_prefix_bytes, segment_ack_pending = _status_effective_acked_prefix(
                segment=item,
                ack_snapshot=ack_snapshot,
                blocker=blocker,
            )
            segment_pending_frames, segment_pending_bytes = _status_pending_prefix_metrics(
                item.decoded,
                acked_prefix_bytes=acked_prefix_bytes,
                error_class="invalid_ack_json",
            )
            pending_frames += segment_pending_frames
            pending_bytes += segment_pending_bytes
            ack_pending = ack_pending or segment_ack_pending

        return _PendingBacklogSnapshot(
            pending_bytes=pending_bytes,
            pending_frames=pending_frames,
            ack_pending=ack_pending,
            first_blocked_segment=blocker.sequence if blocker.present else None,
            first_blocked_offset=blocker.offset if blocker.present else None,
        )
    finally:
        if active_fd is not None:
            _close_fd_quietly(active_fd)
        if lock_held:
            _release_status_lock(runtime.lock_fd)


def _capture_pending_backlog_snapshot(runtime: _AnchoredRuntime) -> _PendingBacklogSnapshot:
    try:
        return _snapshot_pending_backlog(runtime)
    except Exception:
        return _unknown_pending_backlog_snapshot()


def _capture_replay_terminal_backlog(
    runtime: _AnchoredRuntime,
    *,
    ack_pending: bool = False,
) -> _PendingBacklogSnapshot:
    snapshot = _capture_pending_backlog_snapshot(runtime)
    merged_ack_pending = bool(snapshot.ack_pending or ack_pending)
    if merged_ack_pending == snapshot.ack_pending:
        return snapshot
    return _PendingBacklogSnapshot(
        pending_bytes=snapshot.pending_bytes,
        pending_frames=snapshot.pending_frames,
        ack_pending=merged_ack_pending,
        first_blocked_segment=snapshot.first_blocked_segment,
        first_blocked_offset=snapshot.first_blocked_offset,
    )


def _replay_db_call(
    session_db,
    frame: SessionSpoolFrame,
):
    try:
        return session_db.reconcile_bootstrap_and_append_messages_batch(
            frame.record.bootstrap,
            frame.record.batch_messages,
            replay_patience_s=2.0,
        )
    except AppendMessagesBatchConflictError as exc:
        raise SpoolBlockedReplayError(
            "AppendMessagesBatchConflictError",
            frame_offset=frame.frame_offset,
        ) from exc
    except CompressionSessionClosedError as exc:
        raise SpoolBlockedReplayError(
            "CompressionSessionClosedError",
            frame_offset=frame.frame_offset,
        ) from exc
    except CompressionSessionBusyError as exc:
        raise SpoolRetryableReplayError("compression_busy", ack_pending=False) from exc
    except sqlite3.OperationalError as exc:
        err = str(exc).lower()
        if "locked" in err or "busy" in err:
            raise SpoolRetryableReplayError(
                "sqlite_locked_or_busy",
                ack_pending=False,
            ) from exc
        raise


@dataclass(frozen=True)
class _ReplayOwnerLease:
    fd: int
    path: Path


def _is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(path.lstat().st_mode)
    except FileNotFoundError:
        return False



def _require_not_symlink(path: Path) -> None:
    if _is_symlink(path):
        raise SpoolPathSecurityError(f"symlinked fallback spool path refused: {path}")


def _require_existing_dir(path: Path) -> None:
    _require_not_symlink(path)
    try:
        st = path.stat()
    except FileNotFoundError as exc:
        raise SpoolDurabilityError(f"required directory missing: {path}") from exc
    if not stat.S_ISDIR(st.st_mode):
        raise SpoolPathSecurityError(f"fallback spool path is not a directory: {path}")
    if os.name == "posix":
        os.chmod(path, ROOT_MODE)


def _require_existing_file(path: Path) -> None:
    _require_not_symlink(path)
    try:
        st = path.stat()
    except FileNotFoundError as exc:
        raise SpoolDurabilityError(f"required file missing: {path}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise SpoolPathSecurityError(f"fallback spool path is not a regular file: {path}")
    if os.name == "posix":
        os.chmod(path, FILE_MODE)


def _supports_directory_fsync() -> bool:
    return os.name == "posix"


def _dir_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _file_open_flags(base: int) -> int:
    if hasattr(os, "O_NOFOLLOW"):
        base |= os.O_NOFOLLOW
    return base


def _fsync_fd(fd: int) -> None:
    os.fsync(fd)


def _fsync_directory(path: Path) -> None:
    if not _supports_directory_fsync():
        raise SpoolDurabilityError(
            f"directory fsync unavailable for fallback spool path: {path}"
        )
    _require_existing_dir(path)
    directory_fd = os.open(str(path), _dir_open_flags())
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _ensure_directory(path: Path, *, mode: int = ROOT_MODE) -> bool:
    _require_not_symlink(path)
    if path.exists():
        _require_existing_dir(path)
        return False
    parent = path.parent
    _require_existing_dir(parent)
    try:
        os.mkdir(path, mode)
    except FileExistsError:
        _require_existing_dir(path)
        return False
    try:
        if os.name == "posix":
            os.chmod(path, mode)
        _fsync_directory(parent)
    except OSError as exc:
        raise SpoolDurabilityError(
            f"unable to durably create fallback spool directory {path}: {exc}"
        ) from exc
    return True


def _same_file_stat(lhs: os.stat_result, rhs: os.stat_result) -> bool:
    return lhs.st_dev == rhs.st_dev and lhs.st_ino == rhs.st_ino


def _optional_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _json_compatible(value: Any) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _validate_message_payload(payload: Mapping[str, Any]) -> bool:
    expected_keys = {
        "persistence_message_key",
        "persistence_ordinal",
        "role",
        "content",
        "timestamp",
        "tool_name",
        "tool_calls",
        "tool_call_id",
        "finish_reason",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "codex_reasoning_items",
        "codex_message_items",
        "api_content",
        "display_kind",
        "display_metadata",
    }
    if set(payload.keys()) != expected_keys:
        return False
    if not isinstance(payload.get("persistence_message_key"), str) or not payload.get(
        "persistence_message_key"
    ):
        return False
    ordinal = payload.get("persistence_ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        return False
    if not isinstance(payload.get("role"), str) or not payload.get("role"):
        return False
    if payload.get("content") is not None and not isinstance(payload.get("content"), str):
        return False
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        return False
    if not _optional_str(payload.get("tool_name")):
        return False
    if payload.get("tool_calls") is not None and not isinstance(payload.get("tool_calls"), list):
        return False
    if payload.get("tool_calls") is not None and not _json_compatible(payload.get("tool_calls")):
        return False
    if not _optional_str(payload.get("tool_call_id")):
        return False
    if not _optional_str(payload.get("finish_reason")):
        return False
    if not _optional_str(payload.get("reasoning")):
        return False
    if not _json_compatible(payload.get("reasoning_content")):
        return False
    if not _json_compatible(payload.get("reasoning_details")):
        return False
    if not _json_compatible(payload.get("codex_reasoning_items")):
        return False
    if not _json_compatible(payload.get("codex_message_items")):
        return False
    if payload.get("api_content") is not None and not _json_compatible(payload.get("api_content")):
        return False
    if not _optional_str(payload.get("display_kind")):
        return False
    if payload.get("display_metadata") is not None and not isinstance(
        payload.get("display_metadata"), dict
    ):
        return False
    if payload.get("display_metadata") is not None and not _json_compatible(
        payload.get("display_metadata")
    ):
        return False
    return True


def _validate_payload_schema(payload_obj: Any) -> bool:
    if not isinstance(payload_obj, dict):
        return False
    if set(payload_obj.keys()) != {
        "schema_version",
        "record_type",
        "persist_attempt_id",
        "persist_attempt_unit_index",
        "session",
        "canonical_failure",
        "unit",
    }:
        return False
    if payload_obj.get("schema_version") != 1:
        return False
    if payload_obj.get("record_type") != "session_persistence_unit":
        return False
    persist_attempt_id = payload_obj.get("persist_attempt_id")
    if not isinstance(persist_attempt_id, str) or not re.fullmatch(
        r"[0-9a-f]{32}", persist_attempt_id
    ):
        return False
    attempt_index = payload_obj.get("persist_attempt_unit_index")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 0:
        return False

    session = payload_obj.get("session")
    if not isinstance(session, dict) or set(session.keys()) != {
        "session_id",
        "source",
        "started_at",
        "model",
        "model_config",
        "system_prompt",
        "parent_session_id",
        "cwd",
        "profile_name",
        "user_id",
        "session_key",
        "chat_id",
        "chat_type",
        "thread_id",
    }:
        return False
    if not all(
        _optional_str(session.get(key))
        for key in (
            "session_id",
            "source",
            "model",
            "system_prompt",
            "parent_session_id",
            "cwd",
            "profile_name",
            "user_id",
            "session_key",
            "chat_id",
            "chat_type",
            "thread_id",
        )
    ):
        return False
    if session.get("started_at") is not None and not isinstance(
        session.get("started_at"), (int, float)
    ):
        return False
    if not _json_compatible(session.get("model_config")):
        return False

    canonical_failure = payload_obj.get("canonical_failure")
    if not isinstance(canonical_failure, dict) or set(canonical_failure.keys()) != {
        "stage",
        "error_class",
        "error_message",
        "session_row_created",
    }:
        return False
    if canonical_failure.get("stage") not in {
        "session_row_create",
        "append_messages_batch",
    }:
        return False
    if not isinstance(canonical_failure.get("error_class"), str) or not canonical_failure.get(
        "error_class"
    ):
        return False
    error_message = canonical_failure.get("error_message")
    if error_message is None or not isinstance(error_message, str):
        return False
    if "\n" in error_message or "\r" in error_message:
        return False
    if len(error_message.encode("utf-8", errors="ignore")) > 512:
        return False
    if not isinstance(canonical_failure.get("session_row_created"), bool):
        return False

    unit = payload_obj.get("unit")
    if not isinstance(unit, dict) or set(unit.keys()) != {
        "persistence_unit_id",
        "message_count",
        "messages",
    }:
        return False
    if not isinstance(unit.get("persistence_unit_id"), str) or not unit.get(
        "persistence_unit_id"
    ):
        return False
    message_count = unit.get("message_count")
    messages = unit.get("messages")
    if isinstance(message_count, bool) or not isinstance(message_count, int) or message_count < 1:
        return False
    if not isinstance(messages, list) or len(messages) != message_count:
        return False
    if not all(isinstance(message, dict) and _validate_message_payload(message) for message in messages):
        return False
    ordinals = [message["persistence_ordinal"] for message in messages]
    if ordinals != list(range(len(messages))):
        return False
    keys = [message["persistence_message_key"] for message in messages]
    if len(set(keys)) != len(keys):
        return False
    return True


def _message_payload(message: SessionDBBatchMessage) -> dict[str, Any]:
    return {
        "persistence_message_key": message.persistence_message_key,
        "persistence_ordinal": message.persistence_ordinal,
        "role": message.role,
        "content": message.content,
        "timestamp": message.timestamp,
        "tool_name": message.tool_name,
        "tool_calls": message.tool_calls,
        "tool_call_id": message.tool_call_id,
        "finish_reason": message.finish_reason,
        "reasoning": message.reasoning,
        "reasoning_content": message.reasoning_content,
        "reasoning_details": message.reasoning_details,
        "codex_reasoning_items": message.codex_reasoning_items,
        "codex_message_items": message.codex_message_items,
        "api_content": message.api_content,
        "display_kind": message.display_kind,
        "display_metadata": message.display_metadata,
    }


def _payload_dict_for_record(record: SessionSpoolRecord) -> dict[str, Any]:
    if not isinstance(record.persist_attempt_id, str) or not re.fullmatch(
        r"[0-9a-f]{32}", record.persist_attempt_id
    ):
        raise SpoolDurabilityError("invalid fallback spool persist_attempt_id")
    if (
        isinstance(record.persist_attempt_unit_index, bool)
        or not isinstance(record.persist_attempt_unit_index, int)
        or record.persist_attempt_unit_index < 0
    ):
        raise SpoolDurabilityError("invalid fallback spool persist_attempt_unit_index")
    if not record.batch_messages:
        raise SpoolDurabilityError("fallback spool record requires at least one message")

    unit_id = record.batch_messages[0].persistence_unit_id
    if not isinstance(unit_id, str) or not unit_id:
        raise SpoolDurabilityError("invalid fallback spool persistence_unit_id")
    seen_keys: set[str] = set()
    payload_messages = []
    for expected_ordinal, message in enumerate(record.batch_messages):
        if message.persistence_unit_id != unit_id:
            raise SpoolDurabilityError("mixed persistence_unit_id values are not allowed")
        if message.persistence_ordinal != expected_ordinal:
            raise SpoolDurabilityError(
                "fallback spool messages must be stored in ordinal order 0..n-1"
            )
        if not isinstance(message.persistence_message_key, str) or not message.persistence_message_key:
            raise SpoolDurabilityError("invalid fallback spool persistence_message_key")
        if message.persistence_message_key in seen_keys:
            raise SpoolDurabilityError("duplicate fallback spool persistence_message_key")
        seen_keys.add(message.persistence_message_key)
        payload_messages.append(_message_payload(message))

    payload = {
        "schema_version": 1,
        "record_type": "session_persistence_unit",
        "persist_attempt_id": record.persist_attempt_id,
        "persist_attempt_unit_index": record.persist_attempt_unit_index,
        "session": {
            "session_id": record.bootstrap.session_id,
            "source": record.bootstrap.source,
            "started_at": record.bootstrap.started_at,
            "model": record.bootstrap.model,
            "model_config": record.bootstrap.model_config,
            "system_prompt": record.bootstrap.system_prompt,
            "parent_session_id": record.bootstrap.parent_session_id,
            "cwd": record.bootstrap.cwd,
            "profile_name": record.bootstrap.profile_name,
            "user_id": record.bootstrap.user_id,
            "session_key": record.bootstrap.session_key,
            "chat_id": record.bootstrap.chat_id,
            "chat_type": record.bootstrap.chat_type,
            "thread_id": record.bootstrap.thread_id,
        },
        "canonical_failure": {
            "stage": record.canonical_failure.get("stage"),
            "error_class": record.canonical_failure.get("error_class"),
            "error_message": record.canonical_failure.get("error_message"),
            "session_row_created": record.canonical_failure.get(
                "session_row_created"
            ),
        },
        "unit": {
            "persistence_unit_id": unit_id,
            "message_count": len(payload_messages),
            "messages": payload_messages,
        },
    }
    if not _validate_payload_schema(payload):
        raise SpoolDurabilityError("invalid fallback spool record payload")
    return payload


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise _DuplicateJsonKeyError(key)
        obj[key] = value
    return obj


def _payload_bytes_for_record(record: SessionSpoolRecord) -> bytes:
    payload = json.dumps(
        _payload_dict_for_record(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise SpoolFrameTooLargeError(len(payload), len(payload) + HEADER_SIZE)
    return payload


def _record_from_payload_object(payload_obj: Mapping[str, Any]) -> SessionSpoolRecord:
    session = payload_obj["session"]
    canonical_failure = payload_obj["canonical_failure"]
    unit = payload_obj["unit"]
    batch_messages = tuple(
        SessionDBBatchMessage(
            persistence_unit_id=unit["persistence_unit_id"],
            persistence_message_key=message["persistence_message_key"],
            persistence_ordinal=message["persistence_ordinal"],
            role=message["role"],
            content=message["content"],
            timestamp=message["timestamp"],
            tool_name=message["tool_name"],
            tool_calls=message["tool_calls"],
            tool_call_id=message["tool_call_id"],
            finish_reason=message["finish_reason"],
            reasoning=message["reasoning"],
            reasoning_content=message["reasoning_content"],
            reasoning_details=message["reasoning_details"],
            codex_reasoning_items=message["codex_reasoning_items"],
            codex_message_items=message["codex_message_items"],
            api_content=message["api_content"],
            display_kind=message["display_kind"],
            display_metadata=message["display_metadata"],
        )
        for message in unit["messages"]
    )
    return SessionSpoolRecord(
        bootstrap=SessionSpoolBootstrap(
            session_id=session["session_id"],
            source=session["source"],
            started_at=session["started_at"],
            model=session["model"],
            model_config=session["model_config"],
            system_prompt=session["system_prompt"],
            parent_session_id=session["parent_session_id"],
            cwd=session["cwd"],
            profile_name=session["profile_name"],
            user_id=session["user_id"],
            session_key=session["session_key"],
            chat_id=session["chat_id"],
            chat_type=session["chat_type"],
            thread_id=session["thread_id"],
        ),
        persist_attempt_id=payload_obj["persist_attempt_id"],
        persist_attempt_unit_index=payload_obj["persist_attempt_unit_index"],
        canonical_failure={
            "stage": canonical_failure["stage"],
            "error_class": canonical_failure["error_class"],
            "error_message": canonical_failure["error_message"],
            "session_row_created": canonical_failure["session_row_created"],
        },
        batch_messages=batch_messages,
    )


def _frame_from_payload_bytes(
    payload: bytes,
    *,
    record_kind: int = RECORD_KIND_SESSION_PERSISTENCE_UNIT,
    reserved_bytes: bytes = b"\x00\x00",
) -> bytes:
    payload_len = len(payload)
    if payload_len == 0 or payload_len > MAX_PAYLOAD_BYTES:
        raise SpoolFrameTooLargeError(payload_len, payload_len + HEADER_SIZE)
    if len(reserved_bytes) != 2:
        raise SpoolDurabilityError("reserved header field must be exactly two bytes")
    header_prefix = bytes([FRAME_VERSION, record_kind]) + reserved_bytes + payload_len.to_bytes(
        8, "big"
    )
    digest = blake2s(header_prefix + payload, digest_size=16).digest()
    frame = HEADER_MAGIC + header_prefix + digest + payload
    if len(frame) > MAX_FRAME_BYTES:
        raise SpoolFrameTooLargeError(payload_len, len(frame))
    return frame


def _frame_bytes_for_record(record: SessionSpoolRecord) -> bytes:
    return _frame_from_payload_bytes(_payload_bytes_for_record(record))


def _require_secure_path_primitives() -> None:
    if os.name != "posix":
        raise SpoolPathSecurityError(
            "descriptor-anchored fallback spool path security is unavailable on this platform"
        )
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise SpoolPathSecurityError(
            "descriptor-anchored fallback spool path security requires O_NOFOLLOW and O_DIRECTORY"
        )


def _open_home_dir_fd(home_path: Path) -> int:
    _require_secure_path_primitives()
    try:
        fd = os.open(str(home_path), os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise SpoolDurabilityError(
            f"unable to open HERMES_HOME for fallback spool security: {home_path}: {exc}"
        ) from exc
    home_stat = os.fstat(fd)
    if not stat.S_ISDIR(home_stat.st_mode):
        os.close(fd)
        raise SpoolPathSecurityError(f"HERMES_HOME is not a directory: {home_path}")
    return fd


def _fsync_directory_fd(fd: int, label: Path | str) -> None:
    if not _supports_directory_fsync():
        raise SpoolDurabilityError(
            f"directory fsync unavailable for fallback spool path: {label}"
        )
    try:
        os.fsync(fd)
    except OSError as exc:
        raise SpoolDurabilityError(
            f"unable to fsync fallback spool directory {label}: {exc}"
        ) from exc


def _close_fd_quietly(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _open_dir_at(
    parent_fd: int,
    name: str,
    *,
    full_path: Path,
    mode: int,
    create: bool,
    parent_label: Path | str,
    fsync_parent_on_open_existing: bool = False,
) -> tuple[int, bool]:
    open_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    created = False
    while True:
        try:
            fd = os.open(name, open_flags, dir_fd=parent_fd)
            break
        except FileNotFoundError:
            if not create:
                raise SpoolDurabilityError(
                    f"required fallback spool directory missing: {full_path}"
                )
            try:
                os.mkdir(name, mode, dir_fd=parent_fd)
            except FileExistsError:
                continue
            except OSError as exc:
                raise SpoolDurabilityError(
                    f"unable to create fallback spool directory {full_path}: {exc}"
                ) from exc
            created = True
            continue
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise SpoolPathSecurityError(
                    f"symlinked fallback spool directory refused: {full_path}"
                ) from exc
            raise SpoolDurabilityError(
                f"unable to open fallback spool directory {full_path}: {exc}"
            ) from exc
    try:
        dir_stat = os.fstat(fd)
        if not stat.S_ISDIR(dir_stat.st_mode):
            raise SpoolPathSecurityError(
                f"fallback spool path is not a directory: {full_path}"
            )
        if hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
        if created:
            _fsync_directory_fd(parent_fd, parent_label)
        elif fsync_parent_on_open_existing:
            _fsync_directory_fd(parent_fd, parent_label)
        _assert_entry_matches_fd(parent_fd, name, fd, expect="dir", label=str(full_path))
    except SessionFallbackSpoolError:
        _close_fd_quietly(fd)
        raise
    except OSError as exc:
        _close_fd_quietly(fd)
        raise SpoolDurabilityError(
            f"unable to durably open fallback spool directory {full_path}: {exc}"
        ) from exc
    except BaseException:
        _close_fd_quietly(fd)
        raise
    return fd, created


def _open_file_at(
    parent_fd: int,
    name: str,
    *,
    full_path: Path,
    mode: int,
    create: bool,
    fsync_parent_on_create: bool,
    fsync_file_on_create: bool,
    parent_label: Path | str,
    fsync_parent_on_open_existing: bool = False,
) -> tuple[int, bool]:
    open_flags = os.O_RDWR | os.O_NOFOLLOW
    created = False
    while True:
        try:
            fd = os.open(name, open_flags, dir_fd=parent_fd)
            break
        except FileNotFoundError:
            if not create:
                raise SpoolDurabilityError(
                    f"required fallback spool file missing: {full_path}"
                )
            try:
                fd = os.open(name, open_flags | os.O_CREAT | os.O_EXCL, mode, dir_fd=parent_fd)
            except FileExistsError:
                continue
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise SpoolPathSecurityError(
                        f"symlinked fallback spool file refused: {full_path}"
                    ) from exc
                raise SpoolDurabilityError(
                    f"unable to create fallback spool file {full_path}: {exc}"
                ) from exc
            created = True
            break
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise SpoolPathSecurityError(
                    f"symlinked fallback spool file refused: {full_path}"
                ) from exc
            raise SpoolDurabilityError(
                f"unable to open fallback spool file {full_path}: {exc}"
            ) from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise SpoolPathSecurityError(
                f"fallback spool path is not a regular file: {full_path}"
            )
        if hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
        if created:
            if fsync_file_on_create:
                _fsync_fd(fd)
            if fsync_parent_on_create:
                _fsync_directory_fd(parent_fd, parent_label)
        elif fsync_parent_on_open_existing:
            _fsync_directory_fd(parent_fd, parent_label)
        _assert_entry_matches_fd(parent_fd, name, fd, expect="file", label=str(full_path))
    except SessionFallbackSpoolError:
        _close_fd_quietly(fd)
        raise
    except OSError as exc:
        _close_fd_quietly(fd)
        action = "create" if created else "open"
        raise SpoolDurabilityError(
            f"unable to durably {action} fallback spool file {full_path}: {exc}"
        ) from exc
    except BaseException:
        _close_fd_quietly(fd)
        raise
    return fd, created


def _assert_home_matches_fd(home_path: Path, home_fd: int) -> None:
    try:
        current_stat = home_path.stat()
    except FileNotFoundError as exc:
        raise SpoolDurabilityError(
            f"HERMES_HOME disappeared during fallback spool append: {home_path}"
        ) from exc
    if not _same_file_stat(current_stat, os.fstat(home_fd)):
        raise SpoolPathSecurityError(
            f"HERMES_HOME changed during fallback spool append: {home_path}"
        )


def _assert_entry_matches_fd(
    parent_fd: int,
    name: str,
    fd: int,
    *,
    expect: str,
    label: str,
) -> None:
    try:
        entry_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise SpoolDurabilityError(
            f"fallback spool entry disappeared before durability was confirmed: {label}"
        ) from exc
    except OSError as exc:
        raise SpoolDurabilityError(
            f"unable to restat fallback spool entry {label}: {exc}"
        ) from exc
    target_stat = os.fstat(fd)
    if expect == "dir":
        if not stat.S_ISDIR(entry_stat.st_mode):
            raise SpoolPathSecurityError(f"fallback spool path is not a directory: {label}")
    elif expect == "file":
        if not stat.S_ISREG(entry_stat.st_mode):
            raise SpoolPathSecurityError(f"fallback spool path is not a regular file: {label}")
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(f"unknown expectation: {expect}")
    if not _same_file_stat(entry_stat, target_stat):
        raise SpoolPathSecurityError(
            f"fallback spool entry was swapped during append: {label}"
        )


def _is_lock_contention_error(exc: OSError) -> bool:
    return exc.errno in _LOCK_CONTENTION_ERRNOS


@contextmanager
def _append_lock(lock_fd: int, lock_label: str):
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    locked = False
    try:
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif msvcrt is not None:  # pragma: no cover - Windows-only branch
                    msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover - unsupported platform
                    raise SpoolDurabilityError("no secure file-locking primitive available")
                locked = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SpoolLockTimeoutError(
                        f"timed out waiting for fallback spool append lock: {lock_label}"
                    )
                time.sleep(LOCK_RETRY_SECONDS)
            except OSError as exc:
                if _is_lock_contention_error(exc):
                    if time.monotonic() >= deadline:
                        raise SpoolLockTimeoutError(
                            f"timed out waiting for fallback spool append lock: {lock_label}"
                        ) from exc
                    time.sleep(LOCK_RETRY_SECONDS)
                    continue
                raise SpoolDurabilityError(
                    f"unexpected fallback spool lock failure for {lock_label}: {exc}"
                ) from exc
        yield
    finally:
        if locked:
            try:
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows-only branch
                    msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass


def _read_exact_from_fd(fd: int, *, offset: int, length: int) -> bytes:
    if length <= 0:
        return b""
    chunks = bytearray()
    while len(chunks) < length:
        remaining = length - len(chunks)
        try:
            if hasattr(os, "pread"):
                chunk = os.pread(fd, remaining, offset + len(chunks))
            else:  # pragma: no cover - fallback for runtimes without pread
                os.lseek(fd, offset + len(chunks), os.SEEK_SET)
                chunk = os.read(fd, remaining)
        except InterruptedError:
            continue
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _scan_fd(
    fd: int,
    *,
    max_file_bytes: int = TOTAL_CAP_BYTES,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> SpoolScanResult:
    file_size = os.fstat(fd).st_size
    if file_size <= 0:
        return SpoolScanResult(
            valid_prefix_bytes=0,
            frame_count=0,
            tail_status=SpoolTailStatus.CLEAN,
            tail_offset=None,
        )
    payload_cap = max_frame_bytes - HEADER_SIZE
    budget = max_file_bytes if max_file_bytes > 0 else file_size
    offset = 0
    frame_count = 0
    while offset < file_size:
        if offset + HEADER_SIZE > budget:
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.SCAN_LIMIT_EXCEEDED,
                tail_offset=offset,
            )
        header = _read_exact_from_fd(fd, offset=offset, length=HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.INCOMPLETE_EOF,
                tail_offset=offset,
            )
        if header[:4] != HEADER_MAGIC:
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.BAD_MAGIC,
                tail_offset=offset,
            )
        if header[4] != FRAME_VERSION:
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.BAD_VERSION,
                tail_offset=offset,
            )
        if header[5] != RECORD_KIND_SESSION_PERSISTENCE_UNIT:
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.BAD_RECORD_KIND,
                tail_offset=offset,
            )
        if header[6:8] != b"\x00\x00":
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.NONZERO_RESERVED,
                tail_offset=offset,
            )
        payload_len = int.from_bytes(header[8:16], "big")
        if payload_len == 0 or payload_len > payload_cap:
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.OVERSIZED_LENGTH,
                tail_offset=offset,
            )
        frame_len = HEADER_SIZE + payload_len
        if offset + frame_len > budget:
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.SCAN_LIMIT_EXCEEDED,
                tail_offset=offset,
            )
        payload = _read_exact_from_fd(fd, offset=offset + HEADER_SIZE, length=payload_len)
        if len(payload) < payload_len:
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.INCOMPLETE_EOF,
                tail_offset=offset,
            )
        expected_digest = blake2s(header[4:16] + payload, digest_size=16).digest()
        if header[16:32] != expected_digest:
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.CHECKSUM_MISMATCH,
                tail_offset=offset,
            )
        try:
            payload_obj = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError):
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.INVALID_JSON,
                tail_offset=offset,
            )
        if not _validate_payload_schema(payload_obj):
            return SpoolScanResult(
                valid_prefix_bytes=offset,
                frame_count=frame_count,
                tail_status=SpoolTailStatus.INVALID_SCHEMA,
                tail_offset=offset,
            )
        frame_count += 1
        offset += frame_len
    if offset < file_size:
        return SpoolScanResult(
            valid_prefix_bytes=offset,
            frame_count=frame_count,
            tail_status=SpoolTailStatus.SCAN_LIMIT_EXCEEDED,
            tail_offset=offset,
        )
    return SpoolScanResult(
        valid_prefix_bytes=offset,
        frame_count=frame_count,
        tail_status=SpoolTailStatus.CLEAN,
        tail_offset=None,
    )


def scan_spool(
    path: Path,
    *,
    max_file_bytes: int = TOTAL_CAP_BYTES,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> SpoolScanResult:
    if not path.exists():
        return SpoolScanResult(
            valid_prefix_bytes=0,
            frame_count=0,
            tail_status=SpoolTailStatus.CLEAN,
            tail_offset=None,
        )
    _require_existing_file(path)
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        return _scan_fd(fd, max_file_bytes=max_file_bytes, max_frame_bytes=max_frame_bytes)
    finally:
        os.close(fd)


def _decode_spool_segment_fd(
    fd: int,
    *,
    start_offset: int = 0,
    max_file_bytes: int = TOTAL_CAP_BYTES,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> DecodedSegment:
    if start_offset < 0:
        raise SpoolDurabilityError(f"invalid decode offset for fallback spool segment: {start_offset}")
    file_size = os.fstat(fd).st_size
    payload_cap = max_frame_bytes - HEADER_SIZE
    budget = min(file_size, max_file_bytes if max_file_bytes > 0 else file_size)
    offset = start_offset
    frames: list[SessionSpoolFrame] = []
    while offset < file_size:
        if offset + HEADER_SIZE > budget:
            return DecodedSegment(
                prefix_frames=tuple(frames),
                valid_prefix_bytes=offset,
                tail_status=SpoolTailStatus.SCAN_LIMIT_EXCEEDED,
                tail_offset=offset,
            )
        header = _read_exact_from_fd(fd, offset=offset, length=HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            return DecodedSegment(
                prefix_frames=tuple(frames),
                valid_prefix_bytes=offset,
                tail_status=SpoolTailStatus.INCOMPLETE_EOF,
                tail_offset=offset,
            )
        if header[:4] != HEADER_MAGIC:
            tail_status = SpoolTailStatus.BAD_MAGIC
        elif header[4] != FRAME_VERSION:
            tail_status = SpoolTailStatus.BAD_VERSION
        elif header[5] != RECORD_KIND_SESSION_PERSISTENCE_UNIT:
            tail_status = SpoolTailStatus.BAD_RECORD_KIND
        elif header[6:8] != b"\x00\x00":
            tail_status = SpoolTailStatus.NONZERO_RESERVED
        else:
            tail_status = None
        if tail_status is not None:
            return DecodedSegment(
                prefix_frames=tuple(frames),
                valid_prefix_bytes=offset,
                tail_status=tail_status,
                tail_offset=offset,
            )
        payload_len = int.from_bytes(header[8:16], "big")
        if payload_len == 0 or payload_len > payload_cap:
            return DecodedSegment(
                prefix_frames=tuple(frames),
                valid_prefix_bytes=offset,
                tail_status=SpoolTailStatus.OVERSIZED_LENGTH,
                tail_offset=offset,
            )
        frame_length = HEADER_SIZE + payload_len
        if offset + frame_length > budget:
            return DecodedSegment(
                prefix_frames=tuple(frames),
                valid_prefix_bytes=offset,
                tail_status=SpoolTailStatus.SCAN_LIMIT_EXCEEDED,
                tail_offset=offset,
            )
        payload = _read_exact_from_fd(fd, offset=offset + HEADER_SIZE, length=payload_len)
        if len(payload) < payload_len:
            return DecodedSegment(
                prefix_frames=tuple(frames),
                valid_prefix_bytes=offset,
                tail_status=SpoolTailStatus.INCOMPLETE_EOF,
                tail_offset=offset,
            )
        expected_digest = blake2s(header[4:16] + payload, digest_size=16).digest()
        if header[16:32] != expected_digest:
            return DecodedSegment(
                prefix_frames=tuple(frames),
                valid_prefix_bytes=offset,
                tail_status=SpoolTailStatus.CHECKSUM_MISMATCH,
                tail_offset=offset,
            )
        try:
            payload_obj = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError):
            return DecodedSegment(
                prefix_frames=tuple(frames),
                valid_prefix_bytes=offset,
                tail_status=SpoolTailStatus.INVALID_JSON,
                tail_offset=offset,
            )
        if not _validate_payload_schema(payload_obj):
            return DecodedSegment(
                prefix_frames=tuple(frames),
                valid_prefix_bytes=offset,
                tail_status=SpoolTailStatus.INVALID_SCHEMA,
                tail_offset=offset,
            )
        frames.append(
            SessionSpoolFrame(
                record=_record_from_payload_object(payload_obj),
                frame_offset=offset,
                frame_length=frame_length,
                payload_length=payload_len,
                checksum_hex=header[16:32].hex(),
            )
        )
        offset += frame_length
    return DecodedSegment(
        prefix_frames=tuple(frames),
        valid_prefix_bytes=offset,
        tail_status=SpoolTailStatus.CLEAN,
        tail_offset=None,
    )


def decode_spool_segment(
    path: Path,
    *,
    start_offset: int = 0,
    max_file_bytes: int = TOTAL_CAP_BYTES,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> DecodedSegment:
    if not path.exists():
        return DecodedSegment(
            prefix_frames=(),
            valid_prefix_bytes=0,
            tail_status=SpoolTailStatus.CLEAN,
            tail_offset=None,
        )
    _require_existing_file(path)
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        return _decode_spool_segment_fd(
            fd,
            start_offset=start_offset,
            max_file_bytes=max_file_bytes,
            max_frame_bytes=max_frame_bytes,
        )
    finally:
        os.close(fd)


def _iter_quarantine_entries(quarantine_dir: Path):
    if _CURRENT_QUARANTINE_DIR_FD is not None:
        return list(os.scandir(_CURRENT_QUARANTINE_DIR_FD))
    return list(quarantine_dir.iterdir())


def _next_quarantine_sequence(quarantine_dir: Path) -> int:
    max_seq = 0
    for entry in _iter_quarantine_entries(quarantine_dir):
        name = entry.name if hasattr(entry, "name") else entry.name
        match = re.match(r"^(\d{6})-", name)
        if match:
            max_seq = max(max_seq, int(match.group(1)))
    return max_seq + 1


def _parse_quarantine_spool_name(name: str) -> tuple[int, str, int] | None:
    match = re.fullmatch(r"(\d{6})-([a-z0-9_]+)-vp(\d+)\.spool", name)
    if not match:
        return None
    seq = int(match.group(1))
    status = match.group(2)
    valid_prefix = int(match.group(3))
    return seq, status, valid_prefix


def _quarantine_sidecar_payload_from_file(
    quarantine_dir: Path,
    spool_name: str,
    *,
    directory_fd: int,
) -> dict[str, Any]:
    parsed = _parse_quarantine_spool_name(spool_name)
    if parsed is None:
        raise SpoolDurabilityError(
            f"invalid quarantine spool filename for reconciliation: {spool_name}"
        )
    sequence, expected_status, expected_valid_prefix = parsed
    spool_fd = os.open(spool_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    try:
        spool_stat = os.fstat(spool_fd)
        if not stat.S_ISREG(spool_stat.st_mode):
            raise SpoolPathSecurityError(
                f"quarantine evidence is not a regular file: {quarantine_dir / spool_name}"
            )
        scan = _scan_fd(spool_fd)
    finally:
        os.close(spool_fd)
    if scan.tail_status.value != expected_status or scan.valid_prefix_bytes != expected_valid_prefix:
        raise SpoolDurabilityError(
            "quarantine evidence no longer matches its durable filename metadata: "
            f"{quarantine_dir / spool_name}"
        )
    return {
        "sequence": sequence,
        "tail_status": expected_status,
        "valid_prefix_bytes": expected_valid_prefix,
        "original_size": int(spool_stat.st_size),
        "quarantined_at": float(spool_stat.st_mtime),
    }


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        try:
            chunk = os.write(fd, bytes(view[written:]))
        except InterruptedError:
            continue
        if chunk <= 0:
            raise SpoolDurabilityError("short write while appending fallback spool frame")
        written += chunk


def _write_sidecar_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    directory_fd: int | None = None,
) -> None:
    temp_name = f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    fd = -1
    temp_path = path.parent / temp_name
    if directory_fd is None:
        _require_existing_dir(path.parent)
        fd = os.open(
            str(temp_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            FILE_MODE,
        )
        dir_fd = os.open(str(path.parent), _dir_open_flags())
        cleanup_by_name = False
    else:
        dir_fd = directory_fd
        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            FILE_MODE,
            dir_fd=dir_fd,
        )
        cleanup_by_name = True
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, FILE_MODE)
        data = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        _write_all(fd, data)
        _fsync_fd(fd)
        os.close(fd)
        fd = -1
        try:
            if directory_fd is None:
                os.link(str(temp_path), str(path), follow_symlinks=False)
            else:
                os.link(temp_name, path.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False)
        except FileExistsError as exc:
            raise SpoolPathSecurityError(
                f"fallback spool sidecar destination already exists or was swapped: {path}"
            ) from exc
        _fsync_directory_fd(dir_fd, path.parent)
        if directory_fd is None:
            os.unlink(temp_path)
        else:
            os.unlink(temp_name, dir_fd=dir_fd)
        _fsync_directory_fd(dir_fd, path.parent)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            if cleanup_by_name:
                os.unlink(temp_name, dir_fd=dir_fd)
            else:
                temp_path.unlink()
        except OSError:
            pass
        raise
    finally:
        if directory_fd is None:
            os.close(dir_fd)


def _publish_bytes_file(path: Path, data: bytes, *, directory_fd: int) -> None:
    temp_name = f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    fd = -1
    try:
        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            FILE_MODE,
            dir_fd=directory_fd,
        )
        if hasattr(os, "fchmod"):
            os.fchmod(fd, FILE_MODE)
        _write_all(fd, data)
        _fsync_fd(fd)
        os.close(fd)
        fd = -1
        try:
            os.link(temp_name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
        except FileExistsError as exc:
            raise SpoolPathSecurityError(
                f"fallback spool destination already exists or was swapped: {path}"
            ) from exc
        _fsync_directory_fd(directory_fd, path.parent)
        os.unlink(temp_name, dir_fd=directory_fd)
        _fsync_directory_fd(directory_fd, path.parent)
    except BaseException:
        if fd >= 0:
            _close_fd_quietly(fd)
        try:
            os.unlink(temp_name, dir_fd=directory_fd)
        except OSError:
            pass
        raise


def _reconcile_missing_sidecars(quarantine_dir: Path, *, quarantine_fd: int) -> None:
    for entry in os.scandir(quarantine_fd):
        if entry.is_symlink():
            raise SpoolPathSecurityError(
                f"symlinked quarantine entry refused: {quarantine_dir / entry.name}"
            )
    spool_names = sorted(
        entry.name
        for entry in os.scandir(quarantine_fd)
        if entry.is_file(follow_symlinks=False) and entry.name.endswith(".spool")
    )
    for spool_name in spool_names:
        sidecar_name = f"{spool_name[:-6]}.json"
        try:
            sidecar_stat = os.stat(sidecar_name, dir_fd=quarantine_fd, follow_symlinks=False)
        except FileNotFoundError:
            sidecar_payload = _quarantine_sidecar_payload_from_file(
                quarantine_dir,
                spool_name,
                directory_fd=quarantine_fd,
            )
            _write_sidecar_json(
                quarantine_dir / sidecar_name,
                sidecar_payload,
                directory_fd=quarantine_fd,
            )
            continue
        if not stat.S_ISREG(sidecar_stat.st_mode):
            raise SpoolPathSecurityError(
                f"quarantine sidecar path is not a regular file: {quarantine_dir / sidecar_name}"
            )


def _find_quarantine_hardlink(
    quarantine_fd: int,
    *,
    target_stat: os.stat_result,
) -> str | None:
    for entry in os.scandir(quarantine_fd):
        if not entry.name.endswith(".spool") or not entry.is_file(follow_symlinks=False):
            continue
        entry_stat = entry.stat(follow_symlinks=False)
        if _same_file_stat(entry_stat, target_stat):
            return entry.name
    return None


def _quarantine_spool_bytes(quarantine_dir: Path) -> int:
    total = 0
    if _CURRENT_QUARANTINE_DIR_FD is not None:
        for entry in os.scandir(_CURRENT_QUARANTINE_DIR_FD):
            if entry.is_symlink():
                raise SpoolPathSecurityError(
                    f"symlinked quarantine entry refused: {quarantine_dir / entry.name}"
                )
            if entry.is_file(follow_symlinks=False) and entry.name.endswith(".spool"):
                total += int(entry.stat(follow_symlinks=False).st_size)
        return total
    if not quarantine_dir.exists():
        return 0
    for path in quarantine_dir.glob("*.spool"):
        _require_existing_file(path)
        total += path.stat().st_size
    return total


def _quarantine_active_file(
    active_path: Path,
    quarantine_dir: Path,
    scan_result: SpoolScanResult,
    *,
    runtime: _AnchoredRuntime,
    quarantine_fd: int,
    active_fd: int,
) -> None:
    _assert_entry_matches_fd(
        runtime.root_fd,
        QUARANTINE_DIR_NAME,
        quarantine_fd,
        expect="dir",
        label=str(quarantine_dir),
    )
    active_stat = os.fstat(active_fd)
    duplicate_name = _find_quarantine_hardlink(quarantine_fd, target_stat=active_stat)
    if duplicate_name is not None:
        sidecar_name = f"{duplicate_name[:-6]}.json"
        try:
            os.stat(sidecar_name, dir_fd=quarantine_fd, follow_symlinks=False)
        except FileNotFoundError:
            _write_sidecar_json(
                quarantine_dir / sidecar_name,
                _quarantine_sidecar_payload_from_file(
                    quarantine_dir,
                    duplicate_name,
                    directory_fd=quarantine_fd,
                ),
                directory_fd=quarantine_fd,
            )
        _assert_home_matches_fd(runtime.home_path, runtime.home_fd)
        _assert_entry_matches_fd(
            runtime.home_fd,
            SPOOL_ROOT_NAME,
            runtime.root_fd,
            expect="dir",
            label=str(runtime.root_path),
        )
        _assert_entry_matches_fd(
            runtime.root_fd,
            ACTIVE_SPOOL_NAME,
            active_fd,
            expect="file",
            label=str(active_path),
        )
        os.unlink(ACTIVE_SPOOL_NAME, dir_fd=runtime.root_fd)
        _fsync_directory_fd(runtime.root_fd, runtime.root_path)
        return

    seq = _next_quarantine_sequence(quarantine_dir)
    _assert_entry_matches_fd(
        runtime.root_fd,
        QUARANTINE_DIR_NAME,
        quarantine_fd,
        expect="dir",
        label=str(quarantine_dir),
    )
    while True:
        base = f"{seq:06d}-{scan_result.tail_status.value}-vp{scan_result.valid_prefix_bytes}"
        spool_name = f"{base}.spool"
        sidecar_path = quarantine_dir / f"{base}.json"
        try:
            os.link(
                ACTIVE_SPOOL_NAME,
                spool_name,
                src_dir_fd=runtime.root_fd,
                dst_dir_fd=quarantine_fd,
                follow_symlinks=False,
            )
            break
        except FileExistsError:
            seq += 1
            continue
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise SpoolPathSecurityError(
                    f"symlinked quarantine target refused: {quarantine_dir / spool_name}"
                ) from exc
            raise SpoolDurabilityError(
                f"unable to quarantine fallback spool evidence {quarantine_dir / spool_name}: {exc}"
            ) from exc
    _fsync_directory_fd(quarantine_fd, quarantine_dir)
    _write_sidecar_json(
        sidecar_path,
        {
            "sequence": seq,
            "tail_status": scan_result.tail_status.value,
            "valid_prefix_bytes": scan_result.valid_prefix_bytes,
            "original_size": int(active_stat.st_size),
            "quarantined_at": time.time(),
        },
        directory_fd=quarantine_fd,
    )
    _assert_home_matches_fd(runtime.home_path, runtime.home_fd)
    _assert_entry_matches_fd(
        runtime.home_fd,
        SPOOL_ROOT_NAME,
        runtime.root_fd,
        expect="dir",
        label=str(runtime.root_path),
    )
    _assert_entry_matches_fd(
        runtime.root_fd,
        ACTIVE_SPOOL_NAME,
        active_fd,
        expect="file",
        label=str(active_path),
    )
    os.unlink(ACTIVE_SPOOL_NAME, dir_fd=runtime.root_fd)
    _fsync_directory_fd(runtime.root_fd, runtime.root_path)


def _open_locked_runtime() -> _AnchoredRuntime:
    home_path = Path(get_hermes_home())
    root_path = _spool_root()
    active_path = _active_spool_path()
    quarantine_path = _quarantine_dir()
    home_fd = _open_home_dir_fd(home_path)
    root_fd = -1
    lock_fd = -1
    try:
        root_fd, _ = _open_dir_at(
            home_fd,
            SPOOL_ROOT_NAME,
            full_path=root_path,
            mode=ROOT_MODE,
            create=True,
            parent_label=home_path,
            fsync_parent_on_open_existing=True,
        )
        lock_fd, _ = _open_file_at(
            root_fd,
            LOCK_FILE_NAME,
            full_path=_lock_path(),
            mode=FILE_MODE,
            create=True,
            fsync_parent_on_create=True,
            fsync_file_on_create=False,
            parent_label=root_path,
            fsync_parent_on_open_existing=True,
        )
        return _AnchoredRuntime(
            home_path=home_path,
            root_path=root_path,
            quarantine_path=quarantine_path,
            active_path=active_path,
            home_fd=home_fd,
            root_fd=root_fd,
            lock_fd=lock_fd,
        )
    except BaseException:
        if lock_fd >= 0:
            os.close(lock_fd)
        if root_fd >= 0:
            os.close(root_fd)
        os.close(home_fd)
        raise


def _try_lock_fd_nonblocking(fd: int, label: str) -> bool:
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt is not None:  # pragma: no cover - Windows-only branch
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - unsupported platform
            raise SpoolDurabilityError("no secure file-locking primitive available")
        return True
    except BlockingIOError:
        return False
    except OSError as exc:
        if _is_lock_contention_error(exc):
            return False
        raise SpoolDurabilityError(
            f"unexpected fallback spool lock failure for {label}: {exc}"
        ) from exc


def _try_acquire_replay_owner(runtime: _AnchoredRuntime) -> _ReplayOwnerLease | None:
    owner_fd = -1
    try:
        owner_fd, _ = _open_file_at(
            runtime.root_fd,
            REPLAY_OWNER_LOCK_NAME,
            full_path=_owner_lock_path(),
            mode=FILE_MODE,
            create=True,
            fsync_parent_on_create=True,
            fsync_file_on_create=False,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )
        if not _try_lock_fd_nonblocking(owner_fd, str(_owner_lock_path())):
            os.close(owner_fd)
            return None
        return _ReplayOwnerLease(fd=owner_fd, path=_owner_lock_path())
    except BaseException:
        if owner_fd >= 0:
            _close_fd_quietly(owner_fd)
        raise


def _read_segment_highwater(*, runtime: _AnchoredRuntime, root_fd: int) -> int | None:
    try:
        highwater_fd, _ = _open_file_at(
            root_fd,
            HIGHWATER_FILE_NAME,
            full_path=_segment_sequence_highwater_path(),
            mode=FILE_MODE,
            create=False,
            fsync_parent_on_create=False,
            fsync_file_on_create=False,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=False,
        )
    except SpoolDurabilityError as exc:
        if "missing" in str(exc):
            return None
        raise
    try:
        raw = _read_exact_from_fd(highwater_fd, offset=0, length=os.fstat(highwater_fd).st_size)
    finally:
        os.close(highwater_fd)
    return _parse_highwater_payload_bytes(
        raw,
        label=_segment_sequence_highwater_path(),
    )


def _parse_highwater_payload_bytes(raw: bytes, *, label: Path | str) -> int:
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise SpoolDurabilityError(
            f"invalid fallback spool segment high-water file: {label}"
        ) from exc
    if set(payload.keys()) != {"last_reserved_sequence", "schema_version"}:
        raise SpoolDurabilityError(
            f"invalid fallback spool segment high-water schema: {label}"
        )
    if payload.get("schema_version") != 1:
        raise SpoolDurabilityError(
            f"unsupported fallback spool segment high-water version: {label}"
        )
    encoded = payload.get("last_reserved_sequence")
    if not isinstance(encoded, str) or not re.fullmatch(r"\d{20}", encoded):
        raise SpoolDurabilityError(
            f"invalid fallback spool segment high-water sequence: {label}"
        )
    return int(encoded)


def _write_segment_highwater(
    last_reserved_sequence: int,
    *,
    runtime: _AnchoredRuntime,
    root_fd: int,
) -> None:
    if last_reserved_sequence < 0 or last_reserved_sequence > MAX_SEGMENT_SEQUENCE:
        raise SpoolDurabilityError(
            f"invalid fallback spool segment sequence reservation: {last_reserved_sequence}"
        )
    temp_name = f".{HIGHWATER_FILE_NAME}.{os.getpid()}.{time.time_ns()}.tmp"
    fd = -1
    try:
        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            FILE_MODE,
            dir_fd=root_fd,
        )
        if hasattr(os, "fchmod"):
            os.fchmod(fd, FILE_MODE)
        data = json.dumps(
            {
                "last_reserved_sequence": _format_segment_sequence(last_reserved_sequence),
                "schema_version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        _write_all(fd, data)
        _fsync_fd(fd)
        os.close(fd)
        fd = -1
        _assert_home_matches_fd(runtime.home_path, runtime.home_fd)
        _assert_entry_matches_fd(
            runtime.home_fd,
            SPOOL_ROOT_NAME,
            root_fd,
            expect="dir",
            label=str(runtime.root_path),
        )
        os.replace(temp_name, HIGHWATER_FILE_NAME, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        _fsync_directory_fd(root_fd, runtime.root_path)
        _fsync_directory_fd(runtime.home_fd, runtime.home_path)
    except BaseException:
        if fd >= 0:
            _close_fd_quietly(fd)
        try:
            os.unlink(temp_name, dir_fd=root_fd)
        except OSError:
            pass
        raise


def _sealed_segment_sequences() -> list[int]:
    sealed_dir = _sealed_dir()
    if not sealed_dir.exists():
        return []
    _require_existing_dir(sealed_dir)
    sequences: list[int] = []
    for path in sealed_dir.iterdir():
        name = path.name
        if name in {ACKS_DIR_NAME, BLOCKERS_DIR_NAME}:
            continue
        match = re.fullmatch(r"(\d{20})(?:\.prefix)?\.spool", name)
        if match:
            sequences.append(int(match.group(1)))
            continue
        if name.startswith("."):
            continue
        raise SpoolDurabilityError(f"unrecognized sealed segment artifact: {path}")
    return sequences


def _parse_sealed_segment_sequence(name: str) -> int | None:
    match = re.fullmatch(r"(\d{20})(?:\.prefix)?\.spool", name)
    if not match:
        return None
    return int(match.group(1))


def _parse_ack_sidecar_sequence(name: str) -> int | None:
    match = re.fullmatch(r"(\d{20})(?:\.prefix)?\.spool\.ap\d{20}\.json", name)
    if not match:
        return None
    return int(match.group(1))


def _parse_blocker_sequence(name: str) -> int | None:
    match = re.fullmatch(r"(\d{20})\.blocker\.json", name)
    if not match:
        return None
    return int(match.group(1))


def _parse_replay_quarantine_sequence(name: str) -> int | None:
    match = re.fullmatch(r"seq-(\d{20})-[a-z0-9_]+-vp\d+\.(?:spool|json)", name)
    if not match:
        return None
    return int(match.group(1))


def _is_legacy_quarantine_name(name: str) -> bool:
    return bool(re.fullmatch(r"\d{6}-[A-Za-z0-9_]+-vp\d+\.(?:spool|json)", name))


def _parse_protocol_temp_sequence(
    name: str,
    *,
    dir_fd: int,
    label: Path | str,
) -> int | None:
    match = re.fullmatch(r"\.(.+)\.\d+\.\d+\.tmp", name)
    if not match:
        return None
    inner_name = match.group(1)
    if inner_name == HIGHWATER_FILE_NAME:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        try:
            raw = _read_exact_from_fd(fd, offset=0, length=os.fstat(fd).st_size)
        finally:
            os.close(fd)
        return _parse_highwater_payload_bytes(raw, label=label)
    for parser in (
        _parse_sealed_segment_sequence,
        _parse_ack_sidecar_sequence,
        _parse_blocker_sequence,
        _parse_replay_quarantine_sequence,
    ):
        parsed = parser(inner_name)
        if parsed is not None:
            return parsed
    raise SpoolDurabilityError(f"unrecognized protocol temp artifact: {label}")


def _open_dir_optional(parent_fd: int, name: str, *, full_path: Path) -> int | None:
    try:
        fd = os.open(name, _dir_open_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise SpoolPathSecurityError(
                f"symlinked fallback spool directory refused: {full_path}"
            ) from exc
        raise SpoolDurabilityError(
            f"unable to open fallback spool directory {full_path}: {exc}"
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode):
            raise SpoolPathSecurityError(f"fallback spool path is not a directory: {full_path}")
        if name != ".":
            _assert_entry_matches_fd(parent_fd, name, fd, expect="dir", label=str(full_path))
    except BaseException:
        _close_fd_quietly(fd)
        raise
    return fd


def _open_file_optional(parent_fd: int, name: str, *, full_path: Path) -> int | None:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
            raise SpoolPathSecurityError(f"symlinked fallback spool path refused: {full_path}") from exc
        raise SpoolDurabilityError(f"unable to open fallback spool file {full_path}: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SpoolPathSecurityError(f"fallback spool path is not a regular file: {full_path}")
        _assert_entry_matches_fd(parent_fd, name, fd, expect="file", label=str(full_path))
    except BaseException:
        _close_fd_quietly(fd)
        raise
    return fd


def _try_acquire_status_lock(fd: int) -> bool:
    deadline = time.monotonic() + STATUS_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            elif msvcrt is not None:  # pragma: no cover - Windows-only branch
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - unsupported platform
                raise _StatusInspectionError("append_lock_unavailable")
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(STATUS_LOCK_RETRY_SECONDS)
        except OSError as exc:
            if _is_lock_contention_error(exc):
                if time.monotonic() >= deadline:
                    return False
                time.sleep(STATUS_LOCK_RETRY_SECONDS)
                continue
            raise _StatusInspectionError("append_lock_error") from exc


def _release_status_lock(fd: int) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif msvcrt is not None:  # pragma: no cover - Windows-only branch
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


def _collect_sequences_from_directory(
    *,
    dir_fd: int,
    dir_path: Path,
    parsers: tuple,
    allowed_dirs: set[str],
    legacy_quarantine_ok: bool = False,
    sealed_error_label: bool = False,
    byte_accumulator: list[int] | None = None,
) -> list[int]:
    sequences: list[int] = []
    for entry in os.scandir(dir_fd):
        if entry.is_symlink():
            raise SpoolPathSecurityError(f"symlinked fallback spool path refused: {dir_path / entry.name}")
        entry_stat = entry.stat(follow_symlinks=False)
        if stat.S_ISDIR(entry_stat.st_mode):
            if entry.name in allowed_dirs:
                continue
            raise SpoolPathSecurityError(
                f"unexpected fallback spool directory encountered during sequence inventory: {dir_path / entry.name}"
            )
        if not stat.S_ISREG(entry_stat.st_mode):
            raise SpoolPathSecurityError(f"fallback spool path is not a regular file: {dir_path / entry.name}")
        for parser in parsers:
            parsed = parser(entry.name)
            if parsed is not None:
                sequences.append(parsed)
                if byte_accumulator is not None:
                    byte_accumulator[0] += entry_stat.st_size
                break
        else:
            temp_parsed = _parse_protocol_temp_sequence(
                entry.name,
                dir_fd=dir_fd,
                label=dir_path / entry.name,
            )
            if temp_parsed is not None:
                sequences.append(temp_parsed)
            elif legacy_quarantine_ok and _is_legacy_quarantine_name(entry.name):
                if byte_accumulator is not None:
                    byte_accumulator[0] += entry_stat.st_size
                continue
            elif sealed_error_label:
                raise SpoolDurabilityError(f"unrecognized sealed segment artifact: {dir_path / entry.name}")
            else:
                raise SpoolDurabilityError(f"unrecognized sequence-bearing artifact: {dir_path / entry.name}")
    return sequences


def _durable_capacity_inventory(
    runtime: _AnchoredRuntime,
    root_fd: int,
) -> _DurableCapacityInventory:
    quarantine_bytes = [0]
    other_artifact_bytes = [0]
    sealed_fd: int | None = None
    acks_fd: int | None = None
    blockers_fd: int | None = None
    quarantine_fd: int | None = None
    try:
        for entry in os.scandir(root_fd):
            label = runtime.root_path / entry.name
            if entry.is_symlink():
                raise SpoolPathSecurityError(f"symlinked fallback spool path refused: {label}")
            entry_stat = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(entry_stat.st_mode):
                if entry.name in {SEALED_DIR_NAME, QUARANTINE_DIR_NAME}:
                    continue
                raise SpoolPathSecurityError(
                    f"unexpected fallback spool directory encountered during sequence inventory: {label}"
                )
            if not stat.S_ISREG(entry_stat.st_mode):
                raise SpoolPathSecurityError(f"fallback spool path is not a regular file: {label}")
            if entry.name in {ACTIVE_SPOOL_NAME, LOCK_FILE_NAME, REPLAY_OWNER_LOCK_NAME}:
                continue
            if entry.name == HIGHWATER_FILE_NAME:
                other_artifact_bytes[0] += entry_stat.st_size
                continue
            if entry.name.startswith("."):
                temp_parsed = _parse_protocol_temp_sequence(
                    entry.name,
                    dir_fd=root_fd,
                    label=label,
                )
                if temp_parsed is not None:
                    continue
            raise SpoolDurabilityError(f"unrecognized sequence-bearing artifact: {label}")

        sealed_fd = _open_dir_optional(root_fd, SEALED_DIR_NAME, full_path=_sealed_dir())
        if sealed_fd is not None:
            _collect_sequences_from_directory(
                dir_fd=sealed_fd,
                dir_path=_sealed_dir(),
                parsers=(_parse_sealed_segment_sequence,),
                allowed_dirs={ACKS_DIR_NAME, BLOCKERS_DIR_NAME},
                sealed_error_label=True,
                byte_accumulator=other_artifact_bytes,
            )

            acks_fd = _open_dir_optional(sealed_fd, ACKS_DIR_NAME, full_path=_acks_dir())
            if acks_fd is not None:
                _collect_sequences_from_directory(
                    dir_fd=acks_fd,
                    dir_path=_acks_dir(),
                    parsers=(_parse_ack_sidecar_sequence,),
                    allowed_dirs=set(),
                    byte_accumulator=other_artifact_bytes,
                )

            blockers_fd = _open_dir_optional(sealed_fd, BLOCKERS_DIR_NAME, full_path=_blockers_dir())
            if blockers_fd is not None:
                _collect_sequences_from_directory(
                    dir_fd=blockers_fd,
                    dir_path=_blockers_dir(),
                    parsers=(_parse_blocker_sequence,),
                    allowed_dirs=set(),
                    byte_accumulator=other_artifact_bytes,
                )

        quarantine_fd = _open_dir_optional(root_fd, QUARANTINE_DIR_NAME, full_path=_quarantine_dir())
        if quarantine_fd is not None:
            _collect_sequences_from_directory(
                dir_fd=quarantine_fd,
                dir_path=_quarantine_dir(),
                parsers=(_parse_replay_quarantine_sequence,),
                allowed_dirs=set(),
                legacy_quarantine_ok=True,
                byte_accumulator=quarantine_bytes,
            )

        _assert_home_matches_fd(runtime.home_path, runtime.home_fd)
        _assert_entry_matches_fd(
            runtime.home_fd,
            SPOOL_ROOT_NAME,
            root_fd,
            expect="dir",
            label=str(runtime.root_path),
        )
        if sealed_fd is not None:
            _assert_entry_matches_fd(root_fd, SEALED_DIR_NAME, sealed_fd, expect="dir", label=str(_sealed_dir()))
        if acks_fd is not None:
            assert sealed_fd is not None
            _assert_entry_matches_fd(sealed_fd, ACKS_DIR_NAME, acks_fd, expect="dir", label=str(_acks_dir()))
        if blockers_fd is not None:
            assert sealed_fd is not None
            _assert_entry_matches_fd(
                sealed_fd,
                BLOCKERS_DIR_NAME,
                blockers_fd,
                expect="dir",
                label=str(_blockers_dir()),
            )
        if quarantine_fd is not None:
            _assert_entry_matches_fd(
                root_fd,
                QUARANTINE_DIR_NAME,
                quarantine_fd,
                expect="dir",
                label=str(_quarantine_dir()),
            )
        return _DurableCapacityInventory(
            quarantine_bytes=quarantine_bytes[0],
            other_artifact_bytes=other_artifact_bytes[0],
        )
    finally:
        if quarantine_fd is not None:
            _close_fd_quietly(quarantine_fd)
        if blockers_fd is not None:
            _close_fd_quietly(blockers_fd)
        if acks_fd is not None:
            _close_fd_quietly(acks_fd)
        if sealed_fd is not None:
            _close_fd_quietly(sealed_fd)


def _ordered_status_reasons(reasons: set[str]) -> tuple[str, ...]:
    return tuple(reason for reason in _STATUS_REASON_ORDER if reason in reasons)


def _probe_disk_from_home_fd(
    home_fd: int,
    *,
    threshold_bytes: int,
) -> tuple[int | None, int | None, str, str | None]:
    if not hasattr(os, "fstatvfs"):
        return None, None, "unknown", "disk_probe_failed"
    try:
        statvfs = os.fstatvfs(home_fd)
    except OSError:
        return None, None, "unknown", "disk_probe_failed"
    free_bytes = int(statvfs.f_bavail) * int(statvfs.f_frsize)
    total_bytes = int(statvfs.f_blocks) * int(statvfs.f_frsize)
    state = "low" if free_bytes < max(0, int(threshold_bytes)) else "ok"
    return free_bytes, total_bytes, state, None


def _classify_status_os_error(
    exc: OSError,
    *,
    default_error_class: str = "inspection_error",
) -> str:
    if isinstance(exc, FileNotFoundError):
        return "entry_replaced"
    if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
        return "symlink_refused"
    return default_error_class


def _classify_status_exception(
    exc: BaseException,
    *,
    default_error_class: str = "inspection_error",
) -> str:
    if isinstance(exc, _StatusInspectionError):
        return exc.error_class
    if isinstance(exc, SpoolPathSecurityError):
        return "symlink_refused"
    if isinstance(exc, OSError):
        return _classify_status_os_error(exc, default_error_class=default_error_class)
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, SpoolPathSecurityError):
        return "symlink_refused"
    if isinstance(cause, OSError):
        return _classify_status_os_error(cause, default_error_class=default_error_class)
    return default_error_class


def _status_iter_dir_entries(
    dir_fd: int,
    *,
    default_error_class: str = "inspection_error",
):
    try:
        entries = os.scandir(dir_fd)
    except OSError as exc:
        raise _StatusInspectionError(
            _classify_status_os_error(exc, default_error_class=default_error_class)
        ) from exc
    try:
        while True:
            try:
                entry = next(entries)
            except StopIteration:
                break
            except OSError as exc:
                raise _StatusInspectionError(
                    _classify_status_os_error(exc, default_error_class=default_error_class)
                ) from exc
            yield entry
    finally:
        close = getattr(entries, "close", None)
        if close is not None:
            close()


def _status_entry_is_symlink(
    entry,
    *,
    default_error_class: str = "inspection_error",
) -> bool:
    try:
        return entry.is_symlink()
    except OSError as exc:
        raise _StatusInspectionError(
            _classify_status_os_error(exc, default_error_class=default_error_class)
        ) from exc


def _status_entry_stat(
    entry,
    *,
    default_error_class: str = "inspection_error",
) -> os.stat_result:
    try:
        return entry.stat(follow_symlinks=False)
    except OSError as exc:
        raise _StatusInspectionError(
            _classify_status_os_error(exc, default_error_class=default_error_class)
        ) from exc


def _status_root_has_entries(root_fd: int) -> bool:
    for _entry in _status_iter_dir_entries(root_fd):
        return True
    return False


def _status_pending_prefix_metrics(
    decoded: DecodedSegment,
    *,
    acked_prefix_bytes: int,
    error_class: str,
) -> tuple[int, int]:
    if acked_prefix_bytes < 0 or acked_prefix_bytes > decoded.valid_prefix_bytes:
        raise _StatusInspectionError(error_class)
    pending_frames = 0
    pending_bytes = 0
    for frame in decoded.prefix_frames:
        frame_end = frame.frame_offset + frame.frame_length
        if frame_end <= acked_prefix_bytes:
            continue
        if frame.frame_offset < acked_prefix_bytes:
            raise _StatusInspectionError(error_class)
        pending_frames += 1
        pending_bytes += frame.frame_length
    return pending_frames, pending_bytes


def _status_assert_regular_fd_matches_entry(
    *,
    parent_fd: int,
    name: str,
    entry_stat: os.stat_result,
    fd: int,
    error_class: str,
    default_error_class: str | None = None,
) -> None:
    normalized_default_error_class = (
        error_class if default_error_class is None else default_error_class
    )
    try:
        current_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except (FileNotFoundError, OSError) as exc:
        raise _StatusInspectionError(
            _classify_status_os_error(
                exc,
                default_error_class=normalized_default_error_class,
            )
        ) from exc
    try:
        target_stat = os.fstat(fd)
    except OSError as exc:
        raise _StatusInspectionError(
            _classify_status_os_error(
                exc,
                default_error_class=normalized_default_error_class,
            )
        ) from exc
    if not stat.S_ISREG(current_stat.st_mode) or not stat.S_ISREG(target_stat.st_mode):
        raise _StatusInspectionError(error_class)
    if not _same_file_stat(entry_stat, target_stat) or not _same_file_stat(
        current_stat, target_stat
    ):
        raise _StatusInspectionError(error_class)


def _status_assert_active_fd_matches_entry(
    *,
    root_fd: int,
    entry_stat: os.stat_result,
    fd: int,
) -> None:
    _status_assert_regular_fd_matches_entry(
        parent_fd=root_fd,
        name=ACTIVE_SPOOL_NAME,
        entry_stat=entry_stat,
        fd=fd,
        error_class="entry_replaced",
        default_error_class="inspection_error",
    )


def _status_load_orphan_ack_payload(
    *,
    dir_fd: int,
    entry_name: str,
    expected_segment_name: str,
) -> tuple[int, Mapping[str, Any]]:
    parsed = _parse_ack_sidecar_name(entry_name)
    if parsed is None:
        raise SpoolDurabilityError(f"invalid ack sidecar: {entry_name}")
    _sequence, entry_segment_name, acked_prefix = parsed
    if entry_segment_name != expected_segment_name:
        raise SpoolDurabilityError(f"invalid ack sidecar: {entry_name}")
    fd = os.open(entry_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    try:
        raw = _read_exact_from_fd(fd, offset=0, length=os.fstat(fd).st_size)
    finally:
        os.close(fd)
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError):
        raise SpoolDurabilityError(f"invalid ack sidecar: {entry_name}")
    segment_size_bytes = payload.get("segment_size_bytes")
    if isinstance(segment_size_bytes, bool) or not isinstance(segment_size_bytes, int):
        raise SpoolDurabilityError(f"invalid ack sidecar: {entry_name}")
    validated = _validate_ack_payload(
        raw_bytes=raw,
        payload=payload,
        ack_name=entry_name,
        expected_segment_name=expected_segment_name,
        segment_size_bytes=int(segment_size_bytes),
    )
    return acked_prefix, validated


def _classify_status_ack_os_error(exc: OSError) -> str:
    return _classify_status_os_error(exc, default_error_class="invalid_ack_json")


def _status_scan_ack_sidecars(
    *,
    root_fd: int,
    segment_sizes: Mapping[str, int],
) -> _StatusAckDirectorySnapshot:
    sealed_fd = _open_dir_optional(root_fd, SEALED_DIR_NAME, full_path=_sealed_dir())
    if sealed_fd is None:
        return _StatusAckDirectorySnapshot(winners_by_segment={})
    acks_fd: int | None = None
    try:
        acks_fd = _open_dir_optional(sealed_fd, ACKS_DIR_NAME, full_path=_acks_dir())
        if acks_fd is None:
            return _StatusAckDirectorySnapshot(winners_by_segment={})
        winners: dict[str, Mapping[str, Any]] = {}
        orphan_winners: dict[str, tuple[int, int, Mapping[str, Any], int]] = {}
        blocked_orphan_sequence: int | None = None
        candidate_counts: dict[str, int] = {}
        for entry in _status_iter_dir_entries(
            acks_fd,
            default_error_class="invalid_ack_json",
        ):
            if _status_entry_is_symlink(
                entry,
                default_error_class="invalid_ack_json",
            ):
                raise _StatusInspectionError("symlink_refused")
            entry_stat = _status_entry_stat(
                entry,
                default_error_class="invalid_ack_json",
            )
            if stat.S_ISDIR(entry_stat.st_mode):
                raise _StatusInspectionError("unexpected_artifact")
            if not stat.S_ISREG(entry_stat.st_mode):
                raise _StatusInspectionError("unexpected_artifact")
            if entry.name.startswith("."):
                try:
                    temp_parsed = _parse_protocol_temp_sequence(
                        entry.name,
                        dir_fd=acks_fd,
                        label=_acks_dir() / entry.name,
                    )
                except OSError as exc:
                    raise _StatusInspectionError(_classify_status_ack_os_error(exc)) from exc
                except (SpoolDurabilityError, SpoolPathSecurityError) as exc:
                    raise _StatusInspectionError("unexpected_artifact") from exc
                if temp_parsed is not None:
                    continue
            parsed = _parse_ack_sidecar_name(entry.name)
            if parsed is None:
                raise _StatusInspectionError("unexpected_artifact")
            sequence, segment_name, acked_prefix = parsed
            candidate_counts[segment_name] = candidate_counts.get(segment_name, 0) + 1
            if candidate_counts[segment_name] > 64:
                raise _StatusInspectionError("invalid_ack_json")
            try:
                if segment_name in segment_sizes:
                    _acked_prefix, validated = _load_ack_payload_from_fd(
                        dir_fd=acks_fd,
                        entry_name=entry.name,
                        expected_segment_name=segment_name,
                        segment_size_bytes=segment_sizes[segment_name],
                    )
                    current_winner = winners.get(segment_name)
                    if current_winner is None or acked_prefix > int(
                        current_winner["acked_prefix_bytes"]
                    ):
                        winners[segment_name] = validated
                    continue
                _acked_prefix, validated = _status_load_orphan_ack_payload(
                    dir_fd=acks_fd,
                    entry_name=entry.name,
                    expected_segment_name=segment_name,
                )
            except OSError as exc:
                raise _StatusInspectionError(_classify_status_ack_os_error(exc)) from exc
            except SpoolDurabilityError as exc:
                raise _StatusInspectionError("invalid_ack_json") from exc
            current_orphan = orphan_winners.get(segment_name)
            if current_orphan is None or acked_prefix > current_orphan[1]:
                orphan_winners[segment_name] = (
                    sequence,
                    acked_prefix,
                    validated,
                    int(entry_stat.st_mtime_ns),
                )

        orphan_tombstone_sequence: int | None = None
        orphan_tombstone_mtime_ns: int | None = None
        for sequence, _acked_prefix, validated, mtime_ns in orphan_winners.values():
            validated_acked_prefix = int(validated["acked_prefix_bytes"])
            validated_valid_prefix = int(validated["valid_prefix_bytes"])
            segment_size_bytes = int(validated["segment_size_bytes"])
            if (
                validated_acked_prefix == validated_valid_prefix
                and validated_valid_prefix == segment_size_bytes
            ):
                if (
                    orphan_tombstone_sequence is None
                    or sequence < orphan_tombstone_sequence
                ):
                    orphan_tombstone_sequence = sequence
                    orphan_tombstone_mtime_ns = mtime_ns
                continue
            blocked_orphan_sequence = (
                sequence
                if blocked_orphan_sequence is None
                else min(blocked_orphan_sequence, sequence)
            )
        return _StatusAckDirectorySnapshot(
            winners_by_segment=winners,
            orphan_tombstone_sequence=orphan_tombstone_sequence,
            orphan_tombstone_mtime_ns=orphan_tombstone_mtime_ns,
            blocked_orphan_sequence=blocked_orphan_sequence,
        )
    finally:
        if acks_fd is not None:
            _close_fd_quietly(acks_fd)
        _close_fd_quietly(sealed_fd)


def _status_effective_acked_prefix(
    *,
    segment: _StatusSegmentSnapshot,
    ack_snapshot: _StatusAckDirectorySnapshot,
    blocker: _StatusBlockerSnapshot,
) -> tuple[int, bool]:
    winner = ack_snapshot.winners_by_segment.get(segment.name)
    acked_prefix_bytes = 0
    ack_pending = False
    if winner is not None:
        acked_prefix_bytes = int(winner["acked_prefix_bytes"])
        ack_pending = True
    if blocker.present and blocker.prefix_segment_name == segment.name and blocker.acked_prefix_bytes > 0:
        if blocker.valid_prefix_bytes != segment.size_bytes:
            raise _StatusInspectionError("invalid_blocker_json")
        acked_prefix_bytes = max(acked_prefix_bytes, blocker.acked_prefix_bytes)
        ack_pending = True
    return acked_prefix_bytes, ack_pending


def _status_count_directory_protocol_artifacts(
    *,
    dir_fd: int,
    dir_path: Path,
    parsers: tuple,
    allowed_dirs: set[str],
    limit: int,
    counted: list[int],
    legacy_quarantine_ok: bool = False,
) -> None:
    for entry in _status_iter_dir_entries(dir_fd):
        if _status_entry_is_symlink(entry):
            raise _StatusInspectionError("symlink_refused")
        entry_stat = _status_entry_stat(entry)
        if stat.S_ISDIR(entry_stat.st_mode) and entry.name in allowed_dirs:
            continue
        matched = False
        for parser in parsers:
            if parser(entry.name) is not None:
                matched = True
                break
        if not matched and entry.name.startswith("."):
            try:
                temp_parsed = _parse_protocol_temp_sequence(
                    entry.name,
                    dir_fd=dir_fd,
                    label=dir_path / entry.name,
                )
            except OSError as exc:
                raise _StatusInspectionError(
                    _classify_status_os_error(exc)
                ) from exc
            except SpoolPathSecurityError as exc:
                raise _StatusInspectionError("symlink_refused") from exc
            except SpoolDurabilityError as exc:
                raise _StatusInspectionError("unexpected_artifact") from exc
            if temp_parsed is not None:
                matched = True
        if not matched and legacy_quarantine_ok and _is_legacy_quarantine_name(entry.name):
            matched = True
        if not matched:
            raise _StatusInspectionError("unexpected_artifact")
        counted[0] += 1
        if counted[0] > limit:
            raise _StatusInspectionError("artifact_limit_exceeded")


def _status_count_protocol_artifacts(*, root_fd: int, limit: int) -> None:
    counted = [0]
    for entry in _status_iter_dir_entries(root_fd):
        label = _spool_root() / entry.name
        if _status_entry_is_symlink(entry):
            raise _StatusInspectionError("symlink_refused")
        entry_stat = _status_entry_stat(entry)
        if stat.S_ISDIR(entry_stat.st_mode):
            if entry.name in {SEALED_DIR_NAME, QUARANTINE_DIR_NAME}:
                continue
            raise _StatusInspectionError("unexpected_artifact")
        if not stat.S_ISREG(entry_stat.st_mode):
            raise _StatusInspectionError("unexpected_artifact")
        if entry.name in {LOCK_FILE_NAME, REPLAY_OWNER_LOCK_NAME}:
            continue
        if entry.name in {ACTIVE_SPOOL_NAME, HIGHWATER_FILE_NAME}:
            counted[0] += 1
            if counted[0] > limit:
                raise _StatusInspectionError("artifact_limit_exceeded")
            continue
        if entry.name.startswith("."):
            try:
                temp_parsed = _parse_protocol_temp_sequence(
                    entry.name,
                    dir_fd=root_fd,
                    label=label,
                )
            except OSError as exc:
                raise _StatusInspectionError(
                    _classify_status_os_error(exc)
                ) from exc
            except SpoolPathSecurityError as exc:
                raise _StatusInspectionError("symlink_refused") from exc
            except SpoolDurabilityError as exc:
                raise _StatusInspectionError("unexpected_artifact") from exc
            if temp_parsed is not None:
                counted[0] += 1
                if counted[0] > limit:
                    raise _StatusInspectionError("artifact_limit_exceeded")
                continue
        raise _StatusInspectionError("unexpected_artifact")

    sealed_fd = _open_dir_optional(root_fd, SEALED_DIR_NAME, full_path=_sealed_dir())
    if sealed_fd is not None:
        try:
            _status_count_directory_protocol_artifacts(
                dir_fd=sealed_fd,
                dir_path=_sealed_dir(),
                parsers=(_parse_sealed_segment_sequence,),
                allowed_dirs={ACKS_DIR_NAME, BLOCKERS_DIR_NAME},
                limit=limit,
                counted=counted,
            )
            acks_fd = _open_dir_optional(sealed_fd, ACKS_DIR_NAME, full_path=_acks_dir())
            if acks_fd is not None:
                try:
                    _status_count_directory_protocol_artifacts(
                        dir_fd=acks_fd,
                        dir_path=_acks_dir(),
                        parsers=(_parse_ack_sidecar_sequence,),
                        allowed_dirs=set(),
                        limit=limit,
                        counted=counted,
                    )
                finally:
                    _close_fd_quietly(acks_fd)
            blockers_fd = _open_dir_optional(sealed_fd, BLOCKERS_DIR_NAME, full_path=_blockers_dir())
            if blockers_fd is not None:
                try:
                    _status_count_directory_protocol_artifacts(
                        dir_fd=blockers_fd,
                        dir_path=_blockers_dir(),
                        parsers=(_parse_blocker_sequence,),
                        allowed_dirs=set(),
                        limit=limit,
                        counted=counted,
                    )
                finally:
                    _close_fd_quietly(blockers_fd)
        finally:
            _close_fd_quietly(sealed_fd)

    quarantine_fd = _open_dir_optional(root_fd, QUARANTINE_DIR_NAME, full_path=_quarantine_dir())
    if quarantine_fd is not None:
        try:
            _status_count_directory_protocol_artifacts(
                dir_fd=quarantine_fd,
                dir_path=_quarantine_dir(),
                parsers=(_parse_replay_quarantine_sequence,),
                allowed_dirs=set(),
                limit=limit,
                counted=counted,
                legacy_quarantine_ok=True,
            )
        finally:
            _close_fd_quietly(quarantine_fd)


def _status_collect_segment_snapshots(*, root_fd: int) -> list[_StatusSegmentSnapshot]:
    sealed_fd = _open_dir_optional(root_fd, SEALED_DIR_NAME, full_path=_sealed_dir())
    if sealed_fd is None:
        return []
    try:
        snapshots: list[_StatusSegmentSnapshot] = []
        for entry in _status_iter_dir_entries(sealed_fd):
            if _status_entry_is_symlink(entry):
                raise _StatusInspectionError("symlink_refused")
            entry_stat = _status_entry_stat(entry)
            if stat.S_ISDIR(entry_stat.st_mode):
                if entry.name in {ACKS_DIR_NAME, BLOCKERS_DIR_NAME}:
                    continue
                raise _StatusInspectionError("unexpected_artifact")
            if not stat.S_ISREG(entry_stat.st_mode):
                raise _StatusInspectionError("unexpected_artifact")
            if entry.name.startswith("."):
                try:
                    temp_parsed = _parse_protocol_temp_sequence(
                        entry.name,
                        dir_fd=sealed_fd,
                        label=_sealed_dir() / entry.name,
                    )
                except OSError as exc:
                    raise _StatusInspectionError(
                        _classify_status_os_error(exc)
                    ) from exc
                except SpoolPathSecurityError as exc:
                    raise _StatusInspectionError("symlink_refused") from exc
                except SpoolDurabilityError as exc:
                    raise _StatusInspectionError("unexpected_artifact") from exc
                if temp_parsed is not None:
                    continue
            match = re.fullmatch(r"(\d{20})(?:\.prefix)?\.spool", entry.name)
            if not match:
                raise _StatusInspectionError("unexpected_artifact")
            try:
                segment_fd = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=sealed_fd)
            except OSError as exc:
                raise _StatusInspectionError(_classify_status_os_error(exc)) from exc
            try:
                _status_assert_regular_fd_matches_entry(
                    parent_fd=sealed_fd,
                    name=entry.name,
                    entry_stat=entry_stat,
                    fd=segment_fd,
                    error_class="entry_replaced",
                    default_error_class="inspection_error",
                )
                decoded = _decode_spool_segment_fd(segment_fd)
                if decoded.tail_status is not SpoolTailStatus.CLEAN:
                    raise _StatusInspectionError(decoded.tail_status.value)
                _status_assert_regular_fd_matches_entry(
                    parent_fd=sealed_fd,
                    name=entry.name,
                    entry_stat=entry_stat,
                    fd=segment_fd,
                    error_class="entry_replaced",
                    default_error_class="inspection_error",
                )
                snapshots.append(
                    _StatusSegmentSnapshot(
                        sequence=int(match.group(1)),
                        name=entry.name,
                        size_bytes=decoded.valid_prefix_bytes,
                        frame_count=len(decoded.prefix_frames),
                        mtime_ns=int(entry_stat.st_mtime_ns),
                        decoded=decoded,
                    )
                )
            finally:
                _close_fd_quietly(segment_fd)
        snapshots.sort(key=lambda item: (item.sequence, item.name))
        return snapshots
    finally:
        _close_fd_quietly(sealed_fd)


def _status_load_blocker_snapshot(*, root_fd: int) -> _StatusBlockerSnapshot:
    sealed_fd = _open_dir_optional(root_fd, SEALED_DIR_NAME, full_path=_sealed_dir())
    if sealed_fd is None:
        return _StatusBlockerSnapshot()
    blockers_fd: int | None = None
    quarantine_fd: int | None = None
    prefix_fd = -1
    evidence_spool_fd = -1
    try:
        blockers_fd = _open_dir_optional(sealed_fd, BLOCKERS_DIR_NAME, full_path=_blockers_dir())
        if blockers_fd is None:
            return _StatusBlockerSnapshot()
        blocker_entries: list[tuple[int, str, os.stat_result]] = []
        for entry in _status_iter_dir_entries(blockers_fd):
            if _status_entry_is_symlink(entry):
                raise _StatusInspectionError("symlink_refused")
            entry_stat = _status_entry_stat(entry)
            if stat.S_ISDIR(entry_stat.st_mode):
                raise _StatusInspectionError("unexpected_artifact")
            if not stat.S_ISREG(entry_stat.st_mode):
                raise _StatusInspectionError("unexpected_artifact")
            if entry.name.startswith("."):
                try:
                    temp_parsed = _parse_protocol_temp_sequence(
                        entry.name,
                        dir_fd=blockers_fd,
                        label=_blockers_dir() / entry.name,
                    )
                except OSError as exc:
                    raise _StatusInspectionError(
                        _classify_status_os_error(exc)
                    ) from exc
                except SpoolPathSecurityError as exc:
                    raise _StatusInspectionError("symlink_refused") from exc
                except SpoolDurabilityError as exc:
                    raise _StatusInspectionError("unexpected_artifact") from exc
                if temp_parsed is not None:
                    continue
            sequence = _parse_blocker_sequence(entry.name)
            if sequence is None:
                raise _StatusInspectionError("unexpected_artifact")
            blocker_entries.append((sequence, entry.name, entry_stat))
        if not blocker_entries:
            return _StatusBlockerSnapshot()
        blocker_entries.sort(key=lambda item: item[0])
        sequence, entry_name, entry_stat = blocker_entries[0]
        try:
            payload = _load_canonical_json_entry(
                dir_fd=blockers_fd,
                entry_name=entry_name,
                label=_blockers_dir() / entry_name,
                invalid_message=f"invalid corruption blocker: {_blockers_dir() / entry_name}",
            )
        except (OSError, SpoolDurabilityError, SpoolPathSecurityError) as exc:
            raise _StatusInspectionError(
                _classify_status_exception(
                    exc,
                    default_error_class="invalid_blocker_json",
                )
            ) from exc
        required_fields = {
            "schema_version",
            "segment_sequence",
            "source_kind",
            "tail_status",
            "valid_prefix_bytes",
            "acked_prefix_bytes",
            "blocking_offset",
            "prefix_segment_name",
            "evidence_spool_name",
            "evidence_sidecar_name",
            "original_size_bytes",
        }
        if set(payload.keys()) != required_fields:
            raise _StatusInspectionError("invalid_blocker_json")
        if payload.get("schema_version") != 1 or payload.get("segment_sequence") != _format_segment_sequence(sequence):
            raise _StatusInspectionError("invalid_blocker_json")
        source_kind = payload.get("source_kind")
        if source_kind not in {"active", "sealed"}:
            raise _StatusInspectionError("invalid_blocker_json")
        try:
            tail_status = SpoolTailStatus(str(payload.get("tail_status")))
        except ValueError as exc:
            raise _StatusInspectionError("invalid_blocker_json") from exc
        blocking_offset = payload.get("blocking_offset")
        valid_prefix_bytes = payload.get("valid_prefix_bytes")
        acked_prefix_bytes = payload.get("acked_prefix_bytes")
        original_size_bytes = payload.get("original_size_bytes")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (
                blocking_offset,
                valid_prefix_bytes,
                acked_prefix_bytes,
                original_size_bytes,
            )
        ):
            raise _StatusInspectionError("invalid_blocker_json")
        assert isinstance(blocking_offset, int) and not isinstance(blocking_offset, bool)
        assert isinstance(valid_prefix_bytes, int) and not isinstance(valid_prefix_bytes, bool)
        assert isinstance(acked_prefix_bytes, int) and not isinstance(acked_prefix_bytes, bool)
        assert isinstance(original_size_bytes, int) and not isinstance(
            original_size_bytes, bool
        )
        blocking_offset = int(blocking_offset)
        valid_prefix_bytes = int(valid_prefix_bytes)
        acked_prefix_bytes = int(acked_prefix_bytes)
        original_size_bytes = int(original_size_bytes)
        if (
            blocking_offset < 0
            or valid_prefix_bytes < 0
            or acked_prefix_bytes < 0
            or acked_prefix_bytes > valid_prefix_bytes
            or original_size_bytes < 0
        ):
            raise _StatusInspectionError("invalid_blocker_json")
        prefix_segment_name = payload.get("prefix_segment_name")
        if prefix_segment_name is not None and not isinstance(prefix_segment_name, str):
            raise _StatusInspectionError("invalid_blocker_json")
        evidence_spool_name = payload.get("evidence_spool_name")
        evidence_sidecar_name = payload.get("evidence_sidecar_name")
        if not isinstance(evidence_spool_name, str) or not isinstance(
            evidence_sidecar_name, str
        ):
            raise _StatusInspectionError("invalid_blocker_json")
        sequence_str = _format_segment_sequence(sequence)
        expected_evidence_base = (
            f"seq-{sequence_str}-{tail_status.value}-vp{valid_prefix_bytes}"
        )
        if (
            evidence_spool_name != f"{expected_evidence_base}.spool"
            or evidence_sidecar_name != f"{expected_evidence_base}.json"
        ):
            raise _StatusInspectionError("invalid_blocker_json")

        quarantine_fd = _open_dir_optional(
            root_fd, QUARANTINE_DIR_NAME, full_path=_quarantine_dir()
        )
        if quarantine_fd is None:
            raise _StatusInspectionError("invalid_blocker_json")
        try:
            evidence_payload = _load_canonical_json_entry(
                dir_fd=quarantine_fd,
                entry_name=evidence_sidecar_name,
                label=_quarantine_dir() / evidence_sidecar_name,
                invalid_message=(
                    f"invalid replay evidence sidecar: {_quarantine_dir() / evidence_sidecar_name}"
                ),
            )
        except (OSError, SpoolDurabilityError, SpoolPathSecurityError) as exc:
            raise _StatusInspectionError(
                _classify_status_exception(
                    exc,
                    default_error_class="invalid_blocker_json",
                )
            ) from exc
        expected_evidence_payload = {
            "schema_version": 1,
            "segment_sequence": sequence_str,
            "source_kind": str(source_kind),
            "tail_status": tail_status.value,
            "valid_prefix_bytes": valid_prefix_bytes,
            "original_size_bytes": original_size_bytes,
            "evidence_spool_name": evidence_spool_name,
        }
        if dict(evidence_payload) != expected_evidence_payload:
            raise _StatusInspectionError("invalid_blocker_json")

        try:
            evidence_entry_stat = os.stat(
                evidence_spool_name,
                dir_fd=quarantine_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(evidence_entry_stat.st_mode):
                raise _StatusInspectionError("invalid_blocker_json")
            evidence_spool_fd = os.open(
                evidence_spool_name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=quarantine_fd,
            )
            _status_assert_regular_fd_matches_entry(
                parent_fd=quarantine_fd,
                name=evidence_spool_name,
                entry_stat=evidence_entry_stat,
                fd=evidence_spool_fd,
                error_class="invalid_blocker_json",
            )
            if int(os.fstat(evidence_spool_fd).st_size) != original_size_bytes:
                raise _StatusInspectionError("invalid_blocker_json")
        except (FileNotFoundError, OSError) as exc:
            raise _StatusInspectionError(
                "symlink_refused"
                if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}
                else "invalid_blocker_json"
            ) from exc
        finally:
            if evidence_spool_fd >= 0:
                _close_fd_quietly(evidence_spool_fd)
                evidence_spool_fd = -1

        if prefix_segment_name is None:
            if valid_prefix_bytes != 0 or acked_prefix_bytes != 0 or blocking_offset != 0:
                raise _StatusInspectionError("invalid_blocker_json")
            return _StatusBlockerSnapshot(
                present=True,
                sequence=sequence,
                offset=blocking_offset,
                reason_class=tail_status.value,
                source_kind=str(source_kind),
                mtime_ns=int(entry_stat.st_mtime_ns),
                acked_prefix_bytes=acked_prefix_bytes,
                valid_prefix_bytes=valid_prefix_bytes,
                prefix_segment_name=None,
                zero_prefix=True,
            )

        expected_prefix_name = f"{sequence_str}.prefix.spool"
        if (
            prefix_segment_name != expected_prefix_name
            or valid_prefix_bytes <= 0
            or blocking_offset != valid_prefix_bytes
        ):
            raise _StatusInspectionError("invalid_blocker_json")
        try:
            prefix_entry_stat = os.stat(
                prefix_segment_name,
                dir_fd=sealed_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(prefix_entry_stat.st_mode):
                raise _StatusInspectionError("invalid_blocker_json")
            prefix_fd = os.open(
                prefix_segment_name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=sealed_fd,
            )
            _status_assert_regular_fd_matches_entry(
                parent_fd=sealed_fd,
                name=prefix_segment_name,
                entry_stat=prefix_entry_stat,
                fd=prefix_fd,
                error_class="invalid_blocker_json",
            )
            if int(os.fstat(prefix_fd).st_size) != valid_prefix_bytes:
                raise _StatusInspectionError("invalid_blocker_json")
            decoded_prefix = _decode_spool_segment_fd(prefix_fd)
            if (
                decoded_prefix.tail_status is not SpoolTailStatus.CLEAN
                or decoded_prefix.valid_prefix_bytes != valid_prefix_bytes
            ):
                raise _StatusInspectionError("invalid_blocker_json")
            _status_assert_regular_fd_matches_entry(
                parent_fd=sealed_fd,
                name=prefix_segment_name,
                entry_stat=prefix_entry_stat,
                fd=prefix_fd,
                error_class="invalid_blocker_json",
            )
        except (FileNotFoundError, OSError) as exc:
            raise _StatusInspectionError(
                "symlink_refused"
                if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}
                else "invalid_blocker_json"
            ) from exc
        finally:
            if prefix_fd >= 0:
                _close_fd_quietly(prefix_fd)
                prefix_fd = -1
        return _StatusBlockerSnapshot(
            present=True,
            sequence=sequence,
            offset=blocking_offset,
            reason_class=tail_status.value,
            source_kind=str(source_kind),
            mtime_ns=int(entry_stat.st_mtime_ns),
            acked_prefix_bytes=acked_prefix_bytes,
            valid_prefix_bytes=valid_prefix_bytes,
            prefix_segment_name=prefix_segment_name,
            zero_prefix=False,
        )
    finally:
        if evidence_spool_fd >= 0:
            _close_fd_quietly(evidence_spool_fd)
        if prefix_fd >= 0:
            _close_fd_quietly(prefix_fd)
        if quarantine_fd is not None:
            _close_fd_quietly(quarantine_fd)
        if blockers_fd is not None:
            _close_fd_quietly(blockers_fd)
        _close_fd_quietly(sealed_fd)


def collect_session_fallback_spool_status(*, now: float | None = None) -> SessionFallbackSpoolStatus:
    observed_at = time.time() if now is None else float(now)
    home_path = Path(get_hermes_home())
    root_path = _spool_root()
    active_path = _active_spool_path()
    runtime: _AnchoredRuntime | None = None
    home_fd = -1
    root_fd = -1
    lock_fd = -1
    lock_held = False
    inspection_error_class: str | None = None
    reasons: set[str] = set()
    pending_units = 0
    pending_frames = 0
    pending_bytes = 0
    oldest_pending_age_seconds: float | None = None
    retry_pending = False
    retry_class = None
    cooldown_seconds = 0.0
    ack_pending = False
    blocker = _StatusBlockerSnapshot()
    capacity_used_bytes = 0
    capacity_cap_bytes = max(0, int(TOTAL_CAP_BYTES))
    capacity_remaining_bytes = capacity_cap_bytes
    capacity_state = "ok"
    disk_free_bytes: int | None = None
    disk_total_bytes: int | None = None
    disk_state = "unknown"
    root_missing = False

    try:
        try:
            home_fd = _open_home_dir_fd(home_path)
            runtime = _AnchoredRuntime(
                home_path=home_path,
                root_path=root_path,
                quarantine_path=_quarantine_dir(),
                active_path=active_path,
                home_fd=home_fd,
                root_fd=-1,
                lock_fd=-1,
            )
            maybe_root_fd = _open_dir_optional(home_fd, SPOOL_ROOT_NAME, full_path=root_path)
            if maybe_root_fd is None:
                root_missing = True
            else:
                root_fd = maybe_root_fd

                runtime = _AnchoredRuntime(
                    home_path=home_path,
                    root_path=root_path,
                    quarantine_path=_quarantine_dir(),
                    active_path=active_path,
                    home_fd=home_fd,
                    root_fd=root_fd,
                    lock_fd=-1,
                )

                root_populated = _status_root_has_entries(root_fd)
                maybe_lock_fd = _open_file_optional(root_fd, LOCK_FILE_NAME, full_path=_lock_path())
                if maybe_lock_fd is None and root_populated:
                    inspection_error_class = "missing_append_lock"
                elif maybe_lock_fd is not None:
                    lock_fd = maybe_lock_fd
                    if not _try_acquire_status_lock(lock_fd):
                        inspection_error_class = "append_lock_busy"
                    else:
                        lock_held = True

                if inspection_error_class is None:
                    _status_count_protocol_artifacts(root_fd=root_fd, limit=STATUS_SCAN_ARTIFACT_LIMIT)

                    active_fd = _open_file_optional(root_fd, ACTIVE_SPOOL_NAME, full_path=active_path)
                    active_size = 0
                    active_frames = 0
                    active_mtime_ns: int | None = None
                    try:
                        if active_fd is not None:
                            active_stat = os.fstat(active_fd)
                            _status_assert_active_fd_matches_entry(
                                root_fd=root_fd,
                                entry_stat=active_stat,
                                fd=active_fd,
                            )
                            active_size = int(active_stat.st_size)
                            active_mtime_ns = int(active_stat.st_mtime_ns)
                            active_scan = _scan_fd(active_fd)
                            _status_assert_active_fd_matches_entry(
                                root_fd=root_fd,
                                entry_stat=active_stat,
                                fd=active_fd,
                            )
                            if active_scan.tail_status is not SpoolTailStatus.CLEAN:
                                raise _StatusInspectionError(active_scan.tail_status.value)
                            active_frames = active_scan.frame_count
                    finally:
                        if active_fd is not None:
                            _close_fd_quietly(active_fd)

                    blocker = _status_load_blocker_snapshot(root_fd=root_fd)
                    segments = _status_collect_segment_snapshots(
                        root_fd=root_fd,
                    )
                    ack_snapshot = _status_scan_ack_sidecars(
                        root_fd=root_fd,
                        segment_sizes={item.name: item.size_bytes for item in segments},
                    )
                    if ack_snapshot.blocked_orphan_sequence is not None:
                        raise _StatusInspectionError("invalid_ack_json")

                    inventory = _durable_capacity_inventory(runtime, root_fd)
                    capacity_used_bytes = active_size + inventory.quarantine_bytes + inventory.other_artifact_bytes
                    capacity_remaining_bytes = max(0, capacity_cap_bytes - capacity_used_bytes)
                    if capacity_remaining_bytes == 0:
                        capacity_state = "full"
                    elif capacity_remaining_bytes < MAX_FRAME_BYTES:
                        capacity_state = "constrained"

                    ack_pending = (
                        blocker.acked_prefix_bytes > 0
                        or ack_snapshot.orphan_tombstone_sequence is not None
                    )
                    pending_frames = active_frames
                    pending_units = active_frames
                    pending_bytes = active_size
                    for item in segments:
                        acked_prefix_bytes, segment_ack_pending = _status_effective_acked_prefix(
                            segment=item,
                            ack_snapshot=ack_snapshot,
                            blocker=blocker,
                        )
                        segment_pending_frames, segment_pending_bytes = _status_pending_prefix_metrics(
                            item.decoded,
                            acked_prefix_bytes=acked_prefix_bytes,
                            error_class="invalid_ack_json",
                        )
                        pending_frames += segment_pending_frames
                        pending_units += segment_pending_frames
                        pending_bytes += segment_pending_bytes
                        ack_pending = ack_pending or segment_ack_pending

                    oldest_mtime_ns: int | None = None
                    oldest_sequence: int | None = None
                    first_segment = segments[0] if segments else None
                    blocker_uses_prefix_queue_head = (
                        blocker.present
                        and not blocker.zero_prefix
                        and blocker.sequence is not None
                        and first_segment is not None
                        and first_segment.sequence == blocker.sequence
                        and blocker.prefix_segment_name == first_segment.name
                    )
                    if (
                        blocker.present
                        and blocker.sequence is not None
                        and blocker.mtime_ns is not None
                        and not blocker_uses_prefix_queue_head
                    ):
                        oldest_sequence = blocker.sequence
                        oldest_mtime_ns = blocker.mtime_ns
                    if (
                        ack_snapshot.orphan_tombstone_sequence is not None
                        and ack_snapshot.orphan_tombstone_mtime_ns is not None
                        and (
                            oldest_sequence is None
                            or ack_snapshot.orphan_tombstone_sequence < oldest_sequence
                        )
                    ):
                        oldest_sequence = ack_snapshot.orphan_tombstone_sequence
                        oldest_mtime_ns = ack_snapshot.orphan_tombstone_mtime_ns
                    if first_segment is not None and (
                        oldest_sequence is None
                        or first_segment.sequence < oldest_sequence
                        or blocker_uses_prefix_queue_head
                    ):
                        oldest_sequence = first_segment.sequence
                        oldest_mtime_ns = first_segment.mtime_ns
                    if oldest_mtime_ns is None and active_frames > 0 and active_mtime_ns is not None:
                        oldest_mtime_ns = active_mtime_ns
                    if oldest_mtime_ns is not None:
                        oldest_pending_age_seconds = max(0.0, observed_at - (oldest_mtime_ns / 1_000_000_000))

                    cooldown_state = _REPLAY_COOLDOWNS.get(_replay_root_key(runtime))
                    if cooldown_state is not None:
                        remaining = float(cooldown_state["next_eligible"] - time.monotonic())
                        if remaining > 0:
                            retry_pending = True
                            retry_class = str(cooldown_state.get("retry_class") or "retry_pending")
                            cooldown_seconds = remaining
        except (OSError, SpoolDurabilityError, SpoolPathSecurityError, _StatusInspectionError) as exc:
            inspection_error_class = _classify_status_exception(exc)

        if root_missing and inspection_error_class is None:
            disk_free_bytes, disk_total_bytes, disk_state, _disk_error_class = _probe_disk_from_home_fd(
                home_fd,
                threshold_bytes=capacity_remaining_bytes,
            )
            return SessionFallbackSpoolStatus(
                schema_version=1,
                state="empty",
                reasons=(),
                pending_units=0,
                pending_frames=0,
                pending_bytes=0,
                oldest_pending_age_seconds=None,
                retry_pending=False,
                retry_class=None,
                cooldown_seconds=0.0,
                ack_pending=False,
                blocker_present=False,
                blocker_sequence=None,
                blocker_offset=None,
                blocker_reason_class=None,
                blocker_source_kind=None,
                capacity_used_bytes=0,
                capacity_cap_bytes=capacity_cap_bytes,
                capacity_remaining_bytes=capacity_remaining_bytes,
                capacity_state="ok",
                disk_free_bytes=disk_free_bytes,
                disk_total_bytes=disk_total_bytes,
                disk_headroom_threshold_bytes=capacity_remaining_bytes,
                disk_state=disk_state,
                inspection_error_class=None,
            )

        if pending_frames > 0 or pending_bytes > 0:
            reasons.add("pending_backlog")
        if blocker.present:
            reasons.add("blocker")
        if retry_pending:
            reasons.add("retry_cooldown")
        if ack_pending:
            reasons.add("ack_pending")
        if capacity_state == "constrained":
            reasons.add("capacity_constrained")
        elif capacity_state == "full":
            reasons.add("capacity_full")
        if home_fd >= 0:
            disk_free_bytes, disk_total_bytes, disk_state, disk_error_class = _probe_disk_from_home_fd(
                home_fd,
                threshold_bytes=capacity_remaining_bytes,
            )
            if disk_state == "low":
                reasons.add("disk_low")
            elif disk_error_class is not None and inspection_error_class is None:
                inspection_error_class = disk_error_class
        if inspection_error_class is not None:
            reasons.add("inspection_error")

        state = "degraded" if reasons else "healthy"
        blocker_present = blocker.present
        blocker_sequence = blocker.sequence
        blocker_offset = blocker.offset
        blocker_reason_class = blocker.reason_class
        blocker_source_kind = blocker.source_kind

        return SessionFallbackSpoolStatus(
            schema_version=1,
            state=state,
            reasons=_ordered_status_reasons(reasons),
            pending_units=pending_units,
            pending_frames=pending_frames,
            pending_bytes=pending_bytes,
            oldest_pending_age_seconds=oldest_pending_age_seconds,
            retry_pending=retry_pending,
            retry_class=retry_class,
            cooldown_seconds=cooldown_seconds,
            ack_pending=ack_pending,
            blocker_present=blocker_present,
            blocker_sequence=blocker_sequence,
            blocker_offset=blocker_offset,
            blocker_reason_class=blocker_reason_class,
            blocker_source_kind=blocker_source_kind,
            capacity_used_bytes=capacity_used_bytes,
            capacity_cap_bytes=capacity_cap_bytes,
            capacity_remaining_bytes=capacity_remaining_bytes,
            capacity_state=capacity_state,
            disk_free_bytes=disk_free_bytes,
            disk_total_bytes=disk_total_bytes,
            disk_headroom_threshold_bytes=capacity_remaining_bytes,
            disk_state=disk_state,
            inspection_error_class=inspection_error_class,
        )
    finally:
        if lock_held and lock_fd >= 0:
            _release_status_lock(lock_fd)
        if lock_fd >= 0:
            _close_fd_quietly(lock_fd)
        if root_fd >= 0:
            _close_fd_quietly(root_fd)
        _close_fd_quietly(home_fd)


def _collect_sequence_inventory(*, runtime: _AnchoredRuntime, root_fd: int) -> list[int]:
    sequences: list[int] = []
    highwater = _read_segment_highwater(runtime=runtime, root_fd=root_fd)
    if highwater is not None:
        sequences.append(highwater)

    for entry in os.scandir(root_fd):
        if entry.name in {
            ACTIVE_SPOOL_NAME,
            LOCK_FILE_NAME,
            REPLAY_OWNER_LOCK_NAME,
            QUARANTINE_DIR_NAME,
            SEALED_DIR_NAME,
            HIGHWATER_FILE_NAME,
        }:
            continue
        if entry.name.startswith("."):
            temp_parsed = _parse_protocol_temp_sequence(
                entry.name,
                dir_fd=root_fd,
                label=runtime.root_path / entry.name,
            )
            if temp_parsed is not None:
                sequences.append(temp_parsed)

    sealed_fd = _open_dir_optional(root_fd, SEALED_DIR_NAME, full_path=_sealed_dir())
    if sealed_fd is None:
        return sequences
    try:
        sequences.extend(
            _collect_sequences_from_directory(
                dir_fd=sealed_fd,
                dir_path=_sealed_dir(),
                parsers=(_parse_sealed_segment_sequence,),
                allowed_dirs={ACKS_DIR_NAME, BLOCKERS_DIR_NAME},
                sealed_error_label=True,
            )
        )

        acks_fd = _open_dir_optional(sealed_fd, ACKS_DIR_NAME, full_path=_acks_dir())
        if acks_fd is not None:
            try:
                sequences.extend(
                    _collect_sequences_from_directory(
                        dir_fd=acks_fd,
                        dir_path=_acks_dir(),
                        parsers=(_parse_ack_sidecar_sequence,),
                        allowed_dirs=set(),
                    )
                )
            finally:
                _close_fd_quietly(acks_fd)

        blockers_fd = _open_dir_optional(sealed_fd, BLOCKERS_DIR_NAME, full_path=_blockers_dir())
        if blockers_fd is not None:
            try:
                sequences.extend(
                    _collect_sequences_from_directory(
                        dir_fd=blockers_fd,
                        dir_path=_blockers_dir(),
                        parsers=(_parse_blocker_sequence,),
                        allowed_dirs=set(),
                    )
                )
            finally:
                _close_fd_quietly(blockers_fd)
    finally:
        _close_fd_quietly(sealed_fd)

    quarantine_fd = _open_dir_optional(root_fd, QUARANTINE_DIR_NAME, full_path=_quarantine_dir())
    if quarantine_fd is not None:
        try:
            sequences.extend(
                _collect_sequences_from_directory(
                    dir_fd=quarantine_fd,
                    dir_path=_quarantine_dir(),
                    parsers=(_parse_replay_quarantine_sequence,),
                    allowed_dirs=set(),
                    legacy_quarantine_ok=True,
                )
            )
        finally:
            _close_fd_quietly(quarantine_fd)

    return sequences


def _allocate_next_segment_sequence(*, runtime: _AnchoredRuntime, root_fd: int) -> int:
    baseline = max(_collect_sequence_inventory(runtime=runtime, root_fd=root_fd), default=0)
    if baseline >= MAX_SEGMENT_SEQUENCE:
        raise SpoolDurabilityError("segment_sequence_overflow")
    candidate = baseline + 1
    _write_segment_highwater(candidate, runtime=runtime, root_fd=root_fd)
    return candidate


def _ordered_segment_paths() -> list[tuple[int, Path]]:
    sealed_dir = _sealed_dir()
    if not sealed_dir.exists():
        return []
    _require_existing_dir(sealed_dir)
    segments: list[tuple[int, Path]] = []
    for path in sealed_dir.iterdir():
        match = re.fullmatch(r"(\d{20})(?:\.prefix)?\.spool", path.name)
        if not match:
            continue
        _require_existing_file(path)
        segments.append((int(match.group(1)), path))
    return sorted(segments, key=lambda item: (item[0], item[1].name))


def _ack_path_for_segment(segment_path: Path, acked_prefix_bytes: int) -> Path:
    return _acks_dir() / (
        f"{segment_path.name}.ap{_format_segment_sequence(acked_prefix_bytes)}.json"
    )


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _parse_ack_sidecar_name(name: str) -> tuple[int, str, int] | None:
    match = re.fullmatch(r"(\d{20}(?:\.prefix)?\.spool)\.ap(\d{20})\.json", name)
    if not match:
        return None
    segment_name = match.group(1)
    sequence_match = re.fullmatch(r"(\d{20})(?:\.prefix)?\.spool", segment_name)
    if sequence_match is None:
        return None
    return int(sequence_match.group(1)), segment_name, int(match.group(2))


def _open_acks_dir(runtime: _AnchoredRuntime, *, create: bool) -> tuple[int, int]:
    sealed_fd = _open_dir_at(
        runtime.root_fd,
        SEALED_DIR_NAME,
        full_path=_sealed_dir(),
        mode=ROOT_MODE,
        create=True,
        parent_label=runtime.root_path,
        fsync_parent_on_open_existing=True,
    )[0]
    try:
        acks_fd = _open_dir_at(
            sealed_fd,
            ACKS_DIR_NAME,
            full_path=_acks_dir(),
            mode=ROOT_MODE,
            create=create,
            parent_label=_sealed_dir(),
            fsync_parent_on_open_existing=True,
        )[0]
    except BaseException:
        _close_fd_quietly(sealed_fd)
        raise
    return sealed_fd, acks_fd


def _load_segment_stat_for_ack(runtime: _AnchoredRuntime, segment_name: str) -> tuple[int, int, int]:
    sealed_fd = -1
    segment_fd = -1
    try:
        sealed_fd = _open_dir_at(
            runtime.root_fd,
            SEALED_DIR_NAME,
            full_path=_sealed_dir(),
            mode=ROOT_MODE,
            create=True,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )[0]
        segment_fd = os.open(segment_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=sealed_fd)
        _assert_home_matches_fd(runtime.home_path, runtime.home_fd)
        _assert_entry_matches_fd(runtime.home_fd, SPOOL_ROOT_NAME, runtime.root_fd, expect="dir", label=str(runtime.root_path))
        _assert_entry_matches_fd(runtime.root_fd, SEALED_DIR_NAME, sealed_fd, expect="dir", label=str(_sealed_dir()))
        _assert_entry_matches_fd(sealed_fd, segment_name, segment_fd, expect="file", label=str(_sealed_dir() / segment_name))
        st = os.fstat(segment_fd)
        return sealed_fd, segment_fd, int(st.st_size)
    except BaseException:
        if segment_fd >= 0:
            _close_fd_quietly(segment_fd)
        if sealed_fd >= 0:
            _close_fd_quietly(sealed_fd)
        raise


def _validate_ack_payload(
    *,
    raw_bytes: bytes,
    payload: Mapping[str, Any],
    ack_name: str,
    expected_segment_name: str,
    segment_size_bytes: int,
) -> Mapping[str, Any]:
    required_fields = {
        "schema_version",
        "segment_sequence",
        "segment_name",
        "segment_kind",
        "segment_size_bytes",
        "acked_prefix_bytes",
        "valid_prefix_bytes",
        "tail_status",
        "last_frame_offset",
        "last_frame_length",
        "last_frame_checksum_hex",
    }
    if set(payload.keys()) != required_fields:
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    if payload.get("schema_version") != 1:
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    parsed_name = _parse_ack_sidecar_name(ack_name)
    if parsed_name is None:
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    sequence, segment_name, acked_prefix_from_name = parsed_name
    if payload.get("segment_sequence") != _format_segment_sequence(sequence):
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    if payload.get("segment_name") != expected_segment_name or segment_name != expected_segment_name:
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    segment_kind = payload.get("segment_kind")
    expected_kind = "prefix" if expected_segment_name.endswith(".prefix.spool") else "clean"
    if segment_kind != expected_kind:
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    if payload.get("segment_size_bytes") != segment_size_bytes:
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    acked_prefix_bytes = payload.get("acked_prefix_bytes")
    valid_prefix_bytes = payload.get("valid_prefix_bytes")
    last_frame_offset = payload.get("last_frame_offset")
    last_frame_length = payload.get("last_frame_length")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (acked_prefix_bytes, valid_prefix_bytes, last_frame_offset, last_frame_length)):
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    if acked_prefix_bytes <= 0 or acked_prefix_bytes != acked_prefix_from_name:
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    if valid_prefix_bytes < acked_prefix_bytes or valid_prefix_bytes > segment_size_bytes:
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    if last_frame_offset + last_frame_length != acked_prefix_bytes:
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    checksum_hex = payload.get("last_frame_checksum_hex")
    if not isinstance(checksum_hex, str) or not re.fullmatch(r"[0-9a-f]{32}", checksum_hex):
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    canonical_bytes = _canonical_json_bytes(payload)
    if canonical_bytes != raw_bytes:
        raise SpoolDurabilityError(f"invalid ack sidecar: {ack_name}")
    return payload


def _load_ack_payload_from_fd(
    *,
    dir_fd: int,
    entry_name: str,
    expected_segment_name: str,
    segment_size_bytes: int,
) -> tuple[int, Mapping[str, Any]]:
    parsed = _parse_ack_sidecar_name(entry_name)
    if parsed is None:
        raise SpoolDurabilityError(f"invalid ack sidecar: {entry_name}")
    _sequence, entry_segment_name, acked_prefix = parsed
    if entry_segment_name != expected_segment_name:
        raise SpoolDurabilityError(f"invalid ack sidecar: {entry_name}")
    fd = os.open(entry_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    try:
        raw = _read_exact_from_fd(fd, offset=0, length=os.fstat(fd).st_size)
    finally:
        os.close(fd)
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError):
        raise SpoolDurabilityError(f"invalid ack sidecar: {entry_name}")
    validated = _validate_ack_payload(
        raw_bytes=raw,
        payload=payload,
        ack_name=entry_name,
        expected_segment_name=expected_segment_name,
        segment_size_bytes=segment_size_bytes,
    )
    return acked_prefix, validated


def _cleanup_stale_lower_ack_sidecars(
    *,
    acks_fd: int,
    segment_name: str,
    keep_acked_prefix: int,
) -> None:
    removed_any = False
    for entry in os.scandir(acks_fd):
        parsed = _parse_ack_sidecar_name(entry.name)
        if parsed is None:
            continue
        _sequence, entry_segment_name, acked_prefix = parsed
        if entry_segment_name != segment_name or acked_prefix >= keep_acked_prefix:
            continue
        os.unlink(entry.name, dir_fd=acks_fd)
        removed_any = True
    if removed_any:
        _fsync_directory_fd(acks_fd, _acks_dir())


def _classify_ack_tombstones(*, runtime: _AnchoredRuntime, root_fd: int) -> tuple[set[int], int | None]:
    sealed_fd = -1
    acks_fd = -1
    try:
        sealed_fd = _open_dir_optional(root_fd, SEALED_DIR_NAME, full_path=_sealed_dir())
        existing_segments: set[str] = set()
        if sealed_fd is not None:
            for entry in os.scandir(sealed_fd):
                if entry.is_symlink():
                    raise SpoolPathSecurityError(
                        f"symlinked fallback spool path refused: {_sealed_dir() / entry.name}"
                    )
                match = re.fullmatch(r"\d{20}(?:\.prefix)?\.spool", entry.name)
                if match:
                    st = entry.stat(follow_symlinks=False)
                    if not stat.S_ISREG(st.st_mode):
                        raise SpoolPathSecurityError(
                            f"fallback spool path is not a regular file: {_sealed_dir() / entry.name}"
                        )
                    existing_segments.add(entry.name)
            acks_fd = _open_dir_optional(sealed_fd, ACKS_DIR_NAME, full_path=_acks_dir())
        if acks_fd is None:
            return set(), None
        tombstones: set[int] = set()
        blocked_seq: int | None = None
        for entry in os.scandir(acks_fd):
            if entry.is_symlink():
                raise SpoolDurabilityError(f"invalid ack sidecar: {entry.name}")
            st = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(st.st_mode):
                raise SpoolDurabilityError(f"invalid ack sidecar: {entry.name}")
            if st.st_size > 2048:
                raise SpoolDurabilityError(f"invalid ack sidecar: {entry.name}")
            parsed = _parse_ack_sidecar_name(entry.name)
            if parsed is None:
                continue
            sequence, segment_name, _acked = parsed
            if segment_name in existing_segments:
                continue
            fd = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=acks_fd)
            try:
                raw = _read_exact_from_fd(fd, offset=0, length=os.fstat(fd).st_size)
            finally:
                os.close(fd)
            try:
                payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
            except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError):
                raise SpoolDurabilityError(f"invalid ack sidecar: {entry.name}")
            segment_size_bytes = payload.get("segment_size_bytes")
            if isinstance(segment_size_bytes, bool) or not isinstance(segment_size_bytes, int):
                raise SpoolDurabilityError(f"invalid ack sidecar: {entry.name}")
            validated = _validate_ack_payload(
                raw_bytes=raw,
                payload=payload,
                ack_name=entry.name,
                expected_segment_name=segment_name,
                segment_size_bytes=segment_size_bytes,
            )
            acked_prefix = int(validated["acked_prefix_bytes"])
            valid_prefix = int(validated["valid_prefix_bytes"])
            if acked_prefix == valid_prefix:
                tombstones.add(sequence)
            else:
                blocked_seq = sequence if blocked_seq is None else min(blocked_seq, sequence)
        return tombstones, blocked_seq
    finally:
        if acks_fd is not None and acks_fd >= 0:
            _close_fd_quietly(acks_fd)
        if sealed_fd is not None and sealed_fd >= 0:
            _close_fd_quietly(sealed_fd)


def _load_ack_sidecar_winner(runtime: _AnchoredRuntime, *, segment_path: Path) -> Mapping[str, Any] | None:
    sealed_fd = -1
    segment_fd = -1
    acks_fd = -1
    try:
        sealed_fd, segment_fd, segment_size_bytes = _load_segment_stat_for_ack(runtime, segment_path.name)
        _, acks_fd = _open_acks_dir(runtime, create=True)
        matching_entries: list[tuple[int, Mapping[str, Any]]] = []
        count = 0
        for entry in os.scandir(acks_fd):
            if entry.is_symlink():
                raise SpoolDurabilityError(f"invalid ack sidecar: {entry.name}")
            st = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(st.st_mode):
                raise SpoolDurabilityError(f"invalid ack sidecar: {entry.name}")
            if st.st_size > 2048:
                raise SpoolDurabilityError(f"invalid ack sidecar: {entry.name}")
            parsed = _parse_ack_sidecar_name(entry.name)
            if parsed is None:
                continue
            _sequence, entry_segment_name, acked_prefix = parsed
            if entry_segment_name != segment_path.name:
                continue
            count += 1
            if count > 64:
                raise SpoolDurabilityError("too many ack sidecars")
            _acked_prefix, validated = _load_ack_payload_from_fd(
                dir_fd=acks_fd,
                entry_name=entry.name,
                expected_segment_name=segment_path.name,
                segment_size_bytes=segment_size_bytes,
            )
            matching_entries.append((acked_prefix, validated))
        if not matching_entries:
            return None
        matching_entries.sort(key=lambda item: item[0])
        return matching_entries[-1][1]
    finally:
        if acks_fd >= 0:
            _close_fd_quietly(acks_fd)
        if segment_fd >= 0:
            _close_fd_quietly(segment_fd)
        if sealed_fd >= 0:
            _close_fd_quietly(sealed_fd)


def _publish_ack_sidecar_strict(
    runtime: _AnchoredRuntime,
    *,
    segment_sequence: int,
    segment_path: Path,
    ack_payload: Mapping[str, Any],
) -> None:
    sealed_fd = -1
    segment_fd = -1
    acks_fd = -1
    try:
        sealed_fd, segment_fd, segment_size_bytes = _load_segment_stat_for_ack(runtime, segment_path.name)
        _, acks_fd = _open_acks_dir(runtime, create=True)
        ack_name = f"{segment_path.name}.ap{_format_segment_sequence(int(ack_payload['acked_prefix_bytes']))}.json"
        raw_bytes = _canonical_json_bytes(ack_payload)
        _validate_ack_payload(
            raw_bytes=raw_bytes,
            payload=ack_payload,
            ack_name=ack_name,
            expected_segment_name=segment_path.name,
            segment_size_bytes=segment_size_bytes,
        )
        temp_name = f".{ack_name}.{os.getpid()}.{time.time_ns()}.tmp"
        fd = -1
        try:
            fd = os.open(
                temp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                FILE_MODE,
                dir_fd=acks_fd,
            )
            if hasattr(os, "fchmod"):
                os.fchmod(fd, FILE_MODE)
            _write_all(fd, raw_bytes)
            _fsync_fd(fd)
            os.close(fd)
            fd = -1
            try:
                os.link(temp_name, ack_name, src_dir_fd=acks_fd, dst_dir_fd=acks_fd, follow_symlinks=False)
            except FileExistsError:
                existing_fd = os.open(ack_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=acks_fd)
                try:
                    existing_bytes = _read_exact_from_fd(existing_fd, offset=0, length=os.fstat(existing_fd).st_size)
                finally:
                    os.close(existing_fd)
                if existing_bytes != raw_bytes:
                    raise SpoolDurabilityError(f"conflicting ack sidecar already exists: {_acks_dir() / ack_name}")
                os.unlink(temp_name, dir_fd=acks_fd)
                try:
                    _cleanup_stale_lower_ack_sidecars(
                        acks_fd=acks_fd,
                        segment_name=segment_path.name,
                        keep_acked_prefix=int(ack_payload["acked_prefix_bytes"]),
                    )
                except OSError as exc:
                    if _is_retryable_replay_os_error(exc):
                        raise SpoolRetryableReplayError(
                            "ack_cleanup_busy",
                            ack_pending=True,
                        ) from exc
                    raise
                return
            _fsync_directory_fd(acks_fd, _acks_dir())
            os.unlink(temp_name, dir_fd=acks_fd)
            _fsync_directory_fd(acks_fd, _acks_dir())
            try:
                _cleanup_stale_lower_ack_sidecars(
                    acks_fd=acks_fd,
                    segment_name=segment_path.name,
                    keep_acked_prefix=int(ack_payload["acked_prefix_bytes"]),
                )
            except OSError as exc:
                if _is_retryable_replay_os_error(exc):
                    raise SpoolRetryableReplayError(
                        "ack_cleanup_busy",
                        ack_pending=True,
                    ) from exc
                raise
        except BaseException:
            if fd >= 0:
                _close_fd_quietly(fd)
            try:
                os.unlink(temp_name, dir_fd=acks_fd)
            except OSError:
                pass
            raise
    finally:
        if acks_fd >= 0:
            _close_fd_quietly(acks_fd)
        if segment_fd >= 0:
            _close_fd_quietly(segment_fd)
        if sealed_fd >= 0:
            _close_fd_quietly(sealed_fd)


def _publish_ack_sidecar(
    runtime: _AnchoredRuntime,
    *,
    segment_sequence: int,
    segment_path: Path,
    decoded_segment: DecodedSegment,
    frame: SessionSpoolFrame,
) -> bool:
    acked_prefix_bytes = frame.frame_offset + frame.frame_length
    payload = {
        "schema_version": 1,
        "segment_sequence": _format_segment_sequence(segment_sequence),
        "segment_name": segment_path.name,
        "segment_kind": "prefix" if segment_path.name.endswith(".prefix.spool") else "clean",
        "segment_size_bytes": int(decoded_segment.valid_prefix_bytes),
        "acked_prefix_bytes": acked_prefix_bytes,
        "valid_prefix_bytes": decoded_segment.valid_prefix_bytes,
        "tail_status": decoded_segment.tail_status.value,
        "last_frame_offset": frame.frame_offset,
        "last_frame_length": frame.frame_length,
        "last_frame_checksum_hex": frame.checksum_hex,
    }
    _publish_ack_sidecar_strict(
        runtime,
        segment_sequence=segment_sequence,
        segment_path=segment_path,
        ack_payload=payload,
    )
    return True


def _delete_fully_acked_segment(segment_path: Path) -> None:
    runtime = _open_locked_runtime()
    sealed_fd = -1
    acks_fd = -1
    try:
        sealed_fd = _open_dir_at(
            runtime.root_fd,
            SEALED_DIR_NAME,
            full_path=_sealed_dir(),
            mode=ROOT_MODE,
            create=True,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )[0]
        acks_fd = _open_dir_at(
            sealed_fd,
            ACKS_DIR_NAME,
            full_path=_acks_dir(),
            mode=ROOT_MODE,
            create=True,
            parent_label=_sealed_dir(),
            fsync_parent_on_open_existing=True,
        )[0]
        try:
            os.unlink(segment_path.name, dir_fd=sealed_fd)
        except FileNotFoundError:
            pass
        _fsync_directory_fd(sealed_fd, _sealed_dir())
        removed_any = False
        for entry in os.scandir(acks_fd):
            parsed = _parse_ack_sidecar_name(entry.name)
            if parsed is None:
                continue
            _sequence, entry_segment_name, _acked = parsed
            if entry_segment_name != segment_path.name:
                continue
            os.unlink(entry.name, dir_fd=acks_fd)
            removed_any = True
        if removed_any:
            _fsync_directory_fd(acks_fd, _acks_dir())
    finally:
        if acks_fd >= 0:
            _close_fd_quietly(acks_fd)
        if sealed_fd >= 0:
            _close_fd_quietly(sealed_fd)
        _close_fd_quietly(runtime.lock_fd)
        _close_fd_quietly(runtime.root_fd)
        _close_fd_quietly(runtime.home_fd)


def _ordered_segment_entries(*, runtime: _AnchoredRuntime, root_fd: int) -> list[tuple[int, str, Path]]:
    sealed_fd = _open_dir_optional(root_fd, SEALED_DIR_NAME, full_path=_sealed_dir())
    if sealed_fd is None:
        return []
    try:
        entries: list[tuple[int, str, Path]] = []
        for entry in os.scandir(sealed_fd):
            if entry.is_symlink():
                raise SpoolPathSecurityError(
                    f"symlinked fallback spool path refused: {_sealed_dir() / entry.name}"
                )
            match = re.fullmatch(r"(\d{20})(?:\.prefix)?\.spool", entry.name)
            if not match:
                continue
            st = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(st.st_mode):
                raise SpoolPathSecurityError(
                    f"fallback spool path is not a regular file: {_sealed_dir() / entry.name}"
                )
            entries.append((int(match.group(1)), entry.name, _sealed_dir() / entry.name))
        return sorted(entries, key=lambda item: (item[0], item[1]))
    finally:
        _close_fd_quietly(sealed_fd)


def _first_blocker_sequence(*, runtime: _AnchoredRuntime, root_fd: int) -> int | None:
    sealed_fd = _open_dir_optional(root_fd, SEALED_DIR_NAME, full_path=_sealed_dir())
    if sealed_fd is None:
        return None
    blockers_fd: int | None = None
    try:
        blockers_fd = _open_dir_optional(sealed_fd, BLOCKERS_DIR_NAME, full_path=_blockers_dir())
        if blockers_fd is None:
            return None
        sequences = _collect_sequences_from_directory(
            dir_fd=blockers_fd,
            dir_path=_blockers_dir(),
            parsers=(_parse_blocker_sequence,),
            allowed_dirs=set(),
        )
        return min(sequences) if sequences else None
    finally:
        if blockers_fd is not None:
            _close_fd_quietly(blockers_fd)
        _close_fd_quietly(sealed_fd)


def _load_canonical_json_entry(
    *,
    dir_fd: int,
    entry_name: str,
    label: Path,
    invalid_message: str,
) -> Mapping[str, Any]:
    fd = os.open(entry_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SpoolPathSecurityError(f"fallback spool path is not a regular file: {label}")
        raw = _read_exact_from_fd(fd, offset=0, length=st.st_size)
    finally:
        os.close(fd)
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise SpoolDurabilityError(invalid_message) from exc
    if not isinstance(payload, Mapping):
        raise SpoolDurabilityError(invalid_message)
    if _canonical_json_bytes(payload) != raw:
        raise SpoolDurabilityError(invalid_message)
    return payload


def _classify_blocker_crash_state_error(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        missing_name = Path(getattr(exc, "filename", "") or "").name
        if missing_name.startswith("seq-") and missing_name.endswith(".json"):
            return "missing_replay_evidence_sidecar"
        if missing_name.startswith("seq-") and missing_name.endswith(".spool"):
            return "missing_replay_evidence_spool"
        return "invalid_blocker_relationship"
    if isinstance(exc, SpoolDurabilityError):
        message = str(exc)
        if message.startswith("invalid replay evidence sidecar:"):
            return "invalid_replay_evidence_sidecar"
        return "invalid_blocker_relationship"
    if isinstance(exc, SpoolPathSecurityError):
        return "invalid_blocker_relationship"
    raise TypeError(f"unsupported blocker crash-state exception: {type(exc).__name__}")


def _load_blocker_backed_prefix_replay_state(
    *,
    runtime: _AnchoredRuntime,
    root_fd: int,
    blocker_sequence: int,
) -> _BlockerBackedPrefixReplayState | None:
    sequence_str = _format_segment_sequence(blocker_sequence)
    sealed_fd = _open_dir_optional(root_fd, SEALED_DIR_NAME, full_path=_sealed_dir())
    if sealed_fd is None:
        return None
    blockers_fd: int | None = None
    quarantine_fd: int | None = None
    prefix_segment_fd = -1
    try:
        blockers_fd = _open_dir_optional(sealed_fd, BLOCKERS_DIR_NAME, full_path=_blockers_dir())
        if blockers_fd is None:
            return None
        blocker_name = f"{sequence_str}.blocker.json"
        blocker_path = _blockers_dir() / blocker_name
        blocker_payload = _load_canonical_json_entry(
            dir_fd=blockers_fd,
            entry_name=blocker_name,
            label=blocker_path,
            invalid_message=f"invalid corruption blocker: {blocker_path}",
        )
        required_fields = {
            "schema_version",
            "segment_sequence",
            "source_kind",
            "tail_status",
            "valid_prefix_bytes",
            "acked_prefix_bytes",
            "blocking_offset",
            "prefix_segment_name",
            "evidence_spool_name",
            "evidence_sidecar_name",
            "original_size_bytes",
        }
        if set(blocker_payload.keys()) != required_fields:
            raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=0)
        if blocker_payload.get("schema_version") != 1:
            raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=0)
        if blocker_payload.get("segment_sequence") != sequence_str:
            raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=0)
        source_kind = blocker_payload.get("source_kind")
        if source_kind not in {"active", "sealed"}:
            raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=0)
        try:
            tail_status = SpoolTailStatus(str(blocker_payload.get("tail_status")))
        except ValueError as exc:
            raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=0) from exc
        valid_prefix_bytes = blocker_payload.get("valid_prefix_bytes")
        acked_prefix_bytes = blocker_payload.get("acked_prefix_bytes")
        blocking_offset = blocker_payload.get("blocking_offset")
        original_size_bytes = blocker_payload.get("original_size_bytes")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (valid_prefix_bytes, acked_prefix_bytes, blocking_offset, original_size_bytes)
        ):
            raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=0)
        assert valid_prefix_bytes is not None
        assert acked_prefix_bytes is not None
        assert blocking_offset is not None
        assert original_size_bytes is not None
        valid_prefix_bytes = int(valid_prefix_bytes)
        acked_prefix_bytes = int(acked_prefix_bytes)
        blocking_offset = int(blocking_offset)
        original_size_bytes = int(original_size_bytes)
        if valid_prefix_bytes < 0 or acked_prefix_bytes < 0 or acked_prefix_bytes > valid_prefix_bytes:
            raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)
        prefix_segment_name = blocker_payload.get("prefix_segment_name")
        if prefix_segment_name is None:
            if valid_prefix_bytes != 0 or acked_prefix_bytes != 0 or blocking_offset != 0:
                raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)
            return None
        if not isinstance(prefix_segment_name, str):
            raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)
        expected_prefix_name = f"{sequence_str}.prefix.spool"
        if prefix_segment_name != expected_prefix_name or valid_prefix_bytes <= 0 or blocking_offset != valid_prefix_bytes:
            raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)
        try:
            prefix_segment_path = _sealed_dir() / prefix_segment_name
            try:
                prefix_segment_fd = os.open(prefix_segment_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=sealed_fd)
            except FileNotFoundError:
                if acked_prefix_bytes == valid_prefix_bytes:
                    return None
                raise
            prefix_stat = os.fstat(prefix_segment_fd)
            if not stat.S_ISREG(prefix_stat.st_mode):
                raise SpoolPathSecurityError(f"fallback spool path is not a regular file: {prefix_segment_path}")
            if int(prefix_stat.st_size) != valid_prefix_bytes:
                raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)

            evidence_spool_name = blocker_payload.get("evidence_spool_name")
            evidence_sidecar_name = blocker_payload.get("evidence_sidecar_name")
            if not isinstance(evidence_spool_name, str) or not isinstance(evidence_sidecar_name, str):
                raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)
            expected_evidence_base = f"seq-{sequence_str}-{tail_status.value}-vp{valid_prefix_bytes}"
            if (
                evidence_spool_name != f"{expected_evidence_base}.spool"
                or evidence_sidecar_name != f"{expected_evidence_base}.json"
            ):
                raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)

            quarantine_fd = _open_dir_optional(root_fd, QUARANTINE_DIR_NAME, full_path=_quarantine_dir())
            if quarantine_fd is None:
                raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)
            evidence_payload = _load_canonical_json_entry(
                dir_fd=quarantine_fd,
                entry_name=evidence_sidecar_name,
                label=_quarantine_dir() / evidence_sidecar_name,
                invalid_message=f"invalid replay evidence sidecar: {_quarantine_dir() / evidence_sidecar_name}",
            )
            expected_evidence_payload = {
                "schema_version": 1,
                "segment_sequence": sequence_str,
                "source_kind": source_kind,
                "tail_status": tail_status.value,
                "valid_prefix_bytes": valid_prefix_bytes,
                "original_size_bytes": original_size_bytes,
                "evidence_spool_name": evidence_spool_name,
            }
            if dict(evidence_payload) != expected_evidence_payload:
                raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)
            evidence_spool_fd = os.open(evidence_spool_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=quarantine_fd)
            try:
                evidence_stat = os.fstat(evidence_spool_fd)
                if not stat.S_ISREG(evidence_stat.st_mode):
                    raise SpoolPathSecurityError(
                        f"fallback spool path is not a regular file: {_quarantine_dir() / evidence_spool_name}"
                    )
            finally:
                os.close(evidence_spool_fd)
            if int(evidence_stat.st_size) != original_size_bytes:
                raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)

            os.close(prefix_segment_fd)
            prefix_segment_fd = -1
            decoded_prefix = decode_spool_segment(prefix_segment_path)
            if (
                decoded_prefix.tail_status is not SpoolTailStatus.CLEAN
                or decoded_prefix.valid_prefix_bytes != valid_prefix_bytes
            ):
                raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)

            winner = _load_ack_sidecar_winner(runtime, segment_path=prefix_segment_path)
            effective_acked_prefix = acked_prefix_bytes
            if winner is not None:
                if winner.get("tail_status") != tail_status.value:
                    raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)
                if winner.get("valid_prefix_bytes") != valid_prefix_bytes:
                    raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)
                winner_acked_prefix = winner.get("acked_prefix_bytes")
                if not isinstance(winner_acked_prefix, int) or isinstance(winner_acked_prefix, bool):
                    raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)
                if winner_acked_prefix > valid_prefix_bytes:
                    raise SpoolBlockedReplayError("invalid_blocker_relationship", frame_offset=blocking_offset)
                effective_acked_prefix = winner_acked_prefix

            return _BlockerBackedPrefixReplayState(
                segment_sequence=blocker_sequence,
                blocking_offset=blocking_offset,
                tail_status=tail_status,
                prefix_segment_name=prefix_segment_name,
                prefix_segment_path=prefix_segment_path,
                valid_prefix_bytes=valid_prefix_bytes,
                acked_prefix_bytes=effective_acked_prefix,
            )
        except (FileNotFoundError, SpoolDurabilityError, SpoolPathSecurityError) as exc:
            raise SpoolBlockedReplayError(
                _classify_blocker_crash_state_error(exc),
                frame_offset=blocking_offset,
            ) from exc
    finally:
        if prefix_segment_fd >= 0:
            _close_fd_quietly(prefix_segment_fd)
        if quarantine_fd is not None:
            _close_fd_quietly(quarantine_fd)
        if blockers_fd is not None:
            _close_fd_quietly(blockers_fd)
        _close_fd_quietly(sealed_fd)


def _publish_corrupt_sealed_segment_state(
    runtime: _AnchoredRuntime,
    *,
    segment_sequence: int,
    segment_name: str,
    decoded_segment: DecodedSegment,
) -> None:
    sealed_fd = -1
    blockers_fd = -1
    quarantine_fd = -1
    segment_fd = -1
    try:
        sealed_fd = _open_dir_at(
            runtime.root_fd,
            SEALED_DIR_NAME,
            full_path=_sealed_dir(),
            mode=ROOT_MODE,
            create=True,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )[0]
        blockers_fd = _open_dir_at(
            sealed_fd,
            BLOCKERS_DIR_NAME,
            full_path=_blockers_dir(),
            mode=ROOT_MODE,
            create=True,
            parent_label=_sealed_dir(),
            fsync_parent_on_open_existing=True,
        )[0]
        quarantine_fd = _open_dir_at(
            runtime.root_fd,
            QUARANTINE_DIR_NAME,
            full_path=_quarantine_dir(),
            mode=ROOT_MODE,
            create=True,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )[0]
        segment_fd = os.open(segment_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=sealed_fd)
        _assert_home_matches_fd(runtime.home_path, runtime.home_fd)
        _assert_entry_matches_fd(runtime.home_fd, SPOOL_ROOT_NAME, runtime.root_fd, expect="dir", label=str(runtime.root_path))
        _assert_entry_matches_fd(runtime.root_fd, SEALED_DIR_NAME, sealed_fd, expect="dir", label=str(_sealed_dir()))
        _assert_entry_matches_fd(sealed_fd, segment_name, segment_fd, expect="file", label=str(_sealed_dir() / segment_name))
        _assert_entry_matches_fd(sealed_fd, BLOCKERS_DIR_NAME, blockers_fd, expect="dir", label=str(_blockers_dir()))
        _assert_entry_matches_fd(runtime.root_fd, QUARANTINE_DIR_NAME, quarantine_fd, expect="dir", label=str(_quarantine_dir()))

        segment_stat = os.fstat(segment_fd)
        original_bytes = _read_exact_from_fd(segment_fd, offset=0, length=segment_stat.st_size)
        sequence_str = _format_segment_sequence(segment_sequence)
        evidence_base = f"seq-{sequence_str}-{decoded_segment.tail_status.value}-vp{decoded_segment.valid_prefix_bytes}"
        evidence_spool_name = f"{evidence_base}.spool"
        evidence_sidecar_name = f"{evidence_base}.json"
        prefix_segment_name = (
            f"{sequence_str}.prefix.spool" if decoded_segment.valid_prefix_bytes > 0 else None
        )

        if prefix_segment_name is not None:
            prefix_bytes = original_bytes[: decoded_segment.valid_prefix_bytes]
            _publish_bytes_file(_sealed_dir() / prefix_segment_name, prefix_bytes, directory_fd=sealed_fd)
            _fsync_directory_fd(sealed_fd, _sealed_dir())
            _fsync_directory_fd(runtime.root_fd, runtime.root_path)
            _fsync_directory_fd(runtime.home_fd, runtime.home_path)

        try:
            os.link(
                segment_name,
                evidence_spool_name,
                src_dir_fd=sealed_fd,
                dst_dir_fd=quarantine_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise SpoolPathSecurityError(
                f"fallback spool evidence destination already exists or was swapped: {_quarantine_dir() / evidence_spool_name}"
            ) from exc
        _fsync_directory_fd(quarantine_fd, _quarantine_dir())
        _write_sidecar_json(
            _quarantine_dir() / evidence_sidecar_name,
            {
                "schema_version": 1,
                "segment_sequence": sequence_str,
                "source_kind": "sealed",
                "tail_status": decoded_segment.tail_status.value,
                "valid_prefix_bytes": decoded_segment.valid_prefix_bytes,
                "original_size_bytes": int(segment_stat.st_size),
                "evidence_spool_name": evidence_spool_name,
            },
            directory_fd=quarantine_fd,
        )
        _fsync_directory_fd(quarantine_fd, _quarantine_dir())
        _fsync_directory_fd(runtime.root_fd, runtime.root_path)
        _fsync_directory_fd(runtime.home_fd, runtime.home_path)
        _write_sidecar_json(
            _blockers_dir() / f"{sequence_str}.blocker.json",
            {
                "schema_version": 1,
                "segment_sequence": sequence_str,
                "source_kind": "sealed",
                "tail_status": decoded_segment.tail_status.value,
                "valid_prefix_bytes": decoded_segment.valid_prefix_bytes,
                "acked_prefix_bytes": 0,
                "blocking_offset": decoded_segment.valid_prefix_bytes,
                "prefix_segment_name": prefix_segment_name,
                "evidence_spool_name": evidence_spool_name,
                "evidence_sidecar_name": evidence_sidecar_name,
                "original_size_bytes": int(segment_stat.st_size),
            },
            directory_fd=blockers_fd,
        )
        _fsync_directory_fd(blockers_fd, _blockers_dir())
        _fsync_directory_fd(sealed_fd, _sealed_dir())
        _fsync_directory_fd(runtime.root_fd, runtime.root_path)
        _fsync_directory_fd(runtime.home_fd, runtime.home_path)
        os.unlink(segment_name, dir_fd=sealed_fd)
        _fsync_directory_fd(sealed_fd, _sealed_dir())
        _fsync_directory_fd(runtime.root_fd, runtime.root_path)
        _fsync_directory_fd(runtime.home_fd, runtime.home_path)
    finally:
        if segment_fd >= 0:
            _close_fd_quietly(segment_fd)
        if quarantine_fd >= 0:
            _close_fd_quietly(quarantine_fd)
        if blockers_fd >= 0:
            _close_fd_quietly(blockers_fd)
        if sealed_fd >= 0:
            _close_fd_quietly(sealed_fd)


def _reconcile_active_spool_for_replay(runtime: _AnchoredRuntime) -> dict[str, Any] | None:
    sealed_fd = -1
    blockers_fd = -1
    quarantine_fd = -1
    active_fd = -1
    try:
        sealed_fd, _ = _open_dir_at(
            runtime.root_fd,
            SEALED_DIR_NAME,
            full_path=_sealed_dir(),
            mode=ROOT_MODE,
            create=True,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )
        blockers_fd, _ = _open_dir_at(
            sealed_fd,
            BLOCKERS_DIR_NAME,
            full_path=_blockers_dir(),
            mode=ROOT_MODE,
            create=True,
            parent_label=_sealed_dir(),
            fsync_parent_on_open_existing=True,
        )
        quarantine_fd, _ = _open_dir_at(
            runtime.root_fd,
            QUARANTINE_DIR_NAME,
            full_path=_quarantine_dir(),
            mode=ROOT_MODE,
            create=True,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )
        active_fd, _ = _open_file_at(
            runtime.root_fd,
            ACTIVE_SPOOL_NAME,
            full_path=runtime.active_path,
            mode=FILE_MODE,
            create=True,
            fsync_parent_on_create=True,
            fsync_file_on_create=True,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )
        active_stat = os.fstat(active_fd)
        if active_stat.st_size <= 0:
            return None
        scan = _scan_fd(active_fd)
        if scan.tail_status is SpoolTailStatus.CLEAN:
            return None

        sequence = _allocate_next_segment_sequence(runtime=runtime, root_fd=runtime.root_fd)
        sequence_str = _format_segment_sequence(sequence)
        evidence_base = f"seq-{sequence_str}-{scan.tail_status.value}-vp{scan.valid_prefix_bytes}"
        evidence_spool_name = f"{evidence_base}.spool"
        evidence_sidecar_name = f"{evidence_base}.json"
        prefix_segment_name = (
            f"{sequence_str}.prefix.spool" if scan.valid_prefix_bytes > 0 else None
        )

        _assert_home_matches_fd(runtime.home_path, runtime.home_fd)
        _assert_entry_matches_fd(runtime.home_fd, SPOOL_ROOT_NAME, runtime.root_fd, expect="dir", label=str(runtime.root_path))
        _assert_entry_matches_fd(runtime.root_fd, SEALED_DIR_NAME, sealed_fd, expect="dir", label=str(_sealed_dir()))
        _assert_entry_matches_fd(sealed_fd, BLOCKERS_DIR_NAME, blockers_fd, expect="dir", label=str(_blockers_dir()))
        _assert_entry_matches_fd(runtime.root_fd, QUARANTINE_DIR_NAME, quarantine_fd, expect="dir", label=str(_quarantine_dir()))
        _assert_entry_matches_fd(runtime.root_fd, ACTIVE_SPOOL_NAME, active_fd, expect="file", label=str(runtime.active_path))

        if prefix_segment_name is not None:
            prefix_bytes = _read_exact_from_fd(active_fd, offset=0, length=scan.valid_prefix_bytes)
            _publish_bytes_file(_sealed_dir() / prefix_segment_name, prefix_bytes, directory_fd=sealed_fd)
            _fsync_directory_fd(sealed_fd, _sealed_dir())
            _fsync_directory_fd(runtime.root_fd, runtime.root_path)
            _fsync_directory_fd(runtime.home_fd, runtime.home_path)

        try:
            os.link(
                ACTIVE_SPOOL_NAME,
                evidence_spool_name,
                src_dir_fd=runtime.root_fd,
                dst_dir_fd=quarantine_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise SpoolPathSecurityError(
                f"fallback spool evidence destination already exists or was swapped: {_quarantine_dir() / evidence_spool_name}"
            ) from exc
        _fsync_directory_fd(quarantine_fd, _quarantine_dir())
        _write_sidecar_json(
            _quarantine_dir() / evidence_sidecar_name,
            {
                "schema_version": 1,
                "segment_sequence": sequence_str,
                "source_kind": "active",
                "tail_status": scan.tail_status.value,
                "valid_prefix_bytes": scan.valid_prefix_bytes,
                "original_size_bytes": int(active_stat.st_size),
                "evidence_spool_name": evidence_spool_name,
            },
            directory_fd=quarantine_fd,
        )
        _fsync_directory_fd(quarantine_fd, _quarantine_dir())
        _fsync_directory_fd(runtime.root_fd, runtime.root_path)
        _fsync_directory_fd(runtime.home_fd, runtime.home_path)

        blocker_payload = {
            "schema_version": 1,
            "segment_sequence": sequence_str,
            "source_kind": "active",
            "tail_status": scan.tail_status.value,
            "valid_prefix_bytes": scan.valid_prefix_bytes,
            "acked_prefix_bytes": 0,
            "blocking_offset": scan.valid_prefix_bytes,
            "prefix_segment_name": prefix_segment_name,
            "evidence_spool_name": evidence_spool_name,
            "evidence_sidecar_name": evidence_sidecar_name,
            "original_size_bytes": int(active_stat.st_size),
        }
        _write_sidecar_json(
            _blockers_dir() / f"{sequence_str}.blocker.json",
            blocker_payload,
            directory_fd=blockers_fd,
        )
        _fsync_directory_fd(blockers_fd, _blockers_dir())
        _fsync_directory_fd(sealed_fd, _sealed_dir())
        _fsync_directory_fd(runtime.root_fd, runtime.root_path)
        _fsync_directory_fd(runtime.home_fd, runtime.home_path)

        _assert_home_matches_fd(runtime.home_path, runtime.home_fd)
        _assert_entry_matches_fd(runtime.home_fd, SPOOL_ROOT_NAME, runtime.root_fd, expect="dir", label=str(runtime.root_path))
        _assert_entry_matches_fd(runtime.root_fd, ACTIVE_SPOOL_NAME, active_fd, expect="file", label=str(runtime.active_path))
        os.unlink(ACTIVE_SPOOL_NAME, dir_fd=runtime.root_fd)
        _fsync_directory_fd(runtime.root_fd, runtime.root_path)
        os.close(active_fd)
        active_fd = -1
        active_fd, _ = _open_file_at(
            runtime.root_fd,
            ACTIVE_SPOOL_NAME,
            full_path=runtime.active_path,
            mode=FILE_MODE,
            create=True,
            fsync_parent_on_create=True,
            fsync_file_on_create=True,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )
        _fsync_fd(active_fd)
        _fsync_directory_fd(runtime.root_fd, runtime.root_path)
        _fsync_directory_fd(runtime.home_fd, runtime.home_path)

        return {
            "tail_status": scan.tail_status,
            "valid_prefix_bytes": scan.valid_prefix_bytes,
            "segment_sequence": sequence,
            "prefix_segment_name": prefix_segment_name,
            "evidence_spool_name": evidence_spool_name,
            "evidence_sidecar_name": evidence_sidecar_name,
        }
    finally:
        if active_fd >= 0:
            _close_fd_quietly(active_fd)
        if quarantine_fd >= 0:
            _close_fd_quietly(quarantine_fd)
        if blockers_fd >= 0:
            _close_fd_quietly(blockers_fd)
        if sealed_fd >= 0:
            _close_fd_quietly(sealed_fd)


def _seal_clean_active_spool_for_replay(runtime: _AnchoredRuntime) -> bool:
    sealed_fd = -1
    active_fd = -1
    try:
        sealed_fd, _ = _open_dir_at(
            runtime.root_fd,
            SEALED_DIR_NAME,
            full_path=_sealed_dir(),
            mode=ROOT_MODE,
            create=True,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )
        active_fd, _ = _open_file_at(
            runtime.root_fd,
            ACTIVE_SPOOL_NAME,
            full_path=runtime.active_path,
            mode=FILE_MODE,
            create=True,
            fsync_parent_on_create=True,
            fsync_file_on_create=True,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )
        active_stat = os.fstat(active_fd)
        if active_stat.st_size <= 0:
            return False
        scan = _scan_fd(active_fd)
        if scan.tail_status is not SpoolTailStatus.CLEAN:
            return False

        sequence = _allocate_next_segment_sequence(runtime=runtime, root_fd=runtime.root_fd)
        segment_name = f"{_format_segment_sequence(sequence)}.spool"
        _assert_home_matches_fd(runtime.home_path, runtime.home_fd)
        _assert_entry_matches_fd(
            runtime.home_fd,
            SPOOL_ROOT_NAME,
            runtime.root_fd,
            expect="dir",
            label=str(runtime.root_path),
        )
        _assert_entry_matches_fd(
            runtime.root_fd,
            ACTIVE_SPOOL_NAME,
            active_fd,
            expect="file",
            label=str(runtime.active_path),
        )
        os.rename(
            ACTIVE_SPOOL_NAME,
            segment_name,
            src_dir_fd=runtime.root_fd,
            dst_dir_fd=sealed_fd,
        )
        _fsync_directory_fd(sealed_fd, _sealed_dir())
        os.close(active_fd)
        active_fd = -1

        active_fd, _ = _open_file_at(
            runtime.root_fd,
            ACTIVE_SPOOL_NAME,
            full_path=runtime.active_path,
            mode=FILE_MODE,
            create=True,
            fsync_parent_on_create=True,
            fsync_file_on_create=True,
            parent_label=runtime.root_path,
            fsync_parent_on_open_existing=True,
        )
        _fsync_fd(active_fd)
        _fsync_directory_fd(sealed_fd, _sealed_dir())
        _fsync_directory_fd(runtime.root_fd, runtime.root_path)
        _fsync_directory_fd(runtime.home_fd, runtime.home_path)
        return True
    finally:
        if active_fd >= 0:
            _close_fd_quietly(active_fd)
        if sealed_fd >= 0:
            _close_fd_quietly(sealed_fd)


def replay_to_session_db(session_db, *, trigger: str) -> ReplayRunResult:
    runtime = _open_locked_runtime()
    owner = None
    try:
        cooldown = _cooldown_result(runtime, trigger=trigger)
        if cooldown is not None:
            _log_replay_run_result(runtime, cooldown)
            return cooldown
        owner = _try_acquire_replay_owner(runtime)
        if owner is None:
            pending_snapshot = _capture_replay_terminal_backlog(runtime)
            return ReplayRunResult(
                state=ReplayRunState.OWNER_BUSY,
                trigger=trigger,
                pending_bytes_after=pending_snapshot.pending_bytes,
                pending_frames_after=pending_snapshot.pending_frames,
                first_blocked_segment=pending_snapshot.first_blocked_segment,
                first_blocked_offset=pending_snapshot.first_blocked_offset,
                ack_pending=pending_snapshot.ack_pending,
            )

        try:
            with _append_lock(runtime.lock_fd, str(_lock_path())):
                _ensure_directory(_sealed_dir(), mode=ROOT_MODE)
                _ensure_directory(_acks_dir(), mode=ROOT_MODE)
                _ensure_directory(_blockers_dir(), mode=ROOT_MODE)
                reconciled_active = _reconcile_active_spool_for_replay(runtime)
                if reconciled_active is None:
                    _seal_clean_active_spool_for_replay(runtime)
        except OSError as exc:
            if not _is_retryable_replay_os_error(exc):
                pending_snapshot = _capture_pending_backlog_snapshot(runtime)
                return _log_and_return_replay_result(
                    runtime,
                    ReplayRunResult(
                        state=ReplayRunState.NOT_DURABLE,
                        trigger=trigger,
                        pending_bytes_after=pending_snapshot.pending_bytes,
                        pending_frames_after=pending_snapshot.pending_frames,
                        first_blocked_segment=pending_snapshot.first_blocked_segment,
                        first_blocked_offset=pending_snapshot.first_blocked_offset,
                        ack_pending=pending_snapshot.ack_pending,
                        error_class=_classify_nonretryable_replay_error(exc),
                    ),
                )
            pending_snapshot = _capture_replay_terminal_backlog(runtime)
            cooldown_seconds = _register_replay_cooldown(
                runtime,
                retry_class="spool_prepare_busy",
                ack_pending=pending_snapshot.ack_pending,
            )
            return _log_and_return_replay_result(
                runtime,
                ReplayRunResult(
                    state=ReplayRunState.RETRY_PENDING,
                    trigger=trigger,
                    pending_bytes_after=pending_snapshot.pending_bytes,
                    pending_frames_after=pending_snapshot.pending_frames,
                    first_blocked_segment=pending_snapshot.first_blocked_segment,
                    first_blocked_offset=pending_snapshot.first_blocked_offset,
                    retry_class="spool_prepare_busy",
                    ack_pending=pending_snapshot.ack_pending,
                    cooldown_seconds=cooldown_seconds,
                ),
            )

        _tombstones, blocked_orphan_sequence = _classify_ack_tombstones(
            runtime=runtime,
            root_fd=runtime.root_fd,
        )
        if blocked_orphan_sequence is not None:
            return _log_and_return_replay_result(
                runtime,
                ReplayRunResult(
                    state=ReplayRunState.BLOCKED_INTEGRITY,
                    trigger=trigger,
                    first_blocked_segment=blocked_orphan_sequence,
                ),
            )

        blocker_sequence = _first_blocker_sequence(runtime=runtime, root_fd=runtime.root_fd)
        blocked_prefix_state: _BlockerBackedPrefixReplayState | None = None
        if blocker_sequence is not None:
            try:
                blocked_prefix_state = _load_blocker_backed_prefix_replay_state(
                    runtime=runtime,
                    root_fd=runtime.root_fd,
                    blocker_sequence=blocker_sequence,
                )
            except SpoolBlockedReplayError as exc:
                return _log_and_return_replay_result(
                    runtime,
                    ReplayRunResult(
                        state=ReplayRunState.BLOCKED_INTEGRITY,
                        trigger=trigger,
                        first_blocked_segment=blocker_sequence,
                        first_blocked_offset=exc.frame_offset,
                        error_class=exc.error_class,
                    ),
                )

        ordered_segments = _ordered_segment_entries(runtime=runtime, root_fd=runtime.root_fd)
        if not ordered_segments:
            if blocker_sequence is not None:
                return _log_and_return_replay_result(
                    runtime,
                    ReplayRunResult(
                        state=ReplayRunState.BLOCKED_INTEGRITY,
                        trigger=trigger,
                        first_blocked_segment=blocker_sequence,
                        first_blocked_offset=(
                            blocked_prefix_state.blocking_offset
                            if blocked_prefix_state is not None
                            else None
                        ),
                        error_class=(
                            blocked_prefix_state.tail_status.value
                            if blocked_prefix_state is not None
                            else None
                        ),
                    ),
                )
            _clear_replay_cooldown(runtime)
            return _log_and_return_replay_result(
                runtime,
                ReplayRunResult(state=ReplayRunState.EMPTY, trigger=trigger),
            )

        max_frames, max_bytes, max_seconds = _trigger_budget(trigger)
        start_monotonic = time.monotonic()
        frames_committed = 0
        frames_duplicated = 0
        frames_acked = 0
        frames_decoded = 0
        bytes_decoded = 0
        bytes_acked = 0
        for segment_index, (segment_sequence, segment_name, segment_path) in enumerate(ordered_segments):
            if (
                (max_frames is not None and frames_decoded >= max_frames)
                or (max_bytes is not None and bytes_decoded >= max_bytes)
                or (
                    max_seconds is not None
                    and (time.monotonic() - start_monotonic) >= max_seconds
                )
            ):
                pending_snapshot = _capture_replay_terminal_backlog(runtime)
                return ReplayRunResult(
                    state=ReplayRunState.PARTIALLY_REPLAYED,
                    trigger=trigger,
                    segment_count_seen=len(ordered_segments),
                    frames_decoded=frames_decoded,
                    frames_committed=frames_committed,
                    frames_duplicated=frames_duplicated,
                    frames_acked=frames_acked,
                    bytes_decoded=bytes_decoded,
                    bytes_acked=bytes_acked,
                    pending_bytes_after=pending_snapshot.pending_bytes,
                    pending_frames_after=pending_snapshot.pending_frames,
                    ack_pending=pending_snapshot.ack_pending,
                )
            if blocker_sequence is not None and segment_sequence > blocker_sequence:
                pending_snapshot = _capture_replay_terminal_backlog(runtime)
                return _log_and_return_replay_result(
                    runtime,
                    ReplayRunResult(
                        state=ReplayRunState.BLOCKED_INTEGRITY,
                        trigger=trigger,
                        segment_count_seen=len(ordered_segments),
                        frames_decoded=frames_decoded,
                        frames_committed=frames_committed,
                        frames_duplicated=frames_duplicated,
                        frames_acked=frames_acked,
                        bytes_decoded=bytes_decoded,
                        bytes_acked=bytes_acked,
                        pending_bytes_after=pending_snapshot.pending_bytes,
                        pending_frames_after=pending_snapshot.pending_frames,
                        first_blocked_segment=blocker_sequence,
                        first_blocked_offset=(
                            blocked_prefix_state.blocking_offset
                            if blocked_prefix_state is not None
                            else None
                        ),
                        ack_pending=pending_snapshot.ack_pending,
                        error_class=(
                            blocked_prefix_state.tail_status.value
                            if blocked_prefix_state is not None
                            else None
                        ),
                    ),
                )
            if blocker_sequence is not None and segment_sequence == blocker_sequence:
                if blocked_prefix_state is None:
                    pending_snapshot = _capture_replay_terminal_backlog(runtime)
                    return _log_and_return_replay_result(
                        runtime,
                        ReplayRunResult(
                            state=ReplayRunState.BLOCKED_INTEGRITY,
                            trigger=trigger,
                            segment_count_seen=len(ordered_segments),
                            frames_decoded=frames_decoded,
                            frames_committed=frames_committed,
                            frames_duplicated=frames_duplicated,
                            frames_acked=frames_acked,
                            bytes_decoded=bytes_decoded,
                            bytes_acked=bytes_acked,
                            pending_bytes_after=pending_snapshot.pending_bytes,
                            pending_frames_after=pending_snapshot.pending_frames,
                            first_blocked_segment=blocker_sequence,
                            ack_pending=pending_snapshot.ack_pending,
                        ),
                    )
                if segment_name != blocked_prefix_state.prefix_segment_name:
                    pending_snapshot = _capture_replay_terminal_backlog(runtime)
                    return _log_and_return_replay_result(
                        runtime,
                        ReplayRunResult(
                            state=ReplayRunState.BLOCKED_INTEGRITY,
                            trigger=trigger,
                            segment_count_seen=len(ordered_segments),
                            frames_decoded=frames_decoded,
                            frames_committed=frames_committed,
                            frames_duplicated=frames_duplicated,
                            frames_acked=frames_acked,
                            bytes_decoded=bytes_decoded,
                            bytes_acked=bytes_acked,
                            pending_bytes_after=pending_snapshot.pending_bytes,
                            pending_frames_after=pending_snapshot.pending_frames,
                            first_blocked_segment=blocker_sequence,
                            first_blocked_offset=blocked_prefix_state.blocking_offset,
                            ack_pending=pending_snapshot.ack_pending,
                            error_class="invalid_blocker_relationship",
                        ),
                    )
                decoded = decode_spool_segment(segment_path)
                if (
                    decoded.tail_status is not SpoolTailStatus.CLEAN
                    or decoded.valid_prefix_bytes != blocked_prefix_state.valid_prefix_bytes
                ):
                    pending_snapshot = _capture_replay_terminal_backlog(runtime)
                    return _log_and_return_replay_result(
                        runtime,
                        ReplayRunResult(
                            state=ReplayRunState.BLOCKED_INTEGRITY,
                            trigger=trigger,
                            segment_count_seen=len(ordered_segments),
                            frames_decoded=frames_decoded,
                            frames_committed=frames_committed,
                            frames_duplicated=frames_duplicated,
                            frames_acked=frames_acked,
                            bytes_decoded=bytes_decoded,
                            bytes_acked=bytes_acked,
                            pending_bytes_after=pending_snapshot.pending_bytes,
                            pending_frames_after=pending_snapshot.pending_frames,
                            first_blocked_segment=blocker_sequence,
                            first_blocked_offset=blocked_prefix_state.blocking_offset,
                            ack_pending=pending_snapshot.ack_pending,
                            error_class="invalid_blocker_relationship",
                        ),
                    )
                for frame in decoded.prefix_frames:
                    frame_end = frame.frame_offset + frame.frame_length
                    if frame_end <= blocked_prefix_state.acked_prefix_bytes:
                        continue
                    if frame.frame_offset < blocked_prefix_state.acked_prefix_bytes:
                        pending_snapshot = _capture_replay_terminal_backlog(runtime)
                        return _log_and_return_replay_result(
                            runtime,
                            ReplayRunResult(
                                state=ReplayRunState.BLOCKED_INTEGRITY,
                                trigger=trigger,
                                segment_count_seen=len(ordered_segments),
                                frames_decoded=frames_decoded,
                                frames_committed=frames_committed,
                                frames_duplicated=frames_duplicated,
                                frames_acked=frames_acked,
                                bytes_decoded=bytes_decoded,
                                bytes_acked=bytes_acked,
                                pending_bytes_after=pending_snapshot.pending_bytes,
                                pending_frames_after=pending_snapshot.pending_frames,
                                first_blocked_segment=blocker_sequence,
                                first_blocked_offset=blocked_prefix_state.blocking_offset,
                                ack_pending=pending_snapshot.ack_pending,
                                error_class="invalid_blocker_relationship",
                            ),
                        )
                    frames_decoded += 1
                    bytes_decoded += frame.frame_length
                    try:
                        result = _replay_db_call(session_db, frame)
                    except SpoolBlockedReplayError as exc:
                        pending_snapshot = _capture_replay_terminal_backlog(runtime)
                        return _log_and_return_replay_result(
                            runtime,
                            ReplayRunResult(
                                state=ReplayRunState.BLOCKED_INTEGRITY,
                                trigger=trigger,
                                segment_count_seen=len(ordered_segments),
                                frames_decoded=frames_decoded,
                                frames_committed=frames_committed,
                                frames_duplicated=frames_duplicated,
                                frames_acked=frames_acked,
                                bytes_decoded=bytes_decoded,
                                bytes_acked=bytes_acked,
                                pending_bytes_after=pending_snapshot.pending_bytes,
                                pending_frames_after=pending_snapshot.pending_frames,
                                first_blocked_segment=segment_sequence,
                                first_blocked_offset=exc.frame_offset,
                                ack_pending=pending_snapshot.ack_pending,
                                error_class=exc.error_class,
                            ),
                        )
                    except SpoolRetryableReplayError as exc:
                        cooldown_seconds = _register_replay_cooldown(
                            runtime,
                            retry_class=exc.retry_class,
                            ack_pending=exc.ack_pending,
                        )
                        pending_snapshot = _capture_replay_terminal_backlog(
                            runtime,
                            ack_pending=exc.ack_pending,
                        )
                        return _log_and_return_replay_result(
                            runtime,
                            ReplayRunResult(
                                state=ReplayRunState.RETRY_PENDING,
                                trigger=trigger,
                                segment_count_seen=len(ordered_segments),
                                frames_decoded=frames_decoded,
                                frames_committed=frames_committed,
                                frames_duplicated=frames_duplicated,
                                frames_acked=frames_acked,
                                bytes_decoded=bytes_decoded,
                                bytes_acked=bytes_acked,
                                pending_bytes_after=pending_snapshot.pending_bytes,
                                pending_frames_after=pending_snapshot.pending_frames,
                                retry_class=exc.retry_class,
                                ack_pending=pending_snapshot.ack_pending,
                                cooldown_seconds=cooldown_seconds,
                            ),
                        )
                    if result.inserted_count > 0:
                        frames_committed += 1
                    else:
                        frames_duplicated += 1
                    try:
                        with _append_lock(runtime.lock_fd, str(_lock_path())):
                            if _publish_ack_sidecar(
                                runtime,
                                segment_sequence=segment_sequence,
                                segment_path=segment_path,
                                decoded_segment=decoded,
                                frame=frame,
                            ):
                                frames_acked += 1
                                bytes_acked += frame.frame_length
                    except SpoolRetryableReplayError as exc:
                        cooldown_seconds = _register_replay_cooldown(
                            runtime,
                            retry_class=exc.retry_class,
                            ack_pending=exc.ack_pending,
                        )
                        pending_snapshot = _capture_replay_terminal_backlog(
                            runtime,
                            ack_pending=exc.ack_pending,
                        )
                        return _log_and_return_replay_result(
                            runtime,
                            ReplayRunResult(
                                state=ReplayRunState.RETRY_PENDING,
                                trigger=trigger,
                                segment_count_seen=len(ordered_segments),
                                frames_decoded=frames_decoded,
                                frames_committed=frames_committed,
                                frames_duplicated=frames_duplicated,
                                frames_acked=frames_acked,
                                bytes_decoded=bytes_decoded,
                                bytes_acked=bytes_acked,
                                pending_bytes_after=pending_snapshot.pending_bytes,
                                pending_frames_after=pending_snapshot.pending_frames,
                                retry_class=exc.retry_class,
                                ack_pending=pending_snapshot.ack_pending,
                                cooldown_seconds=cooldown_seconds,
                            ),
                        )
                    except OSError as exc:
                        if _is_retryable_replay_os_error(exc):
                            cooldown_seconds = _register_replay_cooldown(
                                runtime,
                                retry_class="ack_publish_busy",
                                ack_pending=True,
                            )
                            pending_snapshot = _capture_replay_terminal_backlog(
                                runtime,
                                ack_pending=True,
                            )
                            return _log_and_return_replay_result(
                                runtime,
                                ReplayRunResult(
                                    state=ReplayRunState.RETRY_PENDING,
                                    trigger=trigger,
                                    segment_count_seen=len(ordered_segments),
                                    frames_decoded=frames_decoded,
                                    frames_committed=frames_committed,
                                    frames_duplicated=frames_duplicated,
                                    frames_acked=frames_acked,
                                    bytes_decoded=bytes_decoded,
                                    bytes_acked=bytes_acked,
                                    pending_bytes_after=pending_snapshot.pending_bytes,
                                    pending_frames_after=pending_snapshot.pending_frames,
                                    retry_class="ack_publish_busy",
                                    ack_pending=pending_snapshot.ack_pending,
                                    cooldown_seconds=cooldown_seconds,
                                ),
                            )
                        pending_snapshot = _capture_replay_terminal_backlog(
                            runtime,
                            ack_pending=True,
                        )
                        return _log_and_return_replay_result(
                            runtime,
                            ReplayRunResult(
                                state=ReplayRunState.NOT_DURABLE,
                                trigger=trigger,
                                segment_count_seen=len(ordered_segments),
                                frames_decoded=frames_decoded,
                                frames_committed=frames_committed,
                                frames_duplicated=frames_duplicated,
                                frames_acked=frames_acked,
                                bytes_decoded=bytes_decoded,
                                bytes_acked=bytes_acked,
                                pending_bytes_after=pending_snapshot.pending_bytes,
                                pending_frames_after=pending_snapshot.pending_frames,
                                ack_pending=pending_snapshot.ack_pending,
                                error_class=_classify_nonretryable_replay_error(exc),
                            ),
                        )
                pending_snapshot = _capture_replay_terminal_backlog(runtime)
                return _log_and_return_replay_result(
                    runtime,
                    ReplayRunResult(
                        state=ReplayRunState.BLOCKED_INTEGRITY,
                        trigger=trigger,
                        segment_count_seen=len(ordered_segments),
                        frames_decoded=frames_decoded,
                        frames_committed=frames_committed,
                        frames_duplicated=frames_duplicated,
                        frames_acked=frames_acked,
                        bytes_decoded=bytes_decoded,
                        bytes_acked=bytes_acked,
                        pending_bytes_after=pending_snapshot.pending_bytes,
                        pending_frames_after=pending_snapshot.pending_frames,
                        first_blocked_segment=blocker_sequence,
                        first_blocked_offset=blocked_prefix_state.blocking_offset,
                        ack_pending=pending_snapshot.ack_pending,
                        error_class=blocked_prefix_state.tail_status.value,
                    ),
                )
            decoded = decode_spool_segment(segment_path)
            if decoded.tail_status is not SpoolTailStatus.CLEAN:
                for frame in decoded.prefix_frames:
                    frames_decoded += 1
                    bytes_decoded += frame.frame_length
                    try:
                        result = _replay_db_call(session_db, frame)
                    except SpoolBlockedReplayError as exc:
                        return _log_and_return_replay_result(
                            runtime,
                            ReplayRunResult(
                                state=ReplayRunState.BLOCKED_INTEGRITY,
                                trigger=trigger,
                                segment_count_seen=len(ordered_segments),
                                frames_decoded=frames_decoded,
                                frames_committed=frames_committed,
                                frames_duplicated=frames_duplicated,
                                frames_acked=frames_acked,
                                bytes_decoded=bytes_decoded,
                                bytes_acked=bytes_acked,
                                pending_bytes_after=_remaining_segment_bytes(
                                    ordered_segments, segment_index
                                ),
                                first_blocked_segment=segment_sequence,
                                first_blocked_offset=exc.frame_offset,
                                error_class=exc.error_class,
                            ),
                        )
                    except SpoolRetryableReplayError as exc:
                        cooldown_seconds = _register_replay_cooldown(
                            runtime,
                            retry_class=exc.retry_class,
                            ack_pending=exc.ack_pending,
                        )
                        return _log_and_return_replay_result(
                            runtime,
                            ReplayRunResult(
                                state=ReplayRunState.RETRY_PENDING,
                                trigger=trigger,
                                segment_count_seen=len(ordered_segments),
                                frames_decoded=frames_decoded,
                                frames_committed=frames_committed,
                                frames_duplicated=frames_duplicated,
                                frames_acked=frames_acked,
                                bytes_decoded=bytes_decoded,
                                bytes_acked=bytes_acked,
                                pending_bytes_after=_remaining_segment_bytes(
                                    ordered_segments, segment_index
                                ),
                                retry_class=exc.retry_class,
                                ack_pending=exc.ack_pending,
                                cooldown_seconds=cooldown_seconds,
                            ),
                        )
                    if result.inserted_count > 0:
                        frames_committed += 1
                    else:
                        frames_duplicated += 1
                try:
                    with _append_lock(runtime.lock_fd, str(_lock_path())):
                        _publish_corrupt_sealed_segment_state(
                            runtime,
                            segment_sequence=segment_sequence,
                            segment_name=segment_name,
                            decoded_segment=decoded,
                        )
                except OSError as exc:
                    if _is_retryable_replay_os_error(exc):
                        cooldown_seconds = _register_replay_cooldown(
                            runtime,
                            retry_class="corrupt_publish_busy",
                            ack_pending=False,
                        )
                        pending_bytes_after, pending_frames_after = (
                            _measure_remaining_segment_backlog(
                                runtime,
                                ordered_segments=ordered_segments,
                                start_index=segment_index,
                                current_segment_name=segment_name,
                                current_segment_frame_count=len(decoded.prefix_frames),
                                current_segment_pending_bytes=decoded.valid_prefix_bytes,
                            )
                        )
                        return _log_and_return_replay_result(
                            runtime,
                            ReplayRunResult(
                                state=ReplayRunState.RETRY_PENDING,
                                trigger=trigger,
                                segment_count_seen=len(ordered_segments),
                                frames_decoded=frames_decoded,
                                frames_committed=frames_committed,
                                frames_duplicated=frames_duplicated,
                                frames_acked=frames_acked,
                                bytes_decoded=bytes_decoded,
                                bytes_acked=bytes_acked,
                                pending_bytes_after=pending_bytes_after,
                                pending_frames_after=pending_frames_after,
                                retry_class="corrupt_publish_busy",
                                ack_pending=False,
                                cooldown_seconds=cooldown_seconds,
                            ),
                        )
                    if not _is_retryable_replay_os_error(exc):
                        pending_bytes_after, pending_frames_after = (
                            _measure_remaining_segment_backlog(
                                runtime,
                                ordered_segments=ordered_segments,
                                start_index=segment_index,
                                current_segment_name=segment_name,
                                current_segment_frame_count=len(decoded.prefix_frames),
                            )
                        )
                        return _log_and_return_replay_result(
                            runtime,
                            ReplayRunResult(
                                state=ReplayRunState.NOT_DURABLE,
                                trigger=trigger,
                                segment_count_seen=len(ordered_segments),
                                frames_decoded=frames_decoded,
                                frames_committed=frames_committed,
                                frames_duplicated=frames_duplicated,
                                frames_acked=frames_acked,
                                bytes_decoded=bytes_decoded,
                                bytes_acked=bytes_acked,
                                pending_bytes_after=pending_bytes_after,
                                pending_frames_after=pending_frames_after,
                                error_class=_classify_nonretryable_replay_error(exc),
                            ),
                        )
                return _log_and_return_replay_result(
                    runtime,
                    ReplayRunResult(
                        state=ReplayRunState.BLOCKED_INTEGRITY,
                        trigger=trigger,
                        segment_count_seen=len(ordered_segments),
                        frames_decoded=frames_decoded,
                        frames_committed=frames_committed,
                        frames_duplicated=frames_duplicated,
                        frames_acked=frames_acked,
                        bytes_decoded=bytes_decoded,
                        bytes_acked=bytes_acked,
                        first_blocked_segment=segment_sequence,
                        first_blocked_offset=decoded.tail_offset,
                        error_class=decoded.tail_status.value,
                    ),
                )
            for frame in decoded.prefix_frames:
                frames_decoded += 1
                bytes_decoded += frame.frame_length
                try:
                    result = _replay_db_call(session_db, frame)
                except SpoolBlockedReplayError as exc:
                    pending_snapshot = _capture_replay_terminal_backlog(runtime)
                    return _log_and_return_replay_result(
                        runtime,
                        ReplayRunResult(
                            state=ReplayRunState.BLOCKED_INTEGRITY,
                            trigger=trigger,
                            segment_count_seen=len(ordered_segments),
                            frames_decoded=frames_decoded,
                            frames_committed=frames_committed,
                            frames_duplicated=frames_duplicated,
                            frames_acked=frames_acked,
                            bytes_decoded=bytes_decoded,
                            bytes_acked=bytes_acked,
                            pending_bytes_after=pending_snapshot.pending_bytes,
                            pending_frames_after=pending_snapshot.pending_frames,
                            first_blocked_segment=segment_sequence,
                            first_blocked_offset=exc.frame_offset,
                            ack_pending=pending_snapshot.ack_pending,
                            error_class=exc.error_class,
                        ),
                    )
                except SpoolRetryableReplayError as exc:
                    cooldown_seconds = _register_replay_cooldown(
                        runtime,
                        retry_class=exc.retry_class,
                        ack_pending=exc.ack_pending,
                    )
                    pending_snapshot = _capture_replay_terminal_backlog(
                        runtime,
                        ack_pending=exc.ack_pending,
                    )
                    return _log_and_return_replay_result(
                        runtime,
                        ReplayRunResult(
                            state=ReplayRunState.RETRY_PENDING,
                            trigger=trigger,
                            segment_count_seen=len(ordered_segments),
                            frames_decoded=frames_decoded,
                            frames_committed=frames_committed,
                            frames_duplicated=frames_duplicated,
                            frames_acked=frames_acked,
                            bytes_decoded=bytes_decoded,
                            bytes_acked=bytes_acked,
                            pending_bytes_after=pending_snapshot.pending_bytes,
                            pending_frames_after=pending_snapshot.pending_frames,
                            retry_class=exc.retry_class,
                            ack_pending=pending_snapshot.ack_pending,
                            cooldown_seconds=cooldown_seconds,
                        ),
                    )
                if result.inserted_count > 0:
                    frames_committed += 1
                else:
                    frames_duplicated += 1
                try:
                    with _append_lock(runtime.lock_fd, str(_lock_path())):
                        if _publish_ack_sidecar(
                            runtime,
                            segment_sequence=segment_sequence,
                            segment_path=segment_path,
                            decoded_segment=decoded,
                            frame=frame,
                        ):
                            frames_acked += 1
                            bytes_acked += frame.frame_length
                except SpoolRetryableReplayError as exc:
                    cooldown_seconds = _register_replay_cooldown(
                        runtime,
                        retry_class=exc.retry_class,
                        ack_pending=exc.ack_pending,
                    )
                    pending_snapshot = _capture_replay_terminal_backlog(
                        runtime,
                        ack_pending=exc.ack_pending,
                    )
                    return _log_and_return_replay_result(
                        runtime,
                        ReplayRunResult(
                            state=ReplayRunState.RETRY_PENDING,
                            trigger=trigger,
                            segment_count_seen=len(ordered_segments),
                            frames_decoded=frames_decoded,
                            frames_committed=frames_committed,
                            frames_duplicated=frames_duplicated,
                            frames_acked=frames_acked,
                            bytes_decoded=bytes_decoded,
                            bytes_acked=bytes_acked,
                            pending_bytes_after=pending_snapshot.pending_bytes,
                            pending_frames_after=pending_snapshot.pending_frames,
                            retry_class=exc.retry_class,
                            ack_pending=pending_snapshot.ack_pending,
                            cooldown_seconds=cooldown_seconds,
                        ),
                    )
                except OSError as exc:
                    if _is_retryable_replay_os_error(exc):
                        cooldown_seconds = _register_replay_cooldown(
                            runtime,
                            retry_class="ack_publish_busy",
                            ack_pending=True,
                        )
                        pending_snapshot = _capture_replay_terminal_backlog(
                            runtime,
                            ack_pending=True,
                        )
                        return _log_and_return_replay_result(
                            runtime,
                            ReplayRunResult(
                                state=ReplayRunState.RETRY_PENDING,
                                trigger=trigger,
                                segment_count_seen=len(ordered_segments),
                                frames_decoded=frames_decoded,
                                frames_committed=frames_committed,
                                frames_duplicated=frames_duplicated,
                                frames_acked=frames_acked,
                                bytes_decoded=bytes_decoded,
                                bytes_acked=bytes_acked,
                                pending_bytes_after=pending_snapshot.pending_bytes,
                                pending_frames_after=pending_snapshot.pending_frames,
                                retry_class="ack_publish_busy",
                                ack_pending=pending_snapshot.ack_pending,
                                cooldown_seconds=cooldown_seconds,
                            ),
                        )
                    pending_snapshot = _capture_replay_terminal_backlog(
                        runtime,
                        ack_pending=True,
                    )
                    return _log_and_return_replay_result(
                        runtime,
                        ReplayRunResult(
                            state=ReplayRunState.NOT_DURABLE,
                            trigger=trigger,
                            segment_count_seen=len(ordered_segments),
                            frames_decoded=frames_decoded,
                            frames_committed=frames_committed,
                            frames_duplicated=frames_duplicated,
                            frames_acked=frames_acked,
                            bytes_decoded=bytes_decoded,
                            bytes_acked=bytes_acked,
                            pending_bytes_after=pending_snapshot.pending_bytes,
                            pending_frames_after=pending_snapshot.pending_frames,
                            ack_pending=pending_snapshot.ack_pending,
                            error_class=_classify_nonretryable_replay_error(exc),
                        ),
                    )
            try:
                with _append_lock(runtime.lock_fd, str(_lock_path())):
                    _delete_fully_acked_segment(segment_path)
            except OSError as exc:
                pending_snapshot = _capture_replay_terminal_backlog(
                    runtime,
                    ack_pending=True,
                )
                if _is_retryable_replay_os_error(exc):
                    cooldown_seconds = _register_replay_cooldown(
                        runtime,
                        retry_class="ack_cleanup_busy",
                        ack_pending=True,
                    )
                    return _log_and_return_replay_result(
                        runtime,
                        ReplayRunResult(
                            state=ReplayRunState.RETRY_PENDING,
                            trigger=trigger,
                            segment_count_seen=len(ordered_segments),
                            frames_decoded=frames_decoded,
                            frames_committed=frames_committed,
                            frames_duplicated=frames_duplicated,
                            frames_acked=frames_acked,
                            bytes_decoded=bytes_decoded,
                            bytes_acked=bytes_acked,
                            pending_bytes_after=pending_snapshot.pending_bytes,
                            pending_frames_after=pending_snapshot.pending_frames,
                            retry_class="ack_cleanup_busy",
                            ack_pending=pending_snapshot.ack_pending,
                            cooldown_seconds=cooldown_seconds,
                        ),
                    )
                return _log_and_return_replay_result(
                    runtime,
                    ReplayRunResult(
                        state=ReplayRunState.NOT_DURABLE,
                        trigger=trigger,
                        segment_count_seen=len(ordered_segments),
                        frames_decoded=frames_decoded,
                        frames_committed=frames_committed,
                        frames_duplicated=frames_duplicated,
                        frames_acked=frames_acked,
                        bytes_decoded=bytes_decoded,
                        bytes_acked=bytes_acked,
                        pending_bytes_after=pending_snapshot.pending_bytes,
                        pending_frames_after=pending_snapshot.pending_frames,
                        ack_pending=pending_snapshot.ack_pending,
                        error_class=_classify_nonretryable_replay_error(exc),
                    ),
                )
        _clear_replay_cooldown(runtime)
        return _log_and_return_replay_result(
            runtime,
            ReplayRunResult(
                state=ReplayRunState.REPLAYED,
                trigger=trigger,
                segment_count_seen=len(ordered_segments),
                frames_decoded=frames_decoded,
                frames_committed=frames_committed,
                frames_duplicated=frames_duplicated,
                frames_acked=frames_acked,
                bytes_decoded=bytes_decoded,
                bytes_acked=bytes_acked,
            ),
        )
    finally:
        if owner is not None:
            _close_fd_quietly(owner.fd)
        _close_fd_quietly(runtime.lock_fd)
        _close_fd_quietly(runtime.root_fd)
        _close_fd_quietly(runtime.home_fd)


def append_records(records: Sequence[SessionSpoolRecord]) -> SpoolAppendAttemptResult:
    if not records:
        return SpoolAppendAttemptResult(unit_results=())

    frames = []
    for record in records:
        frames.append((record, _frame_bytes_for_record(record)))

    runtime = _open_locked_runtime()
    quarantine_fd = -1
    active_fd = -1
    global _CURRENT_QUARANTINE_DIR_FD
    try:
        with _append_lock(runtime.lock_fd, str(_lock_path())):
            _assert_home_matches_fd(runtime.home_path, runtime.home_fd)
            _assert_entry_matches_fd(
                runtime.home_fd,
                SPOOL_ROOT_NAME,
                runtime.root_fd,
                expect="dir",
                label=str(runtime.root_path),
            )
            quarantine_fd, _ = _open_dir_at(
                runtime.root_fd,
                QUARANTINE_DIR_NAME,
                full_path=runtime.quarantine_path,
                mode=ROOT_MODE,
                create=True,
                parent_label=runtime.root_path,
                fsync_parent_on_open_existing=True,
            )
            _assert_entry_matches_fd(
                runtime.root_fd,
                QUARANTINE_DIR_NAME,
                quarantine_fd,
                expect="dir",
                label=str(runtime.quarantine_path),
            )
            _CURRENT_QUARANTINE_DIR_FD = quarantine_fd
            _reconcile_missing_sidecars(runtime.quarantine_path, quarantine_fd=quarantine_fd)

            active_fd, _ = _open_file_at(
                runtime.root_fd,
                ACTIVE_SPOOL_NAME,
                full_path=runtime.active_path,
                mode=FILE_MODE,
                create=True,
                fsync_parent_on_create=True,
                fsync_file_on_create=True,
                parent_label=runtime.root_path,
                fsync_parent_on_open_existing=True,
            )
            scan = _scan_fd(active_fd)
            if scan.tail_status is not SpoolTailStatus.CLEAN:
                _quarantine_active_file(
                    runtime.active_path,
                    runtime.quarantine_path,
                    scan,
                    runtime=runtime,
                    quarantine_fd=quarantine_fd,
                    active_fd=active_fd,
                )
                os.close(active_fd)
                active_fd = -1
                active_fd, _ = _open_file_at(
                    runtime.root_fd,
                    ACTIVE_SPOOL_NAME,
                    full_path=runtime.active_path,
                    mode=FILE_MODE,
                    create=True,
                    fsync_parent_on_create=True,
                    fsync_file_on_create=True,
                    parent_label=runtime.root_path,
                    fsync_parent_on_open_existing=True,
                )

            active_bytes = os.fstat(active_fd).st_size
            inventory = _durable_capacity_inventory(runtime, runtime.root_fd)
            requested_bytes = sum(len(frame) for _record, frame in frames)
            if (
                active_bytes
                + inventory.quarantine_bytes
                + inventory.other_artifact_bytes
                + requested_bytes
                > TOTAL_CAP_BYTES
            ):
                raise SpoolCapacityError(
                    active_bytes=active_bytes,
                    quarantine_bytes=inventory.quarantine_bytes,
                    other_artifact_bytes=inventory.other_artifact_bytes,
                    requested_bytes=requested_bytes,
                    cap_bytes=TOTAL_CAP_BYTES,
                )

            durable_results: list[SpoolUnitAppendResult] = []
            for record, frame in frames:
                offset = os.lseek(active_fd, 0, os.SEEK_END)
                try:
                    _write_all(active_fd, frame)
                    _fsync_fd(active_fd)
                    _fsync_directory_fd(runtime.root_fd, runtime.root_path)
                    _fsync_directory_fd(runtime.home_fd, runtime.home_path)
                    _assert_home_matches_fd(runtime.home_path, runtime.home_fd)
                    _assert_entry_matches_fd(
                        runtime.home_fd,
                        SPOOL_ROOT_NAME,
                        runtime.root_fd,
                        expect="dir",
                        label=str(runtime.root_path),
                    )
                    _assert_entry_matches_fd(
                        runtime.root_fd,
                        ACTIVE_SPOOL_NAME,
                        active_fd,
                        expect="file",
                        label=str(runtime.active_path),
                    )
                except BaseException as exc:
                    cause = (
                        exc
                        if isinstance(exc, SessionFallbackSpoolError)
                        else SpoolDurabilityError(str(exc))
                    )
                    if durable_results:
                        raise SpoolAppendAttemptPartialError(durable_results, cause) from exc
                    if isinstance(cause, SessionFallbackSpoolError):
                        raise cause from exc
                    raise SpoolDurabilityError(str(exc)) from exc
                receipt = SpoolFrameReceipt(
                    path=str(runtime.active_path),
                    offset=offset,
                    frame_length=len(frame),
                    payload_length=len(frame) - HEADER_SIZE,
                    checksum_hex=frame[16:32].hex(),
                )
                durable_results.append(
                    SpoolUnitAppendResult(
                        persistence_unit_id=record.batch_messages[0].persistence_unit_id,
                        message_keys=tuple(
                            message.persistence_message_key for message in record.batch_messages
                        ),
                        receipt=receipt,
                    )
                )
            return SpoolAppendAttemptResult(unit_results=tuple(durable_results))
    finally:
        _CURRENT_QUARANTINE_DIR_FD = None
        if active_fd >= 0:
            os.close(active_fd)
        if quarantine_fd >= 0:
            os.close(quarantine_fd)
        os.close(runtime.lock_fd)
        os.close(runtime.root_fd)
        os.close(runtime.home_fd)
