"""Crash-safe active gateway run journal and startup recovery classification.

The routing index in ``state.db`` is deliberately not used for this marker:
an active turn changes phase at message frequency, while some installations
have multi-gigabyte state databases.  This module keeps the hot lifecycle bit
in one tiny, atomically-replaced sidecar and stores no user message content.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from agent.replay_cleanup import sanitize_replay_history
from utils import atomic_replace

logger = logging.getLogger(__name__)

ACTIVE_RUN_JOURNAL_NAME = ".active_runs.json"
ACTIVE_RUN_JOURNAL_VERSION = 1
RECOVERY_ATTEMPT_LIMIT = 3

PHASE_EXECUTING = "executing"
PHASE_RESPONSE_READY = "response_ready"
ActiveRunPhase = Literal["executing", "response_ready"]

RECOVERY_AUTO_RESUME = "auto_resume"
RECOVERY_WAIT_FOR_PROCESS = "wait_for_process"
RECOVERY_PAUSE_SIDE_EFFECT = "pause_side_effect_unknown"
RECOVERY_PAUSE_DELIVERY = "pause_delivery_unknown"
RECOVERY_PAUSE_INPUT = "pause_input_missing"
RECOVERY_PAUSE_RETRY_LIMIT = "pause_retry_limit"


@dataclass(frozen=True)
class ActiveRunRecord:
    """Durable identity and phase for one in-flight gateway turn."""

    session_key: str
    run_id: str
    started_at: float
    trigger_message_id: Optional[str] = None
    phase: ActiveRunPhase = PHASE_EXECUTING
    recovery_attempts: int = 0
    last_recovery_boot_id: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Any) -> "ActiveRunRecord":
        if not isinstance(raw, dict):
            raise ValueError("active-run record must be an object")
        phase = str(raw.get("phase") or "")
        if phase not in {PHASE_EXECUTING, PHASE_RESPONSE_READY}:
            raise ValueError(f"invalid active-run phase: {phase!r}")
        session_key = str(raw.get("session_key") or "")
        run_id = str(raw.get("run_id") or "")
        if not session_key or not run_id:
            raise ValueError("active-run record is missing session_key or run_id")
        trigger = raw.get("trigger_message_id")
        return cls(
            session_key=session_key,
            run_id=run_id,
            started_at=float(raw.get("started_at") or 0.0),
            trigger_message_id=str(trigger) if trigger not in {None, ""} else None,
            phase=phase,
            recovery_attempts=max(0, int(raw.get("recovery_attempts") or 0)),
            last_recovery_boot_id=(
                str(raw["last_recovery_boot_id"])
                if raw.get("last_recovery_boot_id")
                else None
            ),
        )


class ActiveRunStore:
    """Thread-safe, atomic sidecar store keyed by gateway ``session_key``."""

    def __init__(
        self, sessions_dir: Path | str, *, filename: str = ACTIVE_RUN_JOURNAL_NAME
    ):
        self.path = Path(sessions_dir) / filename
        self._lock = threading.RLock()
        self._loaded = False
        self._runs: dict[str, ActiveRunRecord] = {}

    def _ensure_loaded_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("active-run journal root must be an object")
            if int(raw.get("version") or 0) != ACTIVE_RUN_JOURNAL_VERSION:
                raise ValueError("unsupported active-run journal version")
            records = raw.get("runs") or {}
            if not isinstance(records, dict):
                raise ValueError("active-run journal runs must be an object")
            loaded: dict[str, ActiveRunRecord] = {}
            for key, value in records.items():
                try:
                    record = ActiveRunRecord.from_dict(value)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping invalid active-run record %r: %s", key, exc
                    )
                    continue
                if str(key) != record.session_key:
                    logger.warning(
                        "Skipping active-run record with mismatched key %r", key
                    )
                    continue
                loaded[record.session_key] = record
            self._runs = loaded
        except Exception as exc:
            # Fail safe: a corrupt optional sidecar must never prevent the
            # gateway from starting.  The next mutation replaces it atomically
            # with a valid journal; state.db and transcripts remain untouched.
            logger.warning("Ignoring corrupt active-run journal %s: %s", self.path, exc)
            self._runs = {}

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": ACTIVE_RUN_JOURNAL_VERSION,
            "runs": {key: asdict(record) for key, record in sorted(self._runs.items())},
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            atomic_replace(tmp_name, self.path)
            try:
                dir_fd = os.open(str(self.path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        finally:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass

    def snapshot(self) -> list[ActiveRunRecord]:
        with self._lock:
            self._ensure_loaded_locked()
            return list(self._runs.values())

    def get(self, session_key: str) -> Optional[ActiveRunRecord]:
        with self._lock:
            self._ensure_loaded_locked()
            return self._runs.get(session_key)

    def begin(
        self,
        session_key: str,
        *,
        trigger_message_id: Optional[str] = None,
        recovery_run_id: Optional[str] = None,
        started_at: Optional[float] = None,
    ) -> ActiveRunRecord:
        """Start a new turn, or reclaim the exact run during startup recovery."""
        if not session_key:
            raise ValueError("session_key is required")
        with self._lock:
            self._ensure_loaded_locked()
            current = self._runs.get(session_key)
            if recovery_run_id and current and current.run_id == recovery_run_id:
                return current
            record = ActiveRunRecord(
                session_key=session_key,
                run_id=uuid.uuid4().hex,
                started_at=time.time() if started_at is None else float(started_at),
                trigger_message_id=(
                    str(trigger_message_id)
                    if trigger_message_id not in {None, ""}
                    else None
                ),
            )
            self._runs[session_key] = record
            self._save_locked()
            return record

    def mark_response_ready(self, session_key: str, run_id: str) -> bool:
        """CAS the matching run from ``executing`` to ``response_ready``."""
        with self._lock:
            self._ensure_loaded_locked()
            current = self._runs.get(session_key)
            if current is None or current.run_id != run_id:
                return False
            if current.phase == PHASE_RESPONSE_READY:
                return True
            self._runs[session_key] = replace(current, phase=PHASE_RESPONSE_READY)
            self._save_locked()
            return True

    def finish(
        self, session_key: str, run_id: str, *, require_ready: bool = True
    ) -> bool:
        """CAS-delete a delivered run without clearing a newer replacement."""
        with self._lock:
            self._ensure_loaded_locked()
            current = self._runs.get(session_key)
            if current is None or current.run_id != run_id:
                return False
            if require_ready and current.phase != PHASE_RESPONSE_READY:
                return False
            del self._runs[session_key]
            self._save_locked()
            return True

    def discard(self, session_key: str, *, run_id: Optional[str] = None) -> bool:
        """Explicitly discard a run for /stop, /new, or stale routing cleanup."""
        with self._lock:
            self._ensure_loaded_locked()
            current = self._runs.get(session_key)
            if current is None or (run_id is not None and current.run_id != run_id):
                return False
            del self._runs[session_key]
            self._save_locked()
            return True

    def record_recovery_attempt(
        self, session_key: str, run_id: str, boot_id: str
    ) -> Optional[ActiveRunRecord]:
        """Increment a run's recovery count at most once for this boot."""
        if not boot_id:
            raise ValueError("boot_id is required")
        with self._lock:
            self._ensure_loaded_locked()
            current = self._runs.get(session_key)
            if current is None or current.run_id != run_id:
                return None
            if current.last_recovery_boot_id == boot_id:
                return current
            updated = replace(
                current,
                recovery_attempts=current.recovery_attempts + 1,
                last_recovery_boot_id=boot_id,
            )
            self._runs[session_key] = updated
            self._save_locked()
            return updated


@dataclass(frozen=True)
class RecoveryDecision:
    disposition: str
    reason: str


def _message_timestamp(message: dict[str, Any]) -> Optional[float]:
    value = message.get("timestamp")
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_trigger_index(
    record: ActiveRunRecord, transcript: list[dict[str, Any]]
) -> Optional[int]:
    if record.trigger_message_id:
        for index in range(len(transcript) - 1, -1, -1):
            message = transcript[index]
            if (
                message.get("role") == "user"
                and str(message.get("message_id") or "") == record.trigger_message_id
            ):
                return index
        return None

    # Legacy/internal events may not carry a platform message id.  Match the
    # first durable user row written around or after journal creation.  A small
    # tolerance covers clocks captured on opposite sides of an async boundary.
    threshold = record.started_at - 5.0
    for index in range(len(transcript) - 1, -1, -1):
        message = transcript[index]
        if message.get("role") != "user":
            continue
        timestamp = _message_timestamp(message)
        if timestamp is not None and timestamp >= threshold:
            return index
    return None


def classify_active_run(
    record: ActiveRunRecord,
    transcript: Iterable[dict[str, Any]],
    *,
    has_active_process: bool = False,
    attempt_limit: int = RECOVERY_ATTEMPT_LIMIT,
) -> RecoveryDecision:
    """Classify whether a crashed turn is safe to continue automatically."""
    messages = [message for message in transcript if isinstance(message, dict)]

    if record.phase == PHASE_RESPONSE_READY:
        return RecoveryDecision(
            RECOVERY_PAUSE_DELIVERY,
            "assistant response was ready but delivery acknowledgement is missing",
        )
    if record.recovery_attempts >= attempt_limit:
        return RecoveryDecision(
            RECOVERY_PAUSE_RETRY_LIMIT,
            f"run was interrupted during {record.recovery_attempts} recovery boots",
        )
    if has_active_process:
        return RecoveryDecision(
            RECOVERY_WAIT_FOR_PROCESS,
            "a recovered background process still owns the continuation",
        )

    trigger_index = _find_trigger_index(record, messages)
    if trigger_index is None:
        return RecoveryDecision(
            RECOVERY_PAUSE_INPUT,
            "triggering request was not durably persisted",
        )

    segment = [
        message
        for message in messages[trigger_index:]
        if message.get("role") not in {"session_meta", "system"}
    ]
    sanitized = sanitize_replay_history(segment)
    if any(
        message.get("role") == "tool" and message.get("effect_disposition") == "unknown"
        for message in sanitized
    ):
        return RecoveryDecision(
            RECOVERY_PAUSE_SIDE_EFFECT,
            "a side-effecting tool may have executed without a durable result",
        )

    # A plain assistant tail means model output reached durable history but the
    # journal never reached response_ready.  Recomputing or re-sending could
    # duplicate work, so take the same conservative delivery-unknown path.
    if sanitized:
        tail = sanitized[-1]
        if tail.get("role") == "assistant" and not tail.get("tool_calls"):
            return RecoveryDecision(
                RECOVERY_PAUSE_DELIVERY,
                "assistant output is durable but its delivery state is unknown",
            )

    return RecoveryDecision(
        RECOVERY_AUTO_RESUME,
        "durable request tail contains no uncertain external side effect",
    )
