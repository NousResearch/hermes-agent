"""Durable webhook queue for gateway restarts and transient dispatch failures.

The queue is intentionally tiny and dependency-free: JSONL records plus a
best-effort cross-process file lock. Webhook ingress can append synchronously
before acknowledging the HTTP request; the gateway cron ticker later replays
pending records through the live WebhookAdapter.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_cli.config import get_hermes_home

QUEUE_DIRNAME = "webhook-retry"
QUEUE_FILENAME = "queue.jsonl"
LOCK_FILENAME = "queue.lock"
DEAD_LETTER_FILENAME = "dead-letter.jsonl"
DEFAULT_MAX_ATTEMPTS = 12
DEFAULT_RETRY_DELAY_SECONDS = 60.0
DEFAULT_STALE_INFLIGHT_SECONDS = 15 * 60.0


class WebhookQueueError(RuntimeError):
    """Raised when the durable webhook queue cannot be read or written."""


def _base_dir() -> Path:
    root = get_hermes_home()
    return root / "data" / QUEUE_DIRNAME


def queue_path() -> Path:
    return _base_dir() / QUEUE_FILENAME


def lock_path() -> Path:
    return _base_dir() / LOCK_FILENAME

def dead_letter_path() -> Path:
    return _base_dir() / DEAD_LETTER_FILENAME


@contextlib.contextmanager
def _locked_queue() -> Iterator[None]:
    base = _base_dir()
    base.mkdir(parents=True, exist_ok=True)
    with lock_path().open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _read_records_unlocked() -> List[Dict[str, Any]]:
    path = queue_path()
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise WebhookQueueError(f"Corrupt webhook queue at line {line_no}: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def _write_records_unlocked(records: List[Dict[str, Any]]) -> None:
    path = queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def make_record(
    *,
    route_name: str,
    delivery_id: str,
    event_type: str,
    payload: Dict[str, Any],
    prompt: str,
    deliver_config: Dict[str, Any],
    message_id: Optional[str] = None,
) -> Dict[str, Any]:
    now = time.time()
    return {
        "id": str(uuid.uuid4()),
        "route_name": route_name,
        "delivery_id": str(delivery_id),
        "event_type": event_type,
        "payload": payload,
        "prompt": prompt,
        "deliver_config": deliver_config,
        "message_id": str(message_id or delivery_id),
        "status": "pending",
        "attempts": 0,
        "queued_at": now,
        "updated_at": now,
        "next_attempt_at": now,
        "last_error": None,
    }


def enqueue(record: Dict[str, Any]) -> str:
    """Append a webhook record unless an active record for its delivery exists."""
    delivery_id = str(record.get("delivery_id") or "")
    route_name = str(record.get("route_name") or "")
    with _locked_queue():
        records = _read_records_unlocked()
        for existing in records:
            if (
                str(existing.get("delivery_id") or "") == delivery_id
                and str(existing.get("route_name") or "") == route_name
                and existing.get("status") in {"pending", "inflight"}
            ):
                return str(existing.get("id") or "")
        records.append(record)
        _write_records_unlocked(records)
    return str(record.get("id") or "")


def claim_due(
    *,
    limit: int = 10,
    now: Optional[float] = None,
    stale_inflight_seconds: float = DEFAULT_STALE_INFLIGHT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> List[Dict[str, Any]]:
    now = time.time() if now is None else now
    claimed: List[Dict[str, Any]] = []
    with _locked_queue():
        records = _read_records_unlocked()
        for record in records:
            if len(claimed) >= limit:
                break
            status = record.get("status", "pending")
            attempts = int(record.get("attempts") or 0)
            next_attempt_at = float(record.get("next_attempt_at") or 0)
            updated_at = float(record.get("updated_at") or 0)
            stale_inflight = status == "inflight" and (now - updated_at) >= stale_inflight_seconds
            due_pending = status == "pending" and next_attempt_at <= now
            if attempts >= max_attempts:
                record["status"] = "failed"
                record["updated_at"] = now
                record["last_error"] = record.get("last_error") or "max attempts exceeded"
                continue
            if due_pending or stale_inflight:
                record["status"] = "inflight"
                record["attempts"] = attempts + 1
                record["updated_at"] = now
                claimed.append(dict(record))
        _write_records_unlocked(records)
    return claimed


def mark_inflight(record_id: str, *, now: Optional[float] = None) -> None:
    """Mark a queued record as actively being processed by a live dispatch."""
    now = time.time() if now is None else now
    with _locked_queue():
        records = _read_records_unlocked()
        for record in records:
            if str(record.get("id") or "") != record_id:
                continue
            record["status"] = "inflight"
            record["attempts"] = int(record.get("attempts") or 0) + 1
            record["updated_at"] = now
            break
        _write_records_unlocked(records)


def mark_done(record_id: str) -> None:
    with _locked_queue():
        records = _read_records_unlocked()
        records = [record for record in records if str(record.get("id") or "") != record_id]
        _write_records_unlocked(records)


def mark_retry(
    record_id: str,
    error: str,
    *,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> None:
    now = time.time()
    with _locked_queue():
        records = _read_records_unlocked()
        for record in records:
            if str(record.get("id") or "") != record_id:
                continue
            attempts = int(record.get("attempts") or 0)
            record["status"] = "failed" if attempts >= max_attempts else "pending"
            record["updated_at"] = now
            record["next_attempt_at"] = now + retry_delay_seconds
            record["last_error"] = str(error)[:1000]
            break
        _write_records_unlocked(records)



def mark_dead_letter(
    record_id: str,
    error: str,
    *,
    classification: str = "terminal",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Move a non-retryable webhook failure into a durable dead-letter log.

    Deterministic protocol failures (schema/validation/auth/permission) should
    not be blindly retried with the same payload. Preserve the original queue
    record plus failure context for autonomous repair/replay tooling, then
    remove it from the active retry queue.
    """
    now = time.time()
    dead_record: Dict[str, Any] | None = None
    with _locked_queue():
        records = _read_records_unlocked()
        kept: List[Dict[str, Any]] = []
        for record in records:
            if str(record.get("id") or "") == record_id and dead_record is None:
                dead_record = dict(record)
                dead_record["status"] = "dead_letter"
                dead_record["dead_lettered_at"] = now
                dead_record["updated_at"] = now
                dead_record["last_error"] = str(error)[:4000]
                dead_record["failure_classification"] = classification
                if details:
                    dead_record["failure_details"] = details
            else:
                kept.append(record)
        _write_records_unlocked(kept)
        if dead_record is not None:
            path = dead_letter_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dead_record, ensure_ascii=False, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

def stats() -> Dict[str, int]:
    with _locked_queue():
        records = _read_records_unlocked()
    counts: Dict[str, int] = {"pending": 0, "inflight": 0, "failed": 0, "total": len(records)}
    for record in records:
        status = str(record.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts
