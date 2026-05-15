"""Async batched JSONL file backend for audit events."""

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from audit.backends.base import AuditBackend

logger = logging.getLogger(__name__)


class LogBackend(AuditBackend):
    """
    Async batched JSONL file backend.

    Events are queued and written in batches to minimize I/O overhead.
    Uses a background thread for non-blocking writes.
    """

    def __init__(
        self,
        log_path: str,
        max_size_mb: int = 500,
        backup_count: int = 10,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        """
        Initialize log backend.

        Args:
            log_path: Directory for audit log files
            max_size_mb: Max size per file before rotation (MB)
            backup_count: Number of backup files to keep
            batch_size: Number of events per batch write
            flush_interval: Max seconds between flushes
        """
        self._log_path = Path(log_path).expanduser().resolve()
        self._max_bytes = max_size_mb * 1024 * 1024
        self._backup_count = backup_count
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        self._queue: queue.Queue = queue.Queue()
        self._running = True
        self._worker = threading.Thread(target=self._flush_loop, daemon=True)
        self._worker.start()

    def emit(self, event: Dict[str, Any]) -> None:
        """Queue event for async batch writing."""
        if not self._running:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("Audit log queue full, dropping event")

    def flush(self) -> None:
        """Force flush any pending events."""
        self._queue.join()

    def close(self) -> None:
        """Stop worker and flush pending events."""
        self._running = False
        self._queue.join()
        self._worker.join(timeout=5.0)

    def _flush_loop(self) -> None:
        """Background thread that batches and writes events."""
        batch: List[Dict[str, Any]] = []
        last_flush = time.time()

        while self._running or not self._queue.empty():
            try:
                event = self._queue.get(timeout=1.0)
                batch.append(event)
                self._queue.task_done()

                # Flush conditions: batch full or time elapsed
                if len(batch) >= self._batch_size or (
                    time.time() - last_flush
                ) > self._flush_interval:
                    self._write_batch(batch)
                    batch = []
                    last_flush = time.time()
            except queue.Empty:
                if batch:
                    self._write_batch(batch)
                    batch = []
                    last_flush = time.time()

        # Final flush
        if batch:
            self._write_batch(batch)

    def _write_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Write batch as JSONL with rotation check."""
        if not batch:
            return

        try:
            self._log_path.mkdir(parents=True, exist_ok=True)
            log_file = self._get_log_file()

            # Check rotation
            self._rotate_if_needed(log_file)

            # Append to file
            with open(log_file, "a", encoding="utf-8") as f:
                for event in batch:
                    f.write(self._format_event(event) + "\n")

        except Exception as e:
            logger.error("Failed to write audit batch: %s", e)

    def _get_log_file(self) -> Path:
        """Get the current log file path (date-based)."""
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self._log_path / f"audit-{date_str}.jsonl"

    def _format_event(self, event: Dict[str, Any]) -> str:
        """Format event as JSON string."""
        import json
        return json.dumps(event, ensure_ascii=False, sort_keys=False)

    def _rotate_if_needed(self, log_file: Path) -> None:
        """Rotate log file if it exceeds max size."""
        if not log_file.exists():
            return

        try:
            size = log_file.stat().st_size
            if size >= self._max_bytes:
                self._rotate_file(log_file)
        except OSError as e:
            logger.warning("Failed to check log file size: %s", e)

    def _rotate_file(self, log_file: Path) -> None:
        """Rotate a log file."""
        if not log_file.exists():
            return

        # Find next available rotation number
        for i in range(1, self._backup_count + 1):
            rotated = log_file.with_suffix(f".jsonl.{i}")
            if not rotated.exists():
                try:
                    os.rename(log_file, rotated)
                    self._chmod_if_needed(rotated)
                    return
                except OSError as e:
                    logger.warning("Failed to rotate log file: %s", e)
                    return

        # All rotations full, delete oldest and rotate
        oldest = log_file.with_suffix(f".jsonl.{self._backup_count}")
        try:
            oldest.unlink()
            for i in range(self._backup_count - 1, 0, -1):
                src = log_file.with_suffix(f".jsonl.{i}")
                dst = log_file.with_suffix(f".jsonl.{i + 1}")
                if src.exists():
                    os.rename(src, dst)
            os.rename(log_file, log_file.with_suffix(".jsonl.1"))
            self._chmod_if_needed(log_file.with_suffix(".jsonl.1"))
        except OSError as e:
            logger.warning("Failed to rotate log file: %s", e)

    def _chmod_if_needed(self, path: Path) -> None:
        """Apply group-writable permissions in managed mode."""
        try:
            from hermes_cli.config import is_managed
            if is_managed():
                os.chmod(path, 0o660)
        except Exception:
            pass