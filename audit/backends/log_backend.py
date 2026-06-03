"""Async batched JSONL file backend for audit events."""

import gzip
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

from audit.backends.base import AuditBackend

logger = logging.getLogger(__name__)

# Default rotation settings (can be overridden via config/env)
DEFAULT_MAX_SIZE_MB = 100
DEFAULT_MAX_AGE_HOURS = 24
DEFAULT_MAX_BACKUPS = 720
DEFAULT_COMPRESS = True
DEFAULT_TIMEZONE = "Asia/Shanghai"


class LogBackend(AuditBackend):
    """
    Async batched JSONL file backend with time + size based rotation.

    Events are queued and written in batches to minimize I/O overhead.
    Uses a background thread for non-blocking writes.

    Rotation logic:
    - Size-based: rotates when file exceeds max_size_mb
    - Naming: {basename}.{timestamp}.gz (compressed after rotation)
    - Cleanup: removes files older than max_age_hours or exceeding max_backups
    """

    def __init__(
        self,
        log_path: str,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
        max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
        max_backups: int = DEFAULT_MAX_BACKUPS,
        compress: bool = DEFAULT_COMPRESS,
        timezone_name: str = DEFAULT_TIMEZONE,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        """
        Initialize log backend.

        Args:
            log_path: Directory for audit log files, or full file path for single-file mode.
                     If contains a filename (e.g., "/logs/audit.jsonl"), uses that directly.
                     If is a directory (e.g., "/logs/audit/"), writes date-based files.
            max_size_mb: Max size per file before rotation (MB)
            max_age_hours: Max hours to retain backup files
            max_backups: Number of backup files to keep (0 = keep all)
            compress: Whether to gzip compress rotated files
            timezone_name: Timezone for timestamp formatting
            batch_size: Number of events per batch write
            flush_interval: Max seconds between flushes
        """
        self._log_path = Path(log_path).expanduser().resolve()
        self._max_bytes = max_size_mb * 1024 * 1024
        self._max_age_hours = max_age_hours
        self._max_backups = max_backups
        self._compress = compress
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        # Determine if we're in directory mode or single-file mode
        if self._log_path.suffix and "." in self._log_path.name:
            # Has extension - treat as a file path, use its directory + basename
            self._base_dir = self._log_path.parent
            self._base_name = self._log_path.stem  # filename without extension
            self._suffix = self._log_path.suffix  # .jsonl
        else:
            # Directory mode - will use date-based naming inside this directory
            self._base_dir = self._log_path
            self._base_name = "audit"
            self._suffix = ".jsonl"

        # Get timezone
        try:
            self._tz = timezone(timedelta(hours=8))  # Default Asia/Shanghai
            if timezone_name != DEFAULT_TIMEZONE:
                import zoneinfo
                self._tz = zoneinfo.ZoneInfo(timezone_name)
        except Exception:
            self._tz = timezone(timedelta(hours=8))

        self._queue: queue.Queue = queue.Queue()
        self._running = True
        self._worker = threading.Thread(target=self._flush_loop, daemon=True)
        self._worker.start()

    def _get_current_file(self) -> Path:
        """Get the current active log file path."""
        return self._base_dir / f"{self._base_name}{self._suffix}"

    def _get_timestamp(self) -> str:
        """Get current timestamp for rotation naming."""
        now = datetime.now(self._tz)
        return now.strftime("%Y-%m-%dT%H-%M-%S")

    def _format_event(self, event: Dict[str, Any]) -> str:
        """Format event as JSON string."""
        import json
        return json.dumps(event, ensure_ascii=False, sort_keys=False)

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
            self._base_dir.mkdir(parents=True, exist_ok=True)
            log_file = self._get_current_file()

            # Check rotation (size-based)
            if log_file.exists() and log_file.stat().st_size >= self._max_bytes:
                self._rotate()

            # Append to file
            with open(log_file, "a", encoding="utf-8") as f:
                for event in batch:
                    f.write(self._format_event(event) + "\n")

            # Cleanup old files (run occasionally)
            self._cleanup_old_files()

        except Exception as e:
            logger.error("Failed to write audit batch: %s", e)

    def _rotate(self) -> None:
        """Rotate the current log file with timestamp naming and optional compression."""
        current_file = self._get_current_file()
        if not current_file.exists():
            return

        timestamp = self._get_timestamp()
        rotated_name = f"{self._base_name}{self._suffix}.{timestamp}"
        rotated_file = self._base_dir / rotated_name

        try:
            # Read content and write compressed
            if self._compress:
                with open(current_file, "rb") as f_in:
                    with gzip.open(rotated_file.with_suffix(".gz"), "wb") as f_out:
                        f_out.write(f_in.read())
                # Remove original uncompressed file
                current_file.unlink()
            else:
                os.rename(current_file, rotated_file)

            self._chmod_if_needed(rotated_file if not self._compress else rotated_file.with_suffix(".gz"))
            logger.info("Rotated audit log to %s", rotated_file.with_suffix(".gz") if self._compress else rotated_file)
        except Exception as e:
            logger.warning("Failed to rotate audit log file: %s", e)

    def _cleanup_old_files(self) -> None:
        """Remove files older than max_age_hours or exceeding max_backups."""
        try:
            cutoff_time = time.time() - (self._max_age_hours * 3600)
            rotated_files = []

            # Find all rotated files (matching pattern)
            pattern = f"{self._base_name}{self._suffix}.*"
            for f in self._base_dir.glob(pattern):
                rotated_files.append((f.stat().st_mtime, f))

            # Sort by modification time (oldest first)
            rotated_files.sort(key=lambda x: x[0])

            # Check age-based cleanup
            for mtime, f in rotated_files:
                if mtime < cutoff_time:
                    try:
                        f.unlink()
                        logger.debug("Removed old audit log: %s", f)
                    except OSError:
                        pass

            # Check count-based cleanup (keep max_backups most recent)
            if self._max_backups > 0 and len(rotated_files) > self._max_backups:
                # Remove oldest beyond max_backups
                for mtime, f in rotated_files[:-self._max_backups]:
                    try:
                        f.unlink()
                        logger.debug("Removed excess audit log: %s", f)
                    except OSError:
                        pass

        except Exception as e:
            logger.warning("Failed to cleanup old audit log files: %s", e)

    def _chmod_if_needed(self, path: Path) -> None:
        """Apply group-writable permissions in managed mode."""
        try:
            from hermes_cli.config import is_managed
            if is_managed():
                os.chmod(path, 0o660)
        except Exception:
            pass