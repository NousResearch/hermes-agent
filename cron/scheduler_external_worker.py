"""Diagnostics for restart-safe external cron workers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# This is an internal child-launch marker, not a user configuration variable.
EXTERNAL_WORKER_STDERR_CAPTURE_ENV = "_HERMES_CRON_CAPTURE_STDERR"
_EXTERNAL_WORKER_STDERR_TAIL_BYTES = 16 * 1024


def report_external_worker_stderr(stderr_path: Path, *, execution_id: str) -> None:
    """Log a bounded, always-redacted tail of pre-ack worker stderr.

    The parent owns cleanup of *stderr_path*. Missing diagnostics are normal when
    a worker exits cleanly or emits nothing, so they are not logged as errors.
    """
    try:
        with stderr_path.open("rb") as stderr_file:
            stderr_file.seek(0, os.SEEK_END)
            size = stderr_file.tell()
            if size == 0:
                return
            stderr_file.seek(max(0, size - _EXTERNAL_WORKER_STDERR_TAIL_BYTES))
            captured = stderr_file.read(_EXTERNAL_WORKER_STDERR_TAIL_BYTES)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning(
            "Could not read captured stderr for cron external worker %s (%s)",
            execution_id,
            type(exc).__name__,
        )
        return

    try:
        from agent.redact import redact_sensitive_text

        text = captured.decode("utf-8", errors="replace")
        if size > _EXTERNAL_WORKER_STDERR_TAIL_BYTES:
            text = (
                f"[stderr truncated; showing the last {_EXTERNAL_WORKER_STDERR_TAIL_BYTES} bytes]\n"
                + text
            )
        # Keep tracebacks readable while defanging terminal-control bytes before
        # they enter the structured log stream.
        text = "".join(
            char if char in "\n\t" or ord(char) >= 0x20 else f"\\x{ord(char):02x}"
            for char in text
        )
        redacted = redact_sensitive_text(text, force=True)
    except Exception as exc:
        # Never emit an unredacted fallback when the redactor itself is unavailable.
        logger.warning(
            "Could not redact captured stderr for cron external worker %s (%s); "
            "diagnostic omitted",
            execution_id,
            type(exc).__name__,
        )
        return

    if redacted and redacted.strip():
        logger.error(
            "Cron external worker %s emitted stderr before ownership acknowledgement:\n%s",
            execution_id,
            redacted.rstrip(),
        )


def disable_external_worker_stderr_capture() -> None:
    """Return child fd 2 to the pre-existing detached-worker null sink."""
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, 2)
        finally:
            os.close(devnull_fd)
    except (OSError, ValueError) as exc:
        # Diagnostics are best effort; the worker must still finish its durable
        # execution rather than fail solely because stderr could not be detached.
        logger.debug(
            "Could not disable external worker stderr capture (%s)",
            type(exc).__name__,
        )
