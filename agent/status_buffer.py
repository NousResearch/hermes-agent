"""Retry/fallback status buffering — 層級：Core Logic Layer (Layer 2)

Extracts buffered status logic from run_agent.py to reduce monolithic size.
Keeps retry/fallback noise from flooding CLI/gateway — messages are only
shown after all retries are exhausted, or silently dropped on success.
"""

import logging
from typing import Callable, List, Tuple, Optional

logger = logging.getLogger(__name__)


class StatusBuffer:
    """Buffer retry/fallback status messages to avoid CLI/gateway noise.

    Stores messages as (kind, text) tuples where kind is:
    - "status" -> replay via emit_status
    - "vprint" -> replay via vprint(force=True)
    - "warn"   -> replay via emit_warning
    """

    def __init__(
        self,
        emit_status: Optional[Callable[[str], None]] = None,
        emit_warning: Optional[Callable[[str], None]] = None,
        vprint: Optional[Callable[[str], None]] = None,
        log_prefix: str = "",
    ):
        self._buffer: List[Tuple[str, str]] = []
        self._emit_status = emit_status
        self._emit_warning = emit_warning
        self._vprint = vprint
        self._log_prefix = log_prefix

    def buffer_status(self, message: str) -> None:
        """Buffer a retry/fallback status message."""
        try:
            self._buffer.append(("status", message))
        except Exception:
            # Never break the retry loop on a buffer hiccup.
            pass

    def buffer_vprint(self, message: str) -> None:
        """Buffer a vprint(force=True) retry/fallback line."""
        try:
            self._buffer.append(("vprint", message))
        except Exception:
            pass

    def buffer_warn(self, message: str) -> None:
        """Buffer a warning retry/fallback line."""
        try:
            self._buffer.append(("warn", message))
        except Exception:
            pass

    def clear(self) -> None:
        """Drop buffered retry messages — call on successful recovery."""
        try:
            self._buffer.clear()
        except Exception:
            pass

    def flush(self) -> None:
        """Emit buffered retry messages — call on terminal failure.

        Surfaces the full retry/fallback trace so the user can see what
        was tried before the turn gave up.
        """
        try:
            if not self._buffer:
                return
            # Drain first so a callback exception doesn't double-emit.
            messages = list(self._buffer)
            self._buffer.clear()
            for kind, msg in messages:
                try:
                    if kind == "status" and self._emit_status:
                        self._emit_status(msg)
                    elif kind == "warn" and self._emit_warning:
                        self._emit_warning(msg)
                    elif kind == "vprint" and self._vprint:
                        self._vprint(f"{self._log_prefix}{msg}", force=True)
                except Exception:
                    pass
        except Exception:
            pass

    @property
    def has_buffered_messages(self) -> bool:
        """Check if there are buffered messages."""
        return len(self._buffer) > 0