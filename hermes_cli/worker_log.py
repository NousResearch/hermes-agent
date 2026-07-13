"""Redacting text streams for durable Kanban worker logs."""

from __future__ import annotations

import atexit
import sys
import threading
from typing import Callable, TextIO

from agent.redact import redact_sensitive_text


def _redact_worker_text(text: str) -> str:
    # Worker logs are durable security boundaries, like debug captures and
    # session exports. Never persist a reusable secret even if display
    # redaction was disabled for the interactive UI.
    return redact_sensitive_text(text, force=True)


class RedactingTextStream:
    """Line-buffered stream that redacts secrets before writing them.

    Buffering complete lines keeps credentials split across adjacent
    ``write()`` calls from escaping detection. Private-key blocks are held
    until their closing marker so the shared multiline redactor sees the
    complete value.
    """

    def __init__(
        self,
        stream: TextIO,
        *,
        redact: Callable[[str], str] = _redact_worker_text,
    ) -> None:
        self._stream = stream
        self._redact = redact
        self._pending = ""
        self._private_key = ""
        self._lock = threading.RLock()
        self._finalized = False

    @staticmethod
    def _starts_private_key(text: str) -> bool:
        return "-----BEGIN" in text and "PRIVATE KEY-----" in text

    @staticmethod
    def _ends_private_key(text: str) -> bool:
        return "-----END" in text and "PRIVATE KEY-----" in text

    def _emit_line(self, line: str) -> None:
        if self._private_key:
            self._private_key += line
            if self._ends_private_key(line):
                self._stream.write(self._redact(self._private_key))
                self._private_key = ""
            return

        if self._starts_private_key(line):
            self._private_key = line
            if self._ends_private_key(line):
                self._stream.write(self._redact(self._private_key))
                self._private_key = ""
            return

        self._stream.write(self._redact(line))

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        with self._lock:
            self._pending += text
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                self._emit_line(line + "\n")
        return len(text)

    def flush(self) -> None:
        # Keep an incomplete line buffered: it may be the first half of a
        # credential split across writes. ``finalize`` drains it at shutdown.
        with self._lock:
            self._stream.flush()

    def finalize(self) -> None:
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            if self._private_key:
                # An unterminated key is still secret. Do not depend on the
                # complete-block regex matching malformed/truncated output.
                self._stream.write("[REDACTED INCOMPLETE PRIVATE KEY]\n")
                self._private_key = ""
                self._pending = ""
            elif self._pending:
                self._stream.write(self._redact(self._pending))
                self._pending = ""
            self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None)

    @property
    def errors(self):
        return getattr(self._stream, "errors", None)

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def install_worker_log_redaction() -> None:
    """Wrap process output once for a dispatcher-spawned Kanban worker."""
    if not isinstance(sys.stdout, RedactingTextStream):
        sys.stdout = RedactingTextStream(sys.stdout)
        atexit.register(sys.stdout.finalize)
    if not isinstance(sys.stderr, RedactingTextStream):
        sys.stderr = RedactingTextStream(sys.stderr)
        atexit.register(sys.stderr.finalize)
