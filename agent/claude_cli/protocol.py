"""Stream-json NDJSON parser for `claude --print --output-format stream-json`.

Pure parser — no I/O, no subprocess management. Consumes raw bytes from
the subprocess stdout in arbitrary chunks and yields parsed JSON events
one line at a time. The parser is line-oriented; chunk boundaries are
buffered until a complete newline-terminated line is seen.

Failure modes raise ``ProtocolError``:
  * malformed JSON on a complete line
  * line exceeds ``max_line_bytes``
  * total events seen exceeds ``max_events``
  * persistent non-JSON content (banners, debug logs) on stdout

The parser refuses further input after raising ProtocolError so the
caller cannot silently splice unrelated events after a protocol desync.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from agent.claude_cli.errors import ProtocolError

logger = logging.getLogger(__name__)

DEFAULT_MAX_LINE_BYTES = 1_000_000
DEFAULT_MAX_EVENTS = 10_000
DEFAULT_NON_JSON_TOLERATED = 3


class StreamJsonParser:
    """Incremental NDJSON parser.

    Usage::

        parser = StreamJsonParser()
        for event in parser.feed(chunk):
            handle(event)
        for event in parser.close():
            handle(event)
    """

    def __init__(
        self,
        *,
        max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
        max_events: int = DEFAULT_MAX_EVENTS,
        non_json_tolerated: int = DEFAULT_NON_JSON_TOLERATED,
    ) -> None:
        self._buffer = bytearray()
        self._max_line_bytes = max_line_bytes
        self._max_events = max_events
        self._non_json_tolerated = non_json_tolerated
        self._event_count = 0
        self._non_json_count = 0
        self._failed = False

    def feed(self, chunk: bytes) -> Iterator[dict[str, Any]]:
        """Append ``chunk`` to the buffer and yield each complete line's event."""
        if self._failed:
            raise ProtocolError("parser is in failed state; no further input accepted")
        self._buffer.extend(chunk)
        while True:
            newline_index = self._buffer.find(b"\n")
            if newline_index == -1:
                if len(self._buffer) > self._max_line_bytes:
                    self._failed = True
                    raise ProtocolError(
                        f"line buffer exceeded {self._max_line_bytes} bytes without newline"
                    )
                return
            line = bytes(self._buffer[:newline_index])
            del self._buffer[: newline_index + 1]
            yield from self._parse_line(line)

    def close(self) -> Iterator[dict[str, Any]]:
        """Flush any pending complete line; do NOT emit an incomplete trailing line.

        A partial line at EOF is silently dropped — it could be a crashed
        process that wrote half a line and exited. Logged at debug level.
        """
        if self._failed:
            return
        if not self._buffer:
            return
        # Defensive: drain any complete lines still in the buffer (should not
        # happen — feed() drains them already, but a future caller might call
        # close() without first calling feed() to completion).
        while b"\n" in self._buffer:
            logger.debug("close() draining unexpected complete line from buffer")
            newline_index = self._buffer.find(b"\n")
            line = bytes(self._buffer[:newline_index])
            del self._buffer[: newline_index + 1]
            yield from self._parse_line(line)
        if self._buffer:
            logger.debug(
                "close() dropping %d bytes of partial trailing line", len(self._buffer)
            )
            self._buffer.clear()

    def _parse_line(self, line: bytes) -> Iterator[dict[str, Any]]:
        line = line.rstrip(b"\r")  # tolerate CRLF
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            self._non_json_count += 1
            logger.debug(
                "non-JSON line on stdout (%d/%d tolerated): %r",
                self._non_json_count,
                self._non_json_tolerated,
                line[:200],
            )
            if self._non_json_count > self._non_json_tolerated:
                self._failed = True
                raise ProtocolError(
                    f"persistent non-JSON content on stdout "
                    f"({self._non_json_count} lines, tolerated {self._non_json_tolerated})"
                )
            return
        if not isinstance(event, dict):
            self._failed = True
            raise ProtocolError(
                f"stream-json event was not a JSON object: {type(event).__name__}"
            )
        self._event_count += 1
        if self._event_count > self._max_events:
            self._failed = True
            raise ProtocolError(
                f"event count exceeded {self._max_events}"
            )
        yield event
