"""Answer a bounded set of blocking terminal queries from PTY subprocesses.

Programs running inside a background PTY session (``terminal(background=true,
pty=true)``) sometimes probe their "terminal" with ANSI queries — device
status reports (``ESC[5n``), window-size queries (``ESC[18t``), cursor
position reports (``ESC[6n``), or DEC private-mode queries
(``ESC[?<mode>$p``). A real terminal emulator answers these on stdin; Hermes'
PTY has no emulator on the master side, so the subprocess either blocks
forever waiting for a reply or the raw query bytes leak into the captured
output as garbage.

This module ports openai/codex#41436 (``terminal_queries.rs``): a tiny
byte-level state machine that scans PTY output for the handled queries,
strips them from the output stream (queries split across read chunks
included), and produces the bounded responses to write back to the
subprocess. Everything else — colors, other escape sequences, partial
UTF-8 — passes through untouched.

Only the POSIX ``ptyprocess`` path uses this. On Windows, ConPTY is a real
console host that answers queries itself.
"""

from __future__ import annotations

_ESC = 0x1B

# DEC private-mode queries carry a numeric mode of bounded length; anything
# longer is not a query we answer (and is passed through untouched).
_MAX_MODE_DIGITS = 10
# Longest handled sequence: ESC [ ? <digits> $ p
_MAX_QUERY_BYTES = _MAX_MODE_DIGITS + 5


class PtyQueryResponder:
    """Incremental scanner for terminal queries in a PTY output stream.

    Feed raw output chunks through :meth:`process`; it returns the chunk with
    any handled queries removed, plus the response bytes to write to the
    subprocess's stdin. Call :meth:`flush` at end-of-stream to recover any
    trailing partial escape sequence that never completed.
    """

    def __init__(self, rows: int = 24, cols: int = 80):
        # Exact-match queries and their responses. Window size reports the
        # PTY's actual spawn dimensions; cursor position reports home (1;1) —
        # we don't emulate a screen, a bounded answer just unblocks the
        # subprocess (same policy as openai/codex#41436).
        self._query_responses: tuple[tuple[bytes, bytes], ...] = (
            # Device status report: terminal operating normally.
            (b"\x1b[5n", b"\x1b[0n"),
            # Window-size query: report the PTY's row/col text area.
            (b"\x1b[18t", b"\x1b[8;%d;%dt" % (rows, cols)),
            # Cursor-position report: row 1, column 1.
            (b"\x1b[6n", b"\x1b[1;1R"),
        )
        self._pending = bytearray()

    def process(self, data: bytes) -> tuple[bytes, bytes]:
        """Scan ``data``; return ``(output_bytes, response_bytes)``."""
        if not self._pending and _ESC not in data:
            return data, b""

        output = bytearray()
        responses = bytearray()
        pending = self._pending

        for byte in data:
            if not pending and byte != _ESC:
                output.append(byte)
                continue

            if byte == _ESC:
                # A fresh ESC aborts any partial sequence — flush it through.
                output += pending
                pending.clear()
            pending.append(byte)

            if (
                len(pending) == 1
                or bytes(pending) == b"\x1b["
                or (
                    pending[1] == ord("[")
                    and not (0x40 <= byte <= 0x7E)
                    and len(pending) < _MAX_QUERY_BYTES
                )
            ):
                # Still accumulating a possible query.
                continue

            seq = bytes(pending)
            matched = False
            for query, response in self._query_responses:
                if seq == query:
                    responses += response
                    matched = True
                    break
            if not matched:
                mode = seq[3:-2]
                if (
                    seq.startswith(b"\x1b[?")
                    and seq.endswith(b"$p")
                    and 0 < len(mode) <= _MAX_MODE_DIGITS
                    and mode.isdigit()
                ):
                    # DEC private-mode query: report mode as unrecognized.
                    responses += b"\x1b[?" + mode + b";0$y"
                else:
                    # Not a handled query — pass the sequence through.
                    output += pending
            pending.clear()

        return bytes(output), bytes(responses)

    def flush(self) -> bytes:
        """Return any incomplete trailing sequence held back by the scanner."""
        tail = bytes(self._pending)
        self._pending.clear()
        return tail
