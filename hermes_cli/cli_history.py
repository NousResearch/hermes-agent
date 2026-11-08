"""Prompt history that persists undecodable input bytes losslessly.

prompt_toolkit decodes terminal stdin with ``errors="surrogateescape"``: a byte
that is not part of valid UTF-8 (e.g. 0x94 from a terminal or keyboard layout
sending a legacy single-byte charset) arrives in the buffer as the lone
surrogate U+DC94. The stock ``FileHistory.store_string`` encodes with strict
UTF-8, so pressing Enter on such a line raises ``UnicodeEncodeError`` inside the
key handler and the line is never written to the history file.

``SurrogateSafeFileHistory`` writes those code points back as the original
bytes (the inverse of the stdin decoding) and reads the file with the same
error handler, so the text round-trips byte-for-byte instead of being dropped
or replaced.
"""

from __future__ import annotations

import codecs
import datetime

from prompt_toolkit.history import FileHistory

_ERROR_HANDLER = "hermes-history-surrogates"


def _encode_surrogates(exc: UnicodeError) -> tuple[bytes, int]:
    """Encode lone surrogates without losing information.

    U+DC80..U+DCFF are escaped raw bytes and become that byte again. Any other
    lone surrogate (e.g. half of an emoji pair from a broken paste) is kept via
    its ``surrogatepass`` byte sequence rather than being discarded.
    """
    if not isinstance(exc, UnicodeEncodeError):
        raise exc
    out = bytearray()
    for char in exc.object[exc.start:exc.end]:
        code = ord(char)
        if 0xDC80 <= code <= 0xDCFF:
            out.append(code - 0xDC00)
        else:
            out += char.encode("utf-8", "surrogatepass")
    return bytes(out), exc.end


codecs.register_error(_ERROR_HANDLER, _encode_surrogates)


def encode_history_text(text: str) -> bytes:
    """UTF-8 encode ``text``, restoring escaped stdin bytes as raw bytes."""
    return text.encode("utf-8", _ERROR_HANDLER)


class SurrogateSafeFileHistory(FileHistory):
    """``FileHistory`` that never fails on, or discards, undecodable input."""

    def store_string(self, string: str) -> None:
        with open(self.filename, "ab") as f:
            f.write(encode_history_text(f"\n# {datetime.datetime.now()}\n"))
            f.writelines(encode_history_text(f"+{line}\n") for line in string.split("\n"))

    def load_history_strings(self):
        strings: list[str] = []
        lines: list[str] = []

        def add() -> None:
            if lines:
                strings.append("".join(lines)[:-1])

        try:
            with open(self.filename, "rb") as f:
                for line_bytes in f:
                    # surrogateescape mirrors the stdin decoder: bytes that are
                    # not valid UTF-8 come back as the same U+DCxx code points.
                    line = line_bytes.decode("utf-8", errors="surrogateescape")
                    if line.startswith("+"):
                        lines.append(line[1:])
                    else:
                        add()
                        lines = []
                add()
        except FileNotFoundError:
            pass

        return reversed(strings)
