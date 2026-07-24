"""Secret input prompts with masked typing feedback."""

from __future__ import annotations

import getpass
import os
import sys
from collections.abc import Callable


_BACKSPACE_CHARS = {"\b", "\x7f"}
_ENTER_CHARS = {"\r", "\n"}
_EOF_CHARS = {"\x04", "\x1a"}


def _collect_masked_input(
    read_char: Callable[[], str],
    write: Callable[[str], object],
    prompt: str,
    *,
    mask: str = "*",
) -> str:
    """Read one secret line while writing a mask character per typed char."""
    value: list[str] = []
    write(prompt)

    while True:
        ch = read_char()
        if ch == "":
            write("\r\n")
            raise EOFError
        if ch in _ENTER_CHARS:
            write("\r\n")
            return "".join(value)
        if ch == "\x03":
            write("\r\n")
            raise KeyboardInterrupt
        if ch in _EOF_CHARS:
            write("\r\n")
            raise EOFError
        if ch in _BACKSPACE_CHARS:
            if value:
                value.pop()
                write("\b \b")
            continue
        if ch == "\x1b":
            # Ignore escape itself. Terminals commonly send escape-prefixed
            # navigation/delete sequences; they should not become secret text.
            continue

        value.append(ch)
        if mask:
            write(mask)


def masked_secret_prompt(prompt: str, *, mask: str = "*") -> str:
    """Prompt for a secret while showing masked typing feedback.

    Falls back to ``getpass.getpass`` when stdin/stdout are not interactive or
    when raw terminal handling is unavailable.
    """
    stdin = sys.stdin
    stdout = sys.stdout

    if not _stream_is_tty(stdin) or not _stream_is_tty(stdout):
        return getpass.getpass(prompt)

    if os.name == "nt":
        try:
            return _masked_secret_prompt_windows(prompt, mask=mask)
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception:
            return getpass.getpass(prompt)

    try:
        return _masked_secret_prompt_posix(prompt, mask=mask)
    except (KeyboardInterrupt, EOFError):
        raise
    except Exception:
        return getpass.getpass(prompt)


def _stream_is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _masked_secret_prompt_windows(prompt: str, *, mask: str) -> str:
    import msvcrt

    def read_char() -> str:
        ch = msvcrt.getwch()
        if ch in {"\x00", "\xe0"}:
            msvcrt.getwch()
            return "\x1b"
        return ch

    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    return _collect_masked_input(read_char, write, prompt, mask=mask)


def _read_posix_raw_char(fd: int) -> str:
    """Read one Unicode character from *fd* using ``os.read``.

    Bypasses ``sys.stdin``'s ``TextIOWrapper`` buffer so it can't hide bytes
    from the escape-sequence handling below.

    When the first byte is ESC (``\\x1b``), a follow-up byte is peeked for
    within a short window:

    * If it introduces a **CSI** (``\\x1b[``) or **SS3** (``\\x1bO``) sequence
      (arrow keys, function keys, Home/End, PageUp/PageDown, etc.), the rest
      of the sequence is consumed and ``"\\x1b"`` is returned — the collector
      then skips the ESC. Every keystroke is masked, so leaving the tail bytes
      in the buffer would silently corrupt the secret.
    * If the follow-up byte is **not** a CSI/SS3 introducer, it's a Meta/Alt-
      encoded key (``ESC + <char>``) or a paste that happens to start with
      ESC. The follow-up byte is returned as an ordinary character so it
      isn't silently dropped along with the ESC.
    * If no follow-up byte arrives promptly, this is a lone Escape keypress;
      ``"\\x1b"`` is returned and the collector skips it.

    Sequence-tail reads use a small ``select`` window (5 ms) rather than a
    blocking ``os.read`` so a malformed sequence can't hang the prompt.

    Multi-byte UTF-8 code points are reassembled by reading continuation
    bytes until the accumulated buffer decodes cleanly (max 4 bytes per
    code point).
    """
    import select

    try:
        b = os.read(fd, 1)
    except OSError:
        return ""
    if not b:
        return ""
    if b != b"\x1b":
        return _decode_utf8_char(fd, b)

    if not select.select([fd], [], [], 0.005)[0]:
        return "\x1b"
    try:
        nxt = os.read(fd, 1)
    except OSError:
        return "\x1b"
    if not nxt:
        return "\x1b"

    if nxt == b"[":
        # CSI: parameter bytes (0x30-0x3F) and intermediates (0x20-0x2F),
        # terminated by a final byte in 0x40-0x7E. Bounded to avoid hanging
        # on a malformed sequence.
        for _ in range(32):
            if not select.select([fd], [], [], 0.005)[0]:
                break
            try:
                tail = os.read(fd, 1)
            except OSError:
                break
            if not tail:
                break
            if 0x40 <= tail[0] <= 0x7E:
                break
        return "\x1b"

    if nxt == b"O":
        # SS3: exactly one final byte follows the introducer.
        if select.select([fd], [], [], 0.005)[0]:
            try:
                os.read(fd, 1)
            except OSError:
                pass
        return "\x1b"

    return _decode_utf8_char(fd, nxt)


def _decode_utf8_char(fd: int, first: bytes) -> str:
    """Decode one UTF-8 code point starting with *first*, pulling continuation
    bytes from *fd* as needed."""
    buf = bytearray(first)
    for _ in range(3):
        try:
            return buf.decode("utf-8")
        except UnicodeDecodeError:
            pass
        try:
            more = os.read(fd, 1)
        except OSError:
            break
        if not more:
            break
        buf.extend(more)
    return buf.decode("utf-8", errors="replace")


def _masked_secret_prompt_posix(prompt: str, *, mask: str) -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)

    def read_char() -> str:
        return _read_posix_raw_char(fd)

    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        return _collect_masked_input(read_char, write, prompt, mask=mask)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
