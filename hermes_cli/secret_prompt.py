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
    from the escape-sequence drain below. When the first byte is ESC (``\\x1b``),
    any bytes belonging to a CSI/SS3 escape sequence (arrow keys, function
    keys, Home/End, PageUp/PageDown, etc.) are drained with a short ``select``
    poll so their tail (e.g. ``b"[A"`` for arrow-up) does not leak into the
    caller's secret buffer. Every keystroke is masked, so without the drain
    the corruption would be silent — the user would just see wrong-password
    errors and no clue why. A 5 ms window is longer than any local terminal
    takes to burst these bytes and shorter than a plausible human ESC-then-
    typed-key interval.

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
    if b == b"\x1b":
        while select.select([fd], [], [], 0.005)[0]:
            try:
                if not os.read(fd, 1):
                    break
            except OSError:
                break
        return "\x1b"
    buf = bytearray(b)
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
