"""Secret input prompts with masked typing feedback."""

from __future__ import annotations

import getpass
import os
import sys
from collections.abc import Callable


_BACKSPACE_CHARS = {"\b", "\x7f"}
_ENTER_CHARS = {"\r", "\n"}
_EOF_CHARS = {"\x04", "\x1a", ""}  # "" == stream closed


def _collect_masked_input(
    read_char: Callable[[], str], write: Callable[[str], object], prompt: str, *, mask: str = "*",
) -> str:
    """Read one secret line while writing a mask character per typed char."""
    value: list[str] = []
    write(prompt)
    while True:
        ch = read_char()
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
            # Terminals send escape-prefixed navigation/delete sequences; they must not become
            # secret text.
            continue
        value.append(ch)
        if mask:
            write(mask)


def masked_secret_prompt(prompt: str, *, mask: str = "*") -> str:
    """Prompt for a secret while showing masked typing feedback.

    Falls back to ``getpass.getpass`` when stdin/stdout are not interactive or when raw terminal
    handling is unavailable.
    """
    if not _stream_is_tty(sys.stdin) or not _stream_is_tty(sys.stdout):
        return getpass.getpass(prompt)
    masked = _masked_secret_prompt_windows if os.name == "nt" else _masked_secret_prompt_posix
    try:
        return masked(prompt, mask=mask)
    except (KeyboardInterrupt, EOFError):
        raise
    except Exception:
        return getpass.getpass(prompt)


def _stream_is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _masked_secret_prompt_windows(prompt: str, *, mask: str) -> str:
    import msvcrt

    def read_char() -> str:
        ch = msvcrt.getwch()
        if ch in {"\x00", "\xe0"}:
            msvcrt.getwch()
            return "\x1b"
        return ch

    return _collect_masked_input(read_char, _write, prompt, mask=mask)


def _masked_secret_prompt_posix(prompt: str, *, mask: str) -> str:
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return _collect_masked_input(lambda: sys.stdin.read(1), _write, prompt, mask=mask)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
