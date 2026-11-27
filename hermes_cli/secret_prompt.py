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
            # A non-text key: each reader reports a whole navigation/delete key or paste marker
            # as one ESC (POSIX: _posix_key_reader; Windows: the scan-code pair), so none of it
            # becomes secret text.
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


def _posix_key_reader(read: Callable[[], str]) -> Callable[[], str]:
    """Wrap a raw one-char terminal reader so each escape sequence comes back as one ``"\\x1b"``.

    A raw-mode terminal sends navigation keys as multi-char sequences — CSI (``ESC [`` params
    final: arrows, Home/End, Delete, the bracketed-paste markers ``ESC[200~``/``ESC[201~``; the
    Linux console's F1-F5 are ``ESC [ [ A``-``E``) and SS3 (``ESC O`` + one char: application-mode
    arrows, F1-F4). Consuming them whole keeps their tails out of the secret; text pasted between
    the paste markers still arrives as typed input. After an ESC that starts neither (a lone ESC,
    Alt+key), the next char is ordinary input, and so is a char that cannot end the sequence
    (EOF, Ctrl+C, Enter after Alt+O, ...) — it is handed back, never swallowed.
    """
    pending: list[str] = []

    def end_sequence(ch: str) -> None:
        if not "\x40" <= ch <= "\x7e":  # not a final byte: keep it as input
            pending.append(ch)

    def read_key() -> str:
        ch = pending.pop() if pending else read()
        if ch != "\x1b":
            return ch
        intro = read()
        if intro == "[":
            ch = read()
            if ch == "[":  # Linux console function key
                ch = read()
            while "\x20" <= ch <= "\x3f":  # parameter and intermediate bytes
                ch = read()
            end_sequence(ch)
        elif intro == "O":
            end_sequence(read())
        else:
            pending.append(intro)
        return "\x1b"

    return read_key


def _masked_secret_prompt_posix(prompt: str, *, mask: str) -> str:
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        read_key = _posix_key_reader(lambda: sys.stdin.read(1))
        return _collect_masked_input(read_key, _write, prompt, mask=mask)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
