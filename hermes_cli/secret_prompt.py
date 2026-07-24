"""Secret input prompts with masked typing feedback."""

from __future__ import annotations

import getpass
import os
import re
import sys
from collections.abc import Callable


_BACKSPACE_CHARS = {"\b", "\x7f"}
_ENTER_CHARS = {"\r", "\n"}
_EOF_CHARS = {"\x04", "\x1a"}
_BRACKETED_PASTE_PATTERN = re.compile(r"(?:\x1b)?\[\s*(?:200|201)~")


def _sanitize_pasted_input(value: str) -> str:
    """Strip leaked bracketed-paste control markers from pasted text."""
    if not isinstance(value, str) or not value:
        return value
    return _BRACKETED_PASTE_PATTERN.sub("", value)


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

    if _should_use_prompt_toolkit_secret_input(stdin, stdout):
        try:
            return _sanitize_pasted_input(_prompt_toolkit_secret_prompt(prompt))
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception:
            pass

    if not _stream_is_tty(stdin) or not _stream_is_tty(stdout):
        return _sanitize_pasted_input(getpass.getpass(prompt))

    if os.name == "nt":
        try:
            return _sanitize_pasted_input(_masked_secret_prompt_windows(prompt, mask=mask))
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception:
            return _sanitize_pasted_input(getpass.getpass(prompt))

    try:
        return _sanitize_pasted_input(_masked_secret_prompt_posix(prompt, mask=mask))
    except (KeyboardInterrupt, EOFError):
        raise
    except Exception:
        return _sanitize_pasted_input(getpass.getpass(prompt))


def _stream_is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _should_use_prompt_toolkit_secret_input(stdin, stdout) -> bool:
    """Use prompt_toolkit for interactive Windows secret prompts.

    On Windows, prompt_toolkit preserves hidden input while allowing paste
    paths that raw ``msvcrt.getwch()`` and ``getpass`` commonly miss.
    """
    return sys.platform == "win32" and _stream_is_tty(stdin) and _stream_is_tty(stdout)


def _prompt_toolkit_secret_prompt(prompt: str) -> str:
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.shortcuts import prompt as pt_prompt

    kb = KeyBindings()

    @kb.add("c-v")
    def _paste_clipboard(event):
        try:
            clip = event.app.clipboard.get_data()
        except Exception:
            return
        text = _sanitize_pasted_input(getattr(clip, "text", "") or "")
        if text:
            event.current_buffer.insert_text(text)

    @kb.add(Keys.BracketedPaste, eager=True)
    def _handle_bracketed_paste(event):
        text = _sanitize_pasted_input(event.data or "")
        if text:
            event.current_buffer.insert_text(text)

    return pt_prompt(
        ANSI(prompt),
        is_password=True,
        key_bindings=kb,
    )


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


def _masked_secret_prompt_posix(prompt: str, *, mask: str) -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)

    def read_char() -> str:
        return sys.stdin.read(1)

    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        return _collect_masked_input(read_char, write, prompt, mask=mask)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
