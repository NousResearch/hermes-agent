"""Terminal keyboard protocol detection and enhanced input sequence handling.

Detects Kitty keyboard protocol and modifyOtherKeys support at startup,
normalizes Ctrl+Backspace / Ctrl+Enter / Alt+key escape sequences for
prompt_toolkit, and manages enabling/disabling enhanced keyboard modes.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.named_commands import get_by_name
from prompt_toolkit.keys import Keys

BACKWARD_KILL_WORD = get_by_name("backward-kill-word").handler
CTRL_BACKSPACE_KEYS = (Keys.Escape, Keys.ControlH)
CTRL_BACKSPACE_ESCAPE_SEQUENCES = (
    "\x1b[127;5u",
    "\x1b[8;5u",
    "\x1b[27;5;127~",
    "\x1b[27;5;8~",
)
CTRL_ENTER_KEYS = Keys.ControlJ
KITTY_KEYBOARD_QUERY = "\x1b[?u"
KITTY_KEYBOARD_ENABLE = "\x1b[>1u"
KITTY_KEYBOARD_DISABLE = "\x1b[<u"
MODIFY_OTHER_KEYS_QUERY = "\x1b[?4m"
MODIFY_OTHER_KEYS_ENABLE = "\x1b[>4;2m"
MODIFY_OTHER_KEYS_DISABLE = "\x1b[>4;0m"
DEVICE_ATTRIBUTES_QUERY = "\x1b[c"
MODE_KITTY = "kitty"
MODE_MODIFY_OTHER_KEYS = "modify_other_keys"
QUERY_TIMEOUT_S = 0.25
_KITTY_QUERY_RESPONSE_RE = re.compile(r"\x1b\[\?(\d+)u")
_MODIFY_OTHER_KEYS_RESPONSE_RE = re.compile(r"\x1b\[>4;(\d+)m")
_DEVICE_ATTRIBUTES_RESPONSE_RE = re.compile(r"\x1b\[\?(?:\d+)(?:;\d+)*c")
_CONTROL_KEY_BY_ASCII = {
    ord(c): getattr(Keys, f"Control{c.upper()}")
    for c in "abcdefghijklmnopqrstuvwxyz"
}


@dataclass(frozen=True)
class TerminalKeyboardCapabilities:
    kitty_supported: bool = False
    modify_other_keys_supported: bool = False


@dataclass(frozen=True)
class TerminalKeyboardDetectionResult:
    capabilities: TerminalKeyboardCapabilities
    pending_input: str = ""


def kitty_key_sequence(codepoint: int, modifiers: int) -> str:
    return f"\x1b[{codepoint};{modifiers}u"


def modify_other_keys_sequence(codepoint: int, modifiers: int) -> str:
    return f"\x1b[27;{modifiers};{codepoint}~"


def parse_capabilities(data: str) -> TerminalKeyboardCapabilities:
    """Parse raw terminal responses for keyboard protocol support."""
    kitty_supported = bool(_KITTY_QUERY_RESPONSE_RE.search(data))
    modify_levels = [
        int(match.group(1))
        for match in _MODIFY_OTHER_KEYS_RESPONSE_RE.finditer(data)
    ]
    # Any response (including level 0) proves the terminal understands the
    # protocol; we will explicitly enable level 2 via set_mode().
    return TerminalKeyboardCapabilities(
        kitty_supported=kitty_supported,
        modify_other_keys_supported=bool(modify_levels),
    )


def _strip_query_responses(data: str) -> str:
    """Remove terminal capability probe responses and preserve other input."""
    result = _KITTY_QUERY_RESPONSE_RE.sub("", data)
    result = _MODIFY_OTHER_KEYS_RESPONSE_RE.sub("", result)
    return _DEVICE_ATTRIBUTES_RESPONSE_RE.sub("", result)


def parse_detection_result(
    data: str,
) -> TerminalKeyboardDetectionResult:
    """Parse terminal capability responses and preserve unrelated bytes."""
    return TerminalKeyboardDetectionResult(
        capabilities=parse_capabilities(data),
        pending_input=_strip_query_responses(data),
    )


def select_mode(
    capabilities: TerminalKeyboardCapabilities,
) -> str | None:
    """Pick the best keyboard enhancement mode supported by the terminal."""
    if capabilities.kitty_supported:
        return MODE_KITTY
    if capabilities.modify_other_keys_supported:
        return MODE_MODIFY_OTHER_KEYS
    return None


def write_sequence(
    sequence: str,
    *,
    writer=None,
) -> None:
    """Emit a raw terminal control sequence."""
    if writer is not None:
        writer(sequence)
        return

    stdout = getattr(sys, "__stdout__", None) or sys.stdout
    if stdout is None or not hasattr(stdout, "write") or not hasattr(stdout, "flush"):
        return
    stdout.write(sequence)
    stdout.flush()


def set_mode(
    mode: str | None,
    *,
    enable: bool,
    writer=None,
) -> None:
    """Enable or disable an enhanced terminal keyboard mode."""
    if mode == MODE_KITTY:
        sequence = KITTY_KEYBOARD_ENABLE if enable else KITTY_KEYBOARD_DISABLE
    elif mode == MODE_MODIFY_OTHER_KEYS:
        sequence = MODIFY_OTHER_KEYS_ENABLE if enable else MODIFY_OTHER_KEYS_DISABLE
    else:
        return
    write_sequence(sequence, writer=writer)


def _get_real_stream(name: str):
    """Return the real stdin/stdout if it is a usable TTY, else None."""
    stream = getattr(sys, f"__{name}__", None) or getattr(sys, name, None)
    if stream is None or not hasattr(stream, "isatty"):
        return None
    try:
        if not stream.isatty():
            return None
    except Exception:
        return None
    return stream


def detect_capabilities(
    *,
    timeout_s: float = QUERY_TIMEOUT_S,
) -> TerminalKeyboardDetectionResult:
    """Best-effort terminal capability probe for keyboard enhancement modes."""
    if os.name == "nt":
        return TerminalKeyboardDetectionResult(TerminalKeyboardCapabilities())

    stdin = _get_real_stream("stdin")
    stdout = _get_real_stream("stdout")
    if stdin is None or stdout is None:
        return TerminalKeyboardDetectionResult(TerminalKeyboardCapabilities())

    try:
        import select
        import termios
        import tty
    except Exception:
        return TerminalKeyboardDetectionResult(TerminalKeyboardCapabilities())

    try:
        stdin_fd = stdin.fileno()
        original_attrs = termios.tcgetattr(stdin_fd)
    except Exception:
        return TerminalKeyboardDetectionResult(TerminalKeyboardCapabilities())

    try:
        tty.setcbreak(stdin_fd)
        write_sequence(
            KITTY_KEYBOARD_QUERY + MODIFY_OTHER_KEYS_QUERY + DEVICE_ATTRIBUTES_QUERY
        )

        buf = bytearray()
        decoded = ""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select([stdin_fd], [], [], max(0.0, remaining))
            if not ready:
                continue
            chunk = os.read(stdin_fd, 1024)
            if not chunk:
                break
            buf.extend(chunk)
            decoded = bytes(buf).decode("utf-8", errors="ignore")
            if _DEVICE_ATTRIBUTES_RESPONSE_RE.search(decoded):
                break

        return parse_detection_result(decoded)
    except Exception:
        return TerminalKeyboardDetectionResult(TerminalKeyboardCapabilities())
    finally:
        try:
            termios.tcsetattr(stdin_fd, termios.TCSANOW, original_attrs)
        except Exception:
            pass


def _detect_terminfo_backspace() -> bytes | None:
    """Return the terminal's declared Backspace byte from terminfo, if known."""
    term = os.getenv("TERM")
    if not term:
        return None

    try:
        import curses

        curses.setupterm(term=term)
        return curses.tigetstr("kbs")
    except Exception:
        return None


def _detect_tty_erase() -> bytes | None:
    """Return the current TTY erase byte, if it is available."""
    if os.name == "nt":
        return None

    stdin = _get_real_stream("stdin")
    if stdin is None or not hasattr(stdin, "fileno"):
        return None

    try:
        import termios

        erase = termios.tcgetattr(stdin.fileno())[6][termios.VERASE]
    except Exception:
        return None

    if isinstance(erase, int):
        erase_byte = bytes([erase])
    elif isinstance(erase, (bytes, bytearray)):
        erase_byte = bytes(erase[:1])
    elif isinstance(erase, str):
        erase_byte = erase.encode("latin-1", errors="ignore")[:1]
    else:
        return None

    return erase_byte if erase_byte in (b"\x08", b"\x7f") else None


def install_ctrl_backspace_sequences(
    ansi_sequences: dict[str, Keys | tuple[Keys, ...]] | None = None,
    *,
    terminfo_backspace: bytes | None = None,
) -> None:
    """Teach prompt_toolkit how to recognize Ctrl+Backspace input sequences.

    Modern terminals may emit CSI-u or modifyOtherKeys sequences for
    Ctrl+Backspace. Legacy terminals only expose ``\\x08`` and ``\\x7f``; in
    that case terminfo's ``kbs`` tells us which byte is Backspace so we can
    treat the other byte as the closest available Ctrl+Backspace signal.
    """

    sequences = ANSI_SEQUENCES if ansi_sequences is None else ansi_sequences

    for sequence in CTRL_BACKSPACE_ESCAPE_SEQUENCES:
        sequences[sequence] = CTRL_BACKSPACE_KEYS

    if terminfo_backspace is None:
        terminfo_backspace = _detect_terminfo_backspace()

    tty_erase = _detect_tty_erase()

    # DEL is a common plain Backspace byte in xterm/tmux sessions even when
    # terminfo advertises ^H. Only remap DEL when the active TTY confirms that
    # Backspace is actually ^H.
    if tty_erase == b"\x08":
        sequences["\x7f"] = CTRL_BACKSPACE_KEYS
    elif tty_erase == b"\x7f":
        sequences["\x08"] = CTRL_BACKSPACE_KEYS
    elif terminfo_backspace == b"\x7f":
        sequences["\x08"] = CTRL_BACKSPACE_KEYS


def install_enhanced_sequences(
    ansi_sequences: dict[str, Keys | tuple[Keys, ...]] | None = None,
) -> None:
    """Normalize Kitty/modifyOtherKeys text-key sequences Hermes relies on.

    This keeps existing Hermes bindings working when enhanced keyboard modes are
    enabled, including Alt-based shortcuts and configurable Ctrl/Alt letter
    bindings such as the default voice push-to-talk key.
    """

    sequences = ANSI_SEQUENCES if ansi_sequences is None else ansi_sequences

    for codepoint in range(32, 127):
        char = chr(codepoint)
        alt_key = (Keys.Escape, char)
        sequences[kitty_key_sequence(codepoint, 3)] = alt_key
        sequences[modify_other_keys_sequence(codepoint, 3)] = alt_key

    for codepoint, control_key in _CONTROL_KEY_BY_ASCII.items():
        sequences[kitty_key_sequence(codepoint, 5)] = control_key
        sequences[modify_other_keys_sequence(codepoint, 5)] = control_key

    # Keep enhanced Ctrl+Enter on Hermes' existing multiline binding (`c-j`)
    # instead of letting it collapse to the plain Enter/submit path.
    sequences[kitty_key_sequence(13, 5)] = CTRL_ENTER_KEYS
    sequences[modify_other_keys_sequence(13, 5)] = CTRL_ENTER_KEYS

    for codepoint in (8, 13, 127):
        alt_key = (Keys.Escape, Keys.ControlH if codepoint in (8, 127) else Keys.ControlM)
        sequences[kitty_key_sequence(codepoint, 3)] = alt_key
        sequences[modify_other_keys_sequence(codepoint, 3)] = alt_key


def parse_input_key_presses(data: str) -> list:
    """Parse raw terminal input into prompt_toolkit key presses."""
    if not data:
        return []

    from prompt_toolkit.input.vt100_parser import Vt100Parser

    key_presses = []
    parser = Vt100Parser(key_presses.append)
    parser.feed(data)
    parser.flush()
    return key_presses


def register_word_delete_keybindings(kb: KeyBindings) -> None:
    """Bind terminal word-delete sequences to prompt_toolkit's native handler.

    Many terminals emit Alt/Meta+Backspace or Esc-prefixed delete sequences
    for Ctrl+Backspace-style whole-word deletion. Bind them explicitly so the
    Hermes TUI behaves consistently inside the custom application wrapper.
    """

    @kb.add("escape", "backspace")
    @kb.add("escape", "c-h")
    @kb.add("escape", "delete")
    def _delete_word(event) -> None:
        BACKWARD_KILL_WORD(event)


def install_all() -> None:
    """Install all keyboard sequence mappings into prompt_toolkit's parser."""
    install_ctrl_backspace_sequences()
    install_enhanced_sequences()
